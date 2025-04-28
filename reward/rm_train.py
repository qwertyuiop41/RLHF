import argparse

from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Union, Iterable
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import gc
from datasets import dataset_dict, load_from_disk
# from datasets import Dataset
from tqdm import trange, tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM,
    GPT2Tokenizer,
    GPT2Model,
    Qwen2Config,
    BitsAndBytesConfig,
)
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import sys

from RLHF.reward.rm import RewardModel, RankRewardModel

class RMTrainer():
    def __init__(self,args):
        self.pretrain_path=args.pretrain_path
        self.train_path=args.train_path
        # self.test_path=args.test_path
        self.use_wandb=args.use_wandb
        self.output_dir=args.output_dir


        self.num_epochs=3
        self.eval_steps=50
        self.save_steps=100
        self.batch_size=16
        self.lr=1e-5



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer=AutoTokenizer.from_pretrained(self.pretrain_path)
        # self.tokenizer.padding_side='left'
        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = self.tokenizer.pad_token


        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=False,  # 不使用8bit
        #     load_in_4bit=False,  # 不使用4bit
        #     load_in_2bit=True,   # 启用 2-bit 量化
        #     bnb_2bit_compute_dtype=torch.float16,  # 计算时使用 float16
        #     bnb_2bit_quant_type="nf2"  # `nf2` 量化格式，适用于 LLM
        # )
        # bnb_config=None

        # 设置 LoRA 配置
        lora_config = LoraConfig(
            r=8,  # Rank，越大表示 LoRA 层越大，消耗显存更多
            lora_alpha=16,  # LoRA scaling factor
            lora_dropout=0.05,  # Dropout 防止过拟合
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 只训练注意力层
            bias="none",
            # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
        )

        self.model = RewardModel(self.pretrain_path,lora_config=lora_config).to(self.device)

        if self.use_wandb:
            wandb.init(project="rlhf-reward-model-1.5b-16", 
                       name="reward-model-training", 
                       dir="reward",
                       config={
                            "model": self.model,
                            "lr": self.lr, 
                            "epochs": self.num_epochs,
                            "batch_size": self.batch_size,
                            "eval_steps": self.eval_steps,
                            "save_steps": self.save_steps,
                            "lr": self.lr,
                        })


        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # data=load_dataset("parquet", data_files=self.train_path,split='train',streaming=True)
        train_data=load_dataset("parquet", data_files=self.train_path,split='train',streaming=True).shuffle(seed=42).take(900)

        test_data=load_dataset("parquet", data_files=self.train_path,split='train',streaming=True).shuffle(seed=42).skip(900).take(100)


        

        self.train_dataset=data_prepare(self.tokenizer,train_data,self.device)
        self.test_dataset=data_prepare(self.tokenizer,test_data,self.device)


        self.train_dataloader=DataLoader(dataset=CustomDataset(self.train_dataset),shuffle=True,batch_size=self.batch_size)
        self.test_dataloader=DataLoader(dataset=CustomDataset(self.test_dataset),shuffle=False,batch_size=self.batch_size)




        max_train_steps = self.num_epochs * len(self.train_dataloader)
        warm_steps = int(0.1 * max_train_steps)

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        self.lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )
        

        os.makedirs(self.output_dir, exist_ok=True)


        


        

    def train(self):
        self.model.train()
        global_step = 0
        best_eval_loss = float('inf')
        best_loss = float('inf')
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            
            for batch in progress_bar:
                # 移动数据到设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                _, loss = self.model(batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], labels=batch["chosen_labels"])

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # 更新损失
                epoch_loss += loss.item()

                

                _,loss = self.model(batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], labels=batch["rejected_labels"])

                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # 更新损失
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            

                
                # 记录到 wandb
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                    })
                # 定期保存模型
                if global_step > 0 and global_step % self.save_steps == 0:
                    checkpoint_dir = os.path.join(self.output_dir, f"rank-checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    model_to_save.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    
                    # 保存优化器状态
                    torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                    print(f"Saved checkpoint at step {global_step}")


                # 评估
                if global_step > 0 and global_step % self.eval_steps == 0:
                    eval_results = self.evaluate()
                    print(f"Step {global_step}: Eval Loss = {eval_results['eval_loss']}, MSE = {eval_results['eval_mse']}")
                    
                    if self.use_wandb:
                        wandb.log({
                            "eval_loss": eval_results['eval_loss'],
                            "eval_mse": eval_results['eval_mse'],
                            "global_step": global_step,
                        })
                    
                    # 保存最佳模型
                    if eval_results['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_results['eval_loss']
                        print(f"New best model with eval_loss: {best_eval_loss}")
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        model_to_save.save_pretrained(os.path.join(self.output_dir, "best_model"))
                        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "best_model"))
                
                global_step += 1

            
            # 每个 epoch 结束后的平均损失
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch} average loss: {avg_epoch_loss}")
            
            if self.use_wandb:
                wandb.log({"rank_epoch_avg_loss": avg_epoch_loss, "epoch": epoch})
                # 保存最终模型
        self.tokenizer.save_pretrained(self.output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.output_dir)
        model_to_save.config.save_pretrained(self.output_dir)
        
        # 结束 wandb
        if self.use_wandb:
            wandb.finish()
        




    def compute_loss(self,predict_rewards):
        print("=============rank loss================")
        # predict_rewards的位置越前面的相对分越高
        # loss设置原因见：https://zhuanlan.zhihu.com/p/610147705
        loss, counts = torch.tensor([0]).to(self.device), 0
        # predict_rewards的位置越前面的相对分越高
        for i in range(len(predict_rewards) - 1):  # 遍历所有前项-后项的得分差
            for j in range(i + 1, len(predict_rewards)):
                diff = nn.functional.logsigmoid(predict_rewards[i] - predict_rewards[j])  # sigmoid到0~1之间 log再全变成负的
                print(f"diff:{diff}")
                loss = loss + diff
                counts += 1
        loss = loss / counts
        return -loss  # 要最大化分差，所以要取负数


    def evaluate(self):
        """评估函数，计算验证集上的 MSE 损失"""
        self.model.eval()
        eval_loss = 0
        all_chosen_preds = []
        all_chosen_labels = []
        all_rejected_preds=[]
        all_rejected_labels=[]
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                        
                chosen_out, chosen_loss = self.model(batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], labels=batch["chosen_labels"])
                eval_loss += chosen_loss.item()

                rejected_out, rejected_loss = self.model(batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], labels=batch["rejected_labels"])
                eval_loss += rejected_loss.item()
                

                # 因为 PyTorch 张量需要先转移到 CPU 上才能被转换为 NumPy 数组
                all_chosen_preds.append(chosen_out.squeeze().cpu().numpy().tolist())
                all_chosen_labels.append(batch["chosen_labels"].squeeze().cpu().numpy().tolist())

                all_rejected_preds.append(rejected_out.squeeze().cpu().numpy().tolist())
                all_rejected_labels.append(batch["rejected_labels"].squeeze().cpu().numpy().tolist())
                
        
        avg_loss = eval_loss / len(self.test_dataloader)
        # print(all_chosen_labels)
        # print(all_chosen_preds)
        # exit()
        mse=0
        mse_chosen=0
        mse_rejected=0
        try:
            mse_chosen = mean_squared_error(all_chosen_labels, all_chosen_preds)
            mse+=mse_chosen
            
        except Exception as e:
            print(e)
            print(all_chosen_labels)
            print(all_chosen_preds)
            
        try:
            mse_rejected = mean_squared_error(all_rejected_labels, all_rejected_preds)
            mse+=mse_rejected
        except Exception as e:
            print(e)
            print(all_rejected_labels)
            print(all_rejected_preds)
        

        if self.use_wandb:
            wandb.log({
                "avg_loss": avg_loss,
                "mse": mse,
                "mse_chosen": mse_chosen if mse_chosen>0 else -1 ,
                "mse_rejected": mse_rejected if mse_rejected>0 else -1,
                "all_chosen_preds":all_chosen_preds,
                "all_chosen_preds":all_chosen_preds,
                "all_rejected_preds":all_rejected_preds,
                "all_rejected_preds":all_rejected_preds,
            })
        
        print(f"Eval Loss: {avg_loss}, MSE: {mse}")
        return {"eval_loss": avg_loss, "eval_mse": mse}


    def predict(self,response_lst):
        # response_lst = ["我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是今世界上保存最完好的哺乳动物之一，也是世界自然保护联盟濒危物种红色名录的保护对象之一。在这里，你可以看到全世界最大的熊猫栖息地成都。成都是中国国家林业局直属的国家重点风景名胜区，是国家森林公园、国家湿地公园和国家地质公园的重要组成部分，是全国重点文物保护单位、全国生态文明建设示范区、中国红色旅游名城、国际生态旅游目的地和国际旅游岛建设先进区。地址：四川省成都市绵阳市成华区成都高新技术产业开发区成华大道1号乘车路线：成都绵阳都江堰雅",
        # "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家二级保护动物，是世界上保存最完整的动物种群之一，也是我国第一个国家级自然保护区。我们是四川省的首批国家重点保护野生动物和珍稀动物基金会的成员，被誉为中国动物保护的摇篮和世界生物多样性保护基地，被中国科学院、中华人民共和国国家林业局授予全国生态文明建设示范区称号，被国务院批准为国家森林城市、国际生态旅游目的地。熊猫基地位于成都市双流区东南部，是国家aaaa级旅游景区，国家地理标志保护单位。熊猫栖息地为亚热带或热带的高山",]

        self.model.eval()
        data = self.tokenizer.batch_encode_plus(response_lst, max_length=512, padding="max_length", truncation=True,return_tensors='pt')
        
        # 移动数据到设备
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        
        with torch.no_grad():
            score = self.model(**data)
        
        return score




class CustomDataset(Dataset):               
    def __init__(self, sample):
        super(CustomDataset, self).__init__()
        self.sample = sample

    def __getitem__(self, item):
        res = {k: v[item] for k, v in self.sample.items()}
        return res

    def __len__(self):
        return len(self.sample['chosen_input_ids'])

def data_prepare(tokenizer,data_lst,device):
    train_data = {} 


    chosen_lst=[data['chosen']for data in data_lst]
    rejected_lst=[data['rejected']for data in data_lst]
    chosen_score=[data['chosen_score']for data in data_lst]
    rejected_score=[data['rejected_score']for data in data_lst]



    chosen_data= tokenizer.batch_encode_plus(chosen_lst, max_length=512, padding="longest", truncation=True,return_tensors='pt').to(device) 
    rejected_data=tokenizer.batch_encode_plus(rejected_lst, max_length=512, padding="longest", truncation=True,return_tensors='pt').to(device) 


    train_data["chosen_input_ids"]=chosen_data["input_ids"]
    train_data["rejected_input_ids"]=rejected_data["input_ids"]
    train_data["chosen_attention_mask"]=chosen_data["attention_mask"]
    train_data["rejected_attention_mask"]=rejected_data["attention_mask"]
    train_data["chosen_labels"] = torch.tensor(chosen_score)
    train_data["rejected_labels"] = torch.tensor(rejected_score)


    return train_data




if __name__=="__main__":
    os.environ["WANDB_MODE"] = "offline"
    parser = argparse.ArgumentParser()

    

    # Models
    parser.add_argument("--pretrain_path", type=str, default='models/Qwen/Qwen2.5-1.5B-Instruct')
    # Dataset
    parser.add_argument("--train_path",default='dataset/preference_dataset_mixture2_and_safe_pku/train.parquet')
    # parser.add_argument("--test_path", default='dataset/hh_rlhf_cn/test.parquet')
    #wandb
    parser.add_argument("--use_wandb", default=True)
    #outputsI
    parser.add_argument("--output_dir", default='reward/ckpt')


    args=parser.parse_args()
    

    rmTrainer=RMTrainer(args)
    rmTrainer.train()
