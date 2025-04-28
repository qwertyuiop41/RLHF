import argparse
import random
import re
import time
import numpy
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader,Dataset

from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model,DataCollatorWithPadding,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
import wandb
from pathlib import Path

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

from tqdm import tqdm, trange
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
    Qwen2Config
)


import sys
sys.path.append("./")


# from policy.policy import PolicyModel
from policy.policy import PolicyModel
from policy.value import ValueModel
from reward.rm import RewardModel
from grpo.gms8k_reward import format_reward,correctness_reward



class DPOTrainer():
    def __init__(self,args):
        self.use_wandb=args.use_wandb
        pretrain_path=args.pretrain_path
        train_path=args.train_path
        test_path=args.test_path
        self.device=args.device
        self.use_wandb=args.use_wandb


        self.max_answer_seq_len=512
        self.lr=1e-5
        self.save_steps=240
        self.eval_steps=60
        self.epoch=3
        self.batch_size=4
        self.best_reward = float('-inf')
        self.warmup_ratio=0.0
        self.beta=0.1 # 一般在0.1到0.5之间
        self.accumulation_steps=10

        

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


        train_data=load_dataset("parquet", data_files=train_path,split='train',streaming=True).shuffle(seed=42).take(300)
        test_data=load_dataset("parquet", data_files=test_path,split='train',streaming=True).shuffle(seed=42).take(30)

        
        self.tokenizer=AutoTokenizer.from_pretrained(pretrain_path)
        self.tokenizer.padding_side='left'


        self.train_dataset=data_prepare(self.tokenizer,train_data,self.device)


        self.test_dataset=data_prepare(self.tokenizer,test_data,self.device)

        self.train_dataloader=DataLoader(dataset=CustomDataset(self.train_dataset),shuffle=True,batch_size=self.batch_size)
        self.test_dataloader=DataLoader(dataset=CustomDataset(self.test_dataset),shuffle=False,batch_size=self.batch_size)

        
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
            r=2,  # Rank，越大表示 LoRA 层越大，消耗显存更多
            lora_alpha=4,  # LoRA scaling factor
            lora_dropout=0.1,  # Dropout 防止过拟合
            target_modules=["q_proj", "v_proj"],  # 只训练注意力层
            bias="none",
            # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
        )


        self.policy_model=PolicyModel(pretrain_path,lora_config).to(self.device)
        self.ref_model=deepcopy(self.policy_model).to(self.device)

        # self.policy_model.cuda().half() 
        # self.ref_model.cuda().half()



        


        # 初始化wandb
        # if args.use_wandb:
        wandb.init(
            project='rlhf-dpo-1.5b-4',
            name=f"dpo-{time.strftime('%Y%m%d-%H%M%S')}",
            dir="dpo",
            config={
                "policy_model": args.pretrain_path,
                "lora_config": lora_config,
                "max_answer_seq_len": self.max_answer_seq_len,
                "lr": self.lr,
                "save_steps": self.save_steps,
                "eval_steps": self.eval_steps,
                "epoch": self.epoch,
                "batch_size": self.batch_size,
                "accumulation_steps":self.accumulation_steps,
            }
        )



        max_train_steps = self.epoch * len(self.train_dataloader)
        warm_steps = int(self.warmup_ratio * max_train_steps)


        no_decay = ["bias", "LayerNorm.weight"]
        self.policy_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.policy_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.policy_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.policy_optimizer = torch.optim.AdamW(self.policy_optimizer_grouped_parameters, lr=self.lr)

        
        self.policy_lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.policy_optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )








    
    def learn(self):
        """
        epoch轮学习
        """
        print("==============dpo learn==============")
        global_step=0
        self.policy_model.train()
        self.ref_model.eval()
        for i in range(self.epoch):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Train epoch [{i + 1}/{self.epoch}]",
            )
            policy_loss_sum = 0
            batch_count = 0
            self.policy_optimizer.zero_grad()
            for batch in pbar:
                policy_loss=self.step(batch)
                
                # # self.policy_model.backward(policy_loss)
                # if (global_step + 1) % self.accumulation_steps == 0:
                #     self.policy_optimizer.step()
                #     self.policy_lr_scheduler.step()

                #     self.policy_optimizer.zero_grad()

                policy_loss_sum += policy_loss.item()


                batch_count += 1
                global_step+=1
                
                # 更新进度条信息
                pbar.set_postfix({
                    'policy_loss': policy_loss.item()
                })
                
                # 记录到wandb
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "learning_rate": self.policy_lr_scheduler.get_last_lr()[0],
                })
                                # 定期保存模型
                if global_step > 0 and global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-steps-{global_step}")

                

                if global_step > 0 and global_step % self.eval_steps == 0:
                    # 在测试集上评估
                    test_reward = self.evaluate()
                    # 保存最佳模型
                    if test_reward > self.best_reward:
                        self.best_reward = test_reward
                        self.save_checkpoint("best_model")
                        print(f"New best model with reward: {test_reward:.4f}")

            avg_policy_loss = policy_loss_sum / max(1, batch_count)

            print(f"Epoch {i+1}/{self.epoch} - Avg Policy Loss: {avg_policy_loss:.4f}")
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": i+1,
                    "avg_policy_loss": avg_policy_loss,
                })

            # 在测试集上评估
            test_reward = self.evaluate()
            
            # 保存模型
            if (i+1) % self.save_steps == 0 or i == self.epoch - 1:
                self.save_checkpoint(f"checkpoint-epoch-{i+1}")
            
            # 保存最佳模型
            if test_reward > self.best_reward:
                self.best_reward = test_reward
                self.save_checkpoint("best_model")
                print(f"New best model with reward: {test_reward:.4f}")

            
        wandb.finish()






    def step(self, batch):
        """
        单步学习
        """
        chosen_input_ids=batch["chosen_input_ids"]
        chosen_attention_mask=batch["chosen_attention_mask"]
        rejected_input_ids=batch["rejected_input_ids"]
        rejected_attention_mask=batch["rejected_attention_mask"]
        prompt_ids=batch["input_ids"]

        policy_chosen_logits=self.policy_model(chosen_input_ids,chosen_attention_mask).logits
        policy_rejected_logits=self.policy_model(rejected_input_ids,rejected_attention_mask).logits

        with torch.no_grad():
            ref_chosen_logits=self.ref_model(chosen_input_ids,chosen_attention_mask).logits
            ref_rejected_logits=self.ref_model(rejected_input_ids,rejected_attention_mask).logits
            

        
        # policy_chosen_logps=gather_log_probs(policy_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
        # ref_chosen_logps=gather_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
        # policy_rejected_logps=gather_log_probs(policy_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])
        # ref_rejected_logps=gather_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

        # start = prompt_ids.size()[-1] - 1

        # loss = self.compute_loss(policy_chosen_logps[:, start:],policy_rejected_logps[:, start:],ref_chosen_logps[:, start:],ref_rejected_logps[:, start:])

        start = prompt_ids.size()[-1] - 1

        loss = self.compute_loss(gather_log_probs(policy_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])[:, start:],
                                    gather_log_probs(policy_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])[:, start:],
                                    gather_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])[:, start:],
                                    gather_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])[:, start:])
      
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()

        self.policy_optimizer.zero_grad()



        return loss



    def compute_loss(self,policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, reference_rejected_logps,):
        """
        计算loss
        """
        print(policy_chosen_logps.shape)
        print(ref_chosen_logps.shape)
        chosen_diff = policy_chosen_logps - ref_chosen_logps
        rejected_diff=policy_rejected_logps-reference_rejected_logps

        # 确定最大长度
        max_len = max(chosen_diff.size(1), rejected_diff.size(1))

        # 计算 padding 量
        pad_chosen = max_len - chosen_diff.size(1)
        pad_rejected = max_len - rejected_diff.size(1)

        # 使用 torch.nn.functional.pad 进行 padding
        # TODO pad 0会对最后log sigmoid结果产生影响，但是trl dpotrainer也是pad 0
        chosen_diff_padded = torch.nn.functional.pad(chosen_diff, (0, pad_chosen))
        rejected_diff_padded = torch.nn.functional.pad(rejected_diff, (0, pad_rejected))

        loss=-torch.nn.functional.logsigmoid(self.beta*chosen_diff_padded-self.beta*rejected_diff_padded)
        return loss.mean()
    


    def evaluate(self):
        """
        在测试集上进行评估
        chosen比rejected分高，则算是正确（不要求直接分数）
        """
        print("==============Evaluating on test set==============")
        self.policy_model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_dataloader, desc="Evaluating")):
            #     # 将数据移到相应设备
            #     for k, v in batch.items():
            #         if isinstance(v, torch.Tensor):
            #             batch[k] = v.to(self.device).half()
                
                chosen_input_ids=batch["chosen_input_ids"]
                chosen_attention_mask=batch["chosen_attention_mask"]
                rejected_input_ids=batch["rejected_input_ids"]
                rejected_attention_mask=batch["rejected_attention_mask"]
                prompt_ids=batch["input_ids"]

                policy_chosen_logits=self.policy_model(chosen_input_ids,chosen_attention_mask).logits
                ref_chosen_logits=self.ref_model(chosen_input_ids,chosen_attention_mask).logits
                policy_rejected_logits=self.policy_model(rejected_input_ids,rejected_attention_mask).logits
                ref_rejected_logits=self.ref_model(rejected_input_ids,rejected_attention_mask).logits

                
                # policy_chosen_logps=gather_log_probs(policy_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
                # ref_chosen_logps=gather_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
                # policy_rejected_logps=gather_log_probs(policy_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])
                # ref_rejected_logps=gather_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

                start = prompt_ids.size()[-1] - 1

                loss = self.compute_loss(gather_log_probs(policy_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])[:, start:],
                                         gather_log_probs(policy_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])[:, start:],
                                         gather_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])[:, start:],
                                         gather_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])[:, start:])
                total_loss += loss.item()
        
        # 计算平均奖励
        avg_loss = total_loss / len(self.test_dataloader)
        print(f"Test set - Average loss: {avg_loss:.4f}")
        

        wandb.log({"test_loss": avg_loss})
        
        
        
        return avg_loss


    
    def save_checkpoint(self, checkpoint_name):
        """
        保存模型检查点
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存模型
        self.policy_model.save_pretrained(checkpoint_dir / "policy_model")
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练配置
        with open(checkpoint_dir / "training_args.json", 'w') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4, default=str)
        
        print(f"Model checkpoint saved to {checkpoint_dir}")
        
        
        
def gather_log_probs(logits, labels):
    """
    获得label的对数概率
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

class CustomDataset(Dataset):               
    def __init__(self, sample):
        super(CustomDataset, self).__init__()
        self.sample = sample

    def __getitem__(self, item):
        res = {k: v[item] for k, v in self.sample.items()}
        return res

    def __len__(self):
        return len(self.sample['input_ids'])



def data_prepare(tokenizer,data_lst,device):
    """
    对于DPO，偏好数据是关键，因为省略掉了reward model的打分，在数据中直接出现chosen和rejected
    """
    # print("==========data prepare==========")

    
    question_lst=[data['prompt']for data in data_lst]
    chosen_lst=[data['chosen']for data in data_lst]
    rejected_lst=[data['rejected']for data in data_lst]

    train_data = tokenizer.batch_encode_plus(question_lst, max_length=512, padding="longest", truncation=True,return_tensors='pt').to(device) 
    chosen_data= tokenizer.batch_encode_plus(chosen_lst, max_length=512, padding="longest", truncation=True,return_tensors='pt').to(device) 
    rejected_data=tokenizer.batch_encode_plus(rejected_lst, max_length=512, padding="longest", truncation=True,return_tensors='pt').to(device) 
    # print(f"train_data:{train_data}")
    # print(f"chosen_data:{chosen_data}")
    # print(f"rejected_data:{rejected_data}")

    train_data["chosen_input_ids"]=torch.cat((train_data["input_ids"],chosen_data["input_ids"]),dim=1)
    train_data["rejected_input_ids"]=torch.cat((train_data["input_ids"],rejected_data["input_ids"]),dim=1)
    train_data["chosen_attention_mask"]=torch.cat((train_data["attention_mask"],chosen_data["attention_mask"]),dim=1)
    train_data["rejected_attention_mask"]=torch.cat((train_data["attention_mask"],rejected_data["attention_mask"]),dim=1)

    # print(train_data)

    return train_data



def set_seed(seed=42):
    random.seed(seed)  # Python 内置的随机数生成器
    numpy.random.seed(seed)  # NumPy 的随机数生成器
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机种子（单卡）
    torch.cuda.manual_seed_all(seed)  # PyTorch 的 GPU 随机种子（多卡）
    torch.backends.cudnn.deterministic = True  # 让 cudnn 以确定性模式运行
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，保证可复现性
    




if __name__=="__main__":
    # import torch
    # print(torch.cuda.is_available())  # 应该是 True
    # print(torch.cuda.device_count())  # 应该 >= 1
    # print(torch.cuda.get_device_name(0))  # 打印设备名

    # exit()







    os.environ["WANDB_MODE"] = "offline"
    parser = argparse.ArgumentParser()

    set_seed(42)        
    

    # Models
    parser.add_argument("--pretrain_path", type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    # Dataset
    parser.add_argument("--train_path",default='dataset/hh_rlhf_cn/train.parquet')
    parser.add_argument("--test_path", default='dataset/hh_rlhf_cn/test.parquet')
    #wandb
    parser.add_argument("--use_wandb", default=True)
    #outputs
    parser.add_argument("--output_dir", default='outputs/hh_rlhf_cn/')
    parser.add_argument("--reward_model", default=None)


    args=parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    print(f"device:{device}")

    args.device=device


    dpoTrainer=DPOTrainer(args)
    dpoTrainer.learn()