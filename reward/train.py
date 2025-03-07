from rm import RewardModel, RankRewardModel
from rm_dataprocess import rank_data_prepare,data_prepare,CustomDataset
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
    Qwen2Config
)
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error
from peft import LoraConfig, get_peft_model


def rank_loss(predict_rewards):
    print("=============rank loss================")
    # predict_rewards的位置越前面的相对分越高
    # loss设置原因见：https://zhuanlan.zhihu.com/p/610147705
    loss, counts = torch.tensor([0]), 0
    # predict_rewards的位置越前面的相对分越高
    for i in range(len(predict_rewards) - 1):  # 遍历所有前项-后项的得分差
        for j in range(i + 1, len(predict_rewards)):
            diff = nn.functional.logsigmoid(predict_rewards[i] - predict_rewards[j])  # sigmoid到0~1之间 log再全变成负的
            print(f"diff:{diff}")
            loss = loss + diff
            counts += 1
            print(loss)
    loss = loss / counts
    return -loss  # 要最大化分差，所以要取负数


def evaluate(model, eval_dataloader, device="cuda",use_wandb=True):
    """评估函数，计算验证集上的 MSE 损失"""
    model.eval()
    eval_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            out, loss = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            eval_loss += loss.item()
            

            # 因为 PyTorch 张量需要先转移到 CPU 上才能被转换为 NumPy 数组
            all_preds.append(out.squeeze().cpu().numpy().tolist())
            all_labels.append(batch["labels"].squeeze().cpu().numpy().tolist())
            
    
    avg_loss = eval_loss / len(eval_dataloader)
    mse = mean_squared_error(all_labels, all_preds)
    if use_wandb:
        wandb.log({
            "avg_loss": avg_loss,
            "mse": mse,
            "all_preds":all_preds,
            "all_labels":all_labels
        })
    
    return {"eval_loss": avg_loss, "eval_mse": mse}


def train(pretrain_path, save_path, num_epochs=50, eval_steps=100, save_steps=500, use_wandb=True, eval_split=0.2, batch_size=1):

    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载配置和模型
    # config = Qwen2Config.from_pretrained(pretrain_path)

    lora_config = LoraConfig(
        r=1,  # Rank，越大表示 LoRA 层越大，消耗显存更多
        lora_alpha=8,  # LoRA scaling factor
        lora_dropout=0.1,  # Dropout 防止过拟合
        target_modules=["q_proj", "v_proj"],  # 只训练注意力层
        bias="none",
        # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
    )
    model = RewardModel(pretrain_path,lora_config=lora_config)
    model.to(device)

    if use_wandb:
        wandb.init(project="rlhf-reward-model", name="reward-model-training", config={
            "model": model,
            "lr": 2e-5, 
            "epochs": num_epochs,
            "batch_size": batch_size,
            "weight_decay": 0.01,
            "eval_steps": eval_steps,
            "save_steps": save_steps
        })

    # 优化器设置
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    
    # 准备数据
    train_data, tokenizer = data_prepare(pretrain_path)
    print(type(train_data))
    print(train_data)
    
    # 划分训练集和验证集
    data_size = len(train_data["input_ids"])
    indices = list(range(data_size))
    np.random.shuffle(indices)
    split = int(np.floor(eval_split * data_size))
    train_indices, eval_indices = indices[split:], indices[:split]
    
    # 创建数据集
    train_dataset = {k: [train_data[k][i] for i in train_indices] for k in train_data.keys()}
    eval_dataset = {k: [train_data[k][i] for i in eval_indices] for k in train_data.keys()}
    
    train_dataloader = DataLoader(dataset=CustomDataset(train_dataset), shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(dataset=CustomDataset(eval_dataset), shuffle=False, batch_size=batch_size)

    # 学习率调度器
    max_train_steps = num_epochs * len(train_dataloader)
    warm_steps = int(0.1 * max_train_steps)  # 10% warmup
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    
    os.makedirs(save_path, exist_ok=True)
    

    model.train()
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            out, loss = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # 记录到 wandb
            if use_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                })
            
            # 评估
            if global_step > 0 and global_step % eval_steps == 0:
                eval_results = evaluate(model, eval_dataloader, device,use_wandb)
                print(f"Step {global_step}: Eval Loss = {eval_results['eval_loss']}, MSE = {eval_results['eval_mse']}")
                
                if use_wandb:
                    wandb.log({
                        "eval_loss": eval_results['eval_loss'],
                        "eval_mse": eval_results['eval_mse'],
                        "global_step": global_step,
                    })
                
                # 保存最佳模型
                if eval_results['eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_results['eval_loss']
                    print(f"New best model with eval_loss: {best_eval_loss}")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(os.path.join(save_path, "best_model"))
                    tokenizer.save_pretrained(os.path.join(save_path, "best_model"))
            
            # 定期保存模型
            if global_step > 0 and global_step % save_steps == 0:
                checkpoint_dir = os.path.join(save_path, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # 保存优化器状态
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                print(f"Saved checkpoint at step {global_step}")
            
            global_step += 1
        
        # 每个 epoch 结束后的平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss}")
        
        if use_wandb:
            wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})
    
    # 保存最终模型
    tokenizer.save_pretrained(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    model_to_save.config.save_pretrained(save_path)
    
    # 结束 wandb
    if use_wandb:
        wandb.finish()
    
    return model


def rank_train(pretrain_path, save_path, num_epochs=10, eval_steps=50, save_steps=200, use_wandb=True, batch_size=1):
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # 设置 LoRA 配置
    lora_config = LoraConfig(
        r=1,  # Rank，越大表示 LoRA 层越大，消耗显存更多
        lora_alpha=8,  # LoRA scaling factor
        lora_dropout=0.1,  # Dropout 防止过拟合
        target_modules=["q_proj", "v_proj"],  # 只训练注意力层
        bias="none",
        # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
    )

    model = RankRewardModel(pretrain_path,lora_config=lora_config)

    if use_wandb:
        wandb.init(project="rlhf-reward-model", name="reward-model-training", config={
            "model": model,
            "lr": 2e-5, 
            "epochs": num_epochs,
            "batch_size": batch_size,
            "weight_decay": 0.01,
            "eval_steps": eval_steps,
            "save_steps": save_steps
        })

    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    train_data, tokenizer = rank_data_prepare(pretrain_path)
    dataloader = DataLoader(dataset=CustomDataset(train_data), shuffle=True, batch_size=batch_size)

    max_train_steps = num_epochs * len(dataloader)
    warm_steps = int(0.1 * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    

    os.makedirs(save_path, exist_ok=True)
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # 移动数据到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            reward_lst = []
            for i in range(len(batch["input_ids"][0])):
                input_ids = batch["input_ids"][0][i].unsqueeze(0).to(device)
                attention_mask = batch["attention_mask"][0][i].unsqueeze(0).to(device)
                
                reward = model(input_ids, attention_mask=attention_mask)
                reward_lst.append(reward)
            
            loss = rank_loss(reward_lst)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 更新损失
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            

            if use_wandb:
                wandb.log({
                    "rank_train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                })
            
            # 定期保存模型
            if global_step > 0 and global_step % save_steps == 0:
                checkpoint_dir = os.path.join(save_path, f"rank-checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # 保存优化器状态
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                print(f"Saved checkpoint at step {global_step}")
            
            # 更新当前最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(os.path.join(save_path, "best_rank_model"))
                tokenizer.save_pretrained(os.path.join(save_path, "best_rank_model"))
                print(f"New best model with loss: {best_loss}")
            
            global_step += 1
        
        # 每个 epoch 结束后的平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss}")
        
        if use_wandb:
            wandb.log({"rank_epoch_avg_loss": avg_epoch_loss, "epoch": epoch})
    
    # 保存最终模型
    tokenizer.save_pretrained(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    model_to_save.config.save_pretrained(save_path)
    
    # 结束 wandb
    if use_wandb:
        wandb.finish()
    
    return model


def predict(model_path):
    text = ["我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是今世界上保存最完好的哺乳动物之一，也是世界自然保护联盟濒危物种红色名录的保护对象之一。在这里，你可以看到全世界最大的熊猫栖息地成都。成都是中国国家林业局直属的国家重点风景名胜区，是国家森林公园、国家湿地公园和国家地质公园的重要组成部分，是全国重点文物保护单位、全国生态文明建设示范区、中国红色旅游名城、国际生态旅游目的地和国际旅游岛建设先进区。地址：四川省成都市绵阳市成华区成都高新技术产业开发区成华大道1号乘车路线：成都绵阳都江堰雅",
            "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家二级保护动物，是世界上保存最完整的动物种群之一，也是我国第一个国家级自然保护区。我们是四川省的首批国家重点保护野生动物和珍稀动物基金会的成员，被誉为中国动物保护的摇篮和世界生物多样性保护基地，被中国科学院、中华人民共和国国家林业局授予全国生态文明建设示范区称号，被国务院批准为国家森林城市、国际生态旅游目的地。熊猫基地位于成都市双流区东南部，是国家aaaa级旅游景区，国家地理标志保护单位。熊猫栖息地为亚热带或热带的高山",]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    data = tokenizer.batch_encode_plus(text, max_length=256, padding="max_length", truncation=True,
                                       return_tensors='pt')
    
    # 移动数据到设备
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    
    with torch.no_grad():
        score = model(**data)
    
    return score


if __name__=="__main__":
    # 配置 wandb (在运行前需要登录 wandb)
    # wandb.login()
    
    # 使用更多参数的训练函数
    train(
        pretrain_path='/home/wsy/NLP/RL/Qwen2.5-0.5B-Instruct',
        save_path='/home/wsy/NLP/RL/RLHF/reward/ckpt',
        num_epochs=2,
        eval_steps=2,
        save_steps=5,
        use_wandb=True,
        batch_size=2
    )