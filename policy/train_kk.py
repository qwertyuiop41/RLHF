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


from policy.policy import PolicyModel
from policy.value import ValueModel
from reward.rm import RewardModel
from grpo.kk import compute_score


class PPOTrainer():
    def __init__(self,policy_model,value_model,reward_model,train_data,test_data,train_dataloader,test_dataloader,tokenizer,device,args):
        self.policy_model=policy_model.to(device)
        self.value_model=value_model.to(device)
        # 不会变的 记住sft之后的内容
        self.ref_model=deepcopy(self.policy_model).to(device)
        if reward_model:
            self.reward_model=reward_model.to(device)
        self.reward_mode="model" if reward_model else "rule"
        self.train_dataset=train_data
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.test_dataset=test_data
        self.tokenizer=tokenizer
        self.device=device
        self.args=args

        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = self.tokenizer.pad_token
        
        self.max_answer_seq_len=1024
        self.lr=1e-6
        # self.save_steps=200
        self.eval_steps=50
        self.gamma = 1.0   # 之前0.95
        self.epoch=2
        self.kl_ctl=0.001   # openrlhf 0.01
        self.clip_reward_value = 1
        self.batch_size=2
        self.test_batch_size=16
        self.lam = 1.0 #之前0.9
        self.cliprange = 0.001
        self.cliprange_value = 0.5  #之前0.001
        self.best_reward = float('-inf')
        self.warmup_ratio=0.0
        self.ppo_epoch=1
        

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 初始化wandb
        # if args.use_wandb:
        wandb.init(
            project=f'rlhf-ppo-3b-kk',
            name=f"ppo-{time.strftime('%Y%m%d-%H%M%S')}",
            dir="policy",
            config={
                "policy_model": args.pretrain_path,
                "value_model": self.value_model,
                "reward_mode": self.reward_mode,
                "reward_model": self.reward_model if self.reward_mode=="model" else "rule",
                "max_answer_seq_len": self.max_answer_seq_len,
                "lr": self.lr,
                # "save_steps": self.save_steps,
                "eval_steps":self.eval_steps,
                "gamma": self.gamma,
                "epoch": self.epoch,
                "kl_ctl": self.kl_ctl,
                "clip_reward_value": self.clip_reward_value,
                "test_batch_size":self.test_batch_size,
                "batch_size": self.batch_size,
                "reward_mode": self.reward_mode,
                "lam": self.lam,
                "cliprange": self.cliprange,
                "cliprange_value": self.cliprange_value
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


        self.value_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.value_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.value_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.value_optimizer = torch.optim.AdamW(self.value_optimizer_grouped_parameters, lr=self.lr)


        self.value_lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.value_optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )


        self.policy_optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr)
        self.value_optimizer=torch.optim.Adam(self.value_model.parameters(),lr=self.lr)




    def generate_experience(self,batch):
        """
        重要性采样
        """
        self.eval()

        # 将数据移到相应设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
                
        pad_token_id = self.tokenizer.pad_token_id
        input_ids=batch["input_ids"]
        gt=batch["labels"]
        prompt=self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        with torch.no_grad():
            
            # sanitize_input_ids(self.tokenizer,batch["input_ids"],self.tokenizer.vocab_size)

            seq = self.policy_model.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.max_answer_seq_len,
                pad_token_id=self.tokenizer.pad_token_id,
                )

                                    
            seq_mask = seq.not_equal(pad_token_id).long()
        
            outputs=self.policy_model(seq, attention_mask=seq_mask)
            outputs_ref=self.ref_model(seq, attention_mask=seq_mask)

            rwd_score=self.compute_reward_score(seq,attention_mask=seq_mask,gt=gt)
            # values 估计的是当前 token 未来的累积奖励,所以不需要最后一个
            # 评论模型 critic_model 返回结果维度是(B,L)，L 维度上第 i 个位置代表从 i 位置到最后的累积奖励，用于辅助评估策略的好坏，舍去最后一个位置的 token
            # 价值函数  V(t) 是基于当前状态评估未来的期望回报，但最后一个 token 通常没有后续的未来信息，因此它的价值估计没有意义。
            # 而且生成任务是自回归的，序列的最后一个 token 不会为后续步骤提供任何预测依据，因为生成已经结束。
            values=self.value_model(seq,attention_mask=seq_mask)[:, :-1]

        logits = outputs.logits
        logits_ref = outputs_ref.logits


        experience={
            'prompts': input_ids,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'value': values,
            'rewards': rwd_score,
            'input_ids': seq,
            "attention_mask": seq_mask,
        }
        return experience







    def eval(self):
        self.policy_model.eval()
        self.value_model.eval()
        if self.reward_mode=="model":
            self.reward_model.eval()
        self.ref_model.eval()


    def learn(self):
        """
        epoch轮学习
        """
        self.policy_model.train()
        self.value_model.train()
        global_step=0
        for i in range(self.epoch):
            # TODO 也不用整个train loader的experience一次性传入，可以一部分一部分的传入
            # 重要性采样是off-policy采样，需要乘以重要性权重个重要性权重p(x)/q(x)
            # 如果要on-policy则每次更新model后都重新采样
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Train epoch [{i + 1}/{self.epoch}]",
            )
            policy_loss_sum, value_loss_sum, reward_sum = 0, 0, 0
            total_seq_length,total_adv=0,0
            batch_count = 0
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            
            
            for batch in pbar:
                experience=self.generate_experience(batch)

                for i in range(self.ppo_epoch):

                    rwd_score=experience["rewards"]
                    policy_loss, value_loss, adv=self.step(experience=experience)
                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    reward_sum += rwd_score.sum().item()
                    total_seq_length+=experience["input_ids"].shape[1]
                    total_adv+=adv.float().mean().item()
                    
                    
                    
                    # 更新进度条信息
                    pbar.set_postfix({
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item(),
                        'reward': rwd_score.float().sum().item()
                    })
                    seq_length=experience["input_ids"].shape[1]
                    # 记录到wandb
                    wandb.log({
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "reward": rwd_score.float().sum().item(),
                        "learning_rate": self.policy_lr_scheduler.get_last_lr()[0],
                        "seq_length": seq_length,
                        "advantages": adv.float().mean().item(),
                    })
                    print(f"policy_loss:{policy_loss.item()},value_loss:{value_loss.item()},reward:{rwd_score.float().sum().item()},seq_length:{seq_length}, adv:{adv.float().mean().item()}")

                    
        

                

                if global_step % self.eval_steps == 0:
                    # 在测试集上评估
                    test_reward = self.evaluate()
                    # 保存最佳模型
                    if test_reward > self.best_reward:
                        self.best_reward = test_reward
                        # self.save_checkpoint("best_model")
                        print(f"New best model with reward: {test_reward:.4f}")
                        wandb.log({"best_reward": test_reward})

                batch_count += 1
                global_step+=1






            avg_policy_loss = policy_loss_sum / max(1, batch_count)
            avg_value_loss = value_loss_sum / max(1, batch_count)
            avg_reward = reward_sum / max(1, batch_count*self.batch_size)
            avg_seq_length= total_seq_length / max(1, batch_count)
            avg_adv= total_adv / max(1, batch_count)

            print(f"Epoch {i+1}/{self.epoch} - Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}, Avg Reward: {avg_reward:.4f}, Avg Seq Length: {avg_seq_length:.4f}, Avg Adv: {avg_adv:.4f}")
            
            # 记录到wandb
            if self.args.use_wandb:
                wandb.log({
                    "epoch": i+1,
                    "avg_policy_loss": avg_policy_loss,
                    "avg_value_loss": avg_value_loss,
                    "avg_reward": avg_reward,
                    "avg_seq_length": avg_seq_length,
                    "avg_adv": avg_adv,
                })

            test_reward = self.evaluate()
            # 保存模型
            # if (i+1) % self.save_steps == 0 or i == self.epoch - 1:
            # self.save_checkpoint(f"checkpoint-epoch-{i+1}")
            # 保存最佳模型
            if test_reward > self.best_reward:
                self.best_reward = test_reward
                # self.save_checkpoint("best_model")
                print(f"New best model with reward: {test_reward:.4f}")
                wandb.log({"best_reward": test_reward})



        wandb.finish()


        

    def step(self,experience):
        """
        一轮学习，调用model
        """
          # train the rlhf mode here
        ### process the old outputs
        prompts = experience['prompts']
        log_probs = experience['logprobs']
        ref_log_probs = experience['ref_logprobs']
        reward_score = experience['rewards']
        values = experience['value']
        attention_mask = experience['attention_mask']
        seq = experience['input_ids']



        start = prompts.size()[-1] - 1
        # # #因为第一个token是输入
        # action_mask = attention_mask[:, 1:]

        old_values = values

        old_values=old_values.squeeze(dim=2)

        
        with torch.no_grad():
            old_rewards,rwd = self.compute_rwd(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               attention_mask)
            # 确保只有生成部分的有效 token 参与训练，忽略 padding 部分。
            ends = start +attention_mask[:, start+1:].sum(1)-1

            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.compute_adv(
                old_values, old_rewards, start)

        action_mask = attention_mask[:, 1:]
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        policy_prob = self.policy_model(**batch).logits
        policy_log_prob = gather_log_probs(policy_prob[:, :-1, :], seq[:, 1:])
        policy_loss = self.policy_loss_fn(policy_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])

        policy_loss.backward()
        # # self.policy_model.backward(policy_loss)
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()


        value = self.value_model(**batch)[:, :-1]
        value=value.squeeze(dim=2)
        value_loss = self.value_loss_fn(value[:, start:], old_values[:,start:],
                                        
                                          returns, action_mask[:, start:])
        # self.value_model.backward(value_loss)
        value_loss.backward()
        self.value_optimizer.step()
        self.value_lr_scheduler.step()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        return policy_loss, value_loss,advantages

    def evaluate(self):
        """
        在测试集上进行评估
        """
        print("==============Evaluating on test set==============")
        self.policy_model.eval()
        self.value_model.eval()
        
        total_reward = 0
        generated_examples = []
        num_samples = min(5, len(self.test_dataloader))  # 仅记录少量样本用于展示
        seq_length=0
        
        with torch.no_grad():

            pbar=tqdm(self.test_dataloader, desc="Evaluating")
            for idx, batch in enumerate(pbar):
                # 将数据移到相应设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                input_ids = batch["input_ids"]
                prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                gt=batch["labels"]
                
                try:
                    sanitize_input_ids(self.tokenizer, batch["input_ids"], self.tokenizer.vocab_size)
                    
                    seq = self.policy_model.model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.max_answer_seq_len,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    seq_mask = seq.not_equal(self.tokenizer.pad_token_id).long()
                    
                    # 计算奖励分数
                    reward = self.compute_reward_score(seq, attention_mask=seq_mask,gt=gt)
                    total_reward += reward.float().sum().item()

                    # 更新进度条信息
                    pbar.set_postfix({
                        'reward': reward.float().sum().item()
                    })
                    
                    # 解码生成的回答
                    generations = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
                    seq_length+=seq.shape[1]
                    
                    # 保存一些样本用于展示
                    if idx < num_samples:
                        for p, g, r in zip(prompts, generations, reward):
                            generated_examples.append({
                                "prompt": p,
                                "generation": g,
                                "reward": r.item()
                            })
                
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue
        
        # 计算平均奖励
        avg_reward = total_reward / len(self.test_dataloader)
        seq_length=seq_length/len(self.test_dataloader)
        print(f"Test set - Average reward: {avg_reward:.4f}, test seq length: {seq_length}")
        

        wandb.log({"test_reward": avg_reward,"test_seq_length": seq_length})
        
        # 创建一个表格记录生成样本
        if generated_examples:
            table = wandb.Table(columns=["prompt", "generation", "reward"])
            for example in generated_examples:
                table.add_data(example["prompt"], example["generation"], example["reward"])
            wandb.log({"generation_examples": table})
        
        
        return avg_reward

    # def save_checkpoint(self, checkpoint_name):
    #     """
    #     保存模型检查点
    #     """
    #     checkpoint_dir = self.output_dir / checkpoint_name
    #     checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    #     # 保存模型
    #     self.policy_model.save_pretrained(checkpoint_dir / "policy_model")
    #     self.value_model.save_pretrained(checkpoint_dir / "value_model")
        
    #     # 保存tokenizer
    #     self.tokenizer.save_pretrained(checkpoint_dir)
        
    #     # 保存训练配置
    #     with open(checkpoint_dir / "training_args.json", 'w') as f:
    #         json.dump(vars(self.args), f, indent=2)
        
    #     print(f"Model checkpoint saved to {checkpoint_dir}")




    def compute_reward_score(self,seq,attention_mask,gt):
        """
        prompt+outputs调用reward model
        """
        print("=============compute reward score================")

        if self.reward_mode=="model":

            rwd_score=self.reward_model(seq,attention_mask=attention_mask)
            return rwd_score
                            
        else:
            rwd_score=[]

            
            for s, g in zip(seq,gt):
                seq_text = self.tokenizer.decode(s, skip_special_tokens=True)

                gt_text=self.tokenizer.decode(g, skip_special_tokens=True)
                ground_truth=json.loads(gt_text)

                score=compute_score(seq_text,ground_truth)
                rwd_score.append([score])
            rwd_score=torch.tensor(rwd_score).to(device=self.device)

            return rwd_score


    


    def compute_rwd(self, prompts, log_probs, ref_log_probs, reward_score,action_mask):
        rwd=[]

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start +action_mask[:, start+1:].sum(1)-1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
         # 在L维度上，答案部分每个位置都有KL散度，但是只在最后一个位置加上奖励值
        for j in range(batch_size):
            rewards[j, ends[j]] += reward_clip[j][-1]
            rwd.append(reward_clip[j][-1])

        
        return rewards,torch.tensor(rwd,dtype=torch.float16)
    

    def compute_adv(self, values, rewards, start):
        # https://huggingface.co/blog/deep-rl-a2c
        # values（B，L） critic_model 输出，包含每个 token 上的评分
        # rewards（B，L）reward_model 输出包含了kl散度以及最后一个有效答案 token 的奖励值
        # start 是 answer 开始的位置
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 因为最后时刻的未来value=0
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小，衡量某个动作相对于基准的好坏程度，使用 GAE 平滑计算
            # 这种设计能够使优化既关注当前的即时奖励，又能兼顾未来的长期收益，从而提升整体性能。降低可能因为单步的随机奖励导致估计偏差较大的风险
            # GAE：多步优势估计，当前时刻的优势值 At依赖于未来所有时刻的TD误差δ并通过 γ λ 衰减因子对远期误差进行加权。
            # ​γ：折扣因子，控制未来奖励的重要性。
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 将结果进行反序，也就是扭成正常从左到右的顺序，再进行 stack 组合
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 优势加 values 中有效答案开始后的价值估计得到回报 returns ，后续用来更新 critic_model 
        returns = advantages + values[:, start:]

        return advantages.detach(), returns # (B, start:length)




    def policy_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # 重要性采样权重计算 ratio = exp(log(new)-log(old)) 因为ppo是off policy的，所以需要加上ratio
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss
    


    def value_loss_fn(self, values, old_values, returns, mask):
        # value loss 需要注意的是这里使用裁剪的“老critic_model”的输出约束“新critic_model”不要步子太大。
        """
        values: 实时critic跑出来的预估预期收益（是变动的，随着ppo epoch迭代而改变）
        old_values：老critic跑出来的预估预期收益（是固定值）
        returns：实际预期收益(认为是实际value=Adv+r)
        """
        values_clipped = torch.clamp(values,old_values - self.cliprange_value,old_values + self.cliprange_value)
        values = values.float()
        values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss
    




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
    
    question_lst=[data['prompt'][0]['content']for data in data_lst]

    gt_lst=[json.dumps(data["reward_model"]["ground_truth"])for data in data_lst]

    train_data = tokenizer.batch_encode_plus(question_lst, max_length=1024, padding="longest", truncation=True,return_tensors='pt').to(device) 
    label_data = tokenizer.batch_encode_plus(gt_lst, max_length=1024, padding="longest", truncation=True, return_tensors='pt').to(device) 

    train_data["labels"] = label_data["input_ids"]

    return train_data


def gather_log_probs(logits, labels):
    """
    获得seq_ids的概率
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)



def sanitize_input_ids(tokenizer,input_ids, vocab_size):
    # 将超出词表的Token替换为<unk>
    input_ids[input_ids >= vocab_size] = tokenizer.unk_token_id
    return input_ids

def train(args):

    pretrain_path=args.pretrain_path
    train_path=args.train_path
    test_path=args.test_path
    os.environ["WANDB_MODE"]="offline"




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    print(f"device:{device}")
    train_dataset=load_dataset("parquet", data_files=train_path,split='train',streaming=True).shuffle(seed=42).take(450)


    test_dataset=load_dataset("parquet", data_files=test_path,split='train',streaming=True).shuffle(seed=42).take(50)
    # test_dataset=load_dataset("parquet", data_files=test_path,split='train').shuffle(seed=42).select(range(4))


    
    tokenizer=AutoTokenizer.from_pretrained(pretrain_path)
    tokenizer.padding_side='left'

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=False,  # 不使用8bit
    #     load_in_4bit=False,  # 不使用4bit
    #     load_in_2bit=True,   # 启用 2-bit 量化
    #     bnb_2bit_compute_dtype=torch.float16,  # 计算时使用 float16
    #     bnb_2bit_quant_type="nf2"  # `nf2` 量化格式，适用于 LLM
    # )


    # 设置 LoRA 配置
    lora_config = LoraConfig(
        r=8,  # Rank，越大表示 LoRA 层越大，消耗显存更多
        lora_alpha=16,  # LoRA scaling factor
        lora_dropout=0.1,  # Dropout 防止过拟合
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 只训练注意力层
        bias="none",
        # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
    )


    policy=PolicyModel(pretrain_path,lora_config,bnb_config=None)
    # policy = get_peft_model(policy,lora_config)
    value=ValueModel(pretrain_path,lora_config,bnb_config=None)
    # value = get_peft_model(value,lora_config)
    # rm=RewardModel(pretrain_path,lora_config,bnb_config=None)
    # rm = get_peft_model(rm,lora_config)





    # policy=PolicyModel(config)
    # value=PolicyModel(config)
    # rm=RewardModel(config)

    train_dataset=data_prepare(tokenizer,train_dataset,device)
    test_dataset=data_prepare(tokenizer,test_dataset,device)

    train_dataloader=DataLoader(dataset=CustomDataset(train_dataset),shuffle=True,batch_size=2)
    test_dataloader=DataLoader(dataset=CustomDataset(test_dataset),shuffle=False,batch_size=16)


    ppo=PPOTrainer(policy_model=policy,value_model=value,reward_model=None,
    train_data=train_dataset,test_data=test_dataset,train_dataloader=train_dataloader,test_dataloader=test_dataloader,tokenizer=tokenizer,device=device,args=args)
    ppo.learn()


    
def set_seed(seed=42):
    random.seed(seed)  # Python 内置的随机数生成器
    numpy.random.seed(seed)  # NumPy 的随机数生成器
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机种子（单卡）
    torch.cuda.manual_seed_all(seed)  # PyTorch 的 GPU 随机种子（多卡）
    torch.backends.cudnn.deterministic = True  # 让 cudnn 以确定性模式运行
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，保证可复现性
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    set_seed()

    # Models
    parser.add_argument("--pretrain_path", type=str, default='/HOME/sustc_yqzhang/sustc_yqzhang_1/sy/models/Qwen2.5-3B-Instruct')
    # Dataset
    parser.add_argument("--train_path",default='/HOME/sustc_yqzhang/sustc_yqzhang_1/sy/Logic-RL/data/kk/instruct/3ppl/train.parquet')
    parser.add_argument("--test_path", default='/HOME/sustc_yqzhang/sustc_yqzhang_1/sy/Logic-RL/data/kk/instruct/3ppl/test.parquet')
    #wandb
    parser.add_argument("--use_wandb", default=True)
    #outputs
    parser.add_argument("--output_dir", default='outputs/ppo/kk')


    args=parser.parse_args()

    train(args)