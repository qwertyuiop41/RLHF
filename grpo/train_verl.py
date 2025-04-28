# 模仿verl中的GRPO实现，将kl直接加在reward上，在得到adv和loss进行训练
# 提出的GRPO算法中应该是直接将kl加在loss上(通过reward和adv获得loss)


import argparse
from collections import defaultdict
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
sys.path.append('./')


from policy.policy import PolicyModel
from policy.value import ValueModel
from reward.rm import RewardModel
from grpo.gms8k_reward import format_reward,correctness_reward
from grpo.kk import compute_score

class GRPOTrainer():
    def __init__(self,args):
        pretrain_path=args.pretrain_path
        train_path=args.train_path
        test_path=args.test_path
        self.device=args.device
        self.use_wandb=args.use_wandb


        self.max_answer_seq_len=1600
        self.lr=1e-6
        self.save_steps=200
        self.eval_steps=50
        self.gamma = 1.0  #原本0.95 verl 1.0
        self.epoch=2
        self.kl_ctl=0.001
        self.clip_reward_value = 1.0
        self.batch_size=1
        self.test_batch_size=16
        self.num_generation=6
        self.reward_mode="model" if args.reward_model else "rule"  #{"model","rule"}
        self.reward_fn="kk" #{"kk","gms8k"}
        self.lam = 1.0  #原本 0.9 0.95 verl 1.0
        self.cliprange = 0.5
        self.best_reward = float('-inf')
        self.warmup_ratio=0.0
        self.epsilon = 1e-6   #原本0.00001   verl 1e-6
        self.beta=0.01
        

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


        train_data=load_dataset("parquet", data_files=train_path,split='train',streaming=True).shuffle(seed=42).take(400)
        test_data=load_dataset("parquet", data_files=test_path,split='train',streaming=True).shuffle(seed=42).take(40)


        
        self.tokenizer=AutoTokenizer.from_pretrained(pretrain_path)
        self.tokenizer.padding_side='left'

        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=False,  # 不使用8bit
        #     load_in_4bit=False,  # 不使用4bit
        #     load_in_2bit=True,   # 启用 2-bit 量化
        #     bnb_2bit_compute_dtype=torch.float16,  # 计算时使用 float16
        #     bnb_2bit_quant_type="nf2"  # `nf2` 量化格式，适用于 LLM
        # )
        bnb_config=None


        # 设置 LoRA 配置
        lora_config = LoraConfig(
            r=4,  # Rank，越大表示 LoRA 层越大，消耗显存更多
            lora_alpha=8,  # LoRA scaling factor
            lora_dropout=0.05,  # Dropout 防止过拟合
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 只训练注意力层
            bias="none",
            # task_type="CAUSAL_LM"  # 适用于自回归（decoder-only）模型，如 Qwen
        )


        self.policy_model=PolicyModel(pretrain_path,lora_config,bnb_config=bnb_config).to(self.device)

        self.ref_model=deepcopy(self.policy_model).to(self.device)
        # self.ref_model=self.policy_model




        if args.reward_model:
            self.reward=RewardModel(args.reward_model,lora_config,bnb_config=bnb_config).to(self.device)
        else:
            self.reward=args.reward_fn


        self.train_dataset=data_prepare(self.tokenizer,train_data,self.device)
        self.test_dataset=data_prepare(self.tokenizer,test_data,self.device)

        self.train_dataloader=DataLoader(dataset=CustomDataset(self.train_dataset),shuffle=True,batch_size=self.batch_size)
        self.test_dataloader=DataLoader(dataset=CustomDataset(self.test_dataset),shuffle=False,batch_size=self.test_batch_size)

        
        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = self.tokenizer.pad_token
        


        # 初始化wandb
        # if args.use_wandb:
        wandb.init(
            project='rlhf-grpo-1.5b-4',
            name=f"grpo-{time.strftime('%Y%m%d-%H%M%S')}",
            dir="grpo",
            sync_tensorboard=True,
            config={
                "policy_model": args.pretrain_path,
                "lora_config": lora_config,
                "reward": self.reward,
                "max_answer_seq_len": self.max_answer_seq_len,
                "lr": self.lr,
                "save_steps": self.save_steps,
                "eval_steps": self.eval_steps,
                "gamma": self.gamma,
                "epoch": self.epoch,
                "kl_ctl": self.kl_ctl,
                "clip_reward_value": self.clip_reward_value,
                "batch_size": self.batch_size,
                "reward_mode": self.reward_mode,
                "lam": self.lam,
                "cliprange": self.cliprange,
                "num_generation":self.num_generation,
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

        self.policy_optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr)


    def learn(self):
        """
        epoch轮学习
        grpo on policy
        """
        print("==============grpo learn==============")
        global_step=0
        self.policy_model.train()
        for i in range(self.epoch):

            pbar = tqdm(
                self.train_dataloader,
                desc=f"Train epoch [{i + 1}/{self.epoch}]",
            )

            
            policy_loss_sum, reward_sum = 0, 0
            batch_count = 0
            
            self.policy_optimizer.zero_grad()

            for batch in pbar:
                batch=self.stack_batches(batch)
                """
                grpo是on policy，所以每次都要采样当前模型的completion
                """
                prepared_inputs=self.prepare_inputs(batch)
                rwd_score=prepared_inputs["rewards"].clone().detach()
                # rwd_score=torch.tensor(prepared_inputs["rewards"])
                policy_loss=self.step(prepared_inputs)
                self.policy_optimizer.step()
                self.policy_lr_scheduler.step()
                self.policy_optimizer.zero_grad()
                policy_loss_sum += policy_loss.item()

                reward_sum += rwd_score.sum().item()
                
                
                # 更新进度条信息
                pbar.set_postfix({
                    'policy_loss': policy_loss.item(),
                    'reward': rwd_score.float().sum().item()
                })

                seq_length=prepared_inputs["input_ids"].shape[1]
                
                # 记录到wandb
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "reward": rwd_score.float().sum().item(),
                    "learning_rate": self.policy_lr_scheduler.get_last_lr()[0],
                    "seq_length": seq_length,
                })

                # # 定期保存模型
                # if global_step > 0 and global_step % self.save_steps == 0:
                #     self.save_checkpoint(f"checkpoint-steps-{global_step}")

                

                if global_step % self.eval_steps == 0:
                    # 在测试集上评估
                    test_reward = self.evaluate()
                    # 保存最佳模型
                    if test_reward > self.best_reward:
                        self.best_reward = test_reward
                        self.save_checkpoint("best_model")
                        print(f"New best model with reward: {test_reward:.4f}")

                batch_count += 1
                global_step += 1

            avg_policy_loss = policy_loss_sum / max(1, batch_count)
            avg_reward = reward_sum / max(1, batch_count)

            print(f"Epoch {i+1}/{self.epoch} - Avg Policy Loss: {avg_policy_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": i+1,
                    "avg_policy_loss": avg_policy_loss,
                    "avg_reward": avg_reward,
                })

            # 在测试集上评估
            test_reward = self.evaluate()
            

            self.save_checkpoint(f"checkpoint-epoch-{i+1}")
            
            # 保存最佳模型
            if test_reward > self.best_reward:
                self.best_reward = test_reward
                self.save_checkpoint("best_model")
                print(f"New best model with reward: {test_reward:.4f}")

            
        wandb.finish()
        
    # def prepare_inputs(self,batch):
    #     """
    #     对on policy生成的batch采样 + 打分 + 计算相对优势
    #     """
    #     self.eval()
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor):
    #             batch[k] = v.to(self.device)
                
    #     pad_token_id = self.tokenizer.pad_token_id
        
    #     prompt=self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)[0]
    #     input_ids=batch["input_ids"].repeat_interleave(self.num_generation, dim=0)
    #     with torch.no_grad():
            
    #         # sanitize_input_ids(self.tokenizer,batch["input_ids"],self.tokenizer.vocab_size)

    #         seq = self.policy_model.model.generate(
    #             batch["input_ids"],
    #             attention_mask=batch["attention_mask"],
    #             max_length=self.max_answer_seq_len,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             num_return_sequences=self.num_generation
    #             )

    #         seq_mask = seq.not_equal(pad_token_id).long()
    #         completions =self.get_completion(seq, input_ids, self.tokenizer)
            
        
    #         outputs=self.policy_model(seq, attention_mask=seq_mask)
    #         outputs_ref=self.ref_model(seq, attention_mask=seq_mask)
    #         rwd_score=self.compute_reward_score(seq,attention_mask=seq_mask,completions=completions,labels=batch["labels"])
    #         # print(f"rwd_score:{rwd_score}")
    #         adv=self.compute_adv(rwd_score)
    #         # print(f"adv:{adv}")
            

    #     logits = outputs.logits
    #     logits_ref = outputs_ref.logits

    #     prepared={
    #         'prompts': input_ids,
    #         'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
    #         'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
    #         'rewards': rwd_score,
    #         'advantage':adv,
    #         'input_ids': seq,
    #         "attention_mask": seq_mask,
    #         "completions":completions,
    #     }


    #     return prepared


    def prepare_inputs(self,batch):
        """
        对on policy生成的batch采样 + 打分 + 计算相对优势
        """
        self.eval()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
                
        pad_token_id = self.tokenizer.pad_token_id
        input_ids=batch["input_ids"]
        prompt=self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        with torch.no_grad():
            
            # sanitize_input_ids(self.tokenizer,batch["input_ids"],self.tokenizer.vocab_size)

            seq = self.policy_model.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.max_answer_seq_len,
                # num_return_sequences=self.num_generation,
                pad_token_id=self.tokenizer.pad_token_id,
                )

            seq_mask = seq.not_equal(pad_token_id).long()
            completions =self.get_completion(seq, input_ids, self.tokenizer)
            
        
            outputs=self.policy_model(seq, attention_mask=seq_mask)
            outputs_ref=self.ref_model(seq, attention_mask=seq_mask)

            logits = outputs.logits
            logits_ref = outputs_ref.logits

            rwd_score=self.compute_reward_score(seq,attention_mask=seq_mask,completions=completions,labels=batch["labels"])
            # print(f"rwd_score:{rwd_score}")
            rwd,_=self.compute_rwd(input_ids,logits,
                                              logits_ref, rwd_score,
                                               seq_mask)
            # print(f"rwd:{rwd}")
            adv,_=self.compute_grpo_outcome_advantage(rwd)
            # print(f"adv:{adv}")
            print(adv.shape)
            

        

        prepared={
            'prompts': input_ids,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'rewards': rwd_score,
            'advantage':adv,
            'input_ids': seq,
            "attention_mask": seq_mask,
            "completions":completions,
        }


        return prepared

    def get_completion(self,seq,input_ids, tokenizer):
        """
        从生成的序列中提取 completion 部分。

        Args:
            seq (torch.Tensor): 生成的序列，形状为 (batch_size, seq_len)。
            input_ids (torch.Tensor): 输入的 prompt 的 token IDs，形状为 (batch_size, prompt_len)。
            tokenizer: 用于解码 token IDs 的 tokenizer。

        Returns:
            list: 包含 completion 字符串的列表。
        """
        completions = []
        completions_mask=[]
        batch_size = seq.size(0)

        # 创建一个掩码张量，标记每个位置是否为completion部分
        is_completion = torch.zeros_like(seq, dtype=torch.bool)
        
        # 对于每个样本，将prompt_len之后的位置标记为True
        for i in range(batch_size):
            prompt_len = input_ids[i].size(0)
            if prompt_len < self.max_answer_seq_len:
                is_completion[i, prompt_len:] = True
        
        # 创建一个新的tensor来保存所有completion部分的token IDs
        # 将非completion部分替换为pad token
        completion_ids = seq.clone()
        completion_ids[~is_completion] = tokenizer.pad_token_id
        
        # 使用batch_decode一次性解码所有completion
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # 清理结果（移除可能存在的padding造成的前缀空白）
        completions = [text.strip() for text in decoded]
        return completions
    # def get_completion(self,seq,input_ids, tokenizer):
    #     """
    #     从生成的序列中提取 completion 部分。

    #     Args:
    #         seq (torch.Tensor): 生成的序列，形状为 (batch_size, seq_len)。
    #         input_ids (torch.Tensor): 输入的 prompt 的 token IDs，形状为 (batch_size, prompt_len)。
    #         tokenizer: 用于解码 token IDs 的 tokenizer。

    #     Returns:
    #         list: 包含 completion 字符串的列表。
    #     """
    #     completions = []
    #     completions_mask=[]
    #     batch_size = seq.size(0)

    #     for i in range(batch_size):
    #         # 获取当前样本的 prompt 长度
    #         prompt_len = input_ids[i].size(0)
    #         # 提取 completion 部分的 token IDs
    #         completion_ids = seq[i, prompt_len:]

    #         # 解码 completion 部分
    #         completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    #         completions.append(completion)

    #     # print(completions)
    #     return completions


    def step(self, prepared_inputs):
        """
        单步更新
        """
        # train the rlhf mode here
        ### process the old outputs
        prompts = prepared_inputs['prompts']
        log_probs = prepared_inputs['logprobs']
        ref_log_probs = prepared_inputs['ref_logprobs']
        reward_score = prepared_inputs['rewards']
        attention_mask = prepared_inputs['attention_mask']
        seq = prepared_inputs['input_ids']
        adv=prepared_inputs['advantage']

        start = prompts.size()[-1] - 1
        # #因为第一个token是输入
        action_mask = attention_mask[:, 1:]
        # 确保只有生成部分的有效 token 参与训练，忽略 padding 部分。
        ends = start +attention_mask[:, start+1:].sum(1)-1


        batch = {'input_ids': seq, "attention_mask": attention_mask}

        policy_prob = self.policy_model(**batch).logits
        
            
            

        policy_log_prob = gather_log_probs(policy_prob[:, :-1, :], seq[:, 1:])
        print("policy_log_prob.requires_grad:", policy_log_prob.requires_grad)
        action_mask[:, 0:start]=0
        policy_loss = self.compute_loss(policy_log_prob[:, start:],
                                        log_probs[:, start:],adv[:, start+1:],
                                        action_mask[:, start:])
        policy_loss.backward()
        # # self.policy_model.backward(policy_loss)
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()

        self.policy_optimizer.zero_grad()

        return policy_loss
    
    def compute_reward_score(self,seq,attention_mask,completions,labels):
        """
        根据reward model或者reward function计算rwd
        奖励是sentence-level的，奖励是标量值。
        """
        size=seq.shape[0]
        if self.reward=="model":
            with torch.no_grad():
                rwd_score=self.reward(seq,attention_mask=attention_mask)
            return rwd_score                
        elif self.reward_mode=="rule":
            if self.reward_fn=="gsm8k":
                rwd_score=[0 for i in range(size)]
                for rwd_fn in self.reward:
                    rewards=rwd_fn(completions,labels) 
                    rwd_score = [r1 + r2 for r1, r2 in zip(rwd_score, rewards)]
            elif self.reward_fn=="kk":
                rwd_score=[]
            
                for s, g in zip(seq,labels):
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

    def compute_adv(self,rewards):
        """
        根据reward score计算advantage
        包含过程监督强化学习+GRPO 和 结果监督强化学习+GRPO 两种方式
        Ai,t=(ri-mean(r))/std(r), t对应token-level优势，即一个句子中，每个token对应的优势是一样的。这种方式的好处在于，估计都是从真实的环境reward计算得来，而不是通过价值估计计算而得。
        """
        
        rewards = torch.tensor(rewards, dtype = torch.float).to(self.device)
        A = (rewards - rewards.mean()) / (rewards.std() + self.epsilon)
        return A


    def compute_grpo_outcome_advantage(self, token_level_rewards: torch.Tensor,
                                    epsilon: float = 1e-6):
        """
        Compute advantage for GRPO, operating only on Outcome reward 
        (with only one scalar reward for each response).
        Args:
            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
        
        Returns:
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            Returns: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        response_length = token_level_rewards.shape[-1]
        non_zero_mask = (token_level_rewards != 0)
        scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        # print(scores.shape)

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[i].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[i]) / (id2std[i] + epsilon)
            # scores = scores.unsqueeze(-1).tile([1, response_length])
            # print(scores.shape)

        return scores, scores



    # def compute_loss(self,log_probs,old_log_probs,ref_log_probs,mask,adv):
    #     """
    #     根据advantage和kl散度计算loss
    #     GRPO KL是token-level的
    #     GRPO并没有在奖励中添加KL惩罚，而是通过直接将训练策略和参考策略之间的KL散度添加到损失函数中来进行正则化,从而避免了使得𝐴^𝑖,𝑡的计算变得复杂
    #     当前策略如果和ref策略接近，则kl接近0，loss可能是负数
    #     """
    #     len_oi=mask[:, ].sum(1)
        
    #     # kl
    #     # kl=ref_log_probs.exp() / log_probs.exp()- (ref_log_probs - log_probs) - 1
    #     ratio=torch.exp(log_probs - old_log_probs)
    #     print(ratio.shape)
    #     print(adv.shape)


    #     adv=adv.unsqueeze(dim = 1)  # [a, b ,c] -> [[a], [b], [c]]
    #     print(adv.shape)

    #     loss1=ratio*adv
    #     loss2=reward_clip = torch.clamp(ratio, 1.0 - self.cliprange,
    #                               1.0 + self.cliprange)*adv
    #     loss=(torch.minimum(loss1,loss2)-self.beta*kl)*mask
    #     loss=-(1/self.num_generation)*(1/len_oi.unsqueeze(dim = 1))*loss
    #     loss = loss.sum()

    #     return loss

    def compute_loss(self, logprobs, old_logprobs, advantages, mask):
        # 重要性采样权重计算 ratio = exp(log(new)-log(old)) 因为ppo是off policy的，所以需要加上ratio
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)

        # print(ratio.shape)
        # print(advantages.shape)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    





    def evaluate(self):
        """
        评估当前policy的效果
        """
        print("==============Evaluating on test set==============")
        self.policy_model.eval()
        total_reward = 0
        generated_examples = []
        num_samples = min(5, len(self.test_dataloader))  # 仅记录少量样本用于展示
        seq_length=0
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_dataloader, desc="Evaluating")):
                # 将数据移到相应设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                input_ids = batch["input_ids"]
                prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                try:
                    seq = self.policy_model.model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.max_answer_seq_len,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    seq_mask = seq.not_equal(self.tokenizer.pad_token_id).long()
                    
                    # 计算奖励分数
                    reward = self.compute_reward_score(seq, attention_mask=seq_mask,completions=None, labels=batch["labels"])
                    total_reward += reward.float().sum().item()
                    
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
        # avg_reward = total_reward / len(self.test_dataloader*self.batch_size)
        print(f"Test set - Average reward: {avg_reward:.4f}, test seq length: {seq_length}")
        

        wandb.log({"test_reward": avg_reward,"test_seq_length":seq_length})
        
        # 创建一个表格记录生成样本
        if generated_examples:
            table = wandb.Table(columns=["prompt", "generation", "reward"])
            for example in generated_examples:
                table.add_data(example["prompt"], example["generation"], example["reward"])
            wandb.log({"generation_examples": table})
            wandb.log({"generation_examples": generated_examples})
        
        
        return avg_reward

    def save_checkpoint(self,checkpoint_name):
        """
        保存当前policy
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

    def eval(self):
        self.policy_model.eval()
        if self.reward_mode=="model":
            self.reward_model.eval()
        self.ref_model.eval()

    def stack_batches(self,batch):
        # # 将数据移到相应设备
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         batch[k] = v.to(self.device)
        stack_batch=batch
        for i in range(self.num_generation-1):
            stack_batch={key: torch.cat([stack_batch[key], batch[key]], dim=0) for key in batch}
        return stack_batch



def data_prepare(tokenizer,data_lst,device):
    question_lst=[data['prompt'][0]['content']for data in data_lst]

    gt_lst=[json.dumps(data["reward_model"]["ground_truth"])for data in data_lst]

    train_data = tokenizer.batch_encode_plus(question_lst, max_length=400, padding="longest", truncation=True,return_tensors='pt').to(device) 
    label_data = tokenizer.batch_encode_plus(gt_lst, max_length=400, padding="longest", truncation=True, return_tensors='pt').to(device) 

    train_data["labels"] = label_data["input_ids"]

    return train_data


class CustomDataset(Dataset):               
    def __init__(self, sample):
        super(CustomDataset, self).__init__()
        self.sample = sample

    def __getitem__(self, item):
        res = {k: v[item] for k, v in self.sample.items()}
        return res

    def __len__(self):
        return len(self.sample['input_ids'])



def gather_log_probs(logits, labels):
    """
    获得label的对数概率
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def set_seed(seed=42):
    random.seed(seed)  # Python 内置的随机数生成器
    numpy.random.seed(seed)  # NumPy 的随机数生成器
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机种子（单卡）
    torch.cuda.manual_seed_all(seed)  # PyTorch 的 GPU 随机种子（多卡）
    torch.backends.cudnn.deterministic = True  # 让 cudnn 以确定性模式运行
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，保证可复现性




if __name__=="__main__":
    os.environ["WANDB_MODE"] = "offline"
    parser = argparse.ArgumentParser()


    set_seed(42)

    

    

    # Models
    parser.add_argument("--pretrain_path", type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    # Dataset
    parser.add_argument("--train_path",default='../Logic-RL/data/kk/instruct/3ppl/train.parquet')
    parser.add_argument("--test_path", default='../Logic-RL/data/kk/instruct/3ppl/test.parquet')
    #wandb
    parser.add_argument("--use_wandb", default=True)
    #outputs
    parser.add_argument("--output_dir", default='outputs/kk/')
    parser.add_argument("--reward_model", default=None)


    args=parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    print(f"device:{device}")

    args.device=device
    args.reward_fn=[format_reward,correctness_reward]

    grpoTrainer=GRPOTrainer(args)
    grpoTrainer.learn()