import argparse
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import get_scheduler, AutoTokenizer, PreTrainedModel, AutoModel, AutoModelForCausalLM, BertPreTrainedModel, Qwen2PreTrainedModel, Qwen2Model, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
import wandb
import os
from pathlib import Path

from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
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
sys.path.append("/HOME/sustc_yqzhang/sustc_yqzhang_1/sy")

from RLHF.policy.policy import PolicyModel
from RLHF.policy.value import ValueModel
from RLHF.reward.rm import RewardModel


class PPOTrainer():
    def __init__(self, policy_model, value_model, reward_model, train_data, test_data, train_dataloader, test_dataloader, tokenizer, device, args):
        self.policy_model = policy_model.to(device)
        self.value_model = value_model.to(device)
        self.ref_model = deepcopy(self.policy_model).to(device)
        self.reward_model = reward_model.to(device)
        self.train_dataset = train_data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.test_dataset = test_data
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

        # 创建保存模型的目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 初始化wandb
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model": args.pretrain,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gamma": args.gamma,
                    "kl_ctl": args.kl_ctl,
                    "cliprange": args.cliprange,
                }
            )

        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = self.tokenizer.pad_token
        
        self.max_answer_seq_len = args.max_answer_seq_len
        self.n_updates_per_iteration = 5
        self.clip = args.cliprange  # As recommended by the paper
        self.lr = args.lr
        self.save_freq = args.save_freq
        self.gamma = args.gamma 
        self.epoch = args.epochs
        self.kl_ctl = args.kl_ctl
        self.clip_reward_value = args.clip_reward_value
        self.batch_size = args.batch_size
        self.reward_mode = args.reward_mode  # {"model","rule"}
        self.lam = args.lam
        self.cliprange = args.cliprange
        self.cliprange_value = args.cliprange_value
        self.best_reward = float('-inf')

        max_train_steps = self.epoch * len(self.train_dataloader)
        warm_steps = int(args.warmup_ratio * max_train_steps)

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

    def learn(self):
        """
        epoch轮学习，并在每个epoch结束时保存模型和评估
        """
        print("==============ppo learn==============")
        self.policy_model.train()
        self.value_model.train()
        
        for i in range(self.epoch):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Train epoch [{i + 1}/{self.epoch}]",
            )
            policy_loss_sum, value_loss_sum, reward_sum = 0, 0, 0
            batch_count = 0
            
            for batch in pbar:
                # 将数据移到相应设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                pad_token_id = self.tokenizer.pad_token_id
                input_ids = batch["input_ids"]
                
                try:
                    with torch.no_grad():
                        # 检查输入是否有效
                        sanitize_input_ids(self.tokenizer, batch["input_ids"], self.tokenizer.vocab_size)
                        
                        # 生成回答
                        seq = self.policy_model.model.generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_length=self.max_answer_seq_len,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        seq_mask = seq.not_equal(pad_token_id).long()
                    
                        # 获取模型输出
                        outputs = self.policy_model(seq, attention_mask=seq_mask)
                        outputs_ref = self.ref_model(seq, attention_mask=seq_mask)
                        rwd_score = self.compute_reward_score(seq, attention_mask=seq_mask)
                        values = self.value_model(seq, attention_mask=seq_mask)[:, :-1]

                    logits = outputs.logits
                    logits_ref = outputs_ref.logits

                    # 构建经验数据
                    experience = {
                        'prompts': input_ids,
                        'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
                        'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
                        'value': values,
                        'rewards': rwd_score,
                        'input_ids': seq,
                        "attention_mask": seq_mask,
                    }

                    # 执行一步训练
                    policy_loss, value_loss = self.step(experience=experience)
                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    reward_sum += rwd_score.mean().item()
                    batch_count += 1
                    
                    # 更新进度条信息
                    pbar.set_postfix({
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item(),
                        'reward': rwd_score.mean().item()
                    })
                    
                    # 记录到wandb
                    if self.args.use_wandb:
                        wandb.log({
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "reward": rwd_score.mean().item(),
                            "learning_rate": self.policy_lr_scheduler.get_last_lr()[0],
                        })
                
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            # 计算平均损失
            avg_policy_loss = policy_loss_sum / max(1, batch_count)
            avg_value_loss = value_loss_sum / max(1, batch_count)
            avg_reward = reward_sum / max(1, batch_count)
            
            print(f"Epoch {i+1}/{self.epoch} - Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # 记录到wandb
            if self.args.use_wandb:
                wandb.log({
                    "epoch": i+1,
                    "avg_policy_loss": avg_policy_loss,
                    "avg_value_loss": avg_value_loss,
                    "avg_reward": avg_reward,
                })
            
            # 在测试集上评估
            test_reward = self.evaluate()
            
            # 保存模型
            if (i+1) % self.save_freq == 0 or i == self.epoch - 1:
                self.save_checkpoint(f"checkpoint-epoch-{i+1}")
            
            # 保存最佳模型
            if test_reward > self.best_reward:
                self.best_reward = test_reward
                self.save_checkpoint("best_model")
                print(f"New best model with reward: {test_reward:.4f}")

        # 完成训练后关闭wandb
        if self.args.use_wandb:
            wandb.finish()

    def step(self, experience):
        """
        一轮学习，调用model
        """
        # 处理旧输出
        prompts = experience['prompts']
        log_probs = experience['logprobs']
        ref_log_probs = experience['ref_logprobs']
        reward_score = experience['rewards']
        values = experience['value']
        attention_mask = experience['attention_mask']
        seq = experience['input_ids']

        start = prompts.size()[-1] - 1
        old_values = values
        old_values = old_values.squeeze(dim=2)
        
        with torch.no_grad():
            old_rewards = self.compute_rwd(prompts, log_probs,
                                           ref_log_probs, reward_score,
                                           attention_mask)
            # 确保只有生成部分的有效 token 参与训练，忽略 padding 部分
            ends = start + attention_mask[:, start+1:].sum(1) - 1

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
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()

        value = self.value_model(**batch)[:, :-1]
        value = value.squeeze(dim=2)
        value_loss = self.value_loss_fn(value[:, start:], old_values[:,start:],
                                        returns, action_mask[:, start:])
        value_loss.backward()
        self.value_optimizer.step()
        self.value_lr_scheduler.step()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        return policy_loss, value_loss

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
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_dataloader, desc="Evaluating")):
                # 将数据移到相应设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                input_ids = batch["input_ids"]
                prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                # 生成回答
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
                    reward = self.compute_reward_score(seq, attention_mask=seq_mask)
                    total_reward += reward.mean().item()
                    
                    # 解码生成的回答
                    generations = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
                    
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
        print(f"Test set - Average reward: {avg_reward:.4f}")
        
        # 记录到wandb
        if self.args.use_wandb:
            wandb.log({"test_reward": avg_reward})
            
            # 创建一个表格记录生成样本
            if generated_examples:
                table = wandb.Table(columns=["prompt", "generation", "reward"])
                for example in generated_examples:
                    table.add_data(example["prompt"], example["generation"], example["reward"])
                wandb.log({"generation_examples": table})
        
        self.policy_model.train()
        self.value_model.train()
        
        return avg_reward

    def save_checkpoint(self, checkpoint_name):
        """
        保存模型检查点
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存模型
        self.policy_model.save_pretrained(checkpoint_dir / "policy_model")
        self.value_model.save_pretrained(checkpoint_dir / "value_model")
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练配置
        with open(checkpoint_dir / "training_args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        print(f"Model checkpoint saved to {checkpoint_dir}")

    def compute_reward_score(self, seq, attention_mask):
        """
        prompt+outputs调用reward model
        """
        rwd_score = 0
        if self.reward_mode == "model":
            rwd_score = self.reward_model(seq, attention_mask=attention_mask)
        elif self.reward_mode == "rule":
            gt = input["label"]
            if seq == gt:
                rwd_score = 1.0
            else:
                rwd_score = 0.0
            rwd_score = 1.0
        return rwd_score
    
    def compute_rwd(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        """
        计算奖励，结合KL散度和reward model的分数
        """
        if self.args.verbose:
            print("=============compute rewards================")

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start+1:].sum(1) - 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        
        # 在L维度上，答案部分每个位置都有KL散度，但是只在最后一个位置加上奖励值
        for j in range(batch_size):
            rewards[j, ends[j]] += reward_clip[j][-1]

        return rewards
    
    def compute_adv(self, values, rewards, start):
        """
        计算优势函数
        """
        if self.args.verbose:
            print("================compute advantage================")
        
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 因为最后时刻的未来value=0
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 将结果进行反序，也就是扭成正常从左到右的顺序，再进行 stack 组合
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 优势加 values 中有效答案开始后的价值估计得到回报 returns
        returns = advantages + values[:, start:]

        return advantages.detach(), returns # (B, start:length)

    def policy_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        """
        策略梯度损失函数
        """
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss
    
    def value_loss_fn(self, values, old_values, returns, mask):
        """
        价值函数损失
        """
        if self.args.verbose:
            print("=========value loss=========")
        
        # 使用裁剪的"老critic_model"的输出约束"新critic_model"
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
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
    
def data_prepare(tokenizer, data_lst, device):
    question_lst = [data['prompt'][0]['content'] for data in data_lst]
    gt_lst = [data["reward_model"]["ground_truth"] for data in data_lst]

    train_data = tokenizer.batch_encode_plus(question_lst, max_length=512, padding="longest", truncation=True, return_tensors='pt').to(device) 
    label_data = tokenizer.batch_encode_plus(gt_lst, max_length=512, padding="longest", truncation=True, return_tensors='pt').to(device) 

    return train_data

def gather_log_probs(logits, labels):
    """
    获得seq_ids的概率
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def sanitize_input_ids(tokenizer, input_ids, vocab_size):
    """
    将超出词表的Token替换为<unk>
    """
    input_ids[input_ids >= vocab_size] = tokenizer.unk_token_id
    return input_ids

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 初始化wandb（如果启用）
    if args.use_wandb:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
        if not args.wandb_run_name:
            args.wandb_run_name = f"ppo-{time.strftime('%Y%m%d-%H%M%S')}"
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = load_dataset("parquet", data_files=args.train_dataset_path, split='train', streaming=True).shuffle(seed=args.seed).take(args.train_samples)
    test_dataset = load_dataset("parquet", data_files=args.test_dataset_path, split='train', streaming=True).shuffle(seed=args.seed).take(args.test_samples)
    
    # 加载tokenizer
    print(f"Loading tokenizer from {args.pretrain}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    tokenizer.padding_side = 'left'
    
    # 配置量化（如果启用）
    if args.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.bits == 8,
            load_in_4bit=args.bits == 4,
            load_in_2bit=args.bits == 2,
            bnb_2bit_compute_dtype=torch.float16,
            bnb_2bit_quant_type="nf2"
        )
    else:
        bnb_config = None
    
    # 设置LoRA配置
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else ["q_proj", "v_proj"],
            bias="none",
        )
    else:
        lora_config = None
    
    # 初始化模型
    print("Initializing models...")
    policy = PolicyModel(args.pretrain, lora_config=lora_config)
    value = ValueModel(args.pretrain, lora_config=lora_config)
    rm = RewardModel(args.pretrain, lora_config=lora_config)
    
    # 准备数据
    print("Preparing datasets...")
    train_data = data_prepare(tokenizer, train_dataset, device)
    test_data = data_prepare(tokenizer, test_dataset, device)
    
    train_dataloader = DataLoader(
        dataset=CustomDataset(train_data),
        shuffle=True,
        batch_size=args.batch_size
    )
    
    test_dataloader = DataLoader(
        dataset=CustomDataset(test_data),
        shuffle=False,
        batch_size=args.batch_size
    )
    
    # 创建PPO训练器
    print("Creating PPO trainer...")
    ppo = PPOTrainer(
        policy_model=policy,
        value_model=value,
        reward_model=rm,
        train_data=train_data,
        test_data=test_data,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        device=device,
        args=args
    )
    
    # 开始训练
    print("Starting training...")
    ppo.learn()
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Language Models")
    
    # 模型参数
    parser.add_argument("--pretrain", type=str, default='/HOME/sustc_yqzhang/sustc_yqzhang_1/luoqi/models/Qwen/Qwen2.5-1.5B-Instruct',
                      help="预训练模型的路径")
    
    # 数据集参数
    parser.add_argument("--train_dataset_path", default='/HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/dataset/spider/train.parquet',
                      help="训练数据集路径")
    parser.add_argument("--test_dataset_path", default='/HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/dataset/spider/test.parquet',
                      help="测试数据集路径")
    parser.add_argument("--train_samples", type=int, default=12, help="训练样本数量")
    parser.add_argument("--test_samples", type=int, default=4, help="测试样本数量")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--epochs", type=int, default=1, help="训练的epoch数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--max_answer_seq_len", type=int, default=512, help="生成答案的最大长度")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--lam", type=float, default=0.9, help="GAE lambda参数")
    parser.add_argument("--kl_ctl", type=float, default=0.1, help="KL散度控制系数")
    parser.add_argument("--cliprange", type=float, default=0.05, help="PPO裁剪范围")
    parser.add_argument("--cliprange_value", type=float, default=0.05, help="值函数裁剪范围")
    parser.add_argument("--clip_reward_value", type=float, default=0.5, help="奖励裁剪值")
    parser.add_argument("--reward_mode", type=str, default="model", choices=["model", "rule"], help="奖励计算模式")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="学习率预热比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="是否显示详细日志")
    
    args=parser.parse_args()

    train(args)