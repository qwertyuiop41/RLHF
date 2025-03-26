import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model,Qwen2ForCausalLM,AutoConfig
from peft import get_peft_model

class RewardModel(Qwen2PreTrainedModel):
    def __init__(self, pretrain_path,lora_config=None,bnb_config=None):
        config=AutoConfig.from_pretrained(pretrain_path)
        super(RewardModel, self).__init__(config)
        self.config = config
        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path,quantization_config=bnb_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path)
        if lora_config:
            self.model=get_peft_model(self.model,lora_config)

        # 获取模型的 dtype
        model_dtype = next(self.model.parameters()).dtype
        # 初始化 Linear 层，并设置相同的 dtype
        self.linear = nn.Linear(config.hidden_size, 1).to(model_dtype)
        # 把分数压缩到（0，1）
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()



    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        output = self.linear(outputs)
        logits = self.sigmoid(output)
        batch_indices = torch.arange(logits.shape[0])  # 生成 batch 索引
        # reward应该取最后一位token对应的分数
        last_rewards = logits[batch_indices, -1]  # 获取对应的 reward

        if labels is not None:
            print(labels)
            post_rewards=[]
            for reward,label in zip(last_rewards,labels):
                if label>10 and label<=100:
                    reward = reward* 100
                elif label<=10:
                    reward = reward * 10
                post_rewards.append(reward)
            post_rewards=torch.tensor(post_rewards).to(last_rewards.device)
            print("++++++++++++++++++++++++++++++++++++++")
            print(post_rewards)
            print(labels)
            print("++++++++++++++++++++++++++++++++++++++")
            loss = self.loss_fn(post_rewards, labels)
            return post_rewards, loss
        else:
            return last_rewards
        



class RankRewardModel(Qwen2PreTrainedModel):
    def __init__(self, pretrain_path,lora_config=None,bnb_config=None):
        config=AutoConfig.from_pretrained(pretrain_path)
        super(RankRewardModel, self).__init__(config)
        self.config = config
        # self.model = Qwen2Model(config)
        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path,quantization_config=bnb_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path)
        if lora_config:
            self.model=get_peft_model(self.model,lora_config)

        # 获取模型的 dtype
        model_dtype = next(self.model.parameters()).dtype
        # 初始化 Linear 层，并设置相同的 dtype
        self.linear = nn.Linear(config.hidden_size, 1).to(model_dtype)
        # 把分数压缩到（0，1）
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]
        output = self.linear(outputs)
        logits = self.sigmoid(output)
        batch_indices = torch.arange(logits.shape[0])  # 生成 batch 索引
        # reward应该取最后一位token对应的分数
        rewards = logits[batch_indices, -1]  # 获取对应的 reward

        if labels is not None:
            print(labels)
            post_rewards=[]
            for reward,label in zip(rewards,labels):
                if label>10 and label<=100:
                    reward = reward* 100
                elif label<=10:
                    reward = reward * 10
                post_rewards.append(reward)
            post_rewards=torch.tensor(post_rewards).to(rewards.device)
            print("++++++++++++++++++++++++++++++++++++++")
            print(post_rewards)
            print(labels)
            print("++++++++++++++++++++++++++++++++++++++")
            loss = self.loss_fn(post_rewards, labels)
            return post_rewards, loss
        return rewards

            
            
