import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model,Qwen2ForCausalLM
from peft import get_peft_model

class RewardModel(Qwen2PreTrainedModel):
    def __init__(self, config,lora_config):
        super(RewardModel, self).__init__(config)
        self.config = config
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()
        self.model = Qwen2Model(config)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.model=get_peft_model(self.model,lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        output = self.linear(outputs)
        logits = self.sigmoid(output)
        batch_indices = torch.arange(logits.shape[0])  # 生成 batch 索引
        last_rewards = logits[batch_indices, -1]  # 获取对应的 reward
        # logits=logits[:,-1]
        print(f"reward score:{last_rewards}")
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return last_rewards, loss
        else:
            return last_rewards
        



class RankRewardModel(Qwen2PreTrainedModel):
    def __init__(self, config):
        super(RankRewardModel, self).__init__(config)
        self.config = config
        self.model = Qwen2Model(config)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        # TODO reward应该取最后一位token对应的分数
        # reward = self.linear(outputs).mean(dim=1) 
        reward=self.linear(outputs)[0][-1]
        return reward

            
