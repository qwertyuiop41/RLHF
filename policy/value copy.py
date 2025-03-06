import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model
from peft import get_peft_model

class ValueModel(Qwen2PreTrainedModel):
    def __init__(self, config,lora_config):
        super(ValueModel, self).__init__(config)
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
        # batch_indices = torch.arange(logits.shape[0])  # 生成 batch 索引
        # last_rewards = logits[batch_indices, -1]  # 获取对应的 reward
        # logits=logits[:,-1]
        print(f"value:{logits}")
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            return logits