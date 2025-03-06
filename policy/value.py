import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model,AutoConfig
from peft import get_peft_model

class ValueModel(Qwen2PreTrainedModel):
    def __init__(self, pretrain_path,lora_config,bnb_config=None):
        config=AutoConfig.from_pretrained(pretrain_path)
        super(ValueModel, self).__init__(config)
        self.config = config
        self.loss_fn = nn.MSELoss()
        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path,quantization_config=bnb_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path)
        self.model=get_peft_model(self.model,lora_config)

        # 获取模型的 dtype
        model_dtype = next(self.model.parameters()).dtype
        # 初始化 Linear 层，并设置相同的 dtype
        self.linear = nn.Linear(config.hidden_size, 1).to(model_dtype)

        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]
        output = self.linear(outputs)
        logits = self.sigmoid(output)
        # batch_indices = torch.arange(logits.shape[0])  # 生成 batch 索引
        # last_rewards = logits[batch_indices, -1]  # 获取对应的 reward
        # logits=logits[:,-1]
        # print(f"value:{logits}")
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            return logits