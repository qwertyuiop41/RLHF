import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler,AutoTokenizer,PreTrainedModel,AutoModel,AutoModelForCausalLM, BertPreTrainedModel,Qwen2PreTrainedModel,Qwen2Model,Qwen2ForCausalLM,AutoConfig

from peft import get_peft_model

class PolicyModel(Qwen2PreTrainedModel):
    def __init__(self, pretrain_path,lora_config,bnb_config=None):
        config=AutoConfig.from_pretrained(pretrain_path)
        super(PolicyModel, self).__init__(config)
        self.config = config
        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path,quantization_config=bnb_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_path)
        self.model=get_peft_model(self.model,lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)    
        return outputs
        # if labels is not None:
        #     loss = self.loss_fn(logits, labels)
        #     return logits, loss
        # else:
        #     return logits


        