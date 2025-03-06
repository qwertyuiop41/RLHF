from typing import Optional, Union
from base_model import BaseModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    TrainingArguments
)
from peft import PeftConfig, get_peft_model

class PolicyModel(BaseModel):
    def __init__(self,
                 model_path: Union[str, PreTrainedModel],
                 args: TrainingArguments,
                 peft_config: Optional["PeftConfig"] = None,):
                # Models
        self.policy_model=super.__init__(model_path, args, peft_config)
    


    def forward(self, state):
        pass

    def get_action(self, state):
        pass