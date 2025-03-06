from typing import Optional, Union
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
from peft import PeftConfig, get_peft_model, is_peft_model
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import create_reference_model

from base_model import BaseModel




class RewardModel(BaseModel):
    def __init__(self,
                 model_path: Union[str, PreTrainedModel],
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 peft_config: Optional["PeftConfig"] = None,):
        model_init_kwargs = args.model_init_kwargs or {}
        model=AutoModelForSequenceClassification.from_pretrained(model_path, **model_init_kwargs)


 
       

