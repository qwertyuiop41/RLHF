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




class RefModel(BaseModel):
    def __init__(self,
                 model: Union[str, PreTrainedModel],
                 args: TrainingArguments,
                 peft_config: Optional["PeftConfig"] = None,):
        model_init_kwargs=args.model_init_kwargs or {}
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None


    def apply_kl_penalty():
        pass