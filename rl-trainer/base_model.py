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
from peft import PeftConfig, get_peft_model

class BaseModel():
    def __init__(self,
                model_path: Union[str, PreTrainedModel],
                args: TrainingArguments,
                peft_config: Optional["PeftConfig"] = None,):
            # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_id = model_path
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `config`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)

        # PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)