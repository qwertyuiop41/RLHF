from deepspeed.accelerator import get_accelerator
import torch
print(get_accelerator())
print(torch.device(get_accelerator().device_name()))