# from deepspeed.accelerator import get_accelerator
import torch
# print(get_accelerator())
# print(torch.device(get_accelerator().device_name()))

rwd=[1,2]
rwd=torch.tensor(rwd,dtype=torch.float16)
print(rwd.mean())