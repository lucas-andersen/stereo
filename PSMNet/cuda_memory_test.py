from pynvml import *
import torch

def cuda_info(num):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(num)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'GPU {num}:')
    print(f'total    : {info.total/1e9}GB')
    print(f'free     : {info.free/1e9}GB')
    print(f'used     : {info.used/1e9}GB')
    print(f'Pct free: {info.free/info.total}')

for i in range(torch.cuda.device_count()):
    cuda_info(i)