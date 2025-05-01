import torch
from dotenv import load_dotenv
import os

os.environ["TORCH_LOGS"] ="output_code"

os.environ["TORCH_COMPILE_DEBUG"] ="1"
print(torch.cuda.is_available())
print(torch.cuda.get_device_capability())

def square_3(a):
    return a ** 2

x = torch.randn(6, 6, device='cuda', dtype=torch.float32)

torch.compile(square_3)(x)