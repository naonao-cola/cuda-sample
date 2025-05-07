# TORCH_LOGS="output_code" python reduce_compile.py
import sys
sys.path.append(".")

import os

os.environ["TORCH_LOGS"] ="output_code"

import torch

@torch.compile
def f(a):
    c = torch.sum(a)
    return c

print(f(torch.randn(10).cuda()))
