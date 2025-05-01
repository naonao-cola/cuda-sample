import torch
from torch.utils.cpp_extension import load_inline
import os

import sys
sys.path.append(".")

# 确保 build_directory 存在
# build_directory = './tmp'
# os.makedirs(build_directory, exist_ok=True)

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory='./tmp'
)

print(my_module.hello_world())