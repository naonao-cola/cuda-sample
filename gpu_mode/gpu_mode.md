

## torch profile

谷歌浏览器 打开开发者模式  chrome://tracing 可以打开profile 的 json文件

打开 环境变量 TORCH_COMPILE_DEBUG   可以看到torch.complie 生成的triton 代码

查看生成的代码

Cheat: Generate a triton kernel

TORCH_LOGS="output_code" python compile_square.py

torch.compile(torch.square))

