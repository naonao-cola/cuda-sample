

## torch profile

谷歌浏览器 打开开发者模式  chrome://tracing 可以打开profile 的 json文件

打开 环境变量 TORCH_COMPILE_DEBUG   可以看到torch.complie 生成的triton 代码

查看生成的代码

Cheat: Generate a triton kernel

在命令行使用一下命令，可以查看生成的代码
`TORCH_LOGS="output_code" python compile_square.py`

python文件的代码
```python
torch.compile(torch.square)
```

`os.environ["TORCH_COMPILE_DEBUG"] ="1"` 设置环境变量，可以查看编译过程中的中间代码、诊断错误或性能问题。

## libtorch

libtorch组成讲解之ATen、c10、at、csrc
1.at(ATen)负责声明和定义Tensor运算，是最常用到的命名空间
2.c10是 ATen 的基础，包含了PyTorch的核心抽象、Tensor和Storage数据结构的实际实现
3.torch命名空间下定义的 Tensor 相比于ATen 增加自动求导功能


## jupyter

%time 的计算结果包括：CPU time（CPU运行程序的时间）， Wall time（Wall Clock Time，墙上挂钟的时间，也就是我们感受到的运行时间）



%timeit -r 5 -n 400 nums2=[i+5 for i in nums1]

%%time 与 %time ， %%timeit 与 %timeit 的计算方式相同，区别在于 % 是用于单行代码的命令，%% 是应用于当前单元的命令


## cuda

https://blog.csdn.net/fb_help/article/details/80375858

C/C++程序可通过调用.ptx的信息来运行核函数，完成cuda功能，但.ptx不会显式生成，通过编译时向编译器nvcc输入-ptx（生成ptx文件） 或 -keep（生成全部编译中间文件） 可生成.ptx文件1。


.ptx文件在编译阶段控制kernel内资源的生成或调用，所以，我们可以通过.ptx或--ptxas-options=-v命令获得kernel在编译阶段获得资源的情况。下面讲kernel在编译时能确定什么。而--ptxas-options=-v命令就可以看到所有确定的资源。从而可以确定块在SM上的分配。


--ptxas-options=-v命令的作用是查看kernel在编译阶段确定的所有静态资源，有：寄存器资源，local memory（stack 内资源），共享内存资源，和常量资源和global memory资源。


