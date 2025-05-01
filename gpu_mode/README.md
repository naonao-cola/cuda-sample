


## torch profile

https://github.com/gpu-mode/lectures


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


.ptx文件在编译阶段控制kernel内资源的生成或调用，所以，我们可以通过.ptx或--ptxas-options=-v命令获得kernel在编译阶段获得资源的情况。下面讲kernel在编译时能确定什么。而--ptxas-options=-v命令就可以看到所有确定的资源。从而可以确定块在SM上的分配。


--ptxas-options=-v命令的作用是查看kernel在编译阶段确定的所有静态资源，有：寄存器资源，local memory（stack 内资源），共享内存资源，和常量资源和global memory资源。


获取编译中间步骤

NVCC的dryrun选项可以在控制台上列出所有的编译子命令而不进行真实的编译

```bash
## 生成 ptx  指令
nvcc -O3 -ptx vector_addition.cu
# 更详细的信息
nvcc -O3 -c -arch=sm_86  -lineinfo --source-in-ptx -ptx vector_addition.cu
# 生成目标文件,报告所用资源情况
nvcc -O3 -c -arch=sm_86 -Xptxas -v vector_addition.cu

nvcc -dryrun vector_addition.cu

#$ _NVVM_BRANCH_=nvvm
#$ _NVVM_BRANCH_SUFFIX_=
#$ _SPACE_=
#$ _CUDART_=cudart
#$ _HERE_=/usr/local/cuda/bin
#$ _THERE_=/usr/local/cuda/bin
#$ _TARGET_SIZE_=
#$ _TARGET_DIR_=
#$ _TARGET_DIR_=targets/x86_64-linux
#$ TOP=/usr/local/cuda/bin/..
#$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
#$ LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#$ PATH=/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#$ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"
#$ LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
#$ CUDAFE_FLAGS=
#$ PTXAS_FLAGS=
#$ gcc -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=131 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "vector_addition.cu" -o "/tmp/tmpxft_0000e17d_00000000-5_vector_addition.cpp4.ii"
#$ cudafe++ --c++17 --gnu_version=110400 --display_error_number --orig_src_file_name "vector_addition.cu" --orig_src_path_name "/home/test001/proj/nao/repo/cuda-sample/gpu_mode/002/vector_addition/vector_addition.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.cpp" --stub_file_name "tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_0000e17d_00000000-4_vector_addition.module_id" "/tmp/tmpxft_0000e17d_00000000-5_vector_addition.cpp4.ii"
#$ gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=131 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "vector_addition.cu" -o "/tmp/tmpxft_0000e17d_00000000-9_vector_addition.cpp1.ii"
#$ cicc --c++17 --gnu_version=110400 --display_error_number --orig_src_file_name "vector_addition.cu" --orig_src_path_name "/home/test001/proj/nao/repo/cuda-sample/gpu_mode/002/vector_addition/vector_addition.cu" --allow_managed   -arch compute_52 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_0000e17d_00000000-3_vector_addition.fatbin.c" -tused --module_id_file_name "/tmp/tmpxft_0000e17d_00000000-4_vector_addition.module_id" --gen_c_file_name "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.c" --stub_file_name "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.gpu"  "/tmp/tmpxft_0000e17d_00000000-9_vector_addition.cpp1.ii" -o "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.ptx"
#$ ptxas -arch=sm_52 -m64  "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.ptx"  -o "/tmp/tmpxft_0000e17d_00000000-10_vector_addition.sm_52.cubin"
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=52,file=/tmp/tmpxft_0000e17d_00000000-10_vector_addition.sm_52.cubin" "--image3=kind=ptx,sm=52,file=/tmp/tmpxft_0000e17d_00000000-6_vector_addition.ptx" --embedded-fatbin="/tmp/tmpxft_0000e17d_00000000-3_vector_addition.fatbin.c"
#$ rm /tmp/tmpxft_0000e17d_00000000-3_vector_addition.fatbin
#$ gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_0000e17d_00000000-6_vector_addition.cudafe1.cpp" -o "/tmp/tmpxft_0000e17d_00000000-11_vector_addition.o"
#$ nvlink -m64 --arch=sm_52 --register-link-binaries="/tmp/tmpxft_0000e17d_00000000-7_a_dlink.reg.c"    "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_0000e17d_00000000-11_vector_addition.o"  -lcudadevrt  -o "/tmp/tmpxft_0000e17d_00000000-12_a_dlink.sm_52.cubin" --host-ccbin "gcc"
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=52,file=/tmp/tmpxft_0000e17d_00000000-12_a_dlink.sm_52.cubin" --embedded-fatbin="/tmp/tmpxft_0000e17d_00000000-8_a_dlink.fatbin.c"
#$ rm /tmp/tmpxft_0000e17d_00000000-8_a_dlink.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0000e17d_00000000-8_a_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0000e17d_00000000-7_a_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=131 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda/bin/crt/link.stub" -o "/tmp/tmpxft_0000e17d_00000000-13_a_dlink.o"
#$ g++ -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -m64 -Wl,--start-group "/tmp/tmpxft_0000e17d_00000000-13_a_dlink.o" "/tmp/tmpxft_0000e17d_00000000-11_vector_addition.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "a.out"

```
