#include "../common/common.h"


/**
获取当前CUDA平台上第一台设备的基本信息，
包括SMs的数量、固定内存的字节数、每个块共享内存的字节数等。

Device 0: NVIDIA GeForce RTX 4060 Laptop GPU
  Number of multiprocessors:                     24
  Total amount of constant memory:               64.00 KB
  Total amount of shared memory per block:       48.00 KB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum number of threads per multiprocessor:  1536
  Maximum number of warps per multiprocessor:    48
  
*/
void simpleDeviceQuery();

/**
simpleDivergence演示了GPU上的分歧代码及其对GPU的影响性能和CUDA指标。

** https://langbin.blog.csdn.net/article/details/51348559

** 第一次结果
Data size 4096 Execution Configure (block 128 grid 32)
warmup: 0.000617s
mathKernel1: 0.000220s
mathKernel2: 0.000076s
mathKernel3: 0.000068s
mathKernel4: 0.000098s
** 第二次结果
Data size 4096 Execution Configure (block 128 grid 32)
warmup: 0.000669s
mathKernel1: 0.000152s
mathKernel2: 0.000079s
mathKernel3: 0.000072s
mathKernel4: 0.000146s

*/
void simpleDivergence();