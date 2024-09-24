# cuda-sample



```c++
//02 文件夹
#include "../common/common.h"
/**
 * 在本系统的第一个CUDA设备上显示各种信息，
 * 包括驱动程序版本，运行时版本，计算能力，全局内存字节等。
 */
void test_01();

/**
 * 显示来自主机和设备的线程块和网格的维度。
 */
void test_02();

/**
 * 这个例子有助于可视化线程/块id和之间的关系偏移到数据。对于每个CUDA线程，本例显示块内线程ID，块间块ID，全局坐标a
 * 线程，将计算出的偏移量转换为输入数据，并在此输入数据偏移量。
 */
void test_03();

/**
 * 演示如何从主机定义线程块和块网格的尺寸。
 */
void test_04();

/**
本例演示了GPU和主机上的简单矢量求和。sumArraysOnGPU将矢量和的工作拆分到GPU上的CUDA线程上。为了简单起见，在这个小示例中只使用了一个线程块。
sumArraysOnHost依次遍历主机上的向量元素。有计时器
GPU 比host快5倍左右，与网格的设置有关
*/
void test_05();

/**
本例演示了GPU和主机上的简单矢量求和。上的CUDA线程拆分矢量和的工作GPU。使用一维线程块和一维网格。sumArraysOnHost顺序
遍历宿主上的vector元素。
结果
Matrix size: nx 16384 ny 16384
initialize matrix elapsed 12.689188 sec
sumMatrixOnHost elapsed 1.215950 sec
sumMatrixOnGPU1D <<<(512,1), (32,1)>>> elapsed 0.033776 sec
Arrays match.

*/
void sumMatrixOnGPU_1D_grid_1D_block();

/**
Matrix size: nx 16384 ny 16384
Matrix initialization elapsed 11.103387 sec
sumMatrixOnHost elapsed 1.244238 sec
sumMatrixOnGPU2D <<<(512,16384), (32,1)>>> elapsed 0.054994 sec
Arrays match.
*/
void sumMatrixOnGPU_2D_grid_1D_block();

/**
Matrix size: nx 16384 ny 16384
Matrix initialization elapsed 11.882803 sec
sumMatrixOnHost elapsed 1.209717 sec
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.046039 sec
Arrays match.
*/
void sumMatrixOnGPU_2D_grid_2D_block();
```

```c++
//03 文件夹

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

/**
一个从GPU启动嵌套内核的简单示例。每个线程在执行开始时显示其信息，并在下一个最低嵌套层完成时进行诊断。

Execution Configuration: grid 1 block 8
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
-------> nested execution depth: 1
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
-------> nested execution depth: 2
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0

*/
void nestedHelloWorld();

/**
使用嵌套内核的并行缩减的实现CUDA内核。这个版本在工作中增加了优化nestedReduce.cu。

** 236行有问题
array 1048576 grid 2048 block 512
cpu_reduce: 0.000827s
cpu reduce              elapsed cpu_sum: 1048576
gpu_Neighbored: 0.065077s
gpu Neighbored          elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu_nested: 0.005849s
gpu nested              elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu_nested_without_synchronization: 0.006018s
gpu nestedNosyn         elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu_nested2: 0.001103s
gpu nested2             elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>

*/
void nestedReduce2();

/**
这段代码实现了交错和邻居配对的方法CUDA的并行缩减。对于本例，使用求和操作。一个各种优化并行约简旨在减少分歧也进行了演示，例如展开。

xmake run 03 256
      with array size 16777216  grid 65536 block 256
recursiveReduce: 0.010651s
cpu reduce cpu_sum: 2139353471
reduceNeighbored: 0.032862s
gpu Neighbored  gpu_sum: 2139353471 <<<grid 65536 block 256>>>
reduceNeighboredLess: 0.003544s
gpu Neighbored2 gpu_sum: 2139353471 <<<grid 65536 block 256>>>
reduceInterleaved: 0.003202s
gpu Interleaved gpu_sum: 2139353471 <<<grid 65536 block 256>>>
reduceUnrolling2: 0.001767s
gpu Unrolling2  gpu_sum: 2139353471 <<<grid 32768 block 256>>>
reduceUnrolling4: 0.001155s
gpu Unrolling4  gpu_sum: 2139353471 <<<grid 16384 block 256>>>
reduceUnrolling8: 0.000831s
gpu Unrolling8  gpu_sum: 2139353471 <<<grid 8192 block 256>>>
reduceUnrollWarps8: 0.000951s
gpu UnrollWarp8 gpu_sum: 2139353471 <<<grid 8192 block 256>>>
reduceCompleteUnrollWarps8: 0.000864s
gpu Cmptnroll8  gpu_sum: 2139353471 <<<grid 8192 block 256>>>
reduceCompleteUnroll: 0.000824s
gpu Cmptnroll   gpu_sum: 2139353471 <<<grid 8192 block 256>>>

**********

xmake run 03 512
    with array size 16777216  grid 32768 block 512
recursiveReduce: 0.010152s
cpu reduce cpu_sum: 2139353471
reduceNeighbored: 0.023165s
gpu Neighbored  gpu_sum: 2139353471 <<<grid 32768 block 512>>>
reduceNeighboredLess: 0.003586s
gpu Neighbored2 gpu_sum: 2139353471 <<<grid 32768 block 512>>>
reduceInterleaved: 0.004122s
gpu Interleaved gpu_sum: 2139353471 <<<grid 32768 block 512>>>
reduceUnrolling2: 0.001957s
gpu Unrolling2  gpu_sum: 2139353471 <<<grid 16384 block 512>>>
reduceUnrolling4: 0.001248s
gpu Unrolling4  gpu_sum: 2139353471 <<<grid 8192 block 512>>>
reduceUnrolling8: 0.000824s
gpu Unrolling8  gpu_sum: 2139353471 <<<grid 4096 block 512>>>
reduceUnrollWarps8: 0.000871s
gpu UnrollWarp8 gpu_sum: 2139353471 <<<grid 4096 block 512>>>
reduceCompleteUnrollWarps8: 0.001809s
gpu Cmptnroll8  gpu_sum: 2139353471 <<<grid 4096 block 512>>>
reduceCompleteUnroll: 0.001710s
gpu Cmptnroll   gpu_sum: 2139353471 <<<grid 4096 block 512>>>

******

xmake run 03 1024
with array size 16777216  grid 16384 block 1024
recursiveReduce: 0.008659s
cpu reduce cpu_sum: 2139353471
reduceNeighbored: 0.028250s
gpu Neighbored  gpu_sum: 2139353471 <<<grid 16384 block 1024>>>
reduceNeighboredLess: 0.005869s
gpu Neighbored2 gpu_sum: 2139353471 <<<grid 16384 block 1024>>>
reduceInterleaved: 0.005356s
gpu Interleaved gpu_sum: 2139353471 <<<grid 16384 block 1024>>>
reduceUnrolling2: 0.004216s
gpu Unrolling2  gpu_sum: 2139353471 <<<grid 8192 block 1024>>>
reduceUnrolling4: 0.001749s
gpu Unrolling4  gpu_sum: 2139353471 <<<grid 4096 block 1024>>>
reduceUnrolling8: 0.001181s
gpu Unrolling8  gpu_sum: 2139353471 <<<grid 2048 block 1024>>>
reduceUnrollWarps8: 0.001290s
gpu UnrollWarp8 gpu_sum: 2139353471 <<<grid 2048 block 1024>>>
reduceCompleteUnrollWarps8: 0.001190s
gpu Cmptnroll8  gpu_sum: 2139353471 <<<grid 2048 block 1024>>>
reduceCompleteUnroll: 0.001278s
gpu Cmptnroll   gpu_sum: 2139353471 <<<grid 2048 block 1024>>>

*/
void reduceInteger(int argv1);
```