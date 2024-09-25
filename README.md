# cuda-sample



```c++
//02 文件夹
#include "../common/common.h"
/**
 * 在本系统的第一个CUDA设备上显示各种信息，
 * 包括驱动程序版本，运行时版本，计算能力，全局内存字节等。

 Detected 1 CUDA Capable device(s)
Device 0: "NVIDIA GeForce RTX 4060 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.5 / 12.2
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 8.00 MBytes (8585216000 bytes)
  GPU Clock rate:                                2010 MHz (2.01 GHz)
  Memory Clock rate:                             8001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Max Texture Dimension Size (x,y,z)             1D=(131072), 2D=(131072,65536), 3D=(16384,16384,16384)
  Max Layered Texture Size (dim) x layers        1D=(32768) x 2048, 2D=(32768,32768) x 2048
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes

 */
void test_01();

/**
 * 显示来自主机和设备的线程块和网格的维度。
 grid.x 2 grid.y 1 grid.z 1
block.x 3 block.y 1 block.z 1
threadIdx:(0, 0, 0)
threadIdx:(1, 0, 0)
threadIdx:(2, 0, 0)
threadIdx:(0, 0, 0)
threadIdx:(1, 0, 0)
threadIdx:(2, 0, 0)
blockIdx:(1, 0, 0)
blockIdx:(1, 0, 0)
blockIdx:(1, 0, 0)
blockIdx:(0, 0, 0)
blockIdx:(0, 0, 0)
blockIdx:(0, 0, 0)
blockDim:(3, 1, 1)
blockDim:(3, 1, 1)
blockDim:(3, 1, 1)
blockDim:(3, 1, 1)
blockDim:(3, 1, 1)
blockDim:(3, 1, 1)
gridDim:(2, 1, 1)
gridDim:(2, 1, 1)
gridDim:(2, 1, 1)
gridDim:(2, 1, 1)
gridDim:(2, 1, 1)
gridDim:(2, 1, 1)
 */
void test_02();

/**
 * 这个例子有助于可视化线程/块id和之间的关系偏移到数据。对于每个CUDA线程，本例显示块内线程ID，块间块ID，全局坐标a
 * 线程，将计算出的偏移量转换为输入数据，并在此输入数据偏移量。

 Using Device 0: NVIDIA GeForce RTX 4060 Laptop GPU

Matrix: (8.6)
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47

blockDim:(4, 2, 1)
gridDim:(2, 3, 1)
thread_id (0,0) block_id (1,2) coordinate (4,4) global index 36 ival 36
thread_id (1,0) block_id (1,2) coordinate (5,4) global index 37 ival 37
thread_id (2,0) block_id (1,2) coordinate (6,4) global index 38 ival 38
thread_id (3,0) block_id (1,2) coordinate (7,4) global index 39 ival 39
thread_id (0,1) block_id (1,2) coordinate (4,5) global index 44 ival 44
thread_id (1,1) block_id (1,2) coordinate (5,5) global index 45 ival 45
thread_id (2,1) block_id (1,2) coordinate (6,5) global index 46 ival 46
thread_id (3,1) block_id (1,2) coordinate (7,5) global index 47 ival 47
thread_id (0,0) block_id (0,1) coordinate (0,2) global index 16 ival 16
thread_id (1,0) block_id (0,1) coordinate (1,2) global index 17 ival 17
thread_id (2,0) block_id (0,1) coordinate (2,2) global index 18 ival 18
thread_id (3,0) block_id (0,1) coordinate (3,2) global index 19 ival 19
thread_id (0,1) block_id (0,1) coordinate (0,3) global index 24 ival 24
thread_id (1,1) block_id (0,1) coordinate (1,3) global index 25 ival 25
thread_id (2,1) block_id (0,1) coordinate (2,3) global index 26 ival 26
thread_id (3,1) block_id (0,1) coordinate (3,3) global index 27 ival 27
thread_id (0,0) block_id (0,2) coordinate (0,4) global index 32 ival 32
thread_id (1,0) block_id (0,2) coordinate (1,4) global index 33 ival 33
thread_id (2,0) block_id (0,2) coordinate (2,4) global index 34 ival 34
thread_id (3,0) block_id (0,2) coordinate (3,4) global index 35 ival 35
thread_id (0,1) block_id (0,2) coordinate (0,5) global index 40 ival 40
thread_id (1,1) block_id (0,2) coordinate (1,5) global index 41 ival 41
thread_id (2,1) block_id (0,2) coordinate (2,5) global index 42 ival 42
thread_id (3,1) block_id (0,2) coordinate (3,5) global index 43 ival 43
thread_id (0,0) block_id (1,0) coordinate (4,0) global index  4 ival  4
thread_id (1,0) block_id (1,0) coordinate (5,0) global index  5 ival  5
thread_id (2,0) block_id (1,0) coordinate (6,0) global index  6 ival  6
thread_id (3,0) block_id (1,0) coordinate (7,0) global index  7 ival  7
thread_id (0,1) block_id (1,0) coordinate (4,1) global index 12 ival 12
thread_id (1,1) block_id (1,0) coordinate (5,1) global index 13 ival 13
thread_id (2,1) block_id (1,0) coordinate (6,1) global index 14 ival 14
thread_id (3,1) block_id (1,0) coordinate (7,1) global index 15 ival 15
thread_id (0,0) block_id (1,1) coordinate (4,2) global index 20 ival 20
thread_id (1,0) block_id (1,1) coordinate (5,2) global index 21 ival 21
thread_id (2,0) block_id (1,1) coordinate (6,2) global index 22 ival 22
thread_id (3,0) block_id (1,1) coordinate (7,2) global index 23 ival 23
thread_id (0,1) block_id (1,1) coordinate (4,3) global index 28 ival 28
thread_id (1,1) block_id (1,1) coordinate (5,3) global index 29 ival 29
thread_id (2,1) block_id (1,1) coordinate (6,3) global index 30 ival 30
thread_id (3,1) block_id (1,1) coordinate (7,3) global index 31 ival 31
thread_id (0,0) block_id (0,0) coordinate (0,0) global index  0 ival  0
thread_id (1,0) block_id (0,0) coordinate (1,0) global index  1 ival  1
thread_id (2,0) block_id (0,0) coordinate (2,0) global index  2 ival  2
thread_id (3,0) block_id (0,0) coordinate (3,0) global index  3 ival  3
thread_id (0,1) block_id (0,0) coordinate (0,1) global index  8 ival  8
thread_id (1,1) block_id (0,0) coordinate (1,1) global index  9 ival  9
thread_id (2,1) block_id (0,0) coordinate (2,1) global index 10 ival 10
thread_id (3,1) block_id (0,0) coordinate (3,1) global index 11 ival 11
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
Vector size 16777216
Execution configure <<<32768, 512>>>
sumArraysOnGPU Time elapsed 0.007903 sec
sumArraysOnHost Time elapsed 0.055673 sec 11.700000
Arrays match.

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
cpuRecursiveReduce: 0.000540s
cpu reduce              elapsed cpu_sum: 1048576
reduceNeighbored: 0.016646s
gpu Neighbored          elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpuRecursiveReduce: 0.006180s
gpu nested              elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpuRecursiveReduceNosync: 0.005104s
gpu nestedNosyn         elapsed gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpuRecursiveReduce2: 0.000327s
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

```c++
#include "../common/common.h"

/**
全局内存

使用静态声明的全局变量(devData)来存储的示例设备上的浮点值。
Host:   copied 3.140000 to the global variable
Device: the value of the global variable is 3.140000
Host:   the value changed by the kernel to 5.140000
*/
void globalVariable();

/**
内存复制

使用CUDA的内存复制API将数据传输到和从设备。在这种情况下，cudaMalloc用于在GPU和cudaMemcpy用于将主机内存的内容传输到数组使用cudaMalloc分配。
device 0: NVIDIA GeForce RTX 4060 Laptop GPU memory size 4194304 nbyte 16.00MB
*/
void memTransfer();

/**
锁页内存

使用CUDA的内存复制API将数据传输到和从设备。在这种情况下，cudaMalloc用于在GPU和cudaMemcpy用于将主机内存的内容传输到数组
使用cudaMalloc分配。使用cudaMallocHost来分配主机内存，以创建一个页面锁定的主机阵列。

device 0: NVIDIA GeForce RTX 4060 Laptop GPU memory size 4194304 nbyte 16.00MB canMap 1
*/
void pinMemTransfer();

/**
不对齐读取

这个例子演示了读取不对齐对性能的影响强制在浮点数*上发生不对齐的读取。

 with array size 1048576
sumArraysOnHost: 0.002163s
warmup: 0.013608s
warmup     <<< 1024, 1024 >>> offset    5
readOffset: 0.000237s
readOffset <<< 1024, 1024 >>> offset    5

 with array size 1048576
sumArraysOnHost: 0.002761s
warmup: 0.017136s
warmup     <<< 1024, 1024 >>> offset    7
readOffset: 0.000222s
readOffset <<< 1024, 1024 >>> offset    7

 with array size 1048576
sumArraysOnHost: 0.001934s
warmup: 0.010994s
warmup     <<< 1024, 1024 >>> offset    8
readOffset: 0.000173s
readOffset <<< 1024, 1024 >>> offset    8

 with array size 1048576
sumArraysOnHost: 0.002479s
warmup: 0.011114s
warmup     <<< 1024, 1024 >>> offset    0
readOffset: 0.000277s
readOffset <<< 1024, 1024 >>> offset    0

 with array size 1048576
sumArraysOnHost: 0.001780s
warmup: 0.008403s
warmup     <<< 1024, 1024 >>> offset    9
readOffset: 0.000139s
readOffset <<< 1024, 1024 >>> offset    9

 with array size 1048576
sumArraysOnHost: 0.003487s
warmup: 0.009699s
warmup     <<< 1024, 1024 >>> offset   15
readOffset: 0.000155s
readOffset <<< 1024, 1024 >>> offset   15
*/
void readSegment(int argv1);

/**
这个例子演示了读取不对齐对性能的影响强制在浮点数*上发生不对齐的读取。减少下面还包括通过展开的未对齐读取对性能的影响。
 with array size 1048576
warmup: 0.009393s
warmup     <<< 2048,  512 >>> offset    0
readOffset: 0.000165s
readOffset <<< 2048,  512 >>> offset    0
readOffsetUnroll2: 0.000217s
unroll2    <<< 1024,  512 >>> offset    0
readOffsetUnroll4: 0.000239s
unroll4    <<<  512,  512 >>> offset    0

 with array size 1048576
warmup: 0.008812s
warmup     <<< 2048,  512 >>> offset    5
readOffset: 0.000145s
readOffset <<< 2048,  512 >>> offset    5
readOffsetUnroll2: 0.000189s
unroll2    <<< 1024,  512 >>> offset    5
readOffsetUnroll4: 0.000244s
unroll4    <<<  512,  512 >>> offset    5

 with array size 1048576
warmup: 0.008406s
warmup     <<< 2048,  512 >>> offset    7
readOffset: 0.000251s
readOffset <<< 2048,  512 >>> offset    7
readOffsetUnroll2: 0.000272s
unroll2    <<< 1024,  512 >>> offset    7
readOffsetUnroll4: 0.000261s
unroll4    <<<  512,  512 >>> offset    7

with array size 1048576
warmup: 0.008074s
warmup     <<< 2048,  512 >>> offset   11
readOffset: 0.000143s
readOffset <<< 2048,  512 >>> offset   11
readOffsetUnroll2: 0.000307s
unroll2    <<< 1024,  512 >>> offset   11
readOffsetUnroll4: 0.000276s
unroll4    <<<  512,  512 >>> offset   11

 with array size 1048576
warmup: 0.008596s
warmup     <<< 2048,  512 >>> offset   13
readOffset: 0.000141s
readOffset <<< 2048,  512 >>> offset   13
readOffsetUnroll2: 0.000194s
unroll2    <<< 1024,  512 >>> offset   13
readOffsetUnroll4: 0.000433s
unroll4    <<<  512,  512 >>> offset   13

 with array size 1048576
warmup: 0.006268s
warmup     <<< 2048,  512 >>> offset   15
readOffset: 0.000148s
readOffset <<< 2048,  512 >>> offset   15
readOffsetUnroll2: 0.000238s
unroll2    <<< 1024,  512 >>> offset   15
readOffsetUnroll4: 0.000168s
unroll4    <<<  512,  512 >>> offset   15

*/
void readSegmentUnroll(int argv1);

/**
结构体的数组

一个使用结构数组在设备上存储数据的简单示例。此示例用于研究数据布局对性能的影响GPU。AoS:一次连续的64位读取来获取x和y(最多300个周期)
warmup: 0.029178s
warmup      <<< 32768, 128 >>>
testInnerStruct: 0.001097s
innerstruct <<< 32768, 128 >>>

*/
void simpleMathAoS();
/**
数组的结构体

一个使用数组结构体在设备上存储数据的简单示例。本例主要研究数据布局对GPU性能的影响。SoA:连续读取x和y
warmup2: 0.014121s
warmup2      <<< 32768, 128 >>>
testInnerArray: 0.000793s
innerarray   <<< 32768, 128 >>>
*/
void simpleMathSoA();

/**
同等情况下，零拷贝内存比这杯内存快几十倍。数据量小的情况下，cpu更快

这个例子演示了如何使用零拷贝内存来消除对显式地在主机和设备之间发出内存操作。通过映射主机上，页锁内存进入设备的地址空间，地址可以
直接引用主机阵列，并通过PCIe总线传输其内容。这个例子比较了在有零拷贝内存和没有零拷贝内存的情况下执行向量加法。

Vector size 32768 power 15  nbytes  128 KB
sumArraysOnHost: 0.000081s
sumArrays: 0.007751s
sumArraysZeroCopy: 0.000166s

Vector size 262144 power 18  nbytes    1 MB
sumArraysOnHost: 0.000531s
sumArrays: 0.009757s
sumArraysZeroCopy: 0.000179s
*/
void sumArrayZerocpy(int argv1);


/**
cudaMallocManaged 好像要慢一些

这个例子演示了使用CUDA托管内存来实现矩阵加法。在本例中，可以在主机上解引用任意指针和设备。CUDA将自动管理数据的传输
根据应用程序的需要设置GPU。程序员不需要这样做使用cudaMemcpy, cudaHostGetDevicePointer，或任何其他CUDA API
显式传输数据。此外，由于CUDA管理的内存不强制驻留在一个地方，它可以被转移到最优内存空间，不需要每次通过PCIe总线往返
执行跨设备引用(需要零复制和UVA)。

Matrix size: nx 1024 ny 1024
initialization: 0.046260s
sumMatrixOnHost: 0.000610s
sumMatrixGPU: 0.001150s
sumMatrix on gpu :       <<<(32,32), (32,32)>>>

Matrix size: nx 8192 ny 8192
initialization: 2.791109s
sumMatrixOnHost: 0.063782s
sumMatrixGPU: 0.159918s
sumMatrix on gpu :       <<<(256,256), (32,32)>>>

*/
void sumMatrixGPUManaged(int argv1);


/**
这个要比上一个快很多

这个例子演示了使用显式CUDA内存传输来实现矩阵加法。这段代码与sumMatrixGPUManaged形成对比。
CUDA托管内存用于删除所有显式内存传输，并抽象出物理上独立的地址空间的概念。

Matrix size: nx 4096 ny 4096
initialData: 0.657863s
sumMatrixOnHost: 0.033467s
sumMatrixGPU: 0.001802s
sumMatrix on gpu :       <<<(128,128), (32,32)>>>

*/
void sumMatrixGPUManual(int argv1);


/**
不建议偏移

这个例子演示了写不对齐对性能的影响强制在浮点*上发生不对齐的写操作。

 with array size 1048576
warmup: 0.008561s
warmup      <<< 2048,  512 >>> offset    0
writeOffset: 0.000132s
writeOffset <<< 2048,  512 >>> offset    0
writeOffsetUnroll2: 0.000194s
unroll2     <<< 1024,  512 >>> offset    0
writeOffsetUnroll4: 0.000278s
unroll4     <<<  512,  512 >>> offset    0

增加偏移量结果不对
 with array size 1048576
warmup: 0.008188s
warmup      <<< 2048,  512 >>> offset    1
writeOffset: 0.000142s
writeOffset <<< 2048,  512 >>> offset    1
different on 1th element: host 2.100000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll2: 0.000246s
unroll2     <<< 1024,  512 >>> offset    1
different on 1th element: host 2.100000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll4: 0.000187s
unroll4     <<<  512,  512 >>> offset    1
different on 1th element: host 2.100000 gpu 2.060000
Arrays do not match.

8的偏移也不对
 with array size 1048576
warmup: 0.007886s
warmup      <<< 2048,  512 >>> offset    8
writeOffset: 0.000308s
writeOffset <<< 2048,  512 >>> offset    8
different on 8th element: host 2.480000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll2: 0.000366s
unroll2     <<< 1024,  512 >>> offset    8
different on 8th element: host 2.480000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll4: 0.000174s
unroll4     <<<  512,  512 >>> offset    8
different on 8th element: host 2.480000 gpu 2.060000
Arrays do not match.

16的偏移也不对
 with array size 1048576
warmup: 0.005550s
warmup      <<< 2048,  512 >>> offset   16
writeOffset: 0.000227s
writeOffset <<< 2048,  512 >>> offset   16
different on 16th element: host 2.040000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll2: 0.000216s
unroll2     <<< 1024,  512 >>> offset   16
different on 16th element: host 2.040000 gpu 2.060000
Arrays do not match.

writeOffsetUnroll4: 0.000171s
unroll4     <<<  512,  512 >>> offset   16
different on 16th element: host 2.040000 gpu 2.060000
Arrays do not match.
*/
void writeSegment(int argv1);

/**
//04文件夹
第五种 访存效率最高

应用于矩阵转置的各种内存访问模式优化内核。转置内核:按列读取，按行写入+展开4块

核为0
 with matrix nx 2048 ny 2048 with kernel 0
warmup: 0.007674s
CopyRow        elapsed 0.000268 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 125.211288 GB
 with matrix nx 2048 ny 2048 with kernel 0
warmup: 0.009455s
CopyRow        elapsed 0.000241 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 139.206223 GB
 with matrix nx 2048 ny 2048 with kernel 0
warmup: 0.006636s
CopyRow        elapsed 0.000198 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 169.563232 GB
 with matrix nx 2048 ny 2048 with kernel 0
warmup: 0.009644s
CopyRow        elapsed 0.000424 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 79.154945 GB

核是1
 with matrix nx 2048 ny 2048 with kernel 1
warmup: 0.012380s
CopyCol        elapsed 0.000349 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 96.132164 GB
 with matrix nx 2048 ny 2048 with kernel 1
warmup: 0.009231s
CopyCol        elapsed 0.000286 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 117.379059 GB
with matrix nx 2048 ny 2048 with kernel 1
warmup: 0.008167s
CopyCol        elapsed 0.000362 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 92.651405 GB
 with matrix nx 2048 ny 2048 with kernel 1
warmup: 0.009076s
CopyCol        elapsed 0.000293 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 114.513824 GB

核是2
 with matrix nx 2048 ny 2048 with kernel 2
warmup: 0.008247s
NaiveRow       elapsed 0.000426 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 78.756287 GB
with matrix nx 2048 ny 2048 with kernel 2
warmup: 0.005966s
NaiveRow       elapsed 0.000442 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 75.910187 GB
 with matrix nx 2048 ny 2048 with kernel 2
warmup: 0.009561s
NaiveRow       elapsed 0.000477 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 70.333580 GB

核是3
with matrix nx 2048 ny 2048 with kernel 3
warmup: 0.006614s
NaiveCol       elapsed 0.000207 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 162.139969 GB
with matrix nx 2048 ny 2048 with kernel 3
warmup: 0.008293s
NaiveCol       elapsed 0.000243 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 138.113342 GB
 with matrix nx 2048 ny 2048 with kernel 3
warmup: 0.005356s
NaiveCol       elapsed 0.000234 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 143.463287 GB
with matrix nx 2048 ny 2048 with kernel 3
warmup: 0.008733s
NaiveCol       elapsed 0.000251 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 133.653839 GB

核是4
 with matrix nx 2048 ny 2048 with kernel 4
warmup: 0.006915s
Unroll4Row     elapsed 0.000297 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 112.951439 GB
 with matrix nx 2048 ny 2048 with kernel 4
warmup: 0.005816s
Unroll4Row     elapsed 0.000305 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 109.951164 GB
 with matrix nx 2048 ny 2048 with kernel 4
warmup: 0.005810s
Unroll4Row     elapsed 0.000287 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 116.988770 GB
with matrix nx 2048 ny 2048 with kernel 4
warmup: 0.006243s
Unroll4Row     elapsed 0.000255 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 131.530365 GB

核是5
 with matrix nx 2048 ny 2048 with kernel 5
warmup: 0.006378s
Unroll4Col     elapsed 0.000134 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 250.422577 GB
 with matrix nx 2048 ny 2048 with kernel 5
warmup: 0.006149s
Unroll4Col     elapsed 0.000150 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 223.748001 GB
 with matrix nx 2048 ny 2048 with kernel 5
warmup: 0.008072s
Unroll4Col     elapsed 0.000181 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 185.424881 GB

核是6
 with matrix nx 2048 ny 2048 with kernel 6
warmup: 0.006162s
DiagonalRow    elapsed 0.000285 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 117.673485 GB
with matrix nx 2048 ny 2048 with kernel 6
warmup: 0.006385s
DiagonalRow    elapsed 0.000286 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 117.281242 GB
 with matrix nx 2048 ny 2048 with kernel 6
warmup: 0.008558s
DiagonalRow    elapsed 0.000528 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 63.538368 GB
 with matrix nx 2048 ny 2048 with kernel 6
warmup: 0.005574s
DiagonalRow    elapsed 0.000275 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 122.062004 GB
 with matrix nx 2048 ny 2048 with kernel 6
warmup: 0.006146s
DiagonalRow    elapsed 0.000443 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 75.746765 GB

核是7
with matrix nx 2048 ny 2048 with kernel 7
warmup: 0.011438s
DiagonalCol    elapsed 0.000301 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 111.431107 GB
 with matrix nx 2048 ny 2048 with kernel 7
warmup: 0.006564s
DiagonalCol    elapsed 0.000250 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 134.163483 GB
 with matrix nx 2048 ny 2048 with kernel 7
warmup: 0.008831s
DiagonalCol    elapsed 0.000302 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 111.079315 GB
 with matrix nx 2048 ny 2048 with kernel 7
warmup: 0.006340s
DiagonalCol    elapsed 0.000363 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 92.408066 GB
*/
void transpose(int argv1, int argv2=0, int argv3=0, int argv4=0, int argv5=0);
```