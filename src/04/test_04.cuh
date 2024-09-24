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
这个例子演示了使用CUDA托管内存来实现矩阵加法。在本例中，可以在主机上解引用任意指针和设备。CUDA将自动管理数据的传输
根据应用程序的需要设置GPU。程序员不需要这样做使用cudaMemcpy, cudaHostGetDevicePointer，或任何其他CUDA API
显式传输数据。此外，由于CUDA管理的内存不强制驻留在一个地方，它可以被转移到最优内存空间，不需要每次通过PCIe总线往返
执行跨设备引用(需要零复制和UVA)。

*/
void sumMatrixGPUManaged();