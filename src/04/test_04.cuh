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
void transpose(int argv1, int argv2 = 0, int argv3 = 0, int argv4 = 0, int argv5 = 0);