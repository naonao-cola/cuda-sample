
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