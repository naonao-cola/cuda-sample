
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