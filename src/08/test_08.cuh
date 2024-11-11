#include "../common/common.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
/**
介绍 https://zhuanlan.zhihu.com/p/666391239#:~:text=cublasSg
编译链接需要增加 add_links("cublas")
一个使用cuBLAS执行矩阵-向量乘法的简单示例库和一些随机生成的输入。

cublasStatus_t cublasSetVector(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy);
    n：要复制的元素数量。
    elemSize：每个元素的大小（以字节为单位）。
    hostPtr：指向主机内存中向量数据的指针。
    incx：主机内存中向量的步长（以元素数量为单位），通常设置为 1 表示连续的内存布局。如果向量是稀疏存储或有特殊的布局，可以设置不同的值。
    devicePtr：指向设备内存中目标位置的指针，用于存储复制过来的向量数据。
    incy：设备内存中目标向量的步长（以元素数量为单位），通常也设置为 1。

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *hostPtr, int lda, void *devicePtr, int ldb);
    rows：矩阵的行数。
    cols：矩阵的列数。
    elemSize：每个矩阵元素的大小（以字节为单位）。
    hostPtr：指向主机内存中矩阵数据的指针。
    lda：主机内存中矩阵的主维度（leading dimension），通常等于矩阵的列数，但在某些特殊的存储格式下可能不同。
    devicePtr：指向设备内存中目标位置的指针，用于存储复制过来的矩阵数据。
    ldb：设备内存中矩阵的主维度，通常等于矩阵的列数，但在某些特殊的存储格式下可能不同。

cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
    handle：CUBLAS 库的句柄，通过调用 cublasCreate 创建。
    trans：操作类型，CUBLAS_OP_N（不转置）或 CUBLAS_OP_T（转置），指定矩阵 A 是否转置。
    m：矩阵 A 的行数（如果 trans 为 CUBLAS_OP_N）或列数（如果 trans 为 CUBLAS_OP_T）。
    n：矩阵 A 的列数（如果 trans 为 CUBLAS_OP_N）或行数（如果 trans 为 CUBLAS_OP_T）。
    alpha：标量因子，用于矩阵与向量乘法的结果。
    A：指向设备内存中矩阵 A 的指针。
    lda：矩阵 A 的主维度（leading dimension），通常等于矩阵的列数，但在某些特殊的存储格式下可能不同。
    x：指向设备内存中向量 x 的指针。
    incx：向量 x 的步长（以元素数量为单位），通常设置为 1 表示连续的内存布局。如果向量是稀疏存储或有特殊的布局，可以设置不同的值。
    beta：标量因子，用于向量 y 的初始值。
    y：指向设备内存中结果向量 y 的指针。
    incy：向量 y 的步长（以元素数量为单位），通常设置为 1。

*/
void cublas();


/**
这个函数不支持编译

这个例子说明了OpenACC和CUDA库的使用应用程序。cuRAND用于用随机值填充两个输入矩阵。OpenACC用于使用并行和循环实现矩阵乘法指令。
最后，首先使用cuBLAS对每一行的值求和中所有值的和输出矩阵。
*/
void cuda_openacc();

/**
CUDA 6中引入的多gpu cuFFT XT库的示例用法。这个例子在系统中检测到的所有设备上执行一维正向FFT。,需要多gpu进行编译执行
*/
#include <cufftXt.h>
void cufft_multi();


/**
cuFFT库的一个示例用法。本例执行一维转发FFT。
*/
void cufft();

/**
 *这是一个演示使用cuSPARSE库来执行对随机生成的数据进行稀疏矩阵向量乘法。
真要针对稀疏矩阵，编译失败，未定义的函数。暂时先不管
 */
 #include <cusparse.h>
#include <cusparse_v2.h>
void cusparse();