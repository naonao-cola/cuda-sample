﻿
/*
此示例演示了如何调用CUTLASS GEMM内核，并提供了一个简单的参考矩阵乘核验证其正确性。

CUTALSS Gemm模板在函数CutlassSgemmNN中实例化。这是内核计算通用矩阵积（GEMM）使用单精度浮点运算，并假设
所有矩阵都有列主布局。

线程块块大小选择为128x128x8，为大型矩阵提供了良好的性能。有关可用可调参数的更多说明，请参阅CUTLASS Parallel for All博客文章
在切割。

https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

除了定义和启动SGEMM内核外，此示例不使用任何其他组件或CUTLASS内的公用事业。这些实用程序在其他示例的其他地方进行了演示
在CUTLASS单元测试中很常见。

这个例子刻意地保持了与cutlass-1.3到1.3中的basic_gemm例子的相似性突出显示过渡到cutlass-2.0所需的最小差异量。

Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/


#include <iostream>
#include <sstream>
#include <vector>


#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

#include "cutlass/gemm/device/gemm.h"



/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,                           // Data-type of A matrix
                                                    cutlass::layout::ColumnMajor,    // Layout of A matrix
                                                    float,                           // Data-type of B matrix
                                                    cutlass::layout::ColumnMajor,    // Layout of B matrix
                                                    float,                           // Data-type of C matrix
                                                    cutlass::layout::ColumnMajor>;   // Layout of C matrix

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K},        // Gemm Problem dimensions
                                {A, lda},         // Tensor-ref for source matrix A
                                {B, ldb},         // Tensor-ref for source matrix B
                                {C, ldc},         // Tensor-ref for source matrix C
                                {C, ldc},         // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta});   // Scalars used in the Epilogue


    cutlass::Status status = gemm_operator(args);


    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

//文件中此点之后的源代码是使用CUDA Runtime API的通用CUDA
//以及简单的CUDA内核来初始化矩阵并计算一般矩阵积。

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(float* matrix, int rows, int columns, int seed = 0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < columns) {
        int offset = i + j * rows;
        // Generate arbitrary elements.
        int const k     = 16807;
        int const m     = 16;
        float     value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float* matrix, int rows, int columns, int seed = 0)
{
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
    InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);
    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float** matrix, int rows, int columns, int seed = 0)
{
    cudaError_t result;
    size_t sizeof_matrix = sizeof(float) * rows * columns;
    result = cudaMalloc(reinterpret_cast<void**>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    result = cudaMemset(*matrix, 0, sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    result = InitializeMatrix(*matrix, rows, columns, seed);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}


/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
            accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta)
{
    cudaError_t result;

    // Compute leading dimensions for each matrix.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(float) * ldc * N;

    // Define pointers to matrices in GPU device memory.
    float* A;
    float* B;
    float* C_cutlass;
    float* C_reference;


    result = AllocateMatrix(&A, M, K, 0);

    if (result != cudaSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, K, N, 17);

    if (result != cudaSuccess) {
        cudaFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return result;
    }

    result = AllocateMatrix(&C_reference, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return result;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }


    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Verify.
    //

    // Launch reference GEMM
    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    if (result != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    // Copy to host and verify equivalence.
    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy Reference GEMM results: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Free device memory allocations.
    //

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    //
    // Test for bit equivalence of results.
    //

    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;

        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char* arg[])
{

    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //

    // GEMM problem dimensions.
    int problem[3] = {128, 128, 128};

    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }

    // Scalars used for linear scaling the result of the matrix product.
    float scalars[2] = {1, 0};

    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(arg[i]);
        ss >> scalars[i - 4];
    }

    //
    // Run the CUTLASS GEMM test.
    //

    cudaError_t result = TestCutlassGemm(problem[0],   // GEMM M dimension
                                         problem[1],   // GEMM N dimension
                                         problem[2],   // GEMM K dimension
                                         scalars[0],   // alpha
                                         scalars[1]    // beta
    );

    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }

    // Exit.
    return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
