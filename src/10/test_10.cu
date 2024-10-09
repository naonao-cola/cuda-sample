
#include "test_10.cuh"
#include <nvtx3/nvToolsExt.h>
constexpr size_t M = 200;
constexpr size_t N = 400;
constexpr size_t K = 300;

void cublass_mm(std::vector<float>& mat_a, std::vector<float>& mat_b)
{
    // Show the test matrix data.
    if (M == 2 && N == 4 && K == 3) {
        //      1 2 3
        // A =
        //      4 5 6
        std::iota(mat_a.begin(), mat_a.end(), 1.0f);
        //      1  2  3  4
        // B =  5  6  7  8
        //      9 10 11 12

        std::iota(mat_b.begin(), mat_b.end(), 1.0f);
        //           38 44  50  56
        // C = AB =
        //           83 98 113 128

        std::cout << "A = \n";
        for (size_t i = 0; i < M * K; ++i) {
            std::cout << mat_a[i] << '\t';
            if ((i + 1) % K == 0) {
                std::cout << '\n';
            }
        }
        std::cout << "B = \n";
        for (size_t i = 0; i < K * N; ++i) {
            std::cout << mat_b[i] << '\t';
            if ((i + 1) % N == 0) {
                std::cout << '\n';
            }
        }
    }

    cublasHandle_t handle = 0;
    CHECK_CUBLAS(cublasCreate(&handle));

    float* device_mat_a = nullptr;
    float* device_mat_b = nullptr;
    float* device_mat_c = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mat_a), M * K * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mat_b), K * N * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mat_c), M * N * sizeof(float)));
    /**
    rows：要复制的矩阵的行数。
    cols：要复制的矩阵的列数。
    elemSize：矩阵中每个元素的大小（以字节为单位）。例如，如果矩阵元素是单精度浮点数（float），则elemSize为sizeof(float)。
    hostMat：指向主机内存中矩阵数据的指针。
    lda：主机矩阵中每行的元素数量（通常等于列数，用于可能的填充）。
    deviceMat：指向设备内存中存储矩阵的位置的指针。
    ldb：设备矩阵中每行的元素数量（通常等于列数，用于可能的填充）。
    */
    CHECK_CUBLAS(cublasSetMatrix(M, K, sizeof(float), mat_a.data(), M, device_mat_a, M));
    CHECK_CUBLAS(cublasSetMatrix(K, N, sizeof(float), mat_b.data(), K, device_mat_b, K));

    float alpha = 1.0f;
    float beta  = 0.0f;
    /**
    handle：CUBLAS 库的句柄，通过cublasCreate创建。
    transa、transb：指定矩阵 A 和 B 是否转置，取值可以是CUBLAS_OP_N（不转置）、CUBLAS_OP_T（转置）、CUBLAS_OP_C（共轭转置，仅在复数运算时有效）。
    m：矩阵 C 的行数。
    n：矩阵 C 的列数。
    k：矩阵 A 的列数（同时也是矩阵 B 的行数，如果 A 经过转置，则为 A 的行数）。
    alpha：标量因子，用于矩阵乘法结果。
    A：指向第一个矩阵（如果transa为CUBLAS_OP_N，则是m行k列的矩阵；如果转置，则是k行m列的矩阵）在设备内存中的指针。
    lda：矩阵 A 的主维度大小（如果不转置，通常等于 A 的行数；如果转置，通常等于 A 的列数）。
    B：指向第二个矩阵（如果transb为CUBLAS_OP_N，则是k行n列的矩阵；如果转置，则是n行k列的矩阵）在设备内存中的指针。
    ldb：矩阵 B 的主维度大小（如果不转置，通常等于 B 的行数；如果转置，通常等于 B 的列数）。
    beta：标量因子，用于矩阵 C。
    C：指向结果矩阵在设备内存中的指针，是m行n列的矩阵。
    ldc：矩阵 C 的主维度大小（通常等于 C 的行数）。
    */

    // Performs operation using cublas
    // C = alpha * transa(A)*transb(B) + beta * C
    // `transa` indicates whether the matrix A is transposed or not.
    // `transb` indicates whether the matrix B is transposed or not.
    // A: m x k
    // B: k x n
    // C: m x n
    // LDA, LDB, LDC are the leading dimensions of the three matrices,
    // respectively.
    // If C = A x B is calculated, there is alpha = 1.0, beta = 0.0,
    // transa = CUBLAS_OP_N, transb = CUBLAS_OP_N

    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                          cublasOperation_t transa, cublasOperation_t
    //                          transb, int m, int n, int k, const float *alpha,
    //                          const float *A, int lda,
    //                          const float *B, int ldb,
    //                          const float *beta,
    //                          float *C, int ldc);
    nvtxRangePushA("cublasSgemm");
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, device_mat_b, N, device_mat_a, K, &beta, device_mat_c, N));
    nvtxRangePop();
    std::vector<float> mat_c(M * N, 0.0f);
    CHECK(cudaMemcpy(mat_c.data(), device_mat_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (M == 2 && N == 4 && K == 3) {
        std::cout << "C = AB = \n";
        for (size_t i = 0; i < M * N; ++i) {
            std::cout << mat_c[i] << '\t';
            if ((i + 1) % N == 0) {
                std::cout << '\n';
            }
        }
    }

    cudaFree(device_mat_a);
    cudaFree(device_mat_b);
    cudaFree(device_mat_c);
    cublasDestroy(handle);
}

void test_cublass_mm()
{

    // Initialize the host input matrix;
    std::vector<float> mat_a(M * K, 0.0f);
    std::vector<float> mat_b(K * N, 0.0f);

    // Fill the matrices with random numbers
    std::random_device                 rd;
    std::mt19937                       gen(rd());
    std::uniform_int_distribution<int> dis(0, 100000);
    auto                               rand_num = [&dis, &gen]() { return dis(gen); };
    std::generate(mat_a.begin(), mat_a.end(), rand_num);
    std::generate(mat_b.begin(), mat_b.end(), rand_num);

    cublass_mm(mat_a, mat_b);
    mat_a.clear();
    mat_a.shrink_to_fit();
    mat_b.clear();
    mat_b.shrink_to_fit();
}


#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel1(float* out, const float* in, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        out[idx] = in[idx] + 1.0f;
    }
}

__global__ void kernel2(float* out, const float* in, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        out[idx] = in[idx] * 2.0f;
    }
}

void test_graph(){

    const int    numElements = 1024;
    const size_t size        = numElements * sizeof(float);

    float* h_in  = new float[numElements];
    float* h_out = new float[numElements];

    for (int i = 0; i < numElements; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 捕获CUDA图
    cudaGraph_t     graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
    kernel1<<<1, numElements, 0, stream>>>(d_out, d_in, numElements);
    kernel2<<<1, numElements, 0, stream>>>(d_in, d_out, numElements);
    cudaMemcpyAsync(h_out, d_in, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamEndCapture(stream, &graph);

    // 实例化CUDA图
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // 执行CUDA图
    for (int i = 0; i < 10; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
        cudaGraphLaunch(instance, stream);
        cudaMemcpyAsync(h_out, d_in, size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime( &elapsedTime,start, stop );
        printf("Time to generate:  %3.1f ms\n", elapsedTime);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 清理资源
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}