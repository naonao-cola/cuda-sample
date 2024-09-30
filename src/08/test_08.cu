#include "test_08.cuh"



/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

/*
 * Generate a vector of length N with random single-precision floating-point values between 0 and 100.
 */
void generate_random_vector(int N, float** outX)
{
    int    i;
    double rMax = (double)RAND_MAX;
    float* X    = (float*)malloc(sizeof(float) * N);
    for (i = 0; i < N; i++) {
        int    r  = rand();
        double dr = (double)r;
        X[i]      = (dr / rMax) * 100.0;
    }
    *outX = X;
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float** outA)
{
    int    i, j;
    double rMax = (double)RAND_MAX;
    float* A    = (float*)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++) {
        // For each row
        for (i = 0; i < M; i++) {
            double dr    = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }
    *outA = A;
}

void cublas()
{
    int            i;
    float *        A, *dA;
    float *        X, *dX;
    float *        Y, *dY;
    float          beta;
    float          alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta  = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    CHECK(cudaMalloc((void**)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void**)&dX, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&dY, sizeof(float) * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetVector(N, sizeof(float), X, 1, dX, 1));
    CHECK_CUBLAS(cublasSetVector(M, sizeof(float), Y, 1, dY, 1));
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, dA, M, dX, 1, &beta, dY, 1));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetVector(M, sizeof(float), dY, 1, Y, 1));

    for (i = 0; i < 10; i++) {
        printf("%2.2f\n", Y[i]);
    }

    printf("...\n");

    free(A);
    free(X);
    free(Y);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dY));
    CHECK_CUBLAS(cublasDestroy(handle));
}

#define M 1024
#define N 1024
#define P 1024

void cuda_openacc()
{
    //     int i, j, k;
    //     float* __restrict__ d_A;
    //     float* __restrict__ d_B;
    //     float* __restrict__ d_C;
    //     float*            d_row_sums;
    //     float             total_sum;
    //     curandGenerator_t rand_state    = 0;
    //     cublasHandle_t    cublas_handle = 0;

    //     // Initialize the cuRAND and cuBLAS handles.
    //     CHECK_CURAND(curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT));
    //     CHECK_CUBLAS(cublasCreate(&cublas_handle));

    //     // Allocate GPU memory for the input matrices, output matrix, and row sums.
    //     CHECK(cudaMalloc((void**)&d_A, sizeof(float) * M * N));
    //     CHECK(cudaMalloc((void**)&d_B, sizeof(float) * N * P));
    //     CHECK(cudaMalloc((void**)&d_C, sizeof(float) * M * P));
    //     CHECK(cudaMalloc((void**)&d_row_sums, sizeof(float) * M));

    //     // Generate random values in both input matrices.
    //     CHECK_CURAND(curandGenerateUniform(rand_state, d_A, M * N));
    //     CHECK_CURAND(curandGenerateUniform(rand_state, d_B, N * P));

    // //     // Perform a matrix multiply parallelized across gangs and workers
    // //     //执行跨帮派和工人并行的矩阵乘法
    // #pragma acc parallel loop gang deviceptr(d_A, d_B, d_C)
    //     for (i = 0; i < M; i++) {
    // #pragma acc loop worker vector

    //         for (j = 0; j < P; j++) {
    //             float sum = 0.0f;

    //             for (k = 0; k < N; k++) {
    //                 sum += d_A[i * N + k] * d_B[k * P + j];
    //             }

    //             d_C[i * P + j] = sum;
    //         }
    //     }

    //     /*
    //      * 将cuBLAS设置为设备指针模式，表示所有标量都作为设备指针传递。
    //      */
    //     CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    //     // 对每行中包含的值求和。
    //     for (i = 0; i < M; i++) {
    //         CHECK_CUBLAS(cublasSasum(cublas_handle, P, d_C + (i * P), 1, d_row_sums + i));
    //     }

    //     /*
    // //      * 将cuBLAS设置回主机指针模式，表示所有标量都作为主机指针传递。
    // //      */
    //     CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    //     /*
    //      * 对所有行的和进行最后的求和，得到整个输出矩阵的总和。
    //      */
    //     CHECK_CUBLAS(cublasSasum(cublas_handle, M, d_row_sums, 1, &total_sum));
    //     CHECK(cudaDeviceSynchronize());

    //     // Release device memory
    //     CHECK(cudaFree(d_A));
    //     CHECK(cudaFree(d_B));
    //     CHECK(cudaFree(d_C));
    //     CHECK(cudaFree(d_row_sums));
    //     printf("Total sum = %f\n", total_sum);
}

/*
沿着函数cos(x)创建N个假采样。这些样本将存储为单精度浮点值。
 */
void generate_fake_samples(int n, float** out)
{
    int    i;
    float* result = (float*)malloc(sizeof(float) * n);
    double delta  = M_PI / 4.0;

    for (i = 0; i < n; i++) {
        result[i] = cos(i * delta);
    }

    *out = result;
}

/*
 将长度为实值的向量r转换为复值向量。
 */
void real_to_complex(float* r, cufftComplex** complx, int n)
{
    int i;
    (*complx) = (cufftComplex*)malloc(sizeof(cufftComplex) * n);

    for (i = 0; i < n; i++) {
        (*complx)[i].x = r[i];
        (*complx)[i].y = 0;
    }
}


/*
 * 检索当前系统中所有CUDA设备的设备id。
 */
int getAllGpus(int** gpus)
{
    int i;
    int nGpus;

    CHECK(cudaGetDeviceCount(&nGpus));

    *gpus = (int*)malloc(sizeof(int) * nGpus);

    for (i = 0; i < nGpus; i++) {
        (*gpus)[i] = i;
    }

    return nGpus;
}

void cufft_multi()
{
    // int            i;
    // int            n = 1024;
    // float*         samples;
    // cufftComplex*  complexSamples;
    // int*           gpus;
    // size_t*        workSize;
    // cufftHandle    plan = 0;
    // cudaLibXtDesc* dComplexSamples;

    // int nGPUs = getAllGpus(&gpus);
    // nGPUs     = nGPUs > 2 ? 2 : nGPUs;
    // workSize  = (size_t*)malloc(sizeof(size_t) * nGPUs);

    // // Setup the cuFFT Multi-GPU plan
    // CHECK_CUFFT(cufftCreate(&plan));
    // //CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    // CHECK_CUFFT(cufftXtSetGPUs(plan, 2, gpus));
    // CHECK_CUFFT(cufftMakePlan1d(plan, n, CUFFT_C2C, 1, workSize));

    // // Generate inputs
    // generate_fake_samples(n, &samples);
    // real_to_complex(samples, &complexSamples, n);
    // cufftComplex* complexFreq = (cufftComplex*)malloc(sizeof(cufftComplex) * n);

    // // Allocate memory across multiple GPUs and transfer the inputs into it
    // CHECK_CUFFT(cufftXtMalloc(plan, &dComplexSamples, CUFFT_XT_FORMAT_INPLACE));
    // CHECK_CUFFT(cufftXtMemcpy(plan, dComplexSamples, complexSamples, CUFFT_COPY_HOST_TO_DEVICE));

    // // Execute a complex-to-complex 1D FFT across multiple GPUs
    // CHECK_CUFFT(cufftXtExecDescriptorC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD));

    // // Retrieve the results from multiple GPUs into host memory
    // CHECK_CUFFT(cufftXtMemcpy(plan, complexSamples, dComplexSamples, CUFFT_COPY_DEVICE_TO_HOST));

    // printf("Fourier Coefficients:\n");

    // for (i = 0; i < 30; i++) {
    //     printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x, complexFreq[i].y);
    // }

    // free(gpus);
    // free(samples);
    // free(complexSamples);
    // free(complexFreq);
    // free(workSize);

    // CHECK_CUFFT(cufftXtFree(dComplexSamples));
    // CHECK_CUFFT(cufftDestroy(plan));
}

int nprints = 30;

void cufft()
{
    int           i;
    int           n = 2048;
    float*        samples;
    cufftHandle   plan = 0;
    cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

    // Input Generation
    generate_fake_samples(n, &samples);
    real_to_complex(samples, &complexSamples, n);
    complexFreq = (cufftComplex*)malloc(sizeof(cufftComplex) * n);
    printf("Initial Samples:\n");

    for (i = 0; i < nprints; i++) {
        printf("  %2.4f\n", samples[i]);
    }

    printf("  ...\n");

    // Setup the cuFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    // Allocate device memory
    CHECK(cudaMalloc((void**)&dComplexSamples, sizeof(cufftComplex) * n));

    // Transfer inputs into device memory
    CHECK(cudaMemcpy(dComplexSamples, complexSamples, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice));

    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK(cudaMemcpy(complexFreq, dComplexSamples, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost));

    printf("Fourier Coefficients:\n");

    for (i = 0; i < nprints; i++) {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x, complexFreq[i].y);
    }

    printf("  ...\n");

    free(samples);
    free(complexSamples);
    free(complexFreq);

    CHECK(cudaFree(dComplexSamples));
    CHECK_CUFFT(cufftDestroy(plan));
}

// int M = 1024;
// int N = 1024;

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix_2(int m, int n, float** outA)
{
    int    i, j;
    double rMax     = (double)RAND_MAX;
    float* A        = (float*)malloc(sizeof(float) * m * n);
    int    totalNnz = 0;

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            int    r    = rand();
            float* curr = A + (j * m + i);

            if (r % 3 > 0) {
                *curr = 0.0f;
            }
            else {
                double dr = (double)r;
                *curr     = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f) {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void cusparse()
{
    // int                row;
    // float *            A, *dA;
    // int*               dNnzPerRow;
    // float*             dCsrValA;
    // int*               dCsrRowPtrA;
    // int*               dCsrColIndA;
    // int                totalNnz;
    // float              alpha = 3.0f;
    // float              beta  = 4.0f;
    // float *            dX, *X;
    // float *            dY, *Y;
    // cusparseHandle_t   handle = 0;
    // cusparseMatDescr_t descr  = 0;

    // // Generate input
    // srand(9384);
    // int trueNnz = generate_random_dense_matrix_2(M, N, &A);
    // generate_random_vector(N, &X);
    // generate_random_vector(M, &Y);

    // // Create the cuSPARSE handle
    // CHECK_CUSPARSE(cusparseCreate(&handle));

    // // Allocate device memory for vectors and the dense form of the matrix A
    // CHECK(cudaMalloc((void**)&dX, sizeof(float) * N));
    // CHECK(cudaMalloc((void**)&dY, sizeof(float) * M));
    // CHECK(cudaMalloc((void**)&dA, sizeof(float) * M * N));
    // CHECK(cudaMalloc((void**)&dNnzPerRow, sizeof(int) * M));

    // // Construct a descriptor of the matrix A
    // CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    // CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    // CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // // Transfer the input vectors and dense matrix A to the device
    // CHECK(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(dY, Y, sizeof(float) * M, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    // // Compute the number of non-zero elements in A
    // CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA, M, dNnzPerRow, &totalNnz));

    // if (totalNnz != trueNnz) {
    //     fprintf(stderr, "Difference detected between cuSPARSE NNZ and true value: expected %d but got %d\n", trueNnz, totalNnz);
    // }

    // // Allocate device memory to store the sparse CSR representation of A
    // CHECK(cudaMalloc((void**)&dCsrValA, sizeof(float) * totalNnz));
    // CHECK(cudaMalloc((void**)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    // CHECK(cudaMalloc((void**)&dCsrColIndA, sizeof(int) * totalNnz));

    // // Convert A from a dense formatting to a CSR formatting, using the GPU
    // CHECK_CUSPARSE(cusparseSdense2Csr(handle, M, N, descr, dA, M, dNnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA));

    // // Perform matrix-vector multiplication with the CSR-formatted matrix A
    // CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, totalNnz, &alpha, descr, dCsrValA, dCsrRowPtrA, dCsrColIndA, dX, &beta, dY));

    // // Copy the result vector back to the host
    // CHECK(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));

    // for (row = 0; row < 10; row++) {
    //     printf("%2.2f\n", Y[row]);
    // }

    // printf("...\n");

    // free(A);
    // free(X);
    // free(Y);

    // CHECK(cudaFree(dX));
    // CHECK(cudaFree(dY));
    // CHECK(cudaFree(dA));
    // CHECK(cudaFree(dNnzPerRow));
    // CHECK(cudaFree(dCsrValA));
    // CHECK(cudaFree(dCsrRowPtrA));
    // CHECK(cudaFree(dCsrColIndA));

    // CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    // CHECK_CUSPARSE(cusparseDestroy(handle));
}