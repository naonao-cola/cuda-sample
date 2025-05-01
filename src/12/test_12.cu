#include "test_12.cuh"


/**
 * CUDA device function
 *
 * calculates the sum of val across the group g. The workspace array, x,
 * must be large enough to contain g.size() integers.
 */
__device__ int sumReduction(cg::thread_group g, int* x, int val)
{
    // rank of this thread in the group
    int lane = g.thread_rank();
    // for each iteration of this loop, the number of threads active in the
    // reduction, i, is halved, and each active thread (with index [lane])
    // performs a single summation of it's own value with that
    // of a "partner" (with index [lane+i]).
    for (int i = g.size() / 2; i > 0; i /= 2) {
        // store value for this thread in temporary array
        x[lane] = val;
        // synchronize all threads in group
        g.sync();
        if (lane < i)
            // active threads perform summation of their value with
            // their partner's value
            val += x[lane + i];
        // synchronize all threads in group
        g.sync();
    }
    // master thread in group returns result, and others return -1.
    if (g.thread_rank() == 0)
        return val;
    else
        return -1;
}


/**
 * CUDA kernel device code
 *
 * Creates cooperative groups and performs reductions
 */
__global__ void cgkernel()
{
    // threadBlockGroup includes all threads in the block
    cg::thread_block threadBlockGroup     = cg::this_thread_block();
    int              threadBlockGroupSize = threadBlockGroup.size();

    // workspace array in shared memory required for reduction
    extern __shared__ int workspace[];

    int input, output, expectedOutput;

    // input to reduction, for each thread, is its' rank in the group
    input = threadBlockGroup.thread_rank();

    // expected output from analytical formula (n-1)(n)/2
    // (noting that indexing starts at 0 rather than 1)
    expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

    // perform reduction
    output = sumReduction(threadBlockGroup, workspace, input);

    // master thread in group prints out result
    if (threadBlockGroup.thread_rank() == 0) {
        printf(" Sum of all ranks 0..%d in threadBlockGroup is %d (expected %d)\n\n", (int)threadBlockGroup.size() - 1, output, expectedOutput);

        printf(" Now creating %d groups, each of size 16 threads:\n\n", (int)threadBlockGroup.size() / 16);
    }

    threadBlockGroup.sync();

    // each tiledPartition16 group includes 16 threads
    cg::thread_block_tile<16> tiledPartition16 = cg::tiled_partition<16>(threadBlockGroup);

    // This offset allows each group to have its own unique area in the workspace
    // array
    int workspaceOffset = threadBlockGroup.thread_rank() - tiledPartition16.thread_rank();

    // input to reduction, for each thread, is its' rank in the group
    input = tiledPartition16.thread_rank();

    // expected output from analytical formula (n-1)(n)/2
    // (noting that indexing starts at 0 rather than 1)
    expectedOutput = 15 * 16 / 2;

    // Perform reduction
    output = sumReduction(tiledPartition16, workspace + workspaceOffset, input);

    // each master thread prints out result
    if (tiledPartition16.thread_rank() == 0)
        printf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d "
               "(expected %d)\n",
               output,
               expectedOutput);

    return;
}

void test_cg_01()
{
    cudaError_t err;
    int         blocksPerGrid   = 1;
    int         threadsPerBlock = 64;
    printf("\nLaunching a single block with %d threads...\n\n", threadsPerBlock);
    // we use the optional third argument to specify the size
    // of shared memory required in the kernel
    cgkernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("\n...Done.\n\n");
}

#if __CUDA_ARCH__ >= 700
template<bool writeSquareRoot>
__device__ void reduceBlockData(cuda::barrier<cuda::thread_scope_block>& barrier, cg::thread_block_tile<32>& tile32, double& threadSum, double* result)
{
    extern __shared__ double tmp[];

#    pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        threadSum += tile32.shfl_down(threadSum, offset);
    }
    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = threadSum;
    }

    auto token = barrier.arrive();

    barrier.wait(std::move(token));

    // The warp 0 will perform last round of reduction
    if (tile32.meta_group_rank() == 0) {
        double beta = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;

#    pragma unroll
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            beta += tile32.shfl_down(beta, offset);
        }

        if (tile32.thread_rank() == 0) {
            if (writeSquareRoot)
                *result = sqrt(beta);
            else
                *result = beta;
        }
    }
}
#endif

__global__ void normVecByDotProductAWBarrier(float* vecA, float* vecB, double* partialResults, int size)
{
#if __CUDA_ARCH__ >= 700
#    pragma diag_suppress static_var_with_dynamic_init
    cg::thread_block      cta  = cg::this_thread_block();
    cg::grid_group        grid = cg::this_grid();
    ;
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }

    cg::sync(cta);

    double threadSum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        threadSum += (double)(vecA[i] * vecB[i]);
    }

    // Each thread block performs reduction of partial dotProducts and writes to
    // global mem.
    reduceBlockData<false>(barrier, tile32, threadSum, &partialResults[blockIdx.x]);

    cg::sync(grid);

    // One block performs the final summation of partial dot products
    // of all the thread blocks and writes the sqrt of final dot product.
    if (blockIdx.x == 0) {
        threadSum = 0.0;
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
            threadSum += partialResults[i];
        }
        reduceBlockData<true>(barrier, tile32, threadSum, &partialResults[0]);
    }

    cg::sync(grid);

    const double finalValue = partialResults[0];

    // Perform normalization of vecA & vecB.
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        vecA[i] = (float)vecA[i] / finalValue;
        vecB[i] = (float)vecB[i] / finalValue;
    }
#endif
}

int runNormVecByDotProductAWBarrier(int argc, char** argv, int deviceId)
{
    float * vecA, *d_vecA;
    float * vecB, *d_vecB;
    double* d_partialResults;
    int     size = 10000;

    CHECK(cudaMallocHost(&vecA, sizeof(float) * size));
    CHECK(cudaMallocHost(&vecB, sizeof(float) * size));

    CHECK(cudaMalloc(&d_vecA, sizeof(float) * size));
    CHECK(cudaMalloc(&d_vecB, sizeof(float) * size));

    float baseVal = 2.0;
    for (int i = 0; i < size; i++) {
        vecA[i] = vecB[i] = baseVal;
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK(cudaMemcpyAsync(d_vecA, vecA, sizeof(float) * size, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_vecB, vecB, sizeof(float) * size, cudaMemcpyHostToDevice, stream));

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.
    int minGridSize = 0, blockSize = 0;
    CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)normVecByDotProductAWBarrier, 0, size));

    int smemSize = ((blockSize / 32) + 1) * sizeof(double);

    int numBlocksPerSm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

    int multiProcessorCount = 0;
    CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

    minGridSize = multiProcessorCount * numBlocksPerSm;
    CHECK(cudaMalloc(&d_partialResults, minGridSize * sizeof(double)));

    printf("Launching normVecByDotProductAWBarrier kernel with numBlocks = %d "
           "blockSize = %d\n",
           minGridSize,
           blockSize);

    dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

    void* kernelArgs[] = {(void*)&d_vecA, (void*)&d_vecB, (void*)&d_partialResults, (void*)&size};

    CHECK(cudaLaunchCooperativeKernel((void*)normVecByDotProductAWBarrier, dimGrid, dimBlock, kernelArgs, smemSize, stream));

    CHECK(cudaMemcpyAsync(vecA, d_vecA, sizeof(float) * size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float        expectedResult = (baseVal / sqrt(size * baseVal * baseVal));
    unsigned int matches        = 0;
    for (int i = 0; i < size; i++) {
        if ((vecA[i] - expectedResult) > 0.00001) {
            printf("mismatch at i = %d\n", i);
            break;
        }
        else {
            matches++;
        }
    }

    printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
    CHECK(cudaFree(d_vecA));
    CHECK(cudaFree(d_vecB));
    CHECK(cudaFree(d_partialResults));

    CHECK(cudaFreeHost(vecA));
    CHECK(cudaFreeHost(vecB));
    return matches == size;
}

void test_cg_02()
{
    // printf("%s starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    // int dev = findCudaDevice(argc, (const char**)argv);
    int   argc   = 1;
    char* argv[] = {"test_cg_02"};
    int   dev    = 0;
    int   major  = 0;
    CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

    // Arrive-Wait Barrier require a GPU of Volta (SM7X) architecture or higher.
    if (major < 7) {
        printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
        // exit(EXIT_WAIVED);
        return;
    }

    int supportsCooperativeLaunch = 0;
    CHECK(cudaDeviceGetAttribute(&supportsCooperativeLaunch, cudaDevAttrCooperativeLaunch, dev));

    if (!supportsCooperativeLaunch) {
        printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
               "Waiving the run\n",
               dev);
        // exit(EXIT_WAIVED);
        return;
    }

    int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

    printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
}