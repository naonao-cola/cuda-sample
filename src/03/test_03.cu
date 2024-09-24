#include "test_03.cuh"

void simpleDeviceQuery()
{
    int            iDev = 0;
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp, iDev));
    printf("Device %d: %s\n", iDev, iProp.name);
    printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("  Total number of registers available per block: %d\n", iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);
}
__global__ void mathKernel1(float* c)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if (tid % 2 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel2(float* c)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel3(float* c)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib    = 0.0f;
    bool ipred = (tid % 2 == 0);

    if (ipred) {
        ia = 100.0f;
    }
    if (!ipred) {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel4(float* c)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib  = 0.0f;
    int itid = tid >> 5;
    if (itid & 0x01 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void warmingup(float* c)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}



void simpleDivergence()
{

    int size      = 1 << 12;
    int blocksize = 128;
    // if(argc > 1) blocksize = atoi(argv[1]);
    // if(argc > 2) size      = atoi(argv[2]);
    printf("Data size %d ", size);

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    //
    float* d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));
    // run a warmup kernel to remove overhead
    CHECK(cudaDeviceSynchronize());

    TICK(warmup)
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    TOCK(warmup)

    // run kernel 1
    TICK(mathKernel1)
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    TOCK(mathKernel1)

    // run kernel 3
    TICK(mathKernel2)
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    TOCK(mathKernel2)


    // run kernel 3
    TICK(mathKernel3)
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    TOCK(mathKernel3)

    // run kernel 4
    TICK(mathKernel4)
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    TOCK(mathKernel4)

    CHECK(cudaGetLastError());
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
}


__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1)
        return;
    // reduce block size to half
    int nthreads = iSize >> 1;
    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0) {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

void nestedHelloWorld()
{
    int size      = 8;
    int blocksize = 8;   // initial block size
    int igrid     = 1;
    // if(argc > 1)
    // {
    //     igrid = atoi(argv[1]);
    //     size = igrid * blocksize;
    // }
    // igrid = 4;
    // size = igrid * blocksize;
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configuration: grid %d block %d\n", grid.x, block.x);
    nestedHelloWorld<<<grid, block>>>(block.x, 0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
}

// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int* data, int const size)
{
    // stop condition
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    // call recursively
    return cpuRecursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if (idx >= n)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void gpuRecursiveReduce(int* g_idata, int* g_odata, unsigned int isize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    int* odata = &g_odata[blockIdx.x];
    // stop condition
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // nested invocation
    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        // in place reduction
        idata[tid] += idata[tid + istride];
    }
    // sync at block level
    __syncthreads();
    // nested invocation to generate child grids
    if (tid == 0) {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        // sync all child grids launched in this block
        // cudaDeviceSynchronize();
    }
    // sync at block level again
    __syncthreads();
}

__global__ void gpuRecursiveReduceNosync(int* g_idata, int* g_odata, unsigned int isize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    int* odata = &g_odata[blockIdx.x];
    // stop condition
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // nested invoke
    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
        if (tid == 0) {
            gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
        }
    }
}

__global__ void gpuRecursiveReduce2(int* g_idata, int* g_odata, int iStride, int const iDim)
{
    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * iDim;
    // stop condition
    if (iStride == 1 && threadIdx.x == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + iStride];
    // nested invocation to generate child grids
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gpuRecursiveReduce2<<<gridDim.x, iStride / 2>>>(g_idata, g_odata, iStride / 2, iDim);
    }
}

void nestedReduce2()
{

    // set up device
    // int dev = 0;
    int gpu_sum;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting reduction at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    bool bResult = false;
    // set up execution configuration
    int nblock  = 2048;
    int nthread = 512;   // initial block size

    // if(argc > 1){
    //     nblock = atoi(argv[1]);   // block size from command line argument
    // }

    // if(argc > 2){
    //     nthread = atoi(argv[2]);   // block size from command line argument
    // }

    int size = nblock * nthread;   // total number of elements to reduceNeighbored

    dim3 block(nthread, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory
    size_t bytes   = size * sizeof(int);
    int*   h_idata = (int*)malloc(bytes);
    int*   h_odata = (int*)malloc(grid.x * sizeof(int));
    int*   tmp     = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() & 0xFF);
        h_idata[i] = 1;
    }

    memcpy(tmp, h_idata, bytes);

    // allocate device memory
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));



    // cpu recursive reduction
    TICK(cpu_reduce)
    int cpu_sum = cpuRecursiveReduce(tmp, size);
    TOCK(cpu_reduce)
    printf("cpu reduce\t\telapsed cpu_sum: %d\n", cpu_sum);

    // gpu reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(gpu_Neighbored)
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    TOCK(gpu_Neighbored)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Neighbored\t\telapsed gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);


    // gpu nested reduce kernel
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(gpu_nested)
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    TOCK(gpu_nested)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu nested\t\telapsed gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel without synchronization
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(gpu_nested_without_synchronization)
    gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    TOCK(gpu_nested_without_synchronization)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu nestedNosyn\t\telapsed gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);


    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(gpu_nested2)
    gpuRecursiveReduce2<<<grid, block.x / 2>>>(d_idata, d_odata, block.x / 2, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    TOCK(gpu_nested2)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu nested2\t\telapsed gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));
    CHECK(cudaDeviceReset());
    bResult = (gpu_sum == cpu_sum);
    if (!bResult)
        printf("Test failed!\n");
}

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int* data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n)
        return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n)
        return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

void reduceInteger( int argv1)
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting reduction at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24;   // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if (argv1 > 1) {
        blocksize = argv1;   // block size from command line argument
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes   = size * sizeof(int);
    int*   h_idata = (int*)malloc(bytes);
    int*   h_odata = (int*)malloc(grid.x * sizeof(int));
    int*   tmp     = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);
    int    gpu_sum = 0;

    // allocate device memory
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // cpu reduction
    TICK(recursiveReduce)
    int cpu_sum = recursiveReduce(tmp, size);
    TOCK(recursiveReduce)
    printf("cpu reduce cpu_sum: %d\n", cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceNeighbored)
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceNeighbored)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("gpu Neighbored  gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x,block.x);


    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceNeighboredLess)
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceNeighboredLess)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("gpu Neighbored2 gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x,block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceInterleaved)
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceInterleaved)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("gpu Interleaved gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x,block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceUnrolling2)
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceUnrolling2)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += h_odata[i];

    printf("gpu Unrolling2  gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 2,block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceUnrolling4)
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceUnrolling4)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];

    printf("gpu Unrolling4  gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 4,block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceUnrolling8)
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceUnrolling8)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("gpu Unrolling8  gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 8,block.x);

    for (int i = 0; i < grid.x / 16; i++)
        gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceUnrollWarps8)
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceUnrollWarps8)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 8,block.x);


    // kernel 9: reduceCompleteUnrollWarsp8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    TICK(reduceCompleteUnrollWarps8)
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    TOCK(reduceCompleteUnrollWarps8)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 8,block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    TICK(reduceCompleteUnroll)
    switch (blocksize) {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    TOCK(reduceCompleteUnroll)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x / 8,block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if (!bResult)
        printf("Test failed!\n");
}