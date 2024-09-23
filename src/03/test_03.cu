#include "test_03.cuh"

void simpleDeviceQuery(){
    int iDev = 0;
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp, iDev));
    printf("Device %d: %s\n", iDev, iProp.name);
    printf("  Number of multiprocessors:                     %d\n",iProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %4.2f KB\n",iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n",iProp.sharedMemPerBlock / 1024.0);
    printf("  Total number of registers available per block: %d\n",iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n",iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n",iProp.maxThreadsPerMultiProcessor / 32);


}
__global__ void mathKernel1(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if (tid % 2 == 0){
        ia = 100.0f;
    }
    else{
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0){
        ia = 100.0f;
    }
    else{
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel3(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    bool ipred = (tid % 2 == 0);

    if (ipred){
        ia = 100.0f;
    }
    if (!ipred){
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    int itid = tid >> 5;
    if (itid & 0x01 == 0){
        ia = 100.0f;
    }
    else{
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void warmingup(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0){
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}



void simpleDivergence(){

    int size = 1<< 12;
    int blocksize = 128;
    // if(argc > 1) blocksize = atoi(argv[1]);
    // if(argc > 2) size      = atoi(argv[2]);
    printf("Data size %d ", size);

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    //
    float *d_C;
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