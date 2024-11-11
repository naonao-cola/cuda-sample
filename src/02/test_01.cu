
#include "test_01.cuh"


void test_01()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3), (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
           deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0],
           deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0],
           deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
           deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1],
           deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
    exit(EXIT_SUCCESS);
}


__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // block的维度,就是每个block的线程的个数维度
    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    // grid的维度,就是gird里面的block的个数维度
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

void test_02()
{
    int nElem = 6;
    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();
    // reset device before you leave
    CHECK(cudaDeviceReset());
}


void printMatrix(int* C, const int nx, const int ny)
{
    int* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%3d", ic[ix]);
        }

        ic += nx;
        printf("\n");
    }

    printf("\n");
    return;
}

__global__ void printThreadIndex(int* A, const int nx, const int ny)
{
    int          ix  = threadIdx.x + blockIdx.x * blockDim.x;
    int          iy  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n",
           threadIdx.x,
           threadIdx.y,
           blockIdx.x,
           blockIdx.y,
           ix,
           iy,
           idx,
           A[idx]);
}

void test_03()
{
    // 获取设备信息
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 设置矩阵
    int nx     = 8;
    int ny     = 6;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    // 申请主机端内存
    int* h_A;
    h_A = (int*)malloc(nBytes);

    // 初始化数据并打印
    for (int i = 0; i < nxy; i++)
        h_A[i] = i;
    printMatrix(h_A, nx, ny);

    // 申请设备端内存，并复制到设备端
    int* d_MatA;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    // 网格参数设置，grid 为4列2行  grid 为 2 列 3行
    // x 表示x 方向一行有4个，不是4行的意思
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printf("blockDim:(%d, %d, %d)\n", block.x, block.y, block.z);
    printf("gridDim:(%d, %d, %d)\n", grid.x, grid.y, grid.z);

    // 调用核函数
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    //检查错误释放内存
    CHECK(cudaGetLastError());
    CHECK(cudaFree(d_MatA));
    free(h_A);
    CHECK(cudaDeviceReset());
}


void test_04()
{
    // define total data element
    int nElem = 1024;
    // define grid and block structure
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);
    // reset device before you leave
    CHECK(cudaDeviceReset());
}

//检查结果是否匹配
void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool   match   = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
    return;
}

//初始化数据
void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

//主机端计算
void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

//设备端计算
__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void test_05()
{

    int dev = 0;
    CHECK(cudaSetDevice(dev));
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // 申请内存
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    // 主机端初始化数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 设备端申请内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // 拷贝数据
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    // 网格设置
    int  iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);
    double iStart, iElaps;
    //设备端计算
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU Time elapsed %f sec\n", iElaps);
    // 拷贝数据
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    //     for(int i = 0; i < nElem; i++){
    //        printf("gpu %5.2f at current %d\n", gpuRef[i], i);
    //     }
    // 主机端计算
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec %f \n", iElaps, hostRef[0]);
    //     for(int i = 0; i < nElem; i++){
    //        printf("host %5.2f at current %d\n", hostRef[i], i);
    //     }
    //检查结果
    checkResult(hostRef, gpuRef, nElem);
    //释放设备内存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());
}


void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    float* ia = A;
    float* ib = B;
    float* ic = C;
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
    return;
}

// grid 1D block 1D
__global__ void sumMatrixOnGPU1D(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            int idx   = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

void sumMatrixOnGPU_1D_grid_1D_block()
{
    // set up device
    // int dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("Using Device %d: %s\n", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    int nx     = 1 << 14;
    int ny     = 1 << 14;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialize matrix elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int  dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    iStart = seconds();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nxy);
    CHECK(cudaGetLastError());
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());
}

__global__ void sumMatrixOnGPUMix(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy  = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

void sumMatrixOnGPU_2D_grid_1D_block()
{
    int nx     = 1 << 14;
    int ny     = 1 << 14;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int  dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);


    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);


    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

void sumMatrixOnGPU_2D_grid_2D_block()
{

    int nx     = 1 << 14;
    int ny     = 1 << 14;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    //
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    //
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    //
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    //
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    //
    int  dimx = 32;
    int  dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaGetLastError());
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());
}