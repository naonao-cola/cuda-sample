#include "test_05.cuh"

#define BDIMX 32
#define BDIMY 16
#define IPAD 2

void printData(char* msg, int* in, const int size)
{
    printf("%s: ", msg);
    for (int i = 0; i < size; i++) {
        printf("%4d", in[i]);
        fflush(stdout);
    }
    printf("\n\n");
}

__global__ void setRowReadRow(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol2(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[icol][irow] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColPad(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColDynPad(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}
void checkSmemRectangle(int argv1)
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // cudaSharedMemConfig pConfig;
    // CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    // printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argv1 >= 1)
        iprintf = 1;

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    // allocate device memory
    int* d_C;
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    int* gpuRef = (int*)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setRowReadRow       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setColReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol2<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setColReadCol2      ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setRowReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setRowReadColDyn    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setRowReadColPad    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("setRowReadColDynPad ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(cudaFree(d_C));
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

#define RADIUS 4
#define BDIM 32

// constant memory
__constant__ float coef[RADIUS + 1];

// FD coeffecient
#define a0 0.00000f
#define a1 0.80000f
#define a2 -0.20000f
#define a3 0.03809f
#define a4 -0.00357f

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++) {
        in[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void printData_2(float* in, const int size)
{
    for (int i = RADIUS; i < size; i++) {
        printf("%f ", in[i]);
    }

    printf("\n");
}

void setup_coef_constant(void)
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

void cpu_stencil_1d(float* in, float* out, int isize)
{
    for (int i = RADIUS; i <= isize; i++) {
        float tmp = a1 * (in[i + 1] - in[i - 1]) + a2 * (in[i + 2] - in[i - 2]) + a3 * (in[i + 3] - in[i - 3]) + a4 * (in[i + 4] - in[i - 4]);
        out[i]    = tmp;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int size)
{
    double epsilon = 1.0E-6;
    bool   match   = 1;

    for (int i = RADIUS; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

__global__ void stencil_1d(float* in, float* out, int N)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {

        // index to shared memory for stencil calculatioin
        int sidx = threadIdx.x + RADIUS;

        // Read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS) {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM]   = in[idx + BDIM];
        }

        // Synchronize (ensure all the data is available)
        __syncthreads();

        // Apply the stencil
        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++) {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        // Store the result
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}

void constantStencil()
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting transpose at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up data size
    int isize = 1 << 24;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d ", isize);

    bool iprint = 0;

    // allocate host memory
    float* h_in    = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef  = (float*)malloc(nBytes);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    // initialize host array
    initialData(h_in, isize + 2 * RADIUS);

    // Copy to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // set up constant memory
    setup_coef_constant();

    // launch configuration
    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    dim3 block(BDIM, 1);
    dim3 grid(info.maxGridSize[0] < isize / block.x ? info.maxGridSize[0] : isize / block.x, 1);
    printf("(grid, block) %d,%d \n ", grid.x, block.x);

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, isize);

    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    cpu_stencil_1d(h_in, hostRef, isize);

    // check results
    checkResult(hostRef, gpuRef, isize);

    // print out results
    if (iprint) {
        printData_2(gpuRef, isize);
        printData_2(hostRef, isize);
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

__global__ void stencil_1d(float* in, float* out)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // index to shared memory for stencil calculatioin
    int sidx = threadIdx.x + RADIUS;

    // Read data from global memory into shared memory
    smem[sidx] = in[idx];

    // read halo part to shared memory
    if (threadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM]   = in[idx + BDIM];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    float tmp = 0.0f;
#pragma unroll
    for (int i = 1; i <= RADIUS; i++) {
        tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
    }
    // Store the result
    out[idx] = tmp;
}

__global__ void stencil_1d_read_only(float* in, float* out, const float* __restrict__ dcoef)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // index to shared memory for stencil calculatioin
    int sidx = threadIdx.x + RADIUS;

    // Read data from global memory into shared memory
    smem[sidx] = in[idx];

    // read halo part to shared memory
    if (threadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM]   = in[idx + BDIM];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    float tmp = 0.0f;
#pragma unroll

    for (int i = 1; i <= RADIUS; i++) {
        tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
    }

    // Store the result
    out[idx] = tmp;
}

void constantReadOnly()
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting transpose at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up data size
    int isize = 1 << 24;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d ", isize);

    bool iprint = 0;

    // allocate host memory
    float* h_in    = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef  = (float*)malloc(nBytes);

    // allocate device memory
    float *d_in, *d_out, *d_coef;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));
    CHECK(cudaMalloc((float**)&d_coef, (RADIUS + 1) * sizeof(float)));

    // set up coefficient to global memory
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float), cudaMemcpyHostToDevice);)

    // initialize host array
    initialData(h_in, isize + 2 * RADIUS);

    // Copy to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // set up constant memory
    setup_coef_constant();

    // launch configuration
    dim3 block(BDIM, 1);
    dim3 grid(isize / block.x, 1);
    printf("(grid, block) %d,%d \n ", grid.x, block.x);

    // Launch stencil_1d() kernel on GPU
    TICK(stencil_1d)
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS);
    TOCK(stencil_1d)
    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    TICK(cpu_stencil_1d)
    cpu_stencil_1d(h_in, hostRef, isize);
    TOCK(cpu_stencil_1d)

    // check results
    checkResult(hostRef, gpuRef, isize);

    // launch read only cache kernel
    TICK(stencil_1d_read_only)
    stencil_1d_read_only<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, d_coef);
    TOCK(stencil_1d_read_only)
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, isize);

    // print out results
    if (iprint) {
        printData_2(gpuRef, isize);
        printData_2(hostRef, isize);
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_coef));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

#define DIM 128
extern __shared__ int dsmem[];

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int* data, int const size)
{
    if (size == 1)
        return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmem(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid   = threadIdx.x;
    int*         idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    // in-place reduction in global memory
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

__global__ void reduceSmem(int* g_idata, int* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemDyn(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid   = threadIdx.x;
    int*         idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmemUnroll(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n) {
        int b1       = g_idata[idx];
        int b2       = g_idata[idx + blockDim.x];
        int b3       = g_idata[idx + 2 * blockDim.x];
        int b4       = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
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

__global__ void reduceSmemUnroll(int* g_idata, int* g_odata, unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int tmpSum = 0;

    // boundary check
    if (idx + 3 * blockDim.x <= n) {
        int b1 = g_idata[idx];
        int b2 = g_idata[idx + blockDim.x];
        int b3 = g_idata[idx + 2 * blockDim.x];
        int b4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = b1 + b2 + b3 + b4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4
    int tmpSum = 0;

    if (idx + 3 * blockDim.x < n) {
        int b1 = g_idata[idx];
        int b2 = g_idata[idx + blockDim.x];
        int b3 = g_idata[idx + 2 * blockDim.x];
        int b4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = b1 + b2 + b3 + b4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceNeighboredGmem(int* g_idata, int* g_odata, unsigned int n)
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

__global__ void reduceNeighboredSmem(int* g_idata, int* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n)
        return;

    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            smem[tid] += smem[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

void reduceInteger()
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
    int blocksize = DIM;   // initial block size

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes   = size * sizeof(int);
    int*   h_idata = (int*)malloc(bytes);
    int*   h_odata = (int*)malloc(grid.x * sizeof(int));
    int*   tmp     = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        h_idata[i] = (int)(rand() & 0xFF);

    memcpy(tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // cpu reduction
    TICK(recursiveReduce)
    int cpu_sum = recursiveReduce(tmp, size);
    TOCK(recursiveReduce)
    printf("cpu reduce          : %d\n", cpu_sum);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceNeighboredGmem)
    reduceNeighboredGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceNeighboredGmem)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceNeighboredGmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceNeighboredSmem)
    reduceNeighboredSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceNeighboredSmem)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceNeighboredSmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceGmem)
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceGmem)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceGmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceSmem)
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceSmem)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceSmemDyn)
    reduceSmemDyn<<<grid.x, block, blocksize * sizeof(int)>>>(d_idata, d_odata, size);
    TOCK(reduceSmemDyn)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmemDyn       : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceGmemUnroll)
    reduceGmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    TOCK(reduceGmemUnroll)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];

    printf("reduceGmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceSmemUnroll)
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    TOCK(reduceSmemUnroll)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceSmemUnrollDyn)
    reduceSmemUnrollDyn<<<grid.x / 4, block, DIM * sizeof(int)>>>(d_idata, d_odata, size);
    TOCK(reduceSmemUnrollDyn)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmemDynUnroll4: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

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

#define SMEMDIM 4   // 128/32 = 8


__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 16);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 8);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 4);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 2);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 1);

    return localSum;
}

__global__ void reduceShfl(int* g_idata, int* g_odata, unsigned int n)
{
    // shared memory for each warp sum
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // blcok-wide warp reduce
    int localSum = warpReduce(g_idata[idx]);

    // save warp sum to shared memory
    if (laneIdx == 0)
        smem[warpIdx] = localSum;

    // block synchronization
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize)
        localSum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

    if (warpIdx == 0)
        localSum = warpReduce(localSum);

    // write result for this block to global mem
    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = localSum;
}

__global__ void reduceSmemShfl(int* g_idata, int* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (blockDim.x >= 64 && tid < 32)
        smem[tid] += smem[tid + 32];
    __syncthreads();

    int localSum = smem[tid];
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 16);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 8);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 4);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 2);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 1);

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = localSum;   // smem[0];
}

__global__ void reduceSmemUnrollShfl(int* g_idata, int* g_odata, unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int localSum = 0;

    if (idx + 3 * blockDim.x < n) {
        float b1 = g_idata[idx];
        float b2 = g_idata[idx + blockDim.x];
        float b3 = g_idata[idx + 2 * blockDim.x];
        float b4 = g_idata[idx + 3 * blockDim.x];
        localSum = b1 + b2 + b3 + b4;
    }

    smem[tid] = localSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (blockDim.x >= 64 && tid < 32)
        smem[tid] += smem[tid + 32];
    __syncthreads();

    // unrolling warp
    localSum = smem[tid];
    if (tid < 32) {
        localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 16);
        localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 8);
        localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 4);
        localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 2);
        localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 1);
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = localSum;
}

void reduceIntegerShfl()
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
    int ishift = 24;

    // if (argc > 1)
    //     ishift = atoi(argv[1]);

    int size = 1 << ishift;
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = DIM;   // initial block size

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes   = size * sizeof(int);
    int*   h_idata = (int*)malloc(bytes);
    int*   h_odata = (int*)malloc(grid.x * sizeof(int));
    int*   tmp     = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        h_idata[i] = (int)(rand() & 0xFF);

    memcpy(tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // cpu reduction
    TICK(recursiveReduce)
    int cpu_sum = recursiveReduce(tmp, size);
    TOCK(recursiveReduce)
    printf("cpu reduce          : %d\n", cpu_sum);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceSmem)
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceSmem)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    TICK(reduceShfl)
    reduceShfl<<<grid.x, block>>>(d_idata, d_odata, size);
    TOCK(reduceShfl)
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceShfl          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

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

#define BDIMX 16
#define SEGM 4

__global__ void test_shfl_broadcast(int* d_out, int* d_in, int const srcLane)
{
    int value          = d_in[threadIdx.x];
    value              = __shfl_sync(0xFFFFFFFF, value, srcLane, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap(int* d_out, int* d_in, int const offset)
{
    int value          = d_in[threadIdx.x];
    value              = __shfl_sync(0xFFFFFFFF, value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap_plus(int* d_out, int* d_in, int const offset)
{
    int value = d_in[threadIdx.x];
    value += __shfl_sync(0xFFFFFFFF, value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_up(int* d_out, int* d_in, unsigned int const delta)
{
    int value          = d_in[threadIdx.x];
    value              = __shfl_up_sync(0xFFFFFFFF, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int* d_out, int* d_in, unsigned int const delta)
{
    int value          = d_in[threadIdx.x];
    value              = __shfl_down_sync(0xFFFFFFFF, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int* d_out, int* d_in, int const mask)
{
    int value          = d_in[threadIdx.x];
    value              = __shfl_xor_sync(0xFFFFFFFF, value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int* d_out, int* d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[0] = __shfl_xor_sync(0xFFFFFFFF, value[0], mask, BDIMX);
    value[1] = __shfl_xor_sync(0xFFFFFFFF, value[1], mask, BDIMX);
    value[2] = __shfl_xor_sync(0xFFFFFFFF, value[2], mask, BDIMX);
    value[3] = __shfl_xor_sync(0xFFFFFFFF, value[3], mask, BDIMX);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_int4(int* d_out, int* d_in, int const mask)
{
    int  idx = threadIdx.x * SEGM;
    int4 value;

    value.x = d_in[idx];
    value.y = d_in[idx + 1];
    value.z = d_in[idx + 2];
    value.w = d_in[idx + 3];

    value.x = __shfl_xor_sync(0xFFFFFFFF, value.x, mask, BDIMX);
    value.y = __shfl_xor_sync(0xFFFFFFFF, value.y, mask, BDIMX);
    value.z = __shfl_xor_sync(0xFFFFFFFF, value.z, mask, BDIMX);
    value.w = __shfl_xor_sync(0xFFFFFFFF, value.w, mask, BDIMX);

    d_out[idx]     = value.x;
    d_out[idx + 1] = value.y;
    d_out[idx + 2] = value.z;
    d_out[idx + 3] = value.w;
}



__global__ void test_shfl_xor_element(int* d_out, int* d_in, int const mask, int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[srcIdx] = __shfl_xor_sync(0xFFFFFFFF, value[dstIdx], mask, BDIMX);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_array_swap(int* d_out, int* d_in, int const mask, int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    bool pred = ((threadIdx.x & 1) != mask);

    if (pred) {
        int tmp       = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    value[dstIdx] = __shfl_xor_sync(value[dstIdx], mask, BDIMX);

    if (pred) {
        int tmp       = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}


__inline__ __device__ void swap_old(int* value, int tid, int mask, int srcIdx, int dstIdx)
{
    bool pred = ((tid / mask + 1) == 1);

    if (pred) {
        int tmp       = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    value[dstIdx] = __shfl_xor_sync(0xFFFFFFFF, value[dstIdx], mask, BDIMX);

    if (pred) {
        int tmp       = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }
}

__inline__ __device__ void swap(int* value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx / mask + 1) == 1);

    if (pred) {
        int tmp          = value[firstIdx];
        value[firstIdx]  = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor_sync(0xFFFFFFFF, value[secondIdx], mask, BDIMX);

    if (pred) {
        int tmp          = value[firstIdx];
        value[firstIdx]  = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__ void test_shfl_swap_old(int* d_out, int* d_in, int const mask, int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, srcIdx, dstIdx);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

__global__ void test_shfl_swap(int* d_out, int* d_in, int const mask, int firstIdx, int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, firstIdx, secondIdx);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_array_swap_base(int* d_out, int* d_in, int const mask, int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[dstIdx] = __shfl_xor_sync(0xFFFFFFFF, value[dstIdx], mask, BDIMX);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

__global__ void test_shfl_array(int* d_out, int* d_in, int const offset)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    int lane = (offset + threadIdx.x) % SEGM;
    value[0] = __shfl_sync(0xFFFFFFFF, value[3], lane, BDIMX);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

__global__ void test_shfl_xor_plus(int* d_out, int* d_in, int const mask)
{
    int value = d_in[threadIdx.x];
    value += __shfl_xor_sync(0xFFFFFFFF, value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

void printData(int *in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%2d ", in[i]);
    }

    printf("\n");
}

void simpleShfl()
{
    int  dev       = 0;
    bool iPrintout = 1;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("> %s Starting.", argv[0]);
    printf("at Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = BDIMX;
    int h_inData[BDIMX], h_outData[BDIMX];

    for (int i = 0; i < nElem; i++)
        h_inData[i] = i;

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    size_t nBytes = nElem * sizeof(int);
    int *  d_inData, *d_outData;
    CHECK(cudaMalloc((int**)&d_inData, nBytes));
    CHECK(cudaMalloc((int**)&d_outData, nBytes));

    CHECK(cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice));

    int block = BDIMX;

    // shfl bcast
    TICK(test_shfl_broadcast)
    test_shfl_broadcast<<<1, block>>>(d_outData, d_inData, 2);
    TOCK(test_shfl_broadcast)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl bcast\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    TICK(test_shfl_wrap)
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, -2);
    TOCK(test_shfl_wrap)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl wrap right\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl up
    TICK(test_shfl_up)
    test_shfl_up<<<1, block>>>(d_outData, d_inData, 2);
    TOCK(test_shfl_up)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl up \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    TICK(test_shfl_wrap_2)
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    TOCK(test_shfl_wrap_2)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl wrap left\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    TICK(test_shfl_wrap_3)
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    TOCK(test_shfl_wrap_3)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl wrap 2\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl down
    TICK(test_shfl_down)
    test_shfl_down<<<1, block>>>(d_outData, d_inData, 2);
    TOCK(test_shfl_down)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl down \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor
    TICK(test_shfl_xor)
    test_shfl_xor<<<1, block>>>(d_outData, d_inData, 1);
    TOCK(test_shfl_xor)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if (iPrintout) {
        printf("shfl xor 1\t\t: ");
        printData(h_outData, nElem);
    }

    TICK(test_shfl_xor_2)
    test_shfl_xor<<<1, block>>>(d_outData, d_inData, -8);
    TOCK(test_shfl_xor_2)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl xor -1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - int4
    TICK(test_shfl_xor_int4)
    test_shfl_xor_int4<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    TOCK(test_shfl_xor_int4)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if (iPrintout) {
        printf("shfl int4 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    TICK(test_shfl_xor_array)
    test_shfl_xor_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    TOCK(test_shfl_xor_array)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if (iPrintout) {
        printf("shfl array 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - test_shfl_xor_element
    TICK(test_shfl_xor_element)
    test_shfl_xor_element<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    TOCK(test_shfl_xor_element)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if (iPrintout) {
        printf("shfl idx \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    TICK(test_shfl_xor_array_swap_base)
    test_shfl_xor_array_swap_base<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    TOCK(test_shfl_xor_array_swap_base)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl swap base\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    TICK(test_shfl_xor_array_swap)
    test_shfl_xor_array_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    TOCK(test_shfl_xor_array_swap)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl swap 0 3\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    TICK(test_shfl_swap)
    test_shfl_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    TOCK(test_shfl_swap)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl swap inline\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    TICK(test_shfl_array)
    test_shfl_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    TOCK(test_shfl_array)
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if (iPrintout) {
        printf("shfl array \t\t: ");
        printData(h_outData, nElem);
    }

    // finishing
    CHECK(cudaFree(d_inData));
    CHECK(cudaFree(d_outData));
    CHECK(cudaDeviceReset(););
}