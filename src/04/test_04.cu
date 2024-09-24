#include "test_04.cuh"

__device__ float devData;

__global__ void checkGlobalVariable()
{
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the value
    devData += 2.0f;
}

void globalVariable()
{
    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1, 1>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
}

void memTransfer()
{
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize  = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev, deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory
    float* h_a = (float*)malloc(nbytes);

    // allocate the device memory
    float* d_a;
    CHECK(cudaMalloc((float**)&d_a, nbytes));

    // initialize the host memory
    for (unsigned int i = 0; i < isize; i++)
        h_a[i] = 0.5f;

    // transfer data from the host to the device
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_a));
    free(h_a);

    // reset device
    CHECK(cudaDeviceReset());
}

void pinMemTransfer()
{
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize  = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    // printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB canMap %d\n", dev, deviceProp.name, isize, nbytes / (1024.0f * 1024.0f), deviceProp.canMapHostMemory);

    // allocate pinned host memory
    float* h_a;
    CHECK(cudaMallocHost((float**)&h_a, nbytes));

    // allocate device memory
    float* d_a;
    CHECK(cudaMalloc((float**)&d_a, nbytes));

    // initialize host memory
    memset(h_a, 0, nbytes);

    for (int i = 0; i < isize; i++)
        h_a[i] = 100.10f;

    // transfer data from the host to the device
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFreeHost(h_a));

    // reset device
    CHECK(cudaDeviceReset());
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool   match   = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

void initialData(float* ip, int size)
{
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }
    return;
}


void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++) {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}

__global__ void readOffset(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}

void readSegment(int argv1)
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting reduction at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up array size
    int nElem = 1 << 20;   // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset    = 0;

    if (1)
        offset = argv1;

    if (1)
        blocksize = 1024;

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float* h_A     = (float*)malloc(nBytes);
    float* h_B     = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef  = (float*)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    TICK(sumArraysOnHost)
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);
    TOCK(sumArraysOnHost)

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    //  kernel 1:
    TICK(warmup)
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(warmup)
    printf("warmup     <<< %4d, %4d >>> offset %4d \n", grid.x, block.x, offset);
    CHECK(cudaGetLastError());

    TICK(readOffset)
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(readOffset)
    printf("readOffset <<< %4d, %4d >>> offset %4d \n", grid.x, block.x, offset);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
}

__global__ void readOffsetUnroll2(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + blockDim.x < n) {
        C[i]              = A[k] + B[k];
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void readOffsetUnroll4(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n) {
        C[i]                  = A[k] + B[k];
        C[i + blockDim.x]     = A[k + blockDim.x] + B[k + blockDim.x];
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

void readSegmentUnroll(int argv1)
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s starting reduction at ", argv[0]);
    // printf("device %d: %s ", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up array size
    int nElem = 1 << 20;   // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset    = 0;

    if (1)
        offset = argv1;

    // if (argc > 2)
    //     blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float* h_A     = (float*)malloc(nBytes);
    float* h_B     = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef  = (float*)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    //  kernel 1:
    TICK(warmup)
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(warmup)
    printf("warmup     <<< %4d, %4d >>> offset %4d \n", grid.x, block.x, offset);
    CHECK(cudaGetLastError());

    // kernel 1
    TICK(readOffset)
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(readOffset)
    printf("readOffset <<< %4d, %4d >>> offset %4d \n", grid.x, block.x, offset);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // kernel 2
    TICK(readOffsetUnroll2)
    readOffsetUnroll2<<<grid.x / 2, block>>>(d_A, d_B, d_C, nElem / 2, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(readOffsetUnroll2)
    printf("unroll2    <<< %4d, %4d >>> offset %4d \n", grid.x / 2, block.x, offset);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // kernel 3
    TICK(readOffsetUnroll4)
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem / 4, offset);
    CHECK(cudaDeviceSynchronize());
    TOCK(readOffsetUnroll4)
    printf("unroll4    <<< %4d, %4d >>> offset %4d \n", grid.x / 4, block.x, offset);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
}

#define LEN 1 << 22
struct innerStruct
{
    float x;
    float y;
};
struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initialInnerStruct(innerStruct* ip, int size)
{
    for (int i = 0; i < size; i++) {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
    return;
}

void testInnerStructHost(innerStruct* A, innerStruct* C, const int n)
{
    for (int idx = 0; idx < n; idx++) {
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }
    return;
}

void checkInnerStruct(innerStruct* hostRef, innerStruct* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool   match   = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
            break;
        }
        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].y, gpuRef[i].y);
            break;
        }
    }
    if (!match)
        printf("Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct* data, innerStruct* result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void warmup(innerStruct* data, innerStruct* result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

void simpleMathAoS()
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s test struct of array at ", argv[0]);
    // printf("device %d: %s \n", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // allocate host memory
    int          nElem   = LEN;
    size_t       nBytes  = nElem * sizeof(innerStruct);
    innerStruct* h_A     = (innerStruct*)malloc(nBytes);
    innerStruct* hostRef = (innerStruct*)malloc(nBytes);
    innerStruct* gpuRef  = (innerStruct*)malloc(nBytes);

    // initialize host array
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate device memory
    innerStruct *d_A, *d_C;
    CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for summaryAU: It is blocksize not offset. Thanks.CZ
    int blocksize = 128;

    // if (argc > 1)
    //     blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 1: warmup
    TICK(warmup)
    warmup<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    TOCK(warmup)
    printf("warmup      <<< %3d, %3d >>> \n", grid.x, block.x);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // kernel 2: testInnerStruct
    TICK(testInnerStruct)
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    TOCK(testInnerStruct)
    printf("innerstruct <<< %3d, %3d >>> \n", grid.x, block.x);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // free memories both host and device
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());
}

struct InnerArray
{
    float x[LEN];
    float y[LEN];
};

// functions for inner array outer struct
void initialInnerArray(InnerArray* ip, int size)
{
    for (int i = 0; i < size; i++) {
        ip->x[i] = (float)(rand() & 0xFF) / 100.0f;
        ip->y[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void testInnerArrayHost(InnerArray* A, InnerArray* C, const int n)
{
    for (int idx = 0; idx < n; idx++) {
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
    }
    return;
}

void printfHostResult(InnerArray* C, const int n)
{
    for (int idx = 0; idx < n; idx++) {
        printf("printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
    }
    return;
}

void checkInnerArray(InnerArray* hostRef, InnerArray* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool   match   = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon) {
            match = 0;
            printf("different on x %dth element: host %f gpu %f\n", i, hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon) {
            match = 0;
            printf("different on y %dth element: host %f gpu %f\n", i, hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

__global__ void testInnerArray(InnerArray* data, InnerArray* result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

__global__ void warmup2(InnerArray* data, InnerArray* result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];
        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

void simpleMathSoA()
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s test struct of array at ", argv[0]);
    // printf("device %d: %s \n", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // allocate host memory
    int         nElem   = LEN;
    size_t      nBytes  = sizeof(InnerArray);
    InnerArray* h_A     = (InnerArray*)malloc(nBytes);
    InnerArray* hostRef = (InnerArray*)malloc(nBytes);
    InnerArray* gpuRef  = (InnerArray*)malloc(nBytes);

    // initialize host array
    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    // allocate device memory
    InnerArray *d_A, *d_C;
    CHECK(cudaMalloc((InnerArray**)&d_A, nBytes));
    CHECK(cudaMalloc((InnerArray**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for summary
    int blocksize = 128;

    // if (argc > 1)
    //     blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 1:
    TICK(warmup2)
    warmup2<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    TOCK(warmup2)
    printf("warmup2      <<< %3d, %3d >>> \n", grid.x, block.x);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    TICK(testInnerArray)
    testInnerArray<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    TOCK(testInnerArray)
    printf("innerarray   <<< %3d, %3d >>> \n", grid.x, block.x);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

void sumArrayZerocpy(int argv1)
{

    // set up data size of vectors
    int ipower = 10;

    if (1)
        ipower = argv1;

    int    nElem  = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18) {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower, (float)nBytes / (1024.0f));
    }
    else {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower, (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: using device memory
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    TICK(sumArraysOnHost)
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    TOCK(sumArraysOnHost)

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration
    int  iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    TICK(sumArrays)
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    TOCK(sumArrays)

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zerocopy memory for array A and B,
    //零拷贝内存
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    //将设备内存指向主机内存
    CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));

    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    TICK(sumArraysZeroCopy)
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    TOCK(sumArraysZeroCopy)

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free  memory
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));

    free(hostRef);
    free(gpuRef);

    // reset device
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

// grid 2D block 2D
__global__ void sumMatrixGPU(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

void sumMatrixGPUManaged(int argv1)
{

    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("using Device %d: %s\n", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if (1)
        ishift = argv1;

    nx = ny = 1 << ishift;

    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void**)&A, nBytes));
    CHECK(cudaMallocManaged((void**)&B, nBytes));
    CHECK(cudaMallocManaged((void**)&gpuRef, nBytes););
    CHECK(cudaMallocManaged((void**)&hostRef, nBytes););

    // initialize data at host side
    TICK(initialization)
    initialData(A, nxy);
    initialData(B, nxy);
    TOCK(initialization)

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    TICK(sumMatrixOnHost)
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    TOCK(sumMatrixOnHost)

    // invoke kernel at host side
    int  dimx = 32;
    int  dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    TICK(sumMatrixGPU)
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
    TOCK(sumMatrixGPU)
    printf("sumMatrix on gpu :\t <<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));
    CHECK(cudaDeviceReset());
}

void sumMatrixGPUManual(int argv1)
{
    // set up device
    // int            dev = 0;
    // cudaDeviceProp deviceProp;
    // CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("using Device %d: %s\n", dev, deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    // if (argc > 1)
    //     ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    // initialize data at host side
    TICK(initialData)
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    TOCK(initialData)

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    TICK(sumMatrixOnHost)
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    TOCK(sumMatrixOnHost)


    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    // invoke kernel at host side
    int  dimx = 32;
    int  dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // init device data to 0.0f, then warm-up kernel to obtain accurate timing
    // result
    CHECK(cudaMemset(d_MatA, 0.0f, nBytes));
    CHECK(cudaMemset(d_MatB, 0.0f, nBytes));
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, 1, 1);


    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    TICK(sumMatrixGPU)
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    TOCK(sumMatrixGPU)

    printf("sumMatrix on gpu :\t <<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check kernel error
    CHECK(cudaGetLastError());
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