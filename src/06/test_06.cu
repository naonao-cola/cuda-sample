#include "test_06.cuh"


__global__ void kernel(float* g_data, float value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

int checkResult(float* data, const int n, const float x)
{
    for (int i = 0; i < n; i++) {
        if (data[i] != x) {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

void asyncAPI()
{
    // int            devID = 0;
    // cudaDeviceProp deviceProps;
    // CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    // printf("> %s running on", argv[0]);
    // printf(" CUDA device [%s]\n", deviceProps.name);

    int   num    = 1 << 24;
    int   nbytes = num * sizeof(int);
    float value  = 10.0f;

    // allocate host memory
    float* h_a = 0;
    CHECK(cudaMallocHost((void**)&h_a, nbytes));
    memset(h_a, 0, nbytes);

    // allocate device memory
    float* d_a = 0;
    CHECK(cudaMalloc((void**)&d_a, nbytes));
    CHECK(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 block = dim3(512);
    dim3 grid  = dim3((num + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // print the cpu and gpu times
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = (bool)checkResult(h_a, num, value);

    // release resources
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFree(d_a));
    CHECK(cudaDeviceReset());
    // exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

#define N 300000
#define NSTREAM 4

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("callback from stream %d\n", *((int*)data));
}

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

void simpleCallback()
{
    int n_streams = NSTREAM;

    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("> %s Starting...\n", argv[0]);
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "8", 1);
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", n_streams);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    dim3        block(1);
    dim3        grid(1);
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    int stream_ids[n_streams];

    CHECK(cudaEventRecord(start_event, 0));

    for (int i = 0; i < n_streams; i++) {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        CHECK(cudaStreamAddCallback(streams[i], my_callback, (void*)(stream_ids + i), 0));
    }
    //事件同步函数
    CHECK(cudaEventRecord(stop_event, 0));
    CHECK(cudaEventSynchronize(stop_event));

    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    /*
     * cudaDeviceReset must be called before exiting in order for profiling and
     * tracing tools such as Nsight and Visual Profiler to show complete traces.
     */
    CHECK(cudaDeviceReset());
}

void simpleHyperqBreadth(int argv1, int argv2)
{
    int n_streams = NSTREAM;
    int isize     = 1;
    int iblock    = 1;
    int bigcase   = 1;

    // get argument from command line
    if (argv1 > 1)
        n_streams = argv1;

    if (argv2 > 1)
        bigcase = argv2;

    float elapsed_time;

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char* ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams %d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1) {
        iblock = 512;
        isize  = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with breadth first ordering
    //以广度优先排序调度作业
    for (int i = 0; i < n_streams; i++)
        kernel_1<<<grid, block, 0, streams[i]>>>();

    for (int i = 0; i < n_streams; i++)
        kernel_2<<<grid, block, 0, streams[i]>>>();

    for (int i = 0; i < n_streams; i++)
        kernel_3<<<grid, block, 0, streams[i]>>>();

    for (int i = 0; i < n_streams; i++)
        kernel_4<<<grid, block, 0, streams[i]>>>();

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());
}

void simpleHyperqDependence(int argv1, int argv2)
{
    int n_streams = NSTREAM;
    int isize     = 1;
    int iblock    = 1;
    int bigcase   = 1;

    // get argument from command line
    if (argv1 > 1)
        n_streams = argv1;

    if (argv1 > 1)
        bigcase = argv2;

    float elapsed_time;

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char* ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams %d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1) {
        iblock = 512;
        isize  = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));


    cudaEvent_t* kernelEvent;
    kernelEvent = (cudaEvent_t*)malloc(n_streams * sizeof(cudaEvent_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();

        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        //这个函数会阻塞设置的流，直到指定的事件在设备上发生。一旦事件发生，流中的后续操作可以继续执行。
        CHECK(cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0));
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaEventDestroy(kernelEvent[i]));
    }

    free(streams);
    free(kernelEvent);

    // reset device
    CHECK(cudaDeviceReset());
}

void simpleHyperqDepth(int argv1, int argv2)
{
    int n_streams = NSTREAM;
    int isize     = 1;
    int iblock    = 1;
    int bigcase   = 1;

    // get argument from command line
    if (argv1 > 1)
        n_streams = argv1;

    if (argv2 > 1)
        bigcase = argv2;

    float elapsed_time;

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char* ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1) {
        iblock = 512;
        isize  = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());
}

#include "omp.h"
void simpleHyperqOpenmp()
{
    int n_streams = NSTREAM;
    int isize     = 1;
    int iblock    = 1;
    int bigcase   = 1;

    // get argument from command line
    // if (argc > 1)
    //     n_streams = atoi(argv[1]);

    // if (argc > 2)
    //     bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char* ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1) {
        iblock = 512;
        isize  = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering using OpenMP
    omp_set_num_threads(n_streams);
#pragma omp parallel
    {
        int i = omp_get_thread_num();
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());
}

#define NSTREAM 4
#define BDIM 128

void initialData(float* ip, int size)
{
    int i;

    for (i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float* A, float* B, float* C, const int n)
{
    for (int idx = 0; idx < n; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float* A, float* B, float* C, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        for (int i = 0; i < n; ++i) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

void checkResult(float* hostRef, float* gpuRef, const int n)
{
    double epsilon = 1.0E-8;
    bool   match   = 1;

    for (int i = 0; i < n; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}


void simpleMultiAddBreadth()
{
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "1", 1);
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // invoke kernel at host side
    dim3 block(BDIM);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // sequential operation
    //顺序操作
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n", kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n", itotal, (nBytes * 2e-6) / itotal);

    // grid parallel operation
    int    iElem  = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x        = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventRecord(start, 0));

    // initiate all asynchronous transfers to the device
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
    }

    // launch a kernel in each stream
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6) / execution_time);
    printf(" speedup                : %f \n", ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(hostRef));
    CHECK(cudaFreeHost(gpuRef));

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i) {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaDeviceReset());
}

void simpleMultiAddDepth()
{
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "1", 1);
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // invoke kernel at host side
    dim3 block(BDIM);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // sequential operation
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n", kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n", itotal, (nBytes * 2e-6) / itotal);

    // grid parallel operation
    int    iElem  = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x        = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventRecord(start, 0));

    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
        sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
        CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6) / execution_time);
    printf(" speedup                : %f \n", ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(hostRef));
    CHECK(cudaFreeHost(gpuRef));

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i) {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaDeviceReset());
}