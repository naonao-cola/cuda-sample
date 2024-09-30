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