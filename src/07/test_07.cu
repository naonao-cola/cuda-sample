#include "test_07.cuh"


/**
这个版本的内核使用原子操作来安全地增加a来自多个线程的共享变量。
**/
__global__ void atomics(int* shared_var, int* values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    values_read[tid] = atomicAdd(shared_var, 1);
    for (i = 0; i < iters; i++) {
        atomicAdd(shared_var, 1);
    }
}


/**
这个版本的内核执行与atomics()相同的增量，但在不安全的方式。
**/
__global__ void unsafe(int* shared_var, int* values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N)
        return;

    int old          = *shared_var;
    *shared_var      = old + 1;
    values_read[tid] = old;

    for (i = 0; i < iters; i++) {
        int old     = *shared_var;
        *shared_var = old + 1;
    }
}

/**
 * Utility function for printing the contents of an array.
 **/
static void print_read_results(int* h_arr, int* d_arr, int N, const char* label)
{
    int i;
    int maxNumToPrint = 10;
    int nToPrint      = N > maxNumToPrint ? maxNumToPrint : N;
    CHECK(cudaMemcpy(h_arr, d_arr, nToPrint * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Threads performing %s operations read values", label);

    for (i = 0; i < nToPrint; i++) {
        printf(" %d", h_arr[i]);
    }

    printf("\n");
}

void atomic_ordering()
{
    int  N     = 64;
    int  block = 32;
    int  runs  = 30;
    int  iters = 100000;
    int  r;
    int* d_shared_var;
    int  h_shared_var_atomic, h_shared_var_unsafe;
    int* d_values_read_atomic;
    int* d_values_read_unsafe;
    int* h_values_read;

    CHECK(cudaMalloc((void**)&d_shared_var, sizeof(int)));
    CHECK(cudaMalloc((void**)&d_values_read_atomic, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_values_read_unsafe, N * sizeof(int)));
    h_values_read = (int*)malloc(N * sizeof(int));

    double atomic_mean_time = 0;
    double unsafe_mean_time = 0;

    for (r = 0; r < runs; r++) {
        double start_atomic = seconds();
        CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        atomics<<<N / block, block>>>(d_shared_var, d_values_read_atomic, N, iters);
        CHECK(cudaDeviceSynchronize());
        atomic_mean_time += seconds() - start_atomic;
        CHECK(cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));

        double start_unsafe = seconds();
        CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        unsafe<<<N / block, block>>>(d_shared_var, d_values_read_unsafe, N, iters);
        CHECK(cudaDeviceSynchronize());
        unsafe_mean_time += seconds() - start_unsafe;
        CHECK(cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("In total, %d runs using atomic operations took %f s\n", runs, atomic_mean_time);
    printf("  Using atomic operations also produced an output of %d\n", h_shared_var_atomic);
    printf("In total, %d runs using unsafe operations took %f s\n", runs, unsafe_mean_time);
    printf("  Using unsafe operations also produced an output of %d\n", h_shared_var_unsafe);

    print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
    print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");
}


/**
从设备中保存12.1的单精度和双精度表示到全局内存中。然后将该全局内存复制回主机以供使用后分析。
 **/
__global__ void kernel(float* F, double* D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        *F = 12.1;
        *D = 12.1;
    }
}


void floating_point_accuracy()
{
    float*  deviceF;
    float   h_deviceF;
    double* deviceD;
    double  h_deviceD;

    float  hostF = 12.1;
    double hostD = 12.1;

    CHECK(cudaMalloc((void**)&deviceF, sizeof(float)));
    CHECK(cudaMalloc((void**)&deviceD, sizeof(double)));
    kernel<<<1, 32>>>(deviceF, deviceD);
    CHECK(cudaMemcpy(&h_deviceF, deviceF, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_deviceD, deviceD, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Host single-precision representation of 12.1   = %.20f\n", hostF);
    printf("Host double-precision representation of 12.1   = %.20f\n", hostD);
    printf("Device single-precision representation of 12.1 = %.20f\n", h_deviceF);
    printf("Device double-precision representation of 12.1 = %.20f\n", h_deviceD);
    printf("Device and host single-precision representation equal? %s\n", hostF == h_deviceF ? "yes" : "no");
    printf("Device and host double-precision representation equal? %s\n", hostD == h_deviceD ? "yes" : "no");
}

/**
 * 单精度浮点数的计算内核
 **/
__global__ void lots_of_float_compute(float* inputs, int N, size_t niters, float* outputs)
{
    size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for (; tid < N; tid += nthreads) {
        size_t iter;
        float  val = inputs[tid];

        for (iter = 0; iter < niters; iter++) {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}

/**
 双精度浮点数的计算内核
 **/
__global__ void lots_of_double_compute(double* inputs, int N, size_t niters, double* outputs)
{
    size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for (; tid < N; tid += nthreads) {
        size_t iter;
        double val = inputs[tid];

        for (iter = 0; iter < niters; iter++) {
            val = (val + 5.0) - 101.0;
            val = (val / 3.0) + 102.0;
            val = (val + 1.07) - 103.0;
            val = (val / 1.037) + 104.0;
            val = (val + 3.00) - 105.0;
            val = (val / 0.22) + 106.0;
        }

        outputs[tid] = val;
    }
}


/**
运行一个完整的测试单精度浮点数，包括转移设备的输入，运行单精度内核，复制输出。
 **/
static void run_float_test(size_t N, int niters, int blocksPerGrid, int threadsPerBlock, double* toDeviceTime, double* kernelTime, double* fromDeviceTime, float* sample, int sampleLength)
{
    int    i;
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    h_floatInputs  = (float*)malloc(sizeof(float) * N);
    h_floatOutputs = (float*)malloc(sizeof(float) * N);
    CHECK(cudaMalloc((void**)&d_floatInputs, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&d_floatOutputs, sizeof(float) * N));

    for (i = 0; i < N; i++) {
        h_floatInputs[i] = (float)i;
    }

    double toDeviceStart = seconds();
    CHECK(cudaMemcpy(d_floatInputs, h_floatInputs, sizeof(float) * N, cudaMemcpyHostToDevice));
    *toDeviceTime = seconds() - toDeviceStart;

    double kernelStart = seconds();
    lots_of_float_compute<<<blocksPerGrid, threadsPerBlock>>>(d_floatInputs, N, niters, d_floatOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = seconds() - kernelStart;

    double fromDeviceStart = seconds();
    CHECK(cudaMemcpy(h_floatOutputs, d_floatOutputs, sizeof(float) * N, cudaMemcpyDeviceToHost));
    *fromDeviceTime = seconds() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++) {
        sample[i] = h_floatOutputs[i];
    }

    CHECK(cudaFree(d_floatInputs));
    CHECK(cudaFree(d_floatOutputs));
    free(h_floatInputs);
    free(h_floatOutputs);
}


/**
 * Runs a full test of double-precision floating-point, including transferring
 * inputs to the device, running the single-precision kernel, and copying
 * outputs back.
 **/
static void run_double_test(size_t N, int niters, int blocksPerGrid, int threadsPerBlock, double* toDeviceTime, double* kernelTime, double* fromDeviceTime, double* sample, int sampleLength)
{
    int     i;
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    h_doubleInputs  = (double*)malloc(sizeof(double) * N);
    h_doubleOutputs = (double*)malloc(sizeof(double) * N);
    CHECK(cudaMalloc((void**)&d_doubleInputs, sizeof(double) * N));
    CHECK(cudaMalloc((void**)&d_doubleOutputs, sizeof(double) * N));

    for (i = 0; i < N; i++) {
        h_doubleInputs[i] = (double)i;
    }

    double toDeviceStart = seconds();
    CHECK(cudaMemcpy(d_doubleInputs, h_doubleInputs, sizeof(double) * N, cudaMemcpyHostToDevice));
    *toDeviceTime = seconds() - toDeviceStart;

    double kernelStart = seconds();
    lots_of_double_compute<<<blocksPerGrid, threadsPerBlock>>>(d_doubleInputs, N, niters, d_doubleOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = seconds() - kernelStart;

    double fromDeviceStart = seconds();
    CHECK(cudaMemcpy(h_doubleOutputs, d_doubleOutputs, sizeof(double) * N, cudaMemcpyDeviceToHost));
    *fromDeviceTime = seconds() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++) {
        sample[i] = h_doubleOutputs[i];
    }

    CHECK(cudaFree(d_doubleInputs));
    CHECK(cudaFree(d_doubleOutputs));
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

void floating_point_perf()
{

    int                   i;
    double                meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double                meanDoubleToDeviceTime, meanDoubleKernelTime, meanDoubleFromDeviceTime;
    struct cudaDeviceProp deviceProperties;
    size_t                totalMem, freeMem;
    float*                floatSample;
    double*               doubleSample;
    int                   sampleLength = 10;
    int                   nRuns        = 5;
    int                   nKernelIters = 20;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime = meanDoubleFromDeviceTime = 0.0;

    CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    size_t N               = (freeMem * 0.9 / 2) / sizeof(double);
    int    threadsPerBlock = 256;
    int    blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > deviceProperties.maxGridSize[0]) {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n", blocksPerGrid, threadsPerBlock, N);

    floatSample  = (float*)malloc(sizeof(float) * sampleLength);
    doubleSample = (double*)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++) {
        double toDeviceTime, kernelTime, fromDeviceTime;

        run_float_test(N, nKernelIters, blocksPerGrid, threadsPerBlock, &toDeviceTime, &kernelTime, &fromDeviceTime, floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_double_test(N, nKernelIters, blocksPerGrid, threadsPerBlock, &toDeviceTime, &kernelTime, &fromDeviceTime, doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;

        if (i == 0) {
            int j;
            printf("Input\tDiff Between Single- and Double-Precision\n");
            printf("------\t------\n");

            for (j = 0; j < sampleLength; j++) {
                printf("%d\t%.20e\n", j, fabs(doubleSample[j] - (double)floatSample[j]));
            }

            printf("\n");
        }
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f s\n", meanFloatKernelTime);
    printf("  Copy from device: %f s\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s (%.2fx slower than single-precision)\n", meanDoubleToDeviceTime, meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f s (%.2fx slower than single-precision)\n", meanDoubleKernelTime, meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f s (%.2fx slower than single-precision)\n", meanDoubleFromDeviceTime, meanDoubleFromDeviceTime / meanFloatFromDeviceTime);
}


__global__ void fmad_kernel(double x, double y, double* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        *out = x * x + y;
    }
}

double host_fmad_kernel(double x, double y)
{
    return x * x + y;
}

void fmad()
{
    double *d_out, h_out;
    double  x = 2.891903;
    double  y = -3.980364;

    double host_value = host_fmad_kernel(x, y);
    CHECK(cudaMalloc((void**)&d_out, sizeof(double)));
    fmad_kernel<<<1, 32>>>(x, y, d_out);
    CHECK(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));

    if (host_value == h_out) {
        printf("The device output the same value as the host.\n");
    }
    else {
        printf("The device output a different value than the host, diff=%e.\n", fabs(host_value - h_out));
    }
}

/**
 * Perform iters power operations using the standard powf function.
 使用标准的powf函数执行整数幂运算
 **/
__global__ void standard_kernel(float a, float* out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid == 0) {
        float tmp;

        for (i = 0; i < iters; i++) {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

/**
 * Perform iters power operations using the intrinsic __powf function.
 使用内部__powf函数执行整数幂运算。
 **/
__global__ void intrinsic_kernel(float a, float* out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid == 0) {
        float tmp;

        for (i = 0; i < iters; i++) {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
    }
}

void intrinsic_standard_comp()
{
    int i;
    int runs  = 50;
    int iters = 1000;

    float *d_standard_out, h_standard_out;
    CHECK(cudaMalloc((void**)&d_standard_out, sizeof(float)));

    float *d_intrinsic_out, h_intrinsic_out;
    CHECK(cudaMalloc((void**)&d_intrinsic_out, sizeof(float)));

    float input_value = 8181.25;
    // float  input_value         = 2001.625;
    double mean_intrinsic_time = 0.0;
    double mean_standard_time  = 0.0;

    for (i = 0; i < runs; i++) {
        double start_standard = seconds();
        standard_kernel<<<1, 32>>>(input_value, d_standard_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_standard_time += seconds() - start_standard;

        double start_intrinsic = seconds();
        intrinsic_kernel<<<1, 32>>>(input_value, d_intrinsic_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_intrinsic_time += seconds() - start_intrinsic;
    }

    CHECK(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float), cudaMemcpyDeviceToHost));
    float host_value = powf(input_value, 2.0f);

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n", host_value == h_standard_out ? "Yes" : "No", fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n", host_value == h_intrinsic_out ? "Yes" : "No", fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n", h_standard_out == h_intrinsic_out ? "Yes" : "No", fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f s\n", mean_standard_time);
    printf("Mean execution time for intrinsic function __powf: %f s\n", mean_intrinsic_time);
}

__device__ int myAtomicAdd(int* address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess    = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the guess is incorrect.
    while (oldValue != guess) {
        guess    = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }

    return oldValue;
}

__global__ void kernel(int* sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}

void my_atomic_add()
{
    int  h_sharedInteger;
    int* d_sharedInteger;
    CHECK(cudaMalloc((void**)&d_sharedInteger, sizeof(int)));
    CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int), cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);
}

#ifndef SINGLE_PREC
#    ifndef DOUBLE_PREC
#        define SINGLE_PREC
#    endif
#endif

#ifdef SINGLE_PREC

typedef float real;
#    define MAX_DIST 200.0f
#    define MAX_SPEED 100.0f
#    define MASS 2.0f
#    define DT 0.00001f
#    define LIMIT_DIST 0.000001f
#    define POW(x, y) powf(x, y)
#    define SQRT(x) sqrtf(x)

#else   // SINGLE_PREC

typedef double real;
#    define MAX_DIST 200.0
#    define MAX_SPEED 100.0
#    define MASS 2.0
#    define DT 0.00001
#    define LIMIT_DIST 0.000001
#    define POW(x, y) pow(x, y)
#    define SQRT(x) sqrt(x)

#endif   // SINGLE_PREC

#ifdef VALIDATE
/**
 * Host implementation of the NBody simulation.
 **/
static void h_nbody_update_velocity(real* px, real* py, real* vx, real* vy, real* ax, real* ay, int N, int* exceeded_speed, int id)
{
    real total_ax = 0.0f;
    real total_ay = 0.0f;

    real my_x = px[id];
    real my_y = py[id];

    int i = (id + 1) % N;

    while (i != id) {
        real other_x = px[i];
        real other_y = py[i];

        real rx = other_x - my_x;
        real ry = other_y - my_y;

        real dist2 = rx * rx + ry * ry;

        if (dist2 < LIMIT_DIST) {
            dist2 = LIMIT_DIST;
        }

        real dist6 = dist2 * dist2 * dist2;
        real s     = MASS * (1.0f / SQRT(dist6));
        total_ax += rx * s;
        total_ay += ry * s;

        i = (i + 1) % N;
    }

    ax[id] = total_ax;
    ay[id] = total_ay;

    vx[id] = vx[id] + ax[id];
    vy[id] = vy[id] + ay[id];

    real v = SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0));

    if (v > MAX_SPEED) {
        *exceeded_speed = *exceeded_speed + 1;
    }
}

static void h_nbody_update_position(real* px, real* py, real* vx, real* vy, int N, int* beyond_bounds, int id)
{

    px[id] += (vx[id] * DT);
    py[id] += (vy[id] * DT);

    real dist = SQRT(POW(px[id], 2.0) + POW(py[id], 2.0));

    if (dist > MAX_DIST) {
        *beyond_bounds = 1;
    }
}
#endif   // VALIDATE

/**
 * CUDA implementation of simple NBody.
 **/
__global__ void d_nbody_update_velocity(real* px, real* py, real* vx, real* vy, real* ax, real* ay, int N, int* exceeded_speed)
{
    int  tid      = blockIdx.x * blockDim.x + threadIdx.x;
    real total_ax = 0.0f;
    real total_ay = 0.0f;

    if (tid >= N)
        return;

    real my_x = px[tid];
    real my_y = py[tid];

    int i = (tid + 1) % N;

    while (i != tid) {
        real other_x = px[i];
        real other_y = py[i];

        real rx = other_x - my_x;
        real ry = other_y - my_y;

        real dist2 = rx * rx + ry * ry;

        if (dist2 < LIMIT_DIST) {
            dist2 = LIMIT_DIST;
        }

        real dist6 = dist2 * dist2 * dist2;
        real s     = MASS * (1.0f / SQRT(dist6));
        total_ax += rx * s;
        total_ay += ry * s;

        i = (i + 1) % N;
    }

    ax[tid] = total_ax;
    ay[tid] = total_ay;

    vx[tid] = vx[tid] + ax[tid];
    vy[tid] = vy[tid] + ay[tid];

    real v = SQRT(POW(vx[tid], 2.0) + POW(vy[tid], 2.0));

    if (v > MAX_SPEED) {
        atomicAdd(exceeded_speed, 1);
    }
}

__global__ void d_nbody_update_position(real* px, real* py, real* vx, real* vy, int N, int* beyond_bounds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    px[tid] += (vx[tid] * DT);
    py[tid] += (vy[tid] * DT);
    real dist = SQRT(POW(px[tid], 2.0) + POW(py[tid], 2.0));
    if (dist > MAX_DIST) {
        *beyond_bounds = 1;
    }
}

static void print_points(real* x, real* y, int N)
{
    int i;
    for (i = 0; i < N; i++) {
        printf("%.20e %.20e\n", x[i], y[i]);
    }
}

void nbody()
{
    int   i;
    int   N     = 30720;
    int   block = 256;
    int   iter, niters = 50;
    real *d_px, *d_py;
    real *d_vx, *d_vy;
    real *d_ax, *d_ay;
    real *h_px, *h_py;
    int * d_exceeded_speed, *d_beyond_bounds;
    int   exceeded_speed, beyond_bounds;
#ifdef VALIDATE
    int   id;
    real *host_px, *host_py;
    real *host_vx, *host_vy;
    real *host_ax, *host_ay;
    int   host_exceeded_speed, host_beyond_bounds;
#endif   // VALIDATE

#ifdef SINGLE_PREC
    printf("Using single-precision floating-point values\n");
#else    // SINGLE_PREC
    printf("Using double-precision floating-point values\n");
#endif   // SINGLE_PREC

#ifdef VALIDATE
    printf("Running host simulation. WARNING, this might take a while.\n");
#endif   // VALIDATE

    h_px = (real*)malloc(N * sizeof(real));
    h_py = (real*)malloc(N * sizeof(real));

#ifdef VALIDATE
    host_px = (real*)malloc(N * sizeof(real));
    host_py = (real*)malloc(N * sizeof(real));
    host_vx = (real*)malloc(N * sizeof(real));
    host_vy = (real*)malloc(N * sizeof(real));
    host_ax = (real*)malloc(N * sizeof(real));
    host_ay = (real*)malloc(N * sizeof(real));
#endif   // VALIDATE

    for (i = 0; i < N; i++) {
        real x = (rand() % 200) - 100;
        real y = (rand() % 200) - 100;

        h_px[i] = x;
        h_py[i] = y;
#ifdef VALIDATE
        host_px[i] = x;
        host_py[i] = y;
#endif   // VALIDATE
    }

    CHECK(cudaMalloc((void**)&d_px, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&d_py, N * sizeof(real)));

    CHECK(cudaMalloc((void**)&d_vx, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&d_vy, N * sizeof(real)));

    CHECK(cudaMalloc((void**)&d_ax, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&d_ay, N * sizeof(real)));

    CHECK(cudaMalloc((void**)&d_exceeded_speed, sizeof(int)));
    CHECK(cudaMalloc((void**)&d_beyond_bounds, sizeof(int)));

    CHECK(cudaMemcpy(d_px, h_px, N * sizeof(real), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_py, h_py, N * sizeof(real), cudaMemcpyHostToDevice));

    CHECK(cudaMemset(d_vx, 0x00, N * sizeof(real)));
    CHECK(cudaMemset(d_vy, 0x00, N * sizeof(real)));
#ifdef VALIDATE
    memset(host_vx, 0x00, N * sizeof(real));
    memset(host_vy, 0x00, N * sizeof(real));
#endif   // VALIDATE

    CHECK(cudaMemset(d_ax, 0x00, N * sizeof(real)));
    CHECK(cudaMemset(d_ay, 0x00, N * sizeof(real)));
#ifdef VALIDATE
    memset(host_ax, 0x00, N * sizeof(real));
    memset(host_ay, 0x00, N * sizeof(real));
#endif   // VALIDATE

    double start = seconds();

    for (iter = 0; iter < niters; iter++) {
        CHECK(cudaMemset(d_exceeded_speed, 0x00, sizeof(int)));
        CHECK(cudaMemset(d_beyond_bounds, 0x00, sizeof(int)));

        d_nbody_update_velocity<<<N / block, block>>>(d_px, d_py, d_vx, d_vy, d_ax, d_ay, N, d_exceeded_speed);
        d_nbody_update_position<<<N / block, block>>>(d_px, d_py, d_vx, d_vy, N, d_beyond_bounds);
    }

    CHECK(cudaDeviceSynchronize());
    double exec_time = seconds() - start;

#ifdef VALIDATE

    for (iter = 0; iter < niters; iter++) {
        printf("iter=%d\n", iter);
        host_exceeded_speed = 0;
        host_beyond_bounds  = 0;

#    pragma omp parallel for
        for (id = 0; id < N; id++) {
            h_nbody_update_velocity(host_px, host_py, host_vx, host_vy, host_ax, host_ay, N, &host_exceeded_speed, id);
        }

#    pragma omp parallel for
        for (id = 0; id < N; id++) {
            h_nbody_update_position(host_px, host_py, host_vx, host_vy, N, &host_beyond_bounds, id);
        }
    }

#endif   // VALIDATE

    CHECK(cudaMemcpy(&exceeded_speed, d_exceeded_speed, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&beyond_bounds, d_beyond_bounds, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_px, d_px, N * sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_py, d_py, N * sizeof(real), cudaMemcpyDeviceToHost));

    print_points(h_px, h_py, 10);
    printf("Any points beyond bounds? %s, # points exceeded velocity %d/%d\n", beyond_bounds > 0 ? "true" : "false", exceeded_speed, N);
    printf("Total execution time %f s\n", exec_time);

#ifdef VALIDATE
    double error = 0.0;

    for (i = 0; i < N; i++) {
        double dist = sqrt(pow(h_px[i] - host_px[i], 2.0) + pow(h_py[i] - host_py[i], 2.0));
        error += dist;
    }

    error /= N;
    printf("Error = %.20e\n", error);
#endif   // VALIDATE
}