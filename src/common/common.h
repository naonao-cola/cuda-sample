#ifndef _COMMON_H
#define _COMMON_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>



#define CHECK(call)                                                                      \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        }                                                                                \
    }

#define CHECK_CUBLAS(call)                                                              \
    {                                                                                   \
        cublasStatus_t err;                                                             \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1);                                                                    \
        }                                                                               \
    }

#define CHECK_CURAND(call)                                                              \
    {                                                                                   \
        curandStatus_t err;                                                             \
        if ((err = (call)) != CURAND_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1);                                                                    \
        }                                                                               \
    }

#define CHECK_CUFFT(call)                                                              \
    {                                                                                  \
        cufftResult err;                                                               \
        if ((err = (call)) != CUFFT_SUCCESS) {                                         \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1);                                                                   \
        }                                                                              \
    }

#define CHECK_CUSPARSE(call)                                                                          \
    {                                                                                                 \
        cusparseStatus_t err;                                                                         \
        if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);                      \
            cudaError_t cuda_err = cudaGetLastError();                                                \
            if (cuda_err != cudaSuccess) {                                                            \
                fprintf(stderr, "  CUDA error \"%s\" also detected\n", cudaGetErrorString(cuda_err)); \
            }                                                                                         \
            exit(1);                                                                                  \
        }                                                                                             \
    }

inline double seconds()
{
    struct timeval  tp;
    struct timezone tzp;
    int             i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


#include <chrono>
#include <iostream>
#ifndef __ycm__
#    define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#    define TOCK(x) printf("%s: %lfs\n", #    x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());
#else
#    define TICK(x)
#    define TOCK(x)
#endif


/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CHECK_CUTLASS(status)                                                                                        \
    {                                                                                                                \
        cutlass::Status error = status;                                                                              \
        if (error != cutlass::Status::kSuccess) {                                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                            \
    }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t  _start;
    cudaEvent_t  _stop;

    /// Constructor
    GpuTimer()
        : _stream_id(0)
    {
        CHECK(cudaEventCreate(&_start));
        CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CHECK(cudaEventDestroy(_start));
        CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop() { CHECK(cudaEventRecord(_stop, _stream_id)); }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CHECK(cudaEventSynchronize(_stop));
        CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};


#endif   // _COMMON_H
