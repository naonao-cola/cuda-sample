#include "../common/common.h"
#include <stdio.h>

/*
 * CUDA编程的简单介绍。这个程序打印“Hello”
 *  世界来自GPU!从GPU上运行的10个CUDA线程。
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}


