#include "../common/common.h"

/**
使用CUDA事件来控制GPU上启动的异步工作的示例。在本例中，异步副本和异步内核是使用。CUDA事件用于确定工作何时完成。

CPU executed 52135 iterations while waiting for GPU to finish
*/
void asyncAPI();

/**
使用CUDA回调来触发主机上的工作的示例完成设备上的异步工作。在这个例子中，n_streamsCUDA流被创建并在每个流中异步启动4个内核。
然后，在这些异步内核完成时添加回调打印诊断信息。事件计时器，并且事件同步

> Using Device 0: NVIDIA GeForce RTX 4060 Laptop GPU
> Compute Capability 8.9 hardware with 24 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 8
> with streams = 4
callback from stream 0
callback from stream 1
callback from stream 3
callback from stream 2
Measured time for parallel execution = 2.796s

*/
void simpleCallback();

/**

这个例子演示了将工作提交到CUDA流的广度优先秩序。按广度优先顺序提交工作可以防止错误依赖减少应用程序的并行性。kernel_1 kernel_2,
Kernel_3和kernel_4简单地实现相同的虚拟计算。使用单独的内核可以使这些内核的调度更简单在可视化分析器中可视化。

xmake run 06 4  2

CUDA_DEVICE_MAX_CONNECTIONS = 32
> Using Device 0: NVIDIA GeForce RTX 4060 Laptop GPU with num_streams 4
> Compute Capability 8.9 hardware with 24 multi-processors
> grid 8 block 512
Measured time for parallel execution = 14.485s

*/
void simpleHyperqBreadth(int argv1 = 4, int argv2 = 2);

/**
流间添加依赖
添加流间依赖项的简单示例cudaStreamWaitEvent。这段代码在每个n_streams中启动4个内核流。在每个流完成时记录一个事件(kernelEvent)。
然后在该事件和最后一个流上调用cudaStreamWaitEvent(streams[n_streams - 1])强制最终流中的所有计算只进行当所有其他流完成时执行。

xmake run 06 4  1

CUDA_DEVICE_MAX_CONNECTIONS = 32
> Using Device 0: NVIDIA GeForce RTX 4060 Laptop GPU with num_streams 4
> Compute Capability 8.9 hardware with 24 multi-processors
> grid 8 block 512
Measured time for parallel execution = 20.043s

*/
void simpleHyperqDependence(int argv1 = 4, int argv2 = 2);

/**

*/
void simpleHyperqDepth();