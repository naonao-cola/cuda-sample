[第二章](./doc/02.md)

[第三章](./doc/03.md)

[第四章](./doc/04.md)

[第五章](./doc/05.md)

[第六章](./doc/06.md)

[第七章](./doc/07.md)

[第八章](./doc/08.md)

[第九章](./doc/09.md)

[第十章](./doc/10.md)

```bash
# ncu  为所有用户启用访问权限

https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters

# ncu 使用参考，可以不带参数

https://www.cnblogs.com/chenjambo/articles/using-nsight-compute-to-inspect-your-kernels.html#:~:text=Nsight%20C

https://zhuanlan.zhihu.com/p/715022552

# 参数解释对照

https://zhuanlan.zhihu.com/p/666242337#:~:text=%E7%9B%AE%E5%89%8D%E4%B8%BB%E6%B5%81%E7%9A%84%20CU

# 这个问题是nsight 的版本太低了。需要升级版本。支持显卡的芯片

#Profiling is not supported on device 0. To find out supported GPUs refer --list-chips option.

#这个命令是查看当前的支持的显卡芯片,
ncu --set full -f --list-chips -o 03 ./03
# 生成ncu-rep 文件
ncu --set full -f  -o 03 ./03

# 性能分析 参考
https://zhuanlan.zhihu.com/p/463144086#:~:text=Nsight%20C

# 图吧工具箱 下载gpu-z 工具

```

```c++
ncu ./10

==PROF== Connected to process 75876 (/home/naonao/cxx/cuda-sample/build/linux/x86_64/release/10)
==PROF== Profiling "ampere_sgemm_32x128_nn" - 0: 0%....50%....100% - 9 passes
==PROF== Disconnected from process 75876
[75876] 10@127.0.0.1
  ampere_sgemm_32x128_nn (13, 2, 2)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         6.86
    SM Frequency            cycle/nsecond         1.58
    Elapsed Cycles                  cycle        38921
    Memory Throughput                   %        51.82
    DRAM Throughput                     %        13.44
    Duration                      usecond        24.54
    L1/TEX Cache Throughput             %        68.85
    L2 Cache Throughput                 %        22.56
    SM Active Cycles                cycle     29279.62
    Compute (SM) Throughput             %        51.80
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.5 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     52
    Registers Per Thread             register/thread              57
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           16.38
    Threads                                   thread           13312
    Waves Per SM                                                0.54
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        33.23
    Achieved Active Warps Per SM           warp        15.95
    ------------------------------- ----------- ------------

    OPT   Estimated Speedup: 50.16%
          This kernel's theoretical occupancy (66.7%) is limited by the number of required registers. The difference
          between calculated theoretical (66.7%) and measured achieved occupancy (33.2%) can be the result of warp
          scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between
          warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices Guide
          (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.
```


![](./images/ncu_1.jpg)