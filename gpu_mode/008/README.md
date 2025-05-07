# CUDA Performance Gotchas

```
nvcc -o benchmark file.cu
ncu benchmark
```

```bash
# gpu-mode-008-coalesce
==PROF== Connected to process 90408 (/home/naonao/demo/rep/cuda-sample/build/linux/x86_64/release/benchmark)
==PROF== Profiling "copyDataNonCoalesced" - 0: 0%....50%....100% - 9 passes
==PROF== Profiling "copyDataCoalesced" - 1: 0%....50%....100% - 9 passes
==PROF== Disconnected from process 90408
[90408] benchmark@127.0.0.1
  copyDataNonCoalesced(float *, float *, int) (131072, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.29
    SM Frequency            cycle/nsecond         1.32
    Elapsed Cycles                  cycle     26685092
    Memory Throughput                   %        16.20
    DRAM Throughput                     %         0.26
    Duration                      msecond        20.22
    L1/TEX Cache Throughput             %         0.60
    L2 Cache Throughput                 %        16.20
    SM Active Cycles                cycle  27136250.39
    Compute (SM) Throughput             %         0.52
    ----------------------- ------------- ------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 131072
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread        16777216
    Uses Green Context                                             0
    Waves Per SM                                              390.10
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           32
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        83.30
    Achieved Active Warps Per SM           warp        39.98
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 16.7%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (83.3%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    379885.33
    Total DRAM Elapsed Cycles        cycle    884696064
    Average L1 Active Cycles         cycle  27136250.39
    Total L1 Elapsed Cycles          cycle    881710400
    Average L2 Active Cycles         cycle  29036016.56
    Total L2 Elapsed Cycles          cycle    455769180
    Average SM Active Cycles         cycle  27136250.39
    Total SM Elapsed Cycles          cycle    881710400
    Average SMSP Active Cycles       cycle  27130925.58
    Total SMSP Elapsed Cycles        cycle   3526841600
    -------------------------- ----------- ------------

  copyDataCoalesced(float *, float *, int) (131072, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.29
    SM Frequency            cycle/nsecond         1.32
    Elapsed Cycles                  cycle     16156807
    Memory Throughput                   %        16.16
    DRAM Throughput                     %         0.34
    Duration                      msecond        12.24
    L1/TEX Cache Throughput             %         1.05
    L2 Cache Throughput                 %        16.16
    SM Active Cycles                cycle  17006236.50
    Compute (SM) Throughput             %         0.70
    ----------------------- ------------- ------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 131072
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread        16777216
    Uses Green Context                                             0
    Waves Per SM                                              390.10
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           32
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        78.78
    Achieved Active Warps Per SM           warp        37.82
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 21.22%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (78.8%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    300994.67
    Total DRAM Elapsed Cycles        cycle    535649280
    Average L1 Active Cycles         cycle  17006236.50
    Total L1 Elapsed Cycles          cycle    452507110
    Average L2 Active Cycles         cycle  15589310.67
    Total L2 Elapsed Cycles          cycle    275950854
    Average SM Active Cycles         cycle  17006236.50
    Total SM Elapsed Cycles          cycle    452507110
    Average SMSP Active Cycles       cycle  17003312.09
    Total SMSP Elapsed Cycles        cycle   1810028440
    -------------------------- ----------- ------------

```

```bash
# coarsening
==PROF== Connected to process 94345 (/home/naonao/demo/rep/cuda-sample/build/linux/x86_64/release/benchmark)
==PROF== Profiling "VecAdd" - 0: 0%....50%....100% - 8 passes
==PROF== Profiling "VecAddCoarsened" - 1: 0%....50%....100% - 8 passes
==PROF== Profiling "VecAdd" - 2: 0%....50%....100% - 8 passes
VecAdd execution time: 53.201794 ms
==PROF== Profiling "VecAddCoarsened" - 3: 0%....50%....100% - 8 passes
VecAddCoarsened execution time: 56.024258 ms
==PROF== Disconnected from process 94345
[94345] benchmark@127.0.0.1
  VecAdd(float *, float *, float *) (4, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.14
    SM Frequency            cycle/nsecond         1.29
    Elapsed Cycles                  cycle         3047
    Memory Throughput                   %         1.83
    DRAM Throughput                     %         1.55
    Duration                      usecond         2.37
    L1/TEX Cache Throughput             %         4.77
    L2 Cache Throughput                 %         1.83
    SM Active Cycles                cycle       191.54
    Compute (SM) Throughput             %         0.30
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      4
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread            1024
    Uses Green Context                                             0
    Waves Per SM                                                0.02
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 85.71%
          The grid for this launch is configured to execute only 4 blocks, which is less than the GPU's 28
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel
          concurrently with other workloads, consider reducing the block size to have at least one block per
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
          description for more details on launch configurations.

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        17.04
    Achieved Active Warps Per SM           warp         8.18
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 82.96%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (17.0%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       261.33
    Total DRAM Elapsed Cycles        cycle       101376
    Average L1 Active Cycles         cycle       191.54
    Total L1 Elapsed Cycles          cycle        85944
    Average L2 Active Cycles         cycle       377.33
    Total L2 Elapsed Cycles          cycle        51984
    Average SM Active Cycles         cycle       191.54
    Total SM Elapsed Cycles          cycle        85944
    Average SMSP Active Cycles       cycle       187.33
    Total SMSP Elapsed Cycles        cycle       343776
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 5.37%
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 86.06% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.257%
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 86.14% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.37%
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 86.06% above the average, while the minimum instance value is 100.00% below the
          average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 7.119%
          One or more L2 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 54.48% above the average, while the minimum instance value is 90.19% below the
          average.

  VecAddCoarsened(float *, float *, float *) (2, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.04
    SM Frequency            cycle/nsecond         1.28
    Elapsed Cycles                  cycle         3199
    Memory Throughput                   %         1.90
    DRAM Throughput                     %         1.88
    Duration                      usecond         2.50
    L1/TEX Cache Throughput             %        13.17
    L2 Cache Throughput                 %         1.90
    SM Active Cycles                cycle       110.64
    Compute (SM) Throughput             %         0.26
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      2
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread             512
    Uses Green Context                                             0
    Waves Per SM                                                0.01
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 92.86%
          The grid for this launch is configured to execute only 2 blocks, which is less than the GPU's 28
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel
          concurrently with other workloads, consider reducing the block size to have at least one block per
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
          description for more details on launch configurations.

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        16.11
    Achieved Active Warps Per SM           warp         7.73
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 83.89%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (16.1%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       330.67
    Total DRAM Elapsed Cycles        cycle       105472
    Average L1 Active Cycles         cycle       110.64
    Total L1 Elapsed Cycles          cycle        86110
    Average L2 Active Cycles         cycle       477.56
    Total L2 Elapsed Cycles          cycle        54612
    Average SM Active Cycles         cycle       110.64
    Total SM Elapsed Cycles          cycle        86110
    Average SMSP Active Cycles       cycle       108.26
    Total SMSP Elapsed Cycles        cycle       344440
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 6.876%
          One or more L2 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 43.68% above the average, while the minimum instance value is 65.24% below the
          average.

  VecAdd(float *, float *, float *) (4, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.23
    SM Frequency            cycle/nsecond         1.31
    Elapsed Cycles                  cycle         3054
    Memory Throughput                   %         2.41
    DRAM Throughput                     %         2.41
    Duration                      usecond         2.34
    L1/TEX Cache Throughput             %         4.87
    L2 Cache Throughput                 %         2.19
    SM Active Cycles                cycle       187.61
    Compute (SM) Throughput             %         0.31
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      4
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread            1024
    Uses Green Context                                             0
    Waves Per SM                                                0.02
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 85.71%
          The grid for this launch is configured to execute only 4 blocks, which is less than the GPU's 28
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel
          concurrently with other workloads, consider reducing the block size to have at least one block per
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
          description for more details on launch configurations.

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        17.48
    Achieved Active Warps Per SM           warp         8.39
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 82.52%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (17.5%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle          408
    Total DRAM Elapsed Cycles        cycle       101376
    Average L1 Active Cycles         cycle       187.61
    Total L1 Elapsed Cycles          cycle        82250
    Average L2 Active Cycles         cycle       331.39
    Total L2 Elapsed Cycles          cycle        52092
    Average SM Active Cycles         cycle       187.61
    Total SM Elapsed Cycles          cycle        82250
    Average SMSP Active Cycles       cycle       183.72
    Total SMSP Elapsed Cycles        cycle       329000
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 5.492%
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 85.99% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.381%
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 86.03% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.492%
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 85.99% above the average, while the minimum instance value is 100.00% below the
          average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 6.246%
          One or more L2 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 54.54% above the average, while the minimum instance value is 88.83% below the
          average.

  VecAddCoarsened(float *, float *, float *) (2, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.18
    SM Frequency            cycle/nsecond         1.30
    Elapsed Cycles                  cycle         3236
    Memory Throughput                   %         1.93
    DRAM Throughput                     %         1.85
    Duration                      usecond         2.50
    L1/TEX Cache Throughput             %        13.76
    L2 Cache Throughput                 %         1.93
    SM Active Cycles                cycle       105.89
    Compute (SM) Throughput             %         0.25
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      2
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              28
    Threads                                   thread             512
    Uses Green Context                                             0
    Waves Per SM                                                0.01
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 92.86%
          The grid for this launch is configured to execute only 2 blocks, which is less than the GPU's 28
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel
          concurrently with other workloads, consider reducing the block size to have at least one block per
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
          description for more details on launch configurations.

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        17.03
    Achieved Active Warps Per SM           warp         8.18
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 82.97%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (17.0%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       330.67
    Total DRAM Elapsed Cycles        cycle       107520
    Average L1 Active Cycles         cycle       105.89
    Total L1 Elapsed Cycles          cycle        89024
    Average L2 Active Cycles         cycle       392.67
    Total L2 Elapsed Cycles          cycle        55242
    Average SM Active Cycles         cycle       105.89
    Total SM Elapsed Cycles          cycle        89024
    Average SMSP Active Cycles       cycle       103.79
    Total SMSP Elapsed Cycles        cycle       356096
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 6.856%
          One or more L2 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 53.59% above the average, while the minimum instance value is 90.58% below the
          average.                                                    
```