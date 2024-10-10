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
### wsl 安装perf

```bash
# https://blog.csdn.net/qq_20452055/article/details/108300321
# https://zhuanlan.zhihu.com/p/600483539
# https://cloud.tencent.com/developer/article/2228048

sudo apt-get install linux-tools-common
# WARNING: perf not found for kernel 5.15.153.1-microsoft

#   You may need to install the following packages for this specific kernel:
#     linux-tools-5.15.153.1-microsoft-standard-WSL2
#     linux-cloud-tools-5.15.153.1-microsoft-standard-WSL2

#   You may also want to install one of the following packages to keep up to date:
#     linux-tools-standard-WSL2
#     linux-cloud-tools-standard-WSL2

# 解决方式 编译wsl kernel
sudo apt install build-essential flex bison libssl-dev libelf-dev
git clone https://gitee.com/mirrors/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
make -j8
sudo cp perf /usr/local/bin
# 编译的时候，makefile 提示缺少库，只是少一些检测特性。不用在意
# 编译成功后，即可在此文件夹下找到perf工具，执行成功，也可以自行将perf工具移动到/usr/bin文件夹下方便调用


# perf_event_paranoid setting is 2:
#   -1: Allow use of (almost) all events by all users
#       Ignore mlock limit after perf_event_mlock_kb without CAP_IPC_LOCK
# >= 0: Disallow raw and ftrace function tracepoint access
# >= 1: Disallow CPU event access
# >= 2: Disallow kernel profiling
# To make the adjusted perf_event_paranoid setting permanent preserve it
# in /etc/sysctl.conf (e.g. kernel.perf_event_paranoid = <setting>)

# 修改 /etc/sysctl.conf 文件，并使之生效
sudo /sbin/sysctl -p

# https://zhuanlan.zhihu.com/p/686247554
# https://www.cnblogs.com/conscience-remain/p/16142279.html
# https://blog.csdn.net/youzhangjing_/article/details/124671286

#生成记录
perf record -F 99 -a -g ./11
#查看图标  或者直接查看函数调用占比 perf report -i perf.data
perf report --call-graph none
perf stat -e cache-misses ./11
perf stat -e cpu-clock ./11
# 只查看 11的程序
perf report --call-graph none -c 11
# 生成带svg的图片record
sudo perf timechart record ./11
# 转化为svg
sudo perf timechart
# 查看特定函数的情况
sudo perf annotate -f main
```
![](./images/ncu_1.jpg)