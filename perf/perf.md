## wsl 安装perf


```bash
# https://blog.csdn.net/qq_20452055/article/details/108300321
# https://zhuanlan.zhihu.com/p/600483539
# https://cloud.tencent.com/developer/article/2228048

sudo apt-get install linux-tools-common
# WARNING: perf not found for kernel 5.15.167.4-microsoft
#
#   You may need to install the following packages for this specific kernel:
#     linux-tools-5.15.167.4-microsoft-standard-WSL2
#     linux-cloud-tools-5.15.167.4-microsoft-standard-WSL2
#
#   You may also want to install one of the following packages to keep up to date:
#     linux-tools-standard-WSL2
#     linux-cloud-tools-standard-WSL2

# 解决方式 编译wsl kernel
sudo apt install build-essential flex bison libssl-dev libelf-dev
git clone --depth 1  https://github.moeyy.xyz/https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
make -j8
sudo cp perf /usr/local/bin
# 编译的时候，makefile 提示缺少库，只是少一些检测特性。不用在意
# 编译成功后，即可在此文件夹下找到perf工具，执行成功，也可以自行将perf工具移动到/usr/bin文件夹下方便调用

# 如果某些依赖项无法安装，可以选择禁用相应的功能。如禁用 libtraceevent：
make -j10 NO_LIBTRACEEVENT=1

# 依赖库的问题 https://blog.csdn.net/stallion5632/article/details/141749501
sudo apt update
sudo apt install git build-essential libncurses-dev flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf

sudo apt update
sudo apt install libdw-dev libunwind-dev libslang2-dev liblzma-dev libzstd-dev libcap-dev libnuma-dev libbabeltrace-dev libperl-dev libtraceevent-dev libpfm4-dev libsystemd-dev elfutils libelf-dev systemtap-sdt-dev

# 安装libtraceevent编译工具和依赖
sudo apt update
sudo apt install git build-essential autoconf automake libtool pkg-config

git clone https://git.kernel.org/pub/scm/libs/libtrace/libtraceevent.git
cd libtraceevent

make
sudo make install
sudo ldconfig



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

# -p 指定对进程分析，perf record表示采集系统事件, 没有使用 -e 指定采集事件, 则默认采集 cycles(即 CPU clock 周期), -F 99 表示每秒 99 次, -p 2347 是进程号, 即对哪个进程进行分析, -g 表示记录调用栈, sleep 30 则是持续 30 秒.
perf record -F 99 -p 2347 -g -- sleep 30

```

```bash

# 参考 https://blog.csdn.net/SaberJYang/article/details/123964439

# git clone https://github.com/brendangregg/FlameGraph.git
Cloning into 'FlameGraph'...
remote: Enumerating objects: 961, done.
remote: Total 961 (delta 0), reused 0 (delta 0), pack-reused 961
Receiving objects: 100% (961/961), 1.83 MiB | 118.00 KiB/s, done.
Resolving deltas: 100% (547/547), done.
# ls
# 生成工具在目录下面
FlameGraph

# 解析perf收集的信息
perf script -i perf.data &> perf.unfold
# 生成折叠后的调用栈
./stackcollapse-perf.pl perf.unfold &> perf.folded
# 生成火焰图
./flamegraph.pl perf.folded > perf.svg

# 脚本
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > perf.svg


# 火焰图可以在谷歌浏览器中直接打开，放大缩小，vscode 中不支持放大缩小
```




## uftrace

https://github.com/namhyung/uftrace

```bash
## https://www.bigcatblog.com/uftrace/

## apt 安装， github 有源码安装 需要make install
sudo apt update
sudo apt install uftrace

## 增加编译选项 add_cxflags("-pg")
# 在控制台直接查看
uftrace perf_001

# 先记录5次在查看
uftrace record perf_001  5
uftrace replay

# 不记录 直接查看
uftrace live perf_001  5

# 不想看到 library function 的紀錄， new delete
uftrace --no-libcall perf_001  5

# 如果想看巢狀的 library function 紀錄
uftrace --nest-libcall perf_001  5

# 加上參數 -k 或是 --kernel 來记录 kernel的调用
uftrace -k perf_001  5
# 如果在紀錄中有出現 event 的紀錄，像是下面 linux:sched-out ，可以加上 --no-event 來取消顯示









# 指定层数
uftrace --depth 3 perf_001  5
uftrace -D 3 perf_001  5
# 只追蹤該 function 和其下的 children functions
uftrace -F allocate2 perf_001  5
uftrace --filter allocate2 perf_001  5

# 不追蹤該 function 和其下的 children functions
uftrace -N allocate1 test 3
uftrace --notrace allocate1 test 3

# 只追蹤呼叫該 function 的來源
uftrace -C allocate1 test 3
uftrace --caller-filter allocate1 test 3

# 只不追蹤該 function
uftrace -H allocate2 test 3
uftrace --hide allocate2 test 3

# 過濾花費時間低於設定的閥值的 function
uftrace -t 1us test 3
uftrace --time-filter 1us test 3








# 紀錄 function 傳進去的參數和回傳值
uftrace -A atoi@arg1 test 3
uftrace --argument atoi@arg1 test 3
# 我們可以設定變數型態，來幫助我們更好地了解數據
uftrace -A atoi@arg1/s test 3
uftrace --argument atoi@arg1/s test 3

#顯示 atoi 的回傳為 3，atoi() = 3
uftrace -R atoi@retval test 3
uftrace --retval atoi@retval test 3

# 可以自動記錄參數和回傳值
uftrace -a test 3
uftrace --auto-args test 3


# uftrace report
# Total time：包含自身以及 child functions 所花費的時間
# Self time：只包含自身 function 所花費的時間
# 我們也可以加上參數 -s 來選擇排列順序
# uftrace report -s self
# 預設為 total
# total：依照整個 function 所花費的全部時間由多到少排列
# self：依照自身 function 所花費的時間由多到少排列
# call：依照呼叫次數由多到少排列
Total time   Self time       Calls  Function
==========  ==========  ==========  ====================
54.400 us    1.400 us           1  main
31.900 us    1.000 us           1  allocate1
30.200 us   30.200 us           4  operator new[]
21.100 us   21.100 us           1  atoi
 1.100 us    0.700 us           3  allocate2


Total time   Self time       Calls  Function
  ==========  ==========  ==========  ====================
   41.441 ms   40.793 ms           1  <556cb550d1db>
  489.444 us  489.444 us          25  operator delete
   77.954 us   77.954 us           4  std::__ostream_insert
   58.916 us   58.916 us           1  std::ios_base::Init::Init
   47.301 us   47.301 us          25  operator new
   23.202 us   23.202 us           2  std::basic_ostream::_M_insert
    6.684 us    6.684 us           1  linux:schedule
    3.476 us    3.476 us           4  std::chrono::_V2::system_clock::now
   59.957 us    0.852 us           1  <556cb550d4a4>
    0.189 us    0.189 us           1  __cxa_atexit

# 看程式的 call graph 也可以了解到程式裡的 function 之間的關係
uftrace record test 3
uftrace graph
```