[第二章](./doc/02.md)

[第三章](./doc/03.md)

[第四章](./doc/04.md)

[第五章](./doc/05.md)

[第六章](./doc/06.md)

[第七章](./doc/07.md)

[第八章](./doc/08.md)

[第九章](./doc/09.md)

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

#这个命令是查看当前的支持的显卡芯片
ncu --set full -f --list-chips -o 03 ./03

# 性能分析 参考
https://zhuanlan.zhihu.com/p/463144086#:~:text=Nsight%20C

# 图吧工具箱 下载gpu-z 工具
```


![](./images/ncu_1.jpg)