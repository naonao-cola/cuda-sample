

- [参考文档](#参考文档)
- [cute](#cute)
  - [参考文档](#参考文档-1)


# 参考文档

CUTLASS库使用与优化指北（一）

https://zhuanlan.zhihu.com/p/26869575907

# cute
## 参考文档

0. cute 的 shape stride 解析 [峰子的乐园](https://dingfen.github.io/2024/08/18/2024-8-18-cute/)
1. CuTe, shape 形状计算解析 [cnblog](https://www.cnblogs.com/Edwardlyz/articles/18368114)
2. cute 子块的切割 [zhihu](https://zhuanlan.zhihu.com/p/28356098779)



提供了 `cute::print`  打印

```c++
// 第零个块 第零个线程
// 头文件 cute/util/debug.hpp,
if (thread0()) {
  print(some_cute_object);
}
// 当第n 个块 第n个线程 返回 true
bool thread(int tid, int bid)


cute::print_layout()
cute::print_tensor()
cute::print_latex() // Layout, TiledCopy, and TiledMMA
```
```bash
((2,2),2):((4,2),1)
      0   1
    +---+---+
 0  | 0 | 1 |
    +---+---+
 1  | 4 | 5 |
    +---+---+
 2  | 2 | 3 |
    +---+---+
 3  | 6 | 7 |
    +---+---+
#可以这样理解，在行维度上，我们有未知的子 tensor，该子 tensor 有两行（此为 shape 第一个2），
#行之间的 stride 为 4（所以 stride 第一个数为 4）；然后该子 tensor 在整个大 tensor 中会重
#复两次（此为 shape 的第二个 2），相对应地，子 tensor 间的 stride 为 2（此为 stride 的第二个 2）。

```
```bash
(8,(2,2)):(2,(1,16))
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  1 | 16 | 17 |
    +----+----+----+----+
 1  |  2 |  3 | 18 | 19 |
    +----+----+----+----+
 2  |  4 |  5 | 20 | 21 |
    +----+----+----+----+
 3  |  6 |  7 | 22 | 23 |
    +----+----+----+----+
 4  |  8 |  9 | 24 | 25 |
    +----+----+----+----+
 5  | 10 | 11 | 26 | 27 |
    +----+----+----+----+
 6  | 12 | 13 | 28 | 29 |
    +----+----+----+----+
 7  | 14 | 15 | 30 | 31 |
    +----+----+----+----+

# 在列方向上，Shape 的第一个 2 表示，列内的子 tensor pattern 有两列，第二个 2 表示列一共有两个子 pattern 。
# Stride 的 1 表示在这个子 pattern 内的 stride 为 1，16 表示子 pattern 间的 stride 为 16

Layout (3, (2, 3)):(3, (12, 1))
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
对于Tensor中的索引 17，有如下坐标
Coord: 16
Coord: (1, 5)
Coord: (1, (1, 2))
自然坐标与 Stride 做内积可以得到 index 索引。(1, (1, 2)) 与 (3, (12, 1)) 做内积为17


```
![alt text](v2-dbf1d65ebb1b94080d149c8052dfebef_1440w.webp)


[(2,4),(3,5)]:[(3,6),(1,24)]

阅读方式是冒号前是shape矩阵,冒号后是stride矩阵. 然后, 阅读方式是: (innner_row, outter_row), (inner_col, outter_col). (innner_stride_row, outtter_stride_row),(innner_stride_col, outter_stride_col);

坐标 [(a,b),(c,d)] 形状 [(s1,s2),(s3,s4)]:[(d1,d2),(d3,d4)]

坐标计算 [a,b] (*) [d1,d2]  + [c,d] (*) [d3,d4]

然后根据上面的计算公式: 假设目前给一个坐标[(1,3),(2,4)], 那么计算出来结果应该是:
[1,3] (*) [3,6]  + [2,4] (*) [1,24] = 3 + 18 + 2 +96 = 119

