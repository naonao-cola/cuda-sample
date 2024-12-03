#include "../common/common.h"
#include <stdio.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


/**
 * 此示例是一个简单的代码，说明了线程块内的协作组。代码启动单个线程块创建块中所有线程的协作组，
 * 以及一组拼接分区协作组。对于每一个，它使用一个通用缩减函数，用于计算中所有等级的总和这个群体。
 * 在每种情况下，结果都会与预期答案（使用分析公式计算（n-1）*n）/2，注意排名从零开始）。

 Launching a single block with 64 threads...

 Sum of all ranks 0..63 in threadBlockGroup is 2016 (expected 2016)

 Now creating 4 groups, each of size 16 threads:

   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)

...Done.


 */
void test_cg_01();




#include <cuda/barrier>
#include <cuda_runtime.h>
void test_cg_02();