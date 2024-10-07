#include "../common/common.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
/**
https://zhuanlan.zhihu.com/p/657632577#:~:text=%E9%80%9A%E7%94%A8%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%20(

*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


float testError(void);
float testPerformance(void (*gpuSgemm)(float*, float*, float*, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);


void cpuSgemm(float* a, float* b, float* c, const int M, const int N, const int K);


/*
Max Error = 0.000046

Kernal = naiveSgemm
M N K =    128    128   1024, Time =   0.00007373   0.00010524   0.00021392 s, AVG Performance =   296.9448 Gflops
M N K =    192    192   1024, Time =   0.00013926   0.00014590   0.00016589 s, AVG Performance =   481.9305 Gflops
M N K =    256    256   1024, Time =   0.00020685   0.00022146   0.00024986 s, AVG Performance =   564.4299 Gflops
M N K =    384    384   1024, Time =   0.00041882   0.00045454   0.00050381 s, AVG Performance =   618.7563 Gflops
M N K =    512    512   1024, Time =   0.00074547   0.00077204   0.00096563 s, AVG Performance =   647.6308 Gflops
M N K =    768    768   1024, Time =   0.00165568   0.00188435   0.00238285 s, AVG Performance =   597.0242 Gflops
M N K =   1024   1024   1024, Time =   0.00311603   0.00332503   0.00417587 s, AVG Performance =   601.4983 Gflops
M N K =   1536   1536   1024, Time =   0.00509133   0.00671816   0.00841523 s, AVG Performance =   669.8266 Gflops
M N K =   2048   2048   1024, Time =   0.00898560   0.00976430   0.01047142 s, AVG Performance =   819.3114 Gflops
M N K =   3072   3072   1024, Time =   0.02140160   0.02181140   0.02277888 s, AVG Performance =   825.2563 Gflops
M N K =   4096   4096   1024, Time =   0.03825562   0.03887430   0.04003123 s, AVG Performance =   823.1659 Gflops
M N K =   6144   6144   1024, Time =   0.08832614   0.09031987   0.09316250 s, AVG Performance =   797.1668 Gflops
M N K =   8192   8192   1024, Time =   0.21685453   0.22033111   0.22526976 s, AVG Performance =   580.9438 Gflops
M N K =  12288  12288   1024, Time =   0.61061633   0.61236737   0.61524785 s, AVG Performance =   470.3059 Gflops
M N K =  16384  16384   1024, Time =   1.08129478   1.08677652   1.09356451 s, AVG Performance =   471.1180 Gflops
*/
void test_naive();


float testError(void (*gpuSgemm)(float*, float*, float*, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);


/**
Kernal = sgemm_V1
Max Error = 0.000046
M N K =    128    128   1024, Time =   0.00025498   0.00029261   0.00036438 s, AVG Performance =   106.7982 Gflops
M N K =    192    192   1024, Time =   0.00025498   0.00026317   0.00028365 s, AVG Performance =   267.1707 Gflops
M N K =    256    256   1024, Time =   0.00025600   0.00027534   0.00033075 s, AVG Performance =   453.9776 Gflops
M N K =    384    384   1024, Time =   0.00025498   0.00029210   0.00042803 s, AVG Performance =   962.8473 Gflops
M N K =    512    512   1024, Time =   0.00025702   0.00029777   0.00052723 s, AVG Performance =  1679.1687 Gflops
M N K =    768    768   1024, Time =   0.00046387   0.00058315   0.00123373 s, AVG Performance =  1929.1923 Gflops
M N K =   1024   1024   1024, Time =   0.00069734   0.00072468   0.00090522 s, AVG Performance =  2759.8206 Gflops
M N K =   1536   1536   1024, Time =   0.00135782   0.00153108   0.00234394 s, AVG Performance =  2939.0926 Gflops
M N K =   2048   2048   1024, Time =   0.00245862   0.00281303   0.00397517 s, AVG Performance =  2843.9082 Gflops
M N K =   3072   3072   1024, Time =   0.00538419   0.00580035   0.00686182 s, AVG Performance =  3103.2634 Gflops
M N K =   4096   4096   1024, Time =   0.00733901   0.00872364   0.01109290 s, AVG Performance =  3668.1942 Gflops
M N K =   6144   6144   1024, Time =   0.01735987   0.01766029   0.01790566 s, AVG Performance =  4076.9422 Gflops
M N K =   8192   8192   1024, Time =   0.03020595   0.03144468   0.03200922 s, AVG Performance =  4070.6403 Gflops
M N K =  12288  12288   1024, Time =   0.07000166   0.07106734   0.07186022 s, AVG Performance =  4052.4944 Gflops
M N K =  16384  16384   1024, Time =   0.12459827   0.12609290   0.12739481 s, AVG Performance =  4060.4983 Gflops
*/
void test_gemm_1();


/**
Kernal = sgemm_V2
Max Error = 0.000046
M N K =    128    128   1024, Time =   0.00015770   0.00017287   0.00024413 s, AVG Performance =   180.7679 Gflops
M N K =    192    192   1024, Time =   0.00015962   0.00016902   0.00019558 s, AVG Performance =   415.9991 Gflops
M N K =    256    256   1024, Time =   0.00015565   0.00016628   0.00018941 s, AVG Performance =   751.7223 Gflops
M N K =    384    384   1024, Time =   0.00015872   0.00016741   0.00019750 s, AVG Performance =  1679.9951 Gflops
M N K =    512    512   1024, Time =   0.00015974   0.00017056   0.00020787 s, AVG Performance =  2931.4647 Gflops
M N K =    768    768   1024, Time =   0.00023142   0.00023874   0.00028365 s, AVG Performance =  4712.1919 Gflops
M N K =   1024   1024   1024, Time =   0.00037786   0.00037907   0.00038093 s, AVG Performance =  5276.1318 Gflops
M N K =   1536   1536   1024, Time =   0.00068096   0.00068330   0.00068602 s, AVG Performance =  6585.7257 Gflops
M N K =   2048   2048   1024, Time =   0.00124109   0.00152136   0.00267059 s, AVG Performance =  5258.4639 Gflops
M N K =   3072   3072   1024, Time =   0.00269107   0.00285234   0.00347341 s, AVG Performance =  6310.6170 Gflops
M N K =   4096   4096   1024, Time =   0.00470835   0.00471427   0.00471859 s, AVG Performance =  6787.8985 Gflops
M N K =   6144   6144   1024, Time =   0.00817152   0.00950856   0.01223270 s, AVG Performance =  7572.1269 Gflops
M N K =   8192   8192   1024, Time =   0.01571226   0.01624412   0.01656730 s, AVG Performance =  7879.7751 Gflops
M N K =  12288  12288   1024, Time =   0.03659366   0.03763128   0.03891815 s, AVG Performance =  7653.2070 Gflops
M N K =  16384  16384   1024, Time =   0.06625076   0.06804900   0.06985728 s, AVG Performance =  7523.9904 Gflops
*/
void test_gemm_2();

/**
Kernal = sgemm_V3
Max Error = 0.000046
M N K =    128    128   1024, Time =   0.00010138   0.00011104   0.00017789 s, AVG Performance =   281.4220 Gflops
M N K =    192    192   1024, Time =   0.00010240   0.00012900   0.00022112 s, AVG Performance =   545.0514 Gflops
M N K =    256    256   1024, Time =   0.00010240   0.00013382   0.00033165 s, AVG Performance =   934.0627 Gflops
M N K =    384    384   1024, Time =   0.00010240   0.00010819   0.00012902 s, AVG Performance =  2599.5453 Gflops
M N K =    512    512   1024, Time =   0.00010342   0.00012145   0.00017101 s, AVG Performance =  4117.0426 Gflops
M N K =    768    768   1024, Time =   0.00018941   0.00019516   0.00023450 s, AVG Performance =  5764.5482 Gflops
M N K =   1024   1024   1024, Time =   0.00028339   0.00028546   0.00029491 s, AVG Performance =  7006.2550 Gflops
M N K =   1536   1536   1024, Time =   0.00055910   0.00064142   0.00128307 s, AVG Performance =  7015.6401 Gflops
M N K =   2048   2048   1024, Time =   0.00101171   0.00119674   0.00246170 s, AVG Performance =  6684.8317 Gflops
M N K =   3072   3072   1024, Time =   0.00221594   0.00221920   0.00222310 s, AVG Performance =  8111.0194 Gflops
M N K =   4096   4096   1024, Time =   0.00317338   0.00340589   0.00453632 s, AVG Performance =  9395.4763 Gflops
M N K =   6144   6144   1024, Time =   0.00675635   0.00722473   0.00820634 s, AVG Performance =  9965.7709 Gflops
M N K =   8192   8192   1024, Time =   0.01201766   0.01303153   0.01370522 s, AVG Performance =  9822.3336 Gflops
M N K =  12288  12288   1024, Time =   0.02844774   0.02961224   0.03051418 s, AVG Performance =  9725.7091 Gflops
M N K =  16384  16384   1024, Time =   0.05106381   0.05456701   0.05724262 s, AVG Performance =  9382.9579 Gflops
*/
void test_gemm_3();


/**
Kernal = cublas
Max Error = 0.000198
M N K =    128    128   1024, Time =   0.00001638   0.00045450   0.00433962 s, AVG Performance =    68.7565 Gflops
M N K =    192    192   1024, Time =   0.00002458   0.00008693   0.00025395 s, AVG Performance =   808.8293 Gflops
M N K =    256    256   1024, Time =   0.00003174   0.00034702   0.00308122 s, AVG Performance =   360.2123 Gflops
M N K =    384    384   1024, Time =   0.00006861   0.00010292   0.00023325 s, AVG Performance =  2732.7475 Gflops
M N K =    512    512   1024, Time =   0.00009421   0.00013916   0.00040653 s, AVG Performance =  3592.9452 Gflops
M N K =    768    768   1024, Time =   0.00018330   0.00020344   0.00033894 s, AVG Performance =  5529.8860 Gflops
M N K =   1024   1024   1024, Time =   0.00032038   0.00033318   0.00039424 s, AVG Performance =  6002.8047 Gflops
M N K =   1536   1536   1024, Time =   0.00069306   0.00070873   0.00084480 s, AVG Performance =  6349.3891 Gflops
M N K =   2048   2048   1024, Time =   0.00117555   0.00120535   0.00146534 s, AVG Performance =  6637.0741 Gflops
M N K =   3072   3072   1024, Time =   0.00254566   0.00269085   0.00396883 s, AVG Performance =  6689.3409 Gflops
M N K =   4096   4096   1024, Time =   0.00352870   0.00444866   0.00621053 s, AVG Performance =  7193.1735 Gflops
M N K =   6144   6144   1024, Time =   0.00786842   0.00831548   0.00925472 s, AVG Performance =  8658.5550 Gflops
M N K =   8192   8192   1024, Time =   0.01400525   0.01530317   0.01654579 s, AVG Performance =  8364.2812 Gflops
M N K =  12288  12288   1024, Time =   0.03326566   0.03480494   0.03646361 s, AVG Performance =  8274.6876 Gflops
M N K =  16384  16384   1024, Time =   0.06256845   0.06426501   0.06605824 s, AVG Performance =  7967.0101 Gflops
*/
void test_cublass();