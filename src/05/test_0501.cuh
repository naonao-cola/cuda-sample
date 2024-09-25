#include "../common/common.h"



/**
使用各种优化(包括共享内存、展开和内存填充)对矩形主机数组进行转置的示例内核。

device 0: NVIDIA GeForce RTX 4060 Laptop GPU  with matrix nx 4096 ny 4096
copyGmem elapsed 0.005008 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 26.800760 GB
naiveGmem elapsed 0.001309 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 102.540977 GB
naiveGmemUnroll elapsed 0.001153 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 116.384109 GB
transposeSmem elapsed 0.001665 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 80.617203 GB
transposeSmemPad elapsed 0.001838 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 73.025032 GB
transposeSmemDyn elapsed 0.001801 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 74.523422 GB
transposeSmemPadDyn elapsed 0.001632 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 82.242508 GB
transposeSmemUnroll elapsed 0.001259 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 106.599121 GB
transposeSmemUnrollPad elapsed 0.001226 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 109.480736 GB
transposeSmemUnrollPadDyn elapsed 0.001798 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 74.641998 GB

device 0: NVIDIA GeForce RTX 4060 Laptop GPU  with matrix nx 4096 ny 4096
copyGmem elapsed 0.002995 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 44.813721 GB
naiveGmem elapsed 0.001246 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 107.721001 GB
naiveGmemUnroll elapsed 0.001847 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 72.676216 GB
transposeSmem elapsed 0.001680 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 79.896385 GB
transposeSmemPad elapsed 0.003189 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 42.086571 GB
transposeSmemDyn elapsed 0.001652 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 81.245483 GB
transposeSmemPadDyn elapsed 0.002279 sec <<< grid (256,256) block (16,16)>>> effective bandwidth 58.892139 GB
transposeSmemUnroll elapsed 0.001709 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 78.525589 GB
transposeSmemUnrollPad elapsed 0.001227 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 109.374382 GB
transposeSmemUnrollPadDyn elapsed 0.001510 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 88.891510 GB

*/
void transposeRectangle();