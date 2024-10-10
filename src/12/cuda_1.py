from pycuda.compiler import SourceModule
import sys
from time import time
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.scan import InclusiveScanKernel
from pycuda.reduction import ReductionKernel

InteractiveShell.ast_node_interactivity = "all"


def query_device():
    print(f'The version of PyCUDA: {pycuda.VERSION}')
    print(f'The version of Python: {sys.version}')
    drv.init()
    print('CUDA device query (PyCUDA version) \n')
    print(f'Detected {drv.Device.count()} CUDA Capable device(s) \n')
    for i in range(drv.Device.count()):

        gpu_device = drv.Device(i)
        print(f'Device {i}: {gpu_device.name()}')
        compute_capability = float('%d.%d' % gpu_device.compute_capability())
        print(f'\t Compute Capability: {compute_capability}')
        print(
            f'\t Total Memory: {gpu_device.total_memory()//(1024**2)} megabytes')

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes['MULTIPROCESSOR_COUNT']

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = {5.0: 128, 5.1: 128, 5.2: 128,
                             6.0: 64, 6.1: 128, 6.2: 128, 8.9: 128}.get(compute_capability, None)

        if cuda_cores_per_mp is None:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}")

        print(f'\t ({num_mp}) Multiprocessors, ({cuda_cores_per_mp}) CUDA Cores / Multiprocessor: {num_mp*cuda_cores_per_mp} CUDA Cores')

        device_attributes.pop('MULTIPROCESSOR_COUNT')

        for k in device_attributes.keys():
            print(f'\t {k}: {device_attributes[k]}')


def cuda_1():
    """
    NumPy array 和 gpuarray 之间的相互转换
    """
    host_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    device_data = gpuarray.to_gpu(host_data)
    device_data_x2 = 2 * device_data
    host_data_x2 = device_data_x2.get()
    print(host_data_x2)


def cuda_2():
    """
    gpuarray 的基本运算
    """
    x_host = np.array([1, 2, 3], dtype=np.float32)
    y_host = np.array([1, 1, 1], dtype=np.float32)
    z_host = np.array([2, 2, 2], dtype=np.float32)

    x_device = gpuarray.to_gpu(x_host)
    y_device = gpuarray.to_gpu(y_host)
    z_device = gpuarray.to_gpu(z_host)

    # x_host + y_host
    print((x_device + y_device).get())
    # x_host ** z_host
    print((x_device ** z_device).get())
    # x_host / x_host
    print((x_device / x_device).get())
    # z_host - x_host
    print((z_device - x_device).get())
    # z_host / 2
    print((z_device / 2).get())
    # x_host - 1
    print((x_device - 1).get())


def simple_speed_test():
    host_data = np.float32(np.random.random(50000000))

    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()

    print(f'total time to compute on CPU: {t2 - t1}')

    device_data = gpuarray.to_gpu(host_data)

    t1 = time()
    device_data_2x = device_data * np.float32(2)
    t2 = time()

    from_device = device_data_2x.get()

    print(f'total time to compute on GPU: {t2 - t1}')
    print(
        f'Is the host computation the same as the GPU computation? : {np.allclose(from_device, host_data_2x)}')


# query_device()
# cuda_1()
# cuda_2()
# simple_speed_test()

gpu_2x_ker = ElementwiseKernel(
    "float *in, float *out",
    "out[i] = 2 * in[i];",
    "gpu_2x_ker"
)


def elementwise_kernel_example():
    """
    ElementwiseKernel：按元素运算.ElementWiseKernel 非常类似于 map 函数
    list(map(lambda x: x + 10, [1, 2, 3, 4, 5]))
    第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的迭代器

    ElementwiseKernel 的参数：

    class pycuda.elementwise.ElementwiseKernel(arguments, operation, name="kernel", keep=False, options=[], preamble="")

    arguments：该内核定义的传参。
    operation：该内核定义的内嵌 CUDA C 代码。
    name：定义的内核名称。

    gpuarray.empty_like 用于分配与 device_data 相同形状和类型的内存空间。

    """
    host_data = np.float32(np.random.random(50000000))
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print(f'total time to compute on CPU: {t2 - t1}')

    device_data = gpuarray.to_gpu(host_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()
    from_device = device_data_2x.get()
    print(f'total time to compute on GPU: {t2 - t1}')
    print(
        f'Is the host computation the same as the GPU computation? : {np.allclose(from_device, host_data_2x)}')


# elementwise_kernel_example()
# elementwise_kernel_example()
# elementwise_kernel_example()
# elementwise_kernel_example()
# elementwise_kernel_example()


def inclusive():
    """
    Python 标准包 functools 中的 reduce 函数。 reduce(lambda x, y : x + y, [1, 2, 3, 4])
    与 map 函数不同,reduce 执行迭代的二元运算,只输出一个单值。

    InclusiveScanKernel 类似于 reduce,因为它并非输出单值,输出与输入形状相同。

    [ 1  3  6 10]
    [ 1  3  6 10]
    """
    seq = np.array([1, 2, 3, 4], dtype=np.int32)
    seq_gpu = gpuarray.to_gpu(seq)
    sum_gpu = InclusiveScanKernel(np.int32, "a+b")
    print(sum_gpu(seq_gpu).get())
    print(np.cumsum(seq))


# inclusive()

def inclusive_2():
    """
    查找最大值（最大值向后冒泡）：
    对于 a > b ? a : b ,我们可以想象是做从前往后做一个遍历（实际是并行的）,而对于每个当前元素 cur,都和前一个元素做比较,把最大值赋值给 cur
    这样，最大值就好像“冒泡”一样往后移动，最终取最后一个元素即可。
    """
    seq = np.array([1, 100, -3, -10000, 4, 10000, 66, 14, 21], dtype=np.int32)
    seq_gpu = gpuarray.to_gpu(seq)
    max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
    seq_max_bubble = max_gpu(seq_gpu)
    print(seq_max_bubble)
    print(seq_max_bubble.get()[-1])
    print(np.max(seq))


# inclusive_2()

def reduction():
    """
    实际上,ReductionKernel 就像是执行 ElementWiseKernel 后再执行一个并行扫描内核。一个计算两向量内积的例子：
    """
    a_host = np.array([1, 2, 3], dtype=np.float32)
    b_host = np.array([4, 5, 6], dtype=np.float32)
    print(a_host.dot(b_host))
    dot_prod = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b",
                               map_expr="x[i]*y[i]", arguments="float *x, float *y")
    a_device = gpuarray.to_gpu(a_host)
    b_device = gpuarray.to_gpu(b_host)
    print(dot_prod(a_device, b_device).get())


# reduction()

"""
申请内存的写法
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.elementwise import ElementwiseKernel

# 定义矩阵
A = np.random.randn(3, 3).astype(np.float32)
B = np.random.randn(3, 3).astype(np.float32)

# 将矩阵上传到 GPU
d_A = cuda.mem_alloc(A.nbytes)
d_B = cuda.mem_alloc(B.nbytes)
cuda.memcpy_htod(d_A, A)
cuda.memcpy_htod(d_B, B)

# 定义矩阵乘法的内核函数
matmul_kernel = ElementwiseKernel(
    "float *A, float *B, float *C",
    "C[i] = A[i] * B[i]",
    "matmul_kernel"
)

# 执行矩阵乘法
C = gpuarray.empty_like(A)
matmul_kernel(d_A, d_B, C)

# 从 GPU 获取结果
result = np.empty_like(C.get())
cuda.memcpy_dtoh(result, C)

print(result)
"""

# 定义 CUDA 核函数
mod = SourceModule("""
    __global__ void add(int *a, int *b, int *c) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        c[idx] = a[idx] + b[idx];
    }
""")

def test():
    # 获取核函数
    add_func = mod.get_function("add")

    # 定义输入数据
    a = np.array([1, 2, 3, 4]).astype(np.int32)
    b = np.array([5, 6, 7, 8]).astype(np.int32)
    c = np.zeros_like(a)

    # 将数据上传到 GPU
    d_a = gpuarray.to_gpu(a)
    d_b = gpuarray.to_gpu(b)
    d_c = gpuarray.to_gpu(c)

    # 执行核函数
    block_size = 4
    # 向下取整
    grid_size = len(a) // block_size
    add_func(d_a, d_b, d_c, block=(block_size, 1, 1), grid=(grid_size, 1))


test()
