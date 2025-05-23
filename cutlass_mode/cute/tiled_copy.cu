﻿
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

// This is a simple tutorial showing several ways to partition a tensor into tiles then
// perform efficient, coalesced copies. This example also shows how to vectorize accesses
// which may be a useful optimization or required for certain workloads.
//
// `copy_kernel()` and `copy_kernel_vectorized()` each assume a pair of tensors with
// dimensions (m, n) have been partitioned via `tiled_divide()`.
//
// The result are a part of compatible tensors with dimensions ((M, N), m', n'), where
// (M, N) denotes a statically sized tile, and m' and n' denote the number of such tiles
// within the tensor.
//
// Each statically sized tile is mapped to a CUDA threadblock which performs efficient
// loads and stores to Global Memory.
//
// `copy_kernel()` uses `cute::local_partition()` to partition the tensor and map
// the result to threads using a striped indexing scheme. Threads themselve are arranged
// in a (ThreadShape_M, ThreadShape_N) arrangement which is replicated over the tile.
//
// `copy_kernel_vectorized()` uses `cute::make_tiled_copy()` to perform a similar
// partitioning using `cute::Copy_Atom` to perform vectorization. The actual vector
// size is defined by `ThreadShape`.
//
// This example assumes the overall tensor shape is divisible by the tile size and
// does not perform predication.


//这是一个简单的教程，展示了将张量划分为图块的几种方法执行高效的合并副本。此示例还显示了如何将访问矢量化这可能是一个有用的优化，或者是某些工作负载所必需的。
//
//`copy_kernel（）`和`copy_ernel-vectualized（）`均假设有一对张量维度（m，n）已通过`tiled_divide（）`进行分区。

//结果是具有维度（（M，N），M'，N'）的兼容张量的一部分，其中（M，N）表示静态大小的图块，M'和N'表示此类图块的数量在张量内。
//
//每个静态大小的图块都映射到一个CUDA线程块，该线程块执行高效加载并存储到全局内存。
//
//`copy_kernel（）`使用`cute:：local_partition（）`对张量和映射进行分区使用条带索引方案的线程的结果。线本身是排列的在拼贴块上复制的（ThreadShape_M、ThreadShape_N）排列中。
//
//`copy_kernel_vectorized（）`使用`cute:：make_tiled_copy（）`执行类似的操作使用`cute:：Copy_Atom`进行分区以执行矢量化。实际矢量大小由“ThreadShape”定义。
//
//这个例子假设整个张量形状可以被图块大小整除不执行预测。


/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template<class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
    using namespace cute;

    // Slice the tiled tensors
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

    // Construct a partitioning of the tile among threads with the given thread arrangement.

    // Concept:                         Tensor  ThrLayout       ThrIndex
    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);   // (ThrValM, ThrValN)
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);   // (ThrValM, ThrValN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_tensor to try to match the layout of thr_tile_S
    Tensor fragment = make_tensor_like(thr_tile_S);   // (ThrValM, ThrValN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(thr_tile_S, fragment);
    copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template<class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
    using namespace cute;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S);   // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);   // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D);   // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

/// Main function
int main(int argc, char** argv)
{
    //
    // Given a 2D shape, perform an efficient copy
    //

    using namespace cute;
    using Element = float;

    // Define a tensor shape with dynamic extents (m, n)
    auto tensor_shape = make_shape(256, 512);

    //
    // Allocate and initialize
    //

    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    //
    // Make tensors
    //

    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

    //
    // Tile tensors
    //

    // Define a statically sized block (M, N).
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
        std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
        return -1;
    }
    // Equivalent check to the above
    if (not evenly_divides(tensor_shape, block_shape)) {
        std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
        return -1;
    }

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);   // ((M, N), m', n')
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);   // ((M, N), m', n')

    // Construct a TiledCopy with a specific access pattern.
    //   This version uses a
    //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
    //   (2) Layout-of-Values that each thread will access.

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));   // (32,8) -> thr_idx

    // Value arrangement per thread
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

    // Define `AccessType` which controls the size of the actual memory access instruction.
    using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;   // A very specific access width copy instruction
    // using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
    // using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

    // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
    using Atom = Copy_Atom<CopyOp, Element>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contigous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative value layouts are also possible, though incompatible layouts
    // will result in compile time errors.
    TiledCopy tiled_copy = make_tiled_copy(Atom{},        // Access strategy
                                           thr_layout,    // thread layout (e.g. 32x4 Col-Major)
                                           val_layout);   // value layout (e.g. 4x1)

    //
    // Determine grid and block dimensions
    //

    dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

    //
    // Launch the kernel
    //
    copy_kernel_vectorized<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D, tiled_copy);

    cudaError result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }

    //
    // Verify
    //

    h_D = d_D;

    int32_t       errors      = 0;
    int32_t const kErrorLimit = 10;

    for (size_t i = 0; i < h_D.size(); ++i) {
        if (h_S[i] != h_D[i]) {
            std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Success." << std::endl;

    return 0;
}
