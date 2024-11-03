#include "test_14.cuh"
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,          // Data-type of A matrix
                                                    ColumnMajor,    // Layout of A matrix
                                                    float,          // Data-type of B matrix
                                                    ColumnMajor,    // Layout of B matrix
                                                    float,          // Data-type of C matrix
                                                    ColumnMajor>;   // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm            gemm_operator;
    CutlassGemm::Arguments args({M, N, K},        // Gemm Problem dimensions
                                {A, lda},         // Tensor-ref for source matrix A
                                {B, ldb},         // Tensor-ref for source matrix B
                                {C, ldc},         // Tensor-ref for source matrix C
                                {C, ldc},         // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta});   // Scalars used in the Epilogue

    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(float* matrix, int rows, int columns, int seed = 0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i + j * rows;

        // Generate arbitrary elements.
        int const k     = 16807;
        int const m     = 16;
        float     value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float* matrix, int rows, int columns, int seed = 0)
{

    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);

    InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);

    return cudaGetLastError();
}

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float** matrix, int rows, int columns, int seed = 0)
{
    cudaError_t result;

    size_t sizeof_matrix = sizeof(float) * rows * columns;

    // Allocate device memory.
    result = cudaMalloc(reinterpret_cast<void**>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Clear the allocation.
    result = cudaMemset(*matrix, 0, sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Initialize matrix elements to arbitrary small integers.
    result = InitializeMatrix(*matrix, rows, columns, seed);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}


/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
            accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float* C, int ldc)
{
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return cudaGetLastError();
}

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta)
{
    cudaError_t result;
    // Compute leading dimensions for each matrix.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(float) * ldc * N;

    // Define pointers to matrices in GPU device memory.
    float* A;
    float* B;
    float* C_cutlass;
    float* C_reference;

    //
    // Allocate matrices in GPU device memory with arbitrary seeds.
    //

    result = AllocateMatrix(&A, M, K, 0);

    if (result != cudaSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, K, N, 17);

    if (result != cudaSuccess) {
        cudaFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return result;
    }

    result = AllocateMatrix(&C_reference, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return result;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Launch CUTLASS GEMM.
    //

    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Verify.
    //

    // Launch reference GEMM
    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    if (result != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    // Copy to host and verify equivalence.
    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy Reference GEMM results: " << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Free device memory allocations.
    //

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    //
    // Test for bit equivalence of results.
    //

    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;

        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int test_cutlass_01()
{
    // GEMM problem dimensions.
    int problem[3] = {128, 128, 128};

    // Scalars used for linear scaling the result of the matrix product.
    float scalars[2] = {1, 2};

    //
    // Run the CUTLASS GEMM test.
    //

    cudaError_t result = TestCutlassGemm(problem[0],   // GEMM M dimension
                                         problem[1],   // GEMM N dimension
                                         problem[2],   // GEMM K dimension
                                         scalars[0],   // alpha
                                         scalars[1]    // beta
    );

    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }

    return result == cudaSuccess ? 0 : -1;
}


#define EXAMPLE_MATRIX_ROW 64
#define EXAMPLE_MATRIX_COL 32

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename GmemIterator, typename SmemIterator>
__global__ void kernel_dump(typename GmemIterator::Params params, typename GmemIterator::TensorRef ref)
{
    extern __shared__ Element shared_storage[];

    // Construct the global iterator and load the data to the fragments.
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    GmemIterator gmem_iterator(params, ref.data(), {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL}, tb_thread_id);

    typename GmemIterator::Fragment frag;

    frag.clear();
    gmem_iterator.load(frag);

    // Call dump_fragment() with different parameters.
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nAll threads dump all the elements:\n");
    cutlass::debug::dump_fragment(frag);

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps all the elements:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1);

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps first 16 elements:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16);

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16, /*S = */ 8);

    // Construct the shared iterator and store the data to the shared memory.
    SmemIterator smem_iterator(typename SmemIterator::TensorRef({shared_storage, SmemIterator::Layout::packed({EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL})}), tb_thread_id);

    smem_iterator.store(frag);

    // Call dump_shmem() with different parameters.
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nDump all the elements:\n");
    cutlass::debug::dump_shmem(shared_storage, EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nDump all the elements with a stride of 8:\n");
    cutlass::debug::dump_shmem(shared_storage, EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL, /*S = */ 8);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int test_dump_reg_shmem(){
    // Initialize a 64x32 column major matrix with sequential data (1,2,3...).
    using Element = cutlass::half_t;
    using Layout  = cutlass::layout::ColumnMajor;

    cutlass::HostTensor<Element, Layout> matrix({EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL});
    cutlass::reference::host::BlockFillSequential(matrix.host_data(), matrix.capacity());

    // Dump the matrix.
    std::cout << "Matrix:\n" << matrix.host_view() << "\n";

    // Copy the matrix to the device.
    matrix.sync_device();

    // Define a global iterator, a shared iterator and their thread map.
    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, 32, cutlass::layout::PitchLinearShape<8, 4>, 8>;

    using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element, Layout, 1, ThreadMap>;

    typename GmemIterator::Params params(matrix.layout());

    using SmemIterator = cutlass::transform::threadblock::
        RegularTileIterator<cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, 1, ThreadMap>;

    dim3 grid(1, 1);
    dim3 block(32, 1, 1);

    int smem_size = int(sizeof(Element) * EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

    kernel_dump<Element, GmemIterator, SmemIterator><<<grid, block, smem_size, 0>>>(params, matrix.device_ref());

    cudaError_t result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
        std::cout << "Failed" << std::endl;
    }

    return (result == cudaSuccess ? 0 : -1);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.

template<typename Iterator>
__global__ void copy(typename Iterator::Params dst_params, typename Iterator::Element* dst_pointer, typename Iterator::Params src_params, typename Iterator::Element* src_pointer,
                     cutlass::Coord<2> extent)
{


    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for (size_t i = 0; i < fragment.size(); ++i) {
        fragment[i] = 0;
    }

    src_iterator.load(fragment);
    dst_iterator.store(fragment);


    ++src_iterator;
    ++dst_iterator;

    for (; iterations > 1; --iterations) {

        src_iterator.load(fragment);
        dst_iterator.store(fragment);

        ++src_iterator;
        ++dst_iterator;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes the source tile with sequentially increasing values and performs the copy into
// the destination tile using two PredicatedTileIterators, one to load the data from addressable
// memory into a fragment (regiser-backed array of elements owned by each thread) and another to
// store the data from the fragment back into the addressable memory of the destination tile.

cudaError_t TestTileIterator(int M, int K)
{

    // For this example, we chose a <64, 4> tile shape. The PredicateTileIterator expects
    // PitchLinearShape and PitchLinear layout.
    using Shape        = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout       = cutlass::layout::PitchLinear;
    using Element      = int;
    int const kThreads = 32;

    // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
    // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
    // dimension then along the strided dimension.
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 1, ThreadMap>;


    cutlass::Coord<2> copy_extent  = cutlass::make_Coord(M, K);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M, K);

    // Allocate source and destination tensors
    cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

    Element oob_value = Element(-1);

    // Initialize destination tensor with all -1s
    cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
    // Initialize source tensor with sequentially increasing values
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_params(src_tensor.layout());

    dim3 block(kThreads, 1);
    dim3 grid(1, 1);

    // Launch copy kernel to perform the copy
    copy<Iterator><<<grid, block>>>(dst_params, dst_tensor.device_data(), src_params, src_tensor.device_data(), copy_extent);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "Error - kernel failed." << std::endl;
        return result;
    }

    dst_tensor.sync_host();

    // Verify results
    for (int s = 0; s < alloc_extent[1]; ++s) {
        for (int c = 0; c < alloc_extent[0]; ++c) {

            Element expected = Element(0);

            if (c < copy_extent[0] && s < copy_extent[1]) {
                expected = src_tensor.at({c, s});
            }
            else {
                expected = oob_value;
            }

            Element got   = dst_tensor.at({c, s});
            bool    equal = (expected == got);

            if (!equal) {
                std::cerr << "Error - source tile differs from destination tile." << std::endl;
                return cudaErrorUnknown;
            }
        }
    }

    return cudaSuccess;
}

int test_tile_iterator(){
    cudaError_t result = TestTileIterator(57, 35);
    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }
    return result == cudaSuccess ? 0 : -1;
}