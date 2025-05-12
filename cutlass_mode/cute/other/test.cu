
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

//#include " cute/util/debug.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

void test_001()
{

    using namespace cute;
    cute::device_init(0);


    auto s1 = cute::make_stride(cute::Int<12>{}, cute::Int<1>{});
    cute::print(s1);

    auto s2 = cute::make_shape(cute::Int<2>{}, 4);
    cute::print(s2);

    //默认从上到下 从左到右
    auto shape = cute::Shape<cute::_3, cute::Shape<cute::_2, cute::_3>>{};
    auto ls    = cute::make_layout(shape);
    std::cout << "\r\n 形状 \r\n";
    cute::print(ls);
    std::cout << "\r\n shape 坐标转换 \r\n";
    cute::print(cute::idx2crd(16, shape));                                                              // (1,(1,2))
    cute::print(cute::idx2crd(cute::_16{}, shape));                                                     // (_1,(_1,_2))
    cute::print(cute::idx2crd(cute::make_coord(1, 5), shape));                                          // (1,(1,2))
    cute::print(cute::idx2crd(cute::make_coord(cute::_1{}, 5), shape));                                 // (_1,(1,2))
    cute::print(cute::idx2crd(cute::make_coord(1, cute::make_coord(1, 2)), shape));                     // (1,(1,2))
    cute::print(cute::idx2crd(cute::make_coord(cute::_1{}, cute::make_coord(1, cute::_2{})), shape));   // (_1,(1,_2))


    std::cout << " \r\n layout \r\n";
    auto s2xd4_col = cute::make_layout(cute::make_shape(cute::Int<2>{}, 4), cute::make_stride(cute::Int<12>{}, cute::Int<1>{}));
    cute::print_layout(s2xd4_col);

    cute::Layout s2xh4 = cute::make_layout(cute::make_shape(2, cute::make_shape(2, 2)), cute::make_stride(4, cute::make_stride(2, 1)));
    cute::print_layout(s2xh4);

    cute::Layout ls2 = cute::make_layout(cute::make_shape(cute::make_shape(2, 4), cute::make_shape(3, 5)), cute::make_stride(cute::make_stride(3, 6), cute::make_stride(1, 24)));
    cute::print_layout(ls2);

    std::cout << " \r\n 验证 \r\n";
    cute::print(cute::crd2idx(cute::make_coord(cute::make_coord(cute::_1{}, cute::_3{}), cute::make_coord(cute::_2{}, cute::_4{})),
                              cute::make_shape(cute::make_shape(2, 4), cute::make_shape(3, 5)),
                              cute::make_stride(cute::make_stride(3, 6), cute::make_stride(1, 24))));

    std::cout << " \r\n";

    //[(2,4),(3,5)]:[(3,6),(1,24)]
    cute::print(layout<0>(ls2));      // (2,4):(3,6)
    cute::print(layout<1>(ls2));      // (3,5):(1,24)
    cute::print(layout<1, 0>(ls2));   // 3:1
    cute::print(layout<1, 1>(ls2));   // 5:24
}

void test_002()
{
    // https://blog.csdn.net/qq_33146555/article/details/130658810
    using namespace cute;
    auto ls1 = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};
    auto result = coalesce(ls1);   // _12:_1

    cute::print(ls1);
    std::cout << "\r\n 向左合并 \r\n";
    cute::print(result);
    //复合函数
    Layout a = make_layout(make_shape(Int<10>{}, Int<2>{}), make_stride(Int<16>{}, Int<4>{}));
    Layout b = make_layout(make_shape(Int<5>{}, Int<4>{}), make_stride(Int<1>{}, Int<5>{}));
    // 将b 的结果套用在a里
    /*
    A 矩阵 10行2列 步长16 4
    0    4
    16   20
    32   36
    48   52
    64   68
    80   84
    96   100
    112  116
    128  132
    144  148

    B 矩阵 5行4列  步长 1  5
    0   5  10  15
    1   6  11  16
    2   7  12  17
    3   8  13  18
    4   9  14  19

    结果矩阵，将a矩阵按照b矩阵的序号排列，形状相同， 形状[5,,(2,2)]:[16,(80,4)]
    0  80  4   84
    16 96  20 100
    32 112 36 116
    48 128 52 132
    64 144 68 148
    */
    Layout c = composition(a, b);
    std::cout << "\r\n composition-1 \r\n";
    cute::print(c);


    // https://zhuanlan.zhihu.com/p/28356098779

    //  (12,(4,8)):(59,(13,1))
    auto composition_src_2 = make_layout(make_shape(12, make_shape(4, 8)), make_stride(59, make_stride(13, 1)));
    // (8, 3)
    auto tiler = make_shape(Int<3>{}, Int<8>{});
    // Equivalent to <3:1, 8:1>
    // auto tiler = make_tile(Layout<_3,_1>{},  // Apply 3:1 to mode-0
    //                        Layout<_8,_1>{}); // Apply 8:1 to mode-1

    // (_3,(4,2)):(59,(13,1))
    //取子块，从composition_src_2 里面取tiler 的子块， 结果为 子块的形状跟步长
    auto composition_ret_2 = composition(composition_src_2, tiler);
    std::cout << "\r\n composition-2 \r\n";
    cute::print(composition_ret_2);


    // (12,(4,8)):(59,(13,1))
    auto composition_src_3 = make_layout(make_shape(12, make_shape(4, 8)), make_stride(59, make_stride(13, 1)));
    // <3:4, 8:2>
    auto tiler_3 = make_tile(Layout<_3, _4>{},    // Apply 3:4 to mode-0
                            Layout<_8, _2>{});   // Apply 8:2 to mode-1

    // (_3,(2,4)):(236,(26,1))
    //取子块 子块形状 3 行8列 ，行每隔4个取一个， 列每隔2个取一个。 结果为子块的形状
    auto composition_ret_3 = composition(composition_src_3, tiler_3);
    std::cout << "\r\n composition-3 \r\n";
    cute::print(composition_ret_3);
}

void test_03()
{
    using namespace cute;
    auto layout = Layout<Shape<_4, _2, _3>, Stride<_2, _1, _8>>{};
    cute::print(layout);
    std::cout << "\r\n logical_divide \r\n";
    auto tiler = Layout(Shape<_4>{}, Stride<_2>{});
    auto result = logical_divide(layout, tiler);
    cute::print_layout(result);

    auto layout2 = make_layout(make_shape(9, make_shape(4, 8)), make_stride(59, make_stride(13, 1)));
    //cute::print_layout(layout2);
    std::cout << "\r\n logical_divide_2 \r\n";
    auto tiler_2 = make_tile(Layout(Shape<_3>{}, Stride<_3>{}),
                             Layout(Shape<_2,_4>{}, Stride<_1,_8>{}));
    auto result_2   = logical_divide(layout2, tiler_2);
    cute::print_layout(result_2);
}


void test_04()
{
    using namespace cute;
    Layout tile = Layout<Shape<_2, _2>, Stride<_1, _2>>{};
    cute::print_layout(tile);
    Layout matrix_of_tiles = Layout<Shape<_3, _4>, Stride<_4, _1>>{};
    cute::print_layout(matrix_of_tiles);
    std::cout << "\r\n blocked_product \r\n";
    print_layout(blocked_product(tile, matrix_of_tiles));
    std::cout << "\r\n logical_product \r\n";
    print_layout(logical_product(tile, matrix_of_tiles));

}
int main(int argc, char** argv)
{
    // test_001();
    // test_002();
    //test_03();
    test_04();
}
