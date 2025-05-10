


--  暂时不支持 10.0版本
target("cutlass_mode_cute_sgemm_1")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("sgemm_1.cu")
target_end()


target("cutlass_mode_cute_sgemm_2")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("sgemm_2.cu")
target_end()



target("cutlass_mode_cute_sgemm_sm70")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("sgemm_sm70.cu")
target_end()

target("cutlass_mode_cute_sgemm_sm80")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("sgemm_sm80.cu")
target_end()


target("cutlass_mode_cute_tiled_copy")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("tiled_copy.cu")
target_end()


target("cutlass_mode_cute_other_test")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_packages("openmp")
    add_packages("cutlass")
    add_cuflags("-allow-unsupported-compiler")
    add_cuflags("-rdc=true")
    add_cuflags("--expt-relaxed-constexpr",{public = true})
    add_cuflags("--extended-lambda",{public = true})
    add_cuflags("--ptxas-options=-v",{fource = true})
    set_group("cute_mode")
    add_cuflags(
        "-c",
        "-lineinfo",
        "--source-in-ptx",
        {fource = true})
    add_includedirs("../../3rdparty")
    if is_mode("debug") then
        add_cuflags("-g -G")
    end
    --add_headerfiles("./example_utils.hpp")
    add_files("other/test.cu")
target_end()