set_project("CUDA-SAMPLE")
set_version("1.1.0")
set_languages("c++17")


add_rules("mode.release", "mode.debug", "mode.check","mode.releasedbg")
add_requires("openmp")
add_requires("cutlass")
set_runtimes("MT")

tutorial_list = {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
}

for _, v in pairs(tutorial_list) do
    target(v)
        set_kind("binary")
        add_rules("cuda")
        add_cugencodes("native")
        add_packages("openmp")
        add_packages("cutlass")
        add_cuflags("-allow-unsupported-compiler")
        set_group("test")
        add_headerfiles("src/common/common.h")
        add_includedirs("3rdparty")
        if is_mode("debug") then
            --add_defines("DEBUG")
            --set_symbols("debug")
            --set_optimize("none")
            add_cuflags("-g -G")
        end
        if v =="08"  or v== "09" or v == "10" then
            add_links("cublas","curand","cufft","cusparse")
        end
        if  v == "10" then
            add_links("nvToolsExt")
        end
        add_culdflags("-gencode arch=compute_89,code=sm_89")
        for _, filedir in ipairs(os.filedirs(string.format("src/%s/**", v))) do
            --print(filedir)
            local s = filedir
            if s:endswith(".cuh") or s:endswith(".h") then
                add_headerfiles(filedir)
            end
            if s:endswith(".cu") or s:endswith(".cpp") then
                add_files(filedir)
            end
        end
end
