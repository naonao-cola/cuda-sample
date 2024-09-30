set_project("CUDA-SAMPLE")
set_version("1.1.0")
set_languages("c++17")


add_rules("mode.release", "mode.debug", "mode.check","mode.releasedbg")
set_runtimes("MT")

tutorial_list = {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
}

for _, v in pairs(tutorial_list) do
    target(v)
        set_kind("binary")
        add_rules("cuda")
        add_cugencodes("native")
        --add_cuflags("-allow-unsupported-compiler")
        set_group("test")
        add_headerfiles("src/common/common.h")
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
