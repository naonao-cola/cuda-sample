

cutlass_mode_list = {
        "00_basic_gemm"

}


-- for _, dir in ipairs(os.dirs("$(scriptdir)/*")) do
--     local target_name = path.filename(dir)
--     if os.isdir(target_name) then
--         target(target_name)
--             set_kind("binary")
--             add_rules("cuda")
--             add_cugencodes("native")
--             add_packages("openmp")
--             add_packages("cutlass")
--             add_cuflags("-allow-unsupported-compiler")
--             for _, filepath in ipairs(os.files(target_name .. "/*")) do
--                 local s = filepath
--                 if s:endswith(".cuh") or s:endswith(".h") then
--                     add_headerfiles(filepath)
--                 end
--                 if s:endswith(".cu") or s:endswith(".cpp") then
--                     add_files(filepath)
--                 end
--             end
--             --print(target_name)
--         target_end()
--     end
-- end

for _, v in pairs(cutlass_mode_list) do
    local target_name = "cutlass_mode_" .. v
    target(target_name)
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
        add_includedirs("common/")
        set_group("cutlass_mode")
        add_cuflags(
            "-c",
            "-lineinfo",
            "--source-in-ptx",
            {fource = true})
        add_includedirs("../3rdparty")
        if is_mode("debug") then
            add_cuflags("-g -G")
        end
        --add_culdflags("-gencode arch=compute_86,code=sm_86")
        for _, filedir in ipairs(os.filedirs(string.format("%s/**", v))) do
            local s = filedir
            if s:endswith(".cuh") or s:endswith(".h") then
                add_headerfiles(filedir)
            end
            if s:endswith(".cu") or s:endswith(".cpp") then
                add_files(filedir)
            end
        end
end
