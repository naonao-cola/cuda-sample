
set_languages("cxx11")

gemm_list = {
        "MMult0",
        "MMult1",
        "MMult2",
        "MMult_1x4_3",
        "MMult_1x4_4",
        "MMult_1x4_5",
        "MMult_1x4_6",
        "MMult_1x4_7",
        "MMult_1x4_8",
        "MMult_1x4_9",
        "MMult_4x4_3",
        "MMult_4x4_4",
        "MMult_4x4_5",
        "MMult_4x4_6",
        "MMult_4x4_7",
        "MMult_4x4_8",
        "MMult_4x4_9",
        "MMult_4x4_10",
        "MMult_4x4_11",
        "MMult_4x4_12",
        "MMult_4x4_13",
        "MMult_4x4_14",
        "MMult_4x4_15"
}


rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
    end)

rule_end()


for _, v in pairs(gemm_list) do
    target(v)
        set_kind("binary")
        add_packages("openmp")
        add_includedirs("./utils/")
        add_files("utils/**c")
        add_rules("rule_display")
        add_runenvs("OMP_NUM_THREADS", "1")
        add_runenvs("GOTO_NUM_THREADS", "1")
        add_files("src/test_MMult.c")
        add_files(string.format("src/%s.c", v))
        add_vectorexts("all")  -- 增加指令集扩展
        on_run(function (target)
            os.cp("$(scriptdir)/utils/output_new.m", "$(scriptdir)/utils/output_old.m")
            local file = io.open("$(scriptdir)/utils/output_new.m", "w")
            if file then
                file:write(string.format("version = '%s';\n", v))
                file:close()
            end
            local outdata, errdata = os.iorun(target:targetfile())
            file = io.open("$(scriptdir)/utils/output_new.m", "a")
            if file then
                file:write(outdata)
                file:close()
            end
            cprint("${green}  outdata: %s", outdata)
        end)
end