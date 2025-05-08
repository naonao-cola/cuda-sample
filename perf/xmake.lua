
set_languages("cxx11")

perf_list = {
        "001",

}


rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
    end)

rule_end()


for _, v in pairs(perf_list) do
    local perf_target = "perf_" .. v
    target(perf_target)
        set_kind("binary")
        add_rules("rule_display")
        add_cxflags("-pg")
        -- add_cxflags("-pg")
        for _, filedir in ipairs(os.filedirs(string.format("%s/**", v))) do
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