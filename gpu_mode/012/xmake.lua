

target("12")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("**.cu|torch_extension_template.cu")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})