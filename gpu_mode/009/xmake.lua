


target("gpu-mode-009-01-control_divergence_reduce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("control_divergence_reduce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})



target("gpu-mode-009-02-multistream-reduce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("multistream-reduce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-009-03-reduce_coarsening")
    set_kind("binary")
    set_toolchains("cuda")
    -- set_toolset("cxx", "nvcc")
    -- set_toolset("ld", "nvcc")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("reduce_coarsening.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})
    -- add_cuflags(
    --     "-c",
    --     "-arch=sm_86",
    --     "-lineinfo",
    --     "--source-in-ptx",
    --     "-Xptxas",
    --     "-v",
    --     --"-ptx",
    --     {force = true})
    -- add_cugencodes("compute_86")


target("gpu-mode-009-04-segment_reduce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("segment_reduce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-009-05-shared_reduce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("shared_reduce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-009-06-simple_reduce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("simple_reduce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})