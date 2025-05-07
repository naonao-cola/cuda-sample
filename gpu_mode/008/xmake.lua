


target("gpu-mode-008-01-coalesce")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("coalesce.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-008-02-coarsening")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("coarsening.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-008-03-divergence")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("divergence.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})



target("gpu-mode-008-04-occupancy")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("occupancy.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})


target("gpu-mode-008-05-privatization")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("privatization.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})



target("gpu-mode-008-06-privatization2")
    set_kind("binary")
    add_rules("cuda")
    add_cugencodes("native")
    add_files("privatization2.cu")
    set_basename("benchmark")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})
