


target("vector_addition")
    add_cugencodes("native")
    add_files("vector_addition.cu")
    add_cuflags(
        "--ptxas-options=-v",-- kernel register and memory usage
        {force = true})
