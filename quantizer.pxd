# declaring C++ template functions in quantize1d.h
cdef extern from "quantize1d.h":
    size_t kmeans_cluster[asm_type, value_type](value_type *data, asm_type *cluster_assignments, size_t num_data_points,
            value_type *cluster_centers, size_t num_clusters, size_t max_iterations)

    size_t kmeans_cluster_lazy[asm_type, value_type](value_type *data, asm_type *cluster_assignments, size_t num_data_points,
            value_type *cluster_centers, size_t num_clusters, size_t max_iterations)

