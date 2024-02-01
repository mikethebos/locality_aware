set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(heterogeneous_HEADERS
    heterogeneous/gpu_alltoall.h
    heterogeneous/gpu_alltoallv.h
    heterogeneous/gpu_neighbor_alltoallv.h
    heterogeneous/persistent_gpu.h
    PARENT_SCOPE
    )

set(heterogeneous_SOURCES
    heterogeneous/gpu_alltoall.c
    heterogeneous/gpu_alltoallv.c
    heterogeneous/gpu_neighbor_alltoallv.h
    heterogeneous/persistent_gpu.h
    PARENT_SCOPE
    )

