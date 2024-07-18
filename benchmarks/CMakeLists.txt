include_directories(${mpi_advance_INCDIR})

if(USE_CUDA)
    set_source_files_properties(
        gpu_alltoall.cpp
        gpu_threaded_alltoall.cpp
        gpu_extraproc_alltoall.cpp
        no_thread_launches.cpp
        thread_launches.cpp
        PROPERTIES LANGUAGE CUDA)

    add_executable(gpu_alltoall gpu_alltoall.cpp)
    target_link_libraries(gpu_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(gpu_threaded_alltoall gpu_threaded_alltoall.cpp)
    target_compile_options(gpu_threaded_alltoall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -fopenmp
    >
    )
    target_link_libraries(gpu_threaded_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(gpu_extraproc_alltoall gpu_extraproc_alltoall.cpp)
    target_link_libraries(gpu_extraproc_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(no_thread_launches no_thread_launches.cpp)
    target_compile_options(no_thread_launches PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -fopenmp
    >
    )
    target_link_libraries(no_thread_launches mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(thread_launches thread_launches.cpp)
    target_compile_options(thread_launches PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -fopenmp
    >
    )
    target_link_libraries(thread_launches mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    add_executable(bruck_allgather bruck_allgather.cpp)
    target_link_libraries(bruck_allgather mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoall p2p_alltoall.cpp)
    target_link_libraries(p2p_alltoall mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoallv p2p_alltoallv.cpp)
    target_link_libraries(p2p_alltoallv mpi_advance ${MPI_LIBRARIES})
endif()
