if(USE_CUDA)
    set_source_files_properties(
        test_gpu_alltoall.cpp
        test_gpu_alltoallv.cpp
        test_suitesparse_neighbor_alltoallv_init_matblockvec_gpu.cpp
        test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_gpu_alltoall test_gpu_alltoall.cpp)
target_link_libraries(test_gpu_alltoall mpi_advance gtest pthread)
add_test(GPUAlltoallTest mpirun -n 16 ./test_gpu_alltoall)

add_executable(test_gpu_alltoallv test_gpu_alltoallv.cpp)
target_link_libraries(test_gpu_alltoallv mpi_advance gtest pthread)
add_test(GPUAlltoallvTest mpirun -n 16 ./test_gpu_alltoallv)

add_executable(test_suitesparse_neighbor_alltoallv_init_matblockvec_gpu test_suitesparse_neighbor_alltoallv_init_matblockvec_gpu.cpp)
target_link_libraries(test_suitesparse_neighbor_alltoallv_init_matblockvec_gpu mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX gtest pthread)
add_test(SuitesparseNeighborAlltoallvInitMatblockvecGPUTest mpirun -n 16 ./test_suitesparse_neighbor_alltoallv_init_matblockvec_gpu)

add_executable(test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu.cpp)
target_compile_options(test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
-lineinfo
-Xcompiler -fopenmp
>
)
target_link_libraries(test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX gtest pthread)
add_test(SuitesparseThreadedNeighborAlltoallvInitMatblockvecGPUTest mpirun -n 16 ./test_suitesparse_threaded_neighbor_alltoallv_init_matblockvec_gpu)