// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);
    
    int block_vec_cols = A.global_cols;

    std::vector<int> std_recv_vals, neigh_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;
    std::vector<long> send_indices;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols * block_vec_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols * block_vec_cols; i++)
            send_vals[i] += (rank*1000);
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs * block_vec_cols);
        neigh_recv_vals.resize(A.recv_comm.size_msgs * block_vec_cols);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs * block_vec_cols);
        send_indices.resize(A.send_comm.size_msgs * block_vec_cols);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx = A.send_comm.idx[i];
            for (int j = 0; j < block_vec_cols; j++)
            {
                int send_idx = (i * block_vec_cols) + j;
                alltoallv_send_vals[send_idx] = send_vals[(idx * block_vec_cols) + j];
                send_indices[send_idx] = A.send_comm.idx[i] + A.first_col;
            }
        }
    }
    
    std::vector<int> newSendCounts, newSendDispls, newRecvCounts, newRecvDispls;
    newSendCounts.resize(A.send_comm.counts.size());
    newSendDispls.resize(A.send_comm.counts.size());
    newRecvCounts.resize(A.recv_comm.counts.size());
    newRecvDispls.resize(A.recv_comm.counts.size());
    for (int i = 0; i < A.send_comm.counts.size(); i++)
    {
        int start = A.send_comm.ptr[i];
        int end = A.send_comm.ptr[i+1];
        newSendDispls[i] = start * block_vec_cols;
        newSendCounts[i] = (int)(end - start) * block_vec_cols;
    }
    for (int i = 0; i < A.recv_comm.counts.size(); i++)
    {
        int start = A.recv_comm.ptr[i];
        int end = A.recv_comm.ptr[i+1];
        newRecvDispls[i] = start * block_vec_cols;
        newRecvCounts[i] = (int)(end - start) * block_vec_cols;
    }
    
    int *std_recv_vals_cu, *neigh_recv_vals_cu;
    cudaMalloc((void **)&std_recv_vals_cu, std_recv_vals.size() * sizeof(int));
    cudaMalloc((void **)&neigh_recv_vals_cu, neigh_recv_vals.size() * sizeof(int));
    
    int *alltoallv_send_vals_cu;    
    cudaMalloc((void **)&alltoallv_send_vals_cu, alltoallv_send_vals.size() * sizeof(int));
    cudaMemcpy((void *)alltoallv_send_vals_cu, (void *)(alltoallv_send_vals.data()), alltoallv_send_vals.size() * sizeof(int), cudaMemcpyHostToDevice);

    communicateBlockVec(A, send_vals, std_recv_vals, MPI_INT, block_vec_cols);

    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* neighbor_comm;

    int* s = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
        s = MPI_WEIGHTS_EMPTY;
    int* d = A.send_comm.procs.data();
    if (A.send_comm.n_msgs  == 0)
        d = MPI_WEIGHTS_EMPTY;

    // Standard MPI Dist Graph Create
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            s,
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            d,
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            0,
            &std_comm);
            
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0, 
            &neighbor_comm);
            
    // Standard MPI Implementation of Alltoallv (CUDA)
    MPI_Neighbor_alltoallv(alltoallv_send_vals_cu,
            newSendCounts.data(),
            newSendDispls.data(),
            MPI_INT,
            neigh_recv_vals_cu,
            newRecvCounts.data(),
            newRecvDispls.data(),
            MPI_INT,
            std_comm);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    
    MPIX_Neighbor_alltoallv(alltoallv_send_vals_cu, 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals_cu, 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    
    MPIX_Request *mpixreq;
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals_cu, 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals_cu, 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm,
            MPI_INFO_NULL,
            &mpixreq);
    MPIX_Start(mpixreq);
    MPIX_Wait(mpixreq, MPI_STATUS_IGNORE);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    MPIX_Request_free(mpixreq);
    
    cudaMemcpy((void *)alltoallv_send_vals.data(), (void *)alltoallv_send_vals_cu, alltoallv_send_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    MPIX_Neighbor_alltoallv(alltoallv_send_vals.data(), 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals.data(), 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm);
    cudaMemcpy((void *)neigh_recv_vals_cu, (void *)neigh_recv_vals.data(), neigh_recv_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));

    MPIX_Request *mpixccreq;
    cudaMemcpy((void *)alltoallv_send_vals.data(), (void *)alltoallv_send_vals_cu, alltoallv_send_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(), 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals.data(), 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm,
            MPI_INFO_NULL,
            &mpixccreq);
    MPIX_Start(mpixccreq);
    MPIX_Wait(mpixccreq, MPI_STATUS_IGNORE);
    cudaMemcpy((void *)neigh_recv_vals_cu, (void *)neigh_recv_vals.data(), neigh_recv_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    MPIX_Request_free(mpixccreq);
    
    MPIX_Request *gpureq;
    gpu_aware_neighbor_alltoallv_nonblocking_init(alltoallv_send_vals_cu, 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals_cu, 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm,
            MPI_INFO_NULL,
            &gpureq);
    MPIX_Start(gpureq);
    MPIX_Wait(gpureq, MPI_STATUS_IGNORE);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    MPIX_Request_free(gpureq);
    
    MPIX_Request *copyreq;
    copy_to_cpu_neighbor_alltoallv_nonblocking_init(alltoallv_send_vals_cu, 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals_cu, 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm,
            MPI_INFO_NULL,
            &copyreq);
    MPIX_Start(copyreq);
    MPIX_Wait(copyreq, MPI_STATUS_IGNORE);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    MPIX_Request_free(copyreq);

    MPIX_Request *threadedreq;
    threaded_neighbor_alltoallv_nonblocking_init(alltoallv_send_vals_cu, 
            newSendCounts.data(),
            newSendDispls.data(), 
            MPI_INT,
            neigh_recv_vals_cu, 
            newRecvCounts.data(),
            newRecvDispls.data(), 
            MPI_INT,
            neighbor_comm,
            MPI_INFO_NULL,
            &threadedreq);
    MPIX_Start(threadedreq);
    MPIX_Wait(threadedreq, MPI_STATUS_IGNORE);
    cudaMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    cudaMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(int));
    memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(int));
    MPIX_Request_free(threadedreq);

    MPIX_Comm_free(&neighbor_comm);
    MPI_Comm_free(&std_comm);
    
    cudaFree(std_recv_vals_cu);
    cudaFree(neigh_recv_vals_cu);
    cudaFree(alltoallv_send_vals_cu);
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(RandomCommTest, TestsInTests)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    test_matrix("../../../../test_data/odepa400.pm");
/*
    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");
    test_matrix("../../../../test_data/bcsstk01.pm");
    test_matrix("../../../../test_data/west0132.pm");
    test_matrix("../../../../test_data/gams10a.pm");
    test_matrix("../../../../test_data/gams10am.pm");
    test_matrix("../../../../test_data/D_10.pm");
    test_matrix("../../../../test_data/oscil_dcop_11.pm");
    test_matrix("../../../../test_data/tumorAntiAngiogenesis_4.pm");
    test_matrix("../../../../test_data/ch5-5-b1.pm");
    test_matrix("../../../../test_data/msc01050.pm");
    test_matrix("../../../../test_data/SmaGri.pm");
    test_matrix("../../../../test_data/radfr1.pm");
    test_matrix("../../../../test_data/bibd_49_3.pm");
    test_matrix("../../../../test_data/can_1054.pm");
    test_matrix("../../../../test_data/can_1072.pm");
    test_matrix("../../../../test_data/lp_sctap2.pm");
    test_matrix("../../../../test_data/lp_woodw.pm");
*/
}

