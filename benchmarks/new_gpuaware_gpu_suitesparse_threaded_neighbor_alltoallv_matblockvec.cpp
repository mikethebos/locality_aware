// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.

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

void ASSERT_EQ(double a, double b)
{
    if (a != b)
    {
        fprintf(stderr, "assert error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void test_matrix(const char *filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    int block_vec_cols_pow = 0;
    while (true)
    {
        int block_vec_cols = (int)pow(2, block_vec_cols_pow);
        if (block_vec_cols > (int)pow(2, 10))
        {
            break;
        }

        std::vector<double> std_recv_vals, neigh_recv_vals;
        std::vector<double> send_vals, alltoallv_send_vals;
        std::vector<long> send_indices;

        if (A.on_proc.n_cols)
        {
            send_vals.resize(A.on_proc.n_cols * block_vec_cols);
            std::iota(send_vals.begin(), send_vals.end(), 0);
            for (int i = 0; i < A.on_proc.n_cols * block_vec_cols; i++)
                send_vals[i] += (rank * 1000);
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
            int end = A.send_comm.ptr[i + 1];
            newSendDispls[i] = start * block_vec_cols;
            newSendCounts[i] = (int)(end - start) * block_vec_cols;
        }
        for (int i = 0; i < A.recv_comm.counts.size(); i++)
        {
            int start = A.recv_comm.ptr[i];
            int end = A.recv_comm.ptr[i + 1];
            newRecvDispls[i] = start * block_vec_cols;
            newRecvCounts[i] = (int)(end - start) * block_vec_cols;
        }

        double *std_recv_vals_cu, *neigh_recv_vals_cu;
        gpuMalloc((void **)&std_recv_vals_cu, std_recv_vals.size() * sizeof(double));
        gpuMalloc((void **)&neigh_recv_vals_cu, neigh_recv_vals.size() * sizeof(double));

        double *alltoallv_send_vals_cu;
        gpuMalloc((void **)&alltoallv_send_vals_cu, alltoallv_send_vals.size() * sizeof(double));
        gpuMemcpy((void *)alltoallv_send_vals_cu, (void *)(alltoallv_send_vals.data()), alltoallv_send_vals.size() * sizeof(double), gpuMemcpyHostToDevice);

        communicateBlockVec(A, send_vals, std_recv_vals, MPI_DOUBLE, block_vec_cols);

        MPI_Comm std_comm;
        MPI_Status status;
        MPIX_Comm *neighbor_comm;

        int *s = A.recv_comm.procs.data();
        if (A.recv_comm.n_msgs == 0)
            s = MPI_WEIGHTS_EMPTY;
        int *d = A.send_comm.procs.data();
        if (A.send_comm.n_msgs == 0)
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

        if (rank == 0)
            printf("Testing Size blockVecCols %d, matrix %s\n", block_vec_cols, filename);

        // Standard MPI Implementation of Alltoallv (gpu)
        MPI_Neighbor_alltoallv(alltoallv_send_vals_cu,
                               newSendCounts.data(),
                               newSendDispls.data(),
                               MPI_DOUBLE,
                               neigh_recv_vals_cu,
                               newRecvCounts.data(),
                               newRecvDispls.data(),
                               MPI_DOUBLE,
                               std_comm);
        gpuMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(double), gpuMemcpyDeviceToHost);
        for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
        {
            ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
        }
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int k = 0; k < 2; k++)
        {
            MPI_Neighbor_alltoallv(alltoallv_send_vals_cu,
                                   newSendCounts.data(),
                                   newSendDispls.data(),
                                   MPI_DOUBLE,
                                   neigh_recv_vals_cu,
                                   newRecvCounts.data(),
                                   newRecvDispls.data(),
                                   MPI_DOUBLE,
                                   std_comm);
        }
        double tfinal = (MPI_Wtime() - t0) / 2;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        int niter = (2.0 / t0) + 1;
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < niter; k++)
        {
            MPI_Neighbor_alltoallv(alltoallv_send_vals_cu,
                                   newSendCounts.data(),
                                   newSendDispls.data(),
                                   MPI_DOUBLE,
                                   neigh_recv_vals_cu,
                                   newRecvCounts.data(),
                                   newRecvDispls.data(),
                                   MPI_DOUBLE,
                                   std_comm);
        }
        tfinal = (MPI_Wtime() - t0) / niter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("GPU MPI_Neighbor_alltoallv Time %e; Time All Iters %e\n", t0, t0 * niter);
        gpuMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(double));
        memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(double));
        gpuDeviceSynchronize();

        gpu_aware_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                      newSendCounts.data(),
                                                      newSendDispls.data(),
                                                      MPI_DOUBLE,
                                                      neigh_recv_vals_cu,
                                                      newRecvCounts.data(),
                                                      newRecvDispls.data(),
                                                      MPI_DOUBLE,
                                                      neighbor_comm);
        gpuMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(double), gpuMemcpyDeviceToHost);
        for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
        {
            ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
        }
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < 2; k++)
        {
            gpu_aware_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                        newSendCounts.data(),
                                                        newSendDispls.data(),
                                                        MPI_DOUBLE,
                                                        neigh_recv_vals_cu,
                                                        newRecvCounts.data(),
                                                        newRecvDispls.data(),
                                                        MPI_DOUBLE,
                                                        neighbor_comm);
        }
        tfinal = (MPI_Wtime() - t0) / 2;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        niter = (2.0 / t0) + 1;
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < niter; k++)
        {
            gpu_aware_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                        newSendCounts.data(),
                                                        newSendDispls.data(),
                                                        MPI_DOUBLE,
                                                        neigh_recv_vals_cu,
                                                        newRecvCounts.data(),
                                                        newRecvDispls.data(),
                                                        MPI_DOUBLE,
                                                        neighbor_comm);
        }
        tfinal = (MPI_Wtime() - t0) / niter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("gpu_aware_neighbor_alltoallv_nonblocking_pure Time %e; Time All Iters %e\n", t0, t0 * niter);
        gpuMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(double));
        memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(double));
        gpuDeviceSynchronize();

        gpu_aware_threaded_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                     newSendCounts.data(),
                                                     newSendDispls.data(),
                                                     MPI_DOUBLE,
                                                     neigh_recv_vals_cu,
                                                     newRecvCounts.data(),
                                                     newRecvDispls.data(),
                                                     MPI_DOUBLE,
                                                     neighbor_comm);
        gpuMemcpy((void *)neigh_recv_vals.data(), (void *)neigh_recv_vals_cu, std_recv_vals.size() * sizeof(double), gpuMemcpyDeviceToHost);
        for (int i = 0; i < A.recv_comm.size_msgs * block_vec_cols; i++)
        {
            ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
        }
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < 2; k++)
        {
            gpu_aware_threaded_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                     newSendCounts.data(),
                                                     newSendDispls.data(),
                                                     MPI_DOUBLE,
                                                     neigh_recv_vals_cu,
                                                     newRecvCounts.data(),
                                                     newRecvDispls.data(),
                                                     MPI_DOUBLE,
                                                     neighbor_comm);
        }
        tfinal = (MPI_Wtime() - t0) / 2;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        niter = (2.0 / t0) + 1;
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < niter; k++)
        {
            gpu_aware_threaded_neighbor_alltoallv_nonblocking_pure(alltoallv_send_vals_cu,
                                                     newSendCounts.data(),
                                                     newSendDispls.data(),
                                                     MPI_DOUBLE,
                                                     neigh_recv_vals_cu,
                                                     newRecvCounts.data(),
                                                     newRecvDispls.data(),
                                                     MPI_DOUBLE,
                                                     neighbor_comm);
        }
        tfinal = (MPI_Wtime() - t0) / niter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("gpu_aware_threaded_neighbor_alltoallv_nonblocking_pure Time %e; Time All Iters %e\n", t0, t0 * niter);
        gpuMemset((void *)neigh_recv_vals_cu, 0, neigh_recv_vals.size() * sizeof(double));
        memset((void *)neigh_recv_vals.data(), 0, neigh_recv_vals.size() * sizeof(double));

        MPIX_Comm_free(&neighbor_comm);
        MPI_Comm_free(&std_comm);

        gpuFree(std_recv_vals_cu);
        gpuFree(neigh_recv_vals_cu);
        gpuFree(alltoallv_send_vals_cu);

        block_vec_cols_pow++;
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, nodeRank, nodeSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    
    int rev = 0;
    char **fns = (char **)calloc(argc, sizeof(char *));
    int fn_count = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "r") == 0)
        {
            rev = 1;
        }
        else
        {
            fns[fn_count] = argv[i];
            fn_count++;
        }
    }
    
    if (rev == 0)
    {
        gpuSetDevice(nodeRank);
    }
    else
    {
        gpuSetDevice((nodeSize - nodeRank) - 1);
        printf("rank %d reversed\n", rank);
        fflush(stdout);
    }

    if (fn_count == 0)
    {
        test_matrix("../../test_data/cnr-2000.pm");
        test_matrix("../../test_data/3dtube.pm");
        test_matrix("../../test_data/Goodwin_095.pm");
    }
    else
    {
        for (int i = 0; i < fn_count; ++i)
        {
            test_matrix(fns[i]);
        }
    }

    free((void *)fns);
    MPI_Finalize();
    return 0;
} // end of main() //