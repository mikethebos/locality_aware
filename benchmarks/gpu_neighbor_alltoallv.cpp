#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);

        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int max_i = 20;
        int max_s = pow(2, max_i);
        int max_displ = 32;
        int max_extra_size = 128;
        int n_iter = 100;
        double t0, tfinal;

        int *sendc = (int *)malloc(num_procs * sizeof(int));
        int *dests = (int *)malloc(num_procs * sizeof(int));
        int *destsw = (int *)malloc(num_procs * sizeof(int));
        int *recvc = (int *)malloc(num_procs * sizeof(int));
        int *srcs = (int *)malloc(num_procs * sizeof(int));
        int *srcsw = (int *)malloc(num_procs * sizeof(int));
        int *sendd = (int *)malloc(num_procs * sizeof(int));
        int *recvd = (int *)malloc(num_procs * sizeof(int));
        int *plan = (int *)malloc(num_procs * num_procs * sizeof(int));

        int max_buf_size = (pow(2, max_i - 1) + max_extra_size + max_displ) * num_procs;
        std::vector<double> send_data(max_buf_size);
        std::vector<double> mpi_alltoall(max_buf_size);
        std::vector<double> mpix_alltoall(max_buf_size);

        srand(time(NULL));
        for (int j = 0; j < max_buf_size; j++)
                send_data[j] = rand() + (double)rank;

        double *send_data_d;
        double *recv_data_d;
        cudaMalloc((void **)(&send_data_d), max_buf_size * sizeof(double));
        cudaMalloc((void **)(&recv_data_d), max_buf_size * sizeof(double));
        cudaMemcpy(send_data_d, send_data.data(), max_buf_size * sizeof(double), cudaMemcpyHostToDevice);

        for (int i = 0; i < max_i; i++)
        {
                int seed = 123 + i;
                srand(seed);
                int s = pow(2, i);
                for (int j = 0; j < num_procs * num_procs; j++)
                {
                        plan[j] = s + (rand() % max_extra_size);
                }
                srand(seed + rank);
                int initsdispl = rand() % max_displ;
                int scount = 0;
                for (int dest = 0; dest < num_procs; dest++)
                {
                        int count = plan[rank * num_procs + dest];
                        if (count > 0)
                        {
                                dests[scount] = dest;
                                destsw[scount] = count;
                                sendc[scount] = count;
                                sendd[scount] = initsdispl;
                                initsdispl += count + (rand() % max_displ);
                                scount++;
                        }
                }

                srand(seed + (num_procs * rank));
                int initrdispl = rand() % max_displ;
                int rcount = 0;
                for (int src = 0; src < num_procs; src++)
                {
                        int count = plan[src * num_procs + rank];
                        if (count > 0)
                        {
                                srcs[rcount] = src;
                                srcsw[rcount] = count;
                                recvc[rcount] = count;
                                recvd[rcount] = initrdispl;
                                initrdispl += count + (rand() % max_displ);
                                rcount++;
                        }
                }

                if (rank == 0)
                        printf("Testing (around) Size per proc %d\n", s);

                MPI_Comm mpi_graph;
                MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                               rcount,
                                               srcs,
                                               srcsw,
                                               scount,
                                               dests,
                                               destsw,
                                               MPI_INFO_NULL,
                                               0,
                                               &mpi_graph);

                MPIX_Comm *mpix_graph;
                MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                                rcount,
                                                srcs,
                                                srcsw,
                                                scount,
                                                dests,
                                                destsw,
                                                MPI_INFO_NULL,
                                                0,
                                                &mpix_graph);

                // Standard MPI Implementation
                MPI_Neighbor_alltoallv(send_data_d,
                                       sendc,
                                       sendd,
                                       MPI_DOUBLE,
                                       recv_data_d,
                                       recvc,
                                       recvd,
                                       MPI_DOUBLE,
                                       mpi_graph);
                cudaMemcpy(mpi_alltoall.data(), recv_data_d, initrdispl * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemset(recv_data_d, 0, initrdispl * sizeof(double));

                // MPI Advance : GPU-Aware
                MPIX_Request *gpureq;
                gpu_aware_neighbor_alltoallv_nonblocking_init(send_data_d,
                                                              sendc,
                                                              sendd,
                                                              MPI_DOUBLE,
                                                              recv_data_d,
                                                              recvc,
                                                              recvd,
                                                              MPI_DOUBLE,
                                                              mpix_graph,
                                                              MPI_INFO_NULL,
                                                              &gpureq);
                MPIX_Start(gpureq);
                MPIX_Wait(gpureq, MPI_STATUS_IGNORE);
                cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemset(recv_data_d, 0, initrdispl * sizeof(double));
                for (int j = 0; j < initrdispl; j++)
                {
                        if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
                        {
                                fprintf(stderr,
                                        "Rank %d, idx %d, pmpi %e, GA-NB %e\n",
                                        rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                                return 1;
                        }
                }

                // MPI Advance : copy to cpu
                MPIX_Request *copyreq;
                copy_to_cpu_neighbor_alltoallv_nonblocking_init(send_data_d,
                                                                sendc,
                                                                sendd,
                                                                MPI_DOUBLE,
                                                                recv_data_d,
                                                                recvc,
                                                                recvd,
                                                                MPI_DOUBLE,
                                                                mpix_graph,
                                                                MPI_INFO_NULL,
                                                                &copyreq);
                MPIX_Start(copyreq);
                MPIX_Wait(copyreq, MPI_STATUS_IGNORE);
                cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemset(recv_data_d, 0, initrdispl * sizeof(double));
                for (int j = 0; j < initrdispl; j++)
                {
                        if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
                        {
                                fprintf(stderr,
                                        "Rank %d, idx %d, pmpi %e, CC-NB %e\n",
                                        rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                                return 1;
                        }
                }

                // Time PMPI Alltoall
                MPI_Neighbor_alltoallv(send_data_d,
                                       sendc,
                                       sendd,
                                       MPI_DOUBLE,
                                       recv_data_d,
                                       recvc,
                                       recvd,
                                       MPI_DOUBLE,
                                       mpi_graph);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                for (int k = 0; k < n_iter; k++)
                {
                        MPI_Neighbor_alltoallv(send_data_d,
                                               sendc,
                                               sendd,
                                               MPI_DOUBLE,
                                               recv_data_d,
                                               recvc,
                                               recvd,
                                               MPI_DOUBLE,
                                               mpi_graph);
                }
                tfinal = (MPI_Wtime() - t0) / n_iter;
                MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0)
                        printf("MPI_Neighbor_alltoallv Time %e\n", t0);

                // Time GPU-Aware Pairwise Exchange
                gpu_aware_neighbor_alltoallv_nonblocking_init(send_data_d,
                                                              sendc,
                                                              sendd,
                                                              MPI_DOUBLE,
                                                              recv_data_d,
                                                              recvc,
                                                              recvd,
                                                              MPI_DOUBLE,
                                                              mpix_graph,
                                                              MPI_INFO_NULL,
                                                              &gpureq);
                MPIX_Start(gpureq);
                MPIX_Wait(gpureq, MPI_STATUS_IGNORE);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                for (int k = 0; k < n_iter; k++)
                {
                        // ignore init
                        MPIX_Start(gpureq);
                        MPIX_Wait(gpureq, MPI_STATUS_IGNORE);
                }
                tfinal = (MPI_Wtime() - t0) / n_iter;
                MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0)
                        printf("GPU-Aware Neighbor_alltoallv Time %e\n", t0);

                // Time copy-to-cpu
                copy_to_cpu_neighbor_alltoallv_nonblocking_init(send_data_d,
                                                                sendc,
                                                                sendd,
                                                                MPI_DOUBLE,
                                                                recv_data_d,
                                                                recvc,
                                                                recvd,
                                                                MPI_DOUBLE,
                                                                mpix_graph,
                                                                MPI_INFO_NULL,
                                                                &copyreq);
                MPIX_Start(copyreq);
                MPIX_Wait(copyreq, MPI_STATUS_IGNORE);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                for (int k = 0; k < n_iter; k++)
                {
                        MPIX_Start(copyreq);
                        MPIX_Wait(copyreq, MPI_STATUS_IGNORE);
                }
                tfinal = (MPI_Wtime() - t0) / n_iter;
                MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0)
                        printf("CC Neighbor_alltoallv Time %e\n", t0);

                MPIX_Request_free(gpureq);
                MPIX_Request_free(copyreq);
                MPI_Comm_free(&mpi_graph);
                MPIX_Comm_free(mpix_graph);
        }

        free(plan);
        free(sendc);
        free(dests);
        free(destsw);
        free(recvc);
        free(srcs);
        free(srcsw);
        free(sendd);
        free(recvd);
        cudaFree(send_data_d);
        cudaFree(recv_data_d);

        MPI_Finalize();
        return 0;
}
