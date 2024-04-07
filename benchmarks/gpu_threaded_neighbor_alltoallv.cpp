#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;

    for (int i = 0; i < max_i; i++)
    {
        int seed = 123 + i;
        srand(seed);
        int s = pow(2, i);
        int *sendc = (int *)malloc(num_procs * sizeof(int));
        int *dests = (int *)malloc(num_procs * sizeof(int));
        int *destsw = (int *)malloc(num_procs * sizeof(int));
        int *recvc = (int *)malloc(num_procs * sizeof(int));
        int *srcs = (int *)malloc(num_procs * sizeof(int));
        int *srcsw = (int *)malloc(num_procs * sizeof(int));
        int *sendd = (int *)malloc(num_procs * sizeof(int));
        int *recvd = (int *)malloc(num_procs * sizeof(int));
        int *plan = (int *)malloc(num_procs * num_procs * sizeof(int));
        for (int j = 0; j < num_procs * num_procs; j++)
        {
                plan[j] = rand() % s;
        }
        srand(seed + rank);
        int initsdispl = rand() % 32;
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
                    initsdispl += count + (rand() % 32);
                    scount++;
                }
        }
        
        srand(seed + (num_procs * rank));
        int initrdispl = rand() % 32;
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
                    initrdispl += count + (rand() % 32);
                    rcount++;
                }
        }
        
        free(plan);
        if (rank == 0) printf("Testing Size %d\n", s);

        srand(time(NULL));
        std::vector<double> send_data(initsdispl);
        std::vector<double> mpi_alltoall(initrdispl);
        std::vector<double> mpix_alltoall(initrdispl);
        for (int j = 0; j < initsdispl; j++)
                send_data[j] = rand() + (double)rank;

        double* send_data_d;
        double* recv_data_d;
        cudaMalloc((void**)(&send_data_d), initsdispl*sizeof(double));
        cudaMalloc((void**)(&recv_data_d), initrdispl*sizeof(double));
        cudaMemcpy(send_data_d, send_data.data(), initsdispl*sizeof(double), cudaMemcpyHostToDevice);
        
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
        cudaMemcpy(mpi_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));

        // MPI Advance : copy to cpu
        MPIX_Request *copyreq;
        threaded_neighbor_alltoallv_nonblocking_init(send_data_d,
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
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, CC-TH-NB %e\n", 
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
        if (rank == 0) printf("MPI_Neighbor_alltoallv Time %e\n", t0);

        // Time copy-to-cpu
        threaded_neighbor_alltoallv_nonblocking_init(send_data_d,
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
        if (rank == 0) printf("Threaded CC Neighbor_alltoallv Time %e\n", t0);

        MPIX_Request_free(copyreq);        
        MPI_Comm_free(&mpi_graph);
        MPIX_Comm_free(mpix_graph);
        cudaFree(send_data_d);
        cudaFree(recv_data_d);
        free(sendc);
        free(dests);
        free(destsw);
        free(recvc);
        free(srcs);
        free(srcsw);
        free(sendd);
        free(recvd);
    }

    MPI_Finalize();
    return 0;
}
