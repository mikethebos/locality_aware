#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

// copy recvbuf in case of extra data
// needed if noncontiguous displs (or custom packing)
int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 22;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;

    for (int i = 0; i < max_i; i++)
    {
        int seed = 123 + i;
        srand(seed);
        int s = pow(2, i);

        if (s <= 65536)
        {
                n_iter *= 10;
        }

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
                plan[j] = s;
        }
        srand(seed + rank);
        int initsdispl = 0;
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
                    initsdispl += count;
                    scount++;
                }
        }
        
        srand(seed + (num_procs * rank));
        int initrdispl = 0;
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
                    initrdispl += count;
                    rcount++;
                }
        }
        
        free(plan);        
        if (rank == 0) printf("Testing (around) Size per proc %d\n", s);

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
        
        // MPI CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Neighbor_alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpi_graph);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPI-NB-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance
        MPIX_Neighbor_alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-NB %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance - init
        MPIX_Request *mpixreq;
        MPIX_Neighbor_alltoallv_init(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph,
                MPI_INFO_NULL,
                &mpixreq);
        MPIX_Start(mpixreq);
        MPIX_Wait(mpixreq, MPI_STATUS_IGNORE);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-NB-INIT %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Neighbor_alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-NB-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance - init CC
        MPIX_Request *mpixccreq;
        MPIX_Neighbor_alltoallv_init(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph,
                MPI_INFO_NULL,
                &mpixccreq);
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Start(mpixccreq);
        MPIX_Wait(mpixccreq, MPI_STATUS_IGNORE);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-NB-INIT-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : copy to cpu threaded
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

        // Time MPI Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Neighbor_alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpi_graph);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPI_Neighbor_alltoallv(send_data.data(),
                        sendc,
                        sendd,
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        mpi_graph);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPI_Neighbor_alltoallv CC Time %e\n", t0);

        // Time MPIX Alltoall
        MPIX_Neighbor_alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE,
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                MPIX_Neighbor_alltoallv(send_data_d,
                        sendc,
                        sendd,
                        MPI_DOUBLE,
                        recv_data_d,
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        mpix_graph);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Neighbor_alltoallv Time %e\n", t0);

        // Time MPIX alltoall init
        MPIX_Neighbor_alltoallv_init(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph,
                MPI_INFO_NULL,
                &mpixreq);
        MPIX_Start(mpixreq);
        MPIX_Wait(mpixreq, MPI_STATUS_IGNORE);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                // ignore init
                MPIX_Start(mpixreq);
                MPIX_Wait(mpixreq, MPI_STATUS_IGNORE);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Neighbor_alltoallv_init Time %e\n", t0);
        
        // Time MPIX Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Neighbor_alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPIX_Neighbor_alltoallv(send_data.data(),
                        sendc,
                        sendd,
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        mpix_graph);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Neighbor_alltoallv CC Time %e\n", t0);

        // Time MPIX alltoall init CC
        MPIX_Neighbor_alltoallv_init(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                mpix_graph,
                MPI_INFO_NULL,
                &mpixccreq);
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Start(mpixccreq);
        MPIX_Wait(mpixccreq, MPI_STATUS_IGNORE);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPIX_Start(mpixccreq);
                MPIX_Wait(mpixccreq, MPI_STATUS_IGNORE);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Neighbor_alltoallv_init CC Time %e\n", t0);

        // Time copy-to-cpu threaded
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
        
        MPIX_Comm *locality_comm;
        MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
        
        // alltoall
        // Standard MPI Implementation
        MPI_Alltoall(send_data_d,
                sendc[0],
                MPI_DOUBLE,
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPI-all2all %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Alltoall(send_data.data(),
                sendc[0],
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc[0],
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPI-all2all-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance
        MPIX_Alltoall(send_data_d,
                sendc[0],
                MPI_DOUBLE, 
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-all2all %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Alltoall(send_data.data(),
                sendc[0],
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-all2all-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance : copy to cpu threaded
        threaded_alltoall_nonblocking(send_data_d,
                sendc[0],
                MPI_DOUBLE, 
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, CC-TH-all2all %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // Time PMPI Alltoall
        MPI_Alltoall(send_data_d,
                sendc[0],
                MPI_DOUBLE,
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                MPI_Alltoall(send_data_d,
                        sendc[0],
                        MPI_DOUBLE,
                        recv_data_d,
                        recvc[0],
                        MPI_DOUBLE,
                        MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPI_Alltoall Time %e\n", t0);

        // Time MPI Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Alltoall(send_data.data(),
                sendc[0],
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc[0],
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPI_Alltoall(send_data.data(),
                        sendc[0],
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc[0],
                        MPI_DOUBLE,
                        MPI_COMM_WORLD);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPI_Alltoall CC Time %e\n", t0);

        // Time MPIX Alltoall
        MPIX_Alltoall(send_data_d,
                sendc[0],
                MPI_DOUBLE,
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                MPIX_Alltoall(send_data_d,
                        sendc[0],
                        MPI_DOUBLE,
                        recv_data_d,
                        recvc[0],
                        MPI_DOUBLE,
                        locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoall Time %e\n", t0);

        // Time MPIX Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Alltoall(send_data.data(),
                sendc[0],
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPIX_Alltoall(send_data.data(),
                        sendc[0],
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc[0],
                        MPI_DOUBLE,
                        locality_comm);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoall CC Time %e\n", t0);
        
        // Time copy-to-cpu threaded
        threaded_alltoall_nonblocking(send_data_d,
                sendc[0],
                MPI_DOUBLE, 
                recv_data_d,
                recvc[0],
                MPI_DOUBLE,
                locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                threaded_alltoall_nonblocking(send_data_d,
                        sendc[0],
                        MPI_DOUBLE, 
                        recv_data_d,
                        recvc[0],
                        MPI_DOUBLE,
                        locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded CC alltoall Time %e\n", t0);
        
        // alltoallv
        // Standard MPI Implementation
        MPI_Alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE,
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPI-all2allv %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPI-all2allv-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance
        MPIX_Alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-all2allv %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, MPIX-all2allv-CC %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        // MPI Advance : copy to cpu threaded
        threaded_alltoallv_nonblocking(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, initrdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        for (int j = 0; j < initrdispl; j++)
	{
            if (fabs(mpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, CC-TH-all2allv %e\n", 
                         rank, j, mpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // Time PMPI Alltoall
        MPI_Alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE,
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                MPI_Alltoallv(send_data_d,
                        sendc,
                        sendd,
                        MPI_DOUBLE,
                        recv_data_d,
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPI_Alltoallv Time %e\n", t0);

        // Time MPI Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPI_Alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPI_Alltoallv(send_data.data(),
                        sendc,
                        sendd,
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        MPI_COMM_WORLD);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPI_Alltoallv CC Time %e\n", t0);

        // Time MPIX Alltoall
        MPIX_Alltoallv(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE,
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                MPIX_Alltoallv(send_data_d,
                        sendc,
                        sendd,
                        MPI_DOUBLE,
                        recv_data_d,
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoallv Time %e\n", t0);

        // Time MPIX Alltoall CC
        cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                cudaMemcpyDeviceToHost);
        MPIX_Alltoallv(send_data.data(),
                sendc,
                sendd,
                MPI_DOUBLE, 
                mpix_alltoall.data(),
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemset(recv_data_d, 0, initrdispl*sizeof(double));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                cudaMemcpy(send_data.data(), send_data_d, initsdispl*sizeof(double),
                        cudaMemcpyDeviceToHost);
                MPIX_Alltoallv(send_data.data(),
                        sendc,
                        sendd,
                        MPI_DOUBLE, 
                        mpix_alltoall.data(),
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        locality_comm);
                cudaMemcpy(recv_data_d, mpix_alltoall.data(), initrdispl*sizeof(double),
                        cudaMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoallv CC Time %e\n", t0);

        // Time copy-to-cpu threaded
        threaded_alltoallv_nonblocking(send_data_d,
                sendc,
                sendd,
                MPI_DOUBLE, 
                recv_data_d,
                recvc,
                recvd,
                MPI_DOUBLE,
                locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
                threaded_alltoallv_nonblocking(send_data_d,
                        sendc,
                        sendd,
                        MPI_DOUBLE, 
                        recv_data_d,
                        recvc,
                        recvd,
                        MPI_DOUBLE,
                        locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded CC alltoallv Time %e\n", t0);

        if (s <= 65536) 
       	{
       	       	n_iter /= 10;
       	} 

        
        MPIX_Request_free(mpixreq);
        MPIX_Request_free(mpixccreq);
        MPIX_Request_free(copyreq);
        MPI_Comm_free(&mpi_graph);
        MPIX_Comm_free(mpix_graph);
        MPIX_Comm_free(locality_comm);
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
