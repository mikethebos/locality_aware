#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include <omp.h>

void alltoall(double* send_data, double* recv_data, int n, int start, int stop, int step)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int src, dest;
    for (int i = start; i < stop; i += step)
    {
        dest = rank - i; 
        if (dest < 0) dest += num_procs;
        src = rank + i;
        if (src >= num_procs)
            src -= num_procs;
        int send_pos = dest*n;
        int recv_pos = src*n;
        
        MPI_Sendrecv(send_data + send_pos, n, MPI_DOUBLE, dest, 0, recv_data + recv_pos, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void alltoall_nonblocking_bench(double* send_data, double* recv_data, int n, int start, int stop, int step, MPI_Request *reqs)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int src, dest;
    int count = 0;
    for (int i = start; i < stop; i += step)
    {
        dest = rank - i; 
        if (dest < 0) dest += num_procs;
        src = rank + i;
        if (src >= num_procs)
            src -= num_procs;
        int send_pos = dest*n;
        int recv_pos = src*n;
        
        MPI_Isend(send_data + send_pos, n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &(reqs[count]));
        count++;
        MPI_Irecv(recv_data + recv_pos, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &(reqs[count]));
        count++;
    }
    
    MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
}

int compare(std::vector<double>& std_alltoall, std::vector<double>& new_alltoall, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(std_alltoall[i] - new_alltoall[i]) > 1e-10)
        {
            return i;
        }
    }
    return -1;
}

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int arg_nt = omp_get_max_threads();

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

//     omp_set_num_threads(10);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*num_procs);
    std::vector<double> recv_data(max_s*num_procs);
    std::vector<double> pmpi_alltoall(max_s*num_procs);
    std::vector<double> mpix_alltoall(max_s*num_procs);
    for (int j = 0; j < max_s*num_procs; j++)
        send_data[j] = rand();

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    int gpu_rank;
    MPI_Comm_rank(locality_comm->local_comm, &gpu_rank);
    gpuSetDevice(gpu_rank);

    double* send_data_d;
    double* recv_data_d;
    cudaMalloc((void**)(&send_data_d), max_s*num_procs*sizeof(double));
    cudaMalloc((void**)(&recv_data_d), max_s*num_procs*sizeof(double));
    cudaMemcpy(send_data_d, send_data.data(), max_s*num_procs*sizeof(double), cudaMemcpyHostToDevice);
    
    MPI_Request *reqs = (MPI_Request *) malloc(arg_nt * 2 * num_procs * sizeof(MPI_Request));

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        int n_iter = max_n_iter;
        if (s > 4096) n_iter /= 10;

        // Standard MPI Implementation
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(pmpi_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));

        // MPI Advance : Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-PE %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : Copy To CPU Nonblocking (P2P)
        threaded_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-NB %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // Copy-to-CPU 10Thread Alltoall
        gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads);
        }
        gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        int err = compare(pmpi_alltoall, mpix_alltoall, s*num_procs);
        if (err >= 0)
        {   
            printf("%dThreads MPIX Error at IDX %d, rank %d\n", arg_nt, err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Copy-to-CPU 10Thread Alltoall
        gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall_nonblocking_bench(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads, &(reqs[thread_id * 2 * num_procs]));
        }
        gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(pmpi_alltoall, mpix_alltoall, s*num_procs);
        if (err >= 0)
        {   
            printf("%dThreads MPIX Nonblocking Error at IDX %d, rank %d\n", arg_nt, err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Time PMPI Alltoall
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Alltoall(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI_Alltoall Time %e\n", t0);

        // Time Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            threaded_alltoall_pairwise(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    locality_comm,
                    (char *)send_data.data(),
                    (char *)recv_data.data());
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded Pairwise Exchange Time %e\n", t0);

        // Time Threaded Nonblocking
        threaded_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            threaded_alltoall_nonblocking(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    locality_comm,
                    (char *)send_data.data(),
                    (char *)recv_data.data());
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded Nonblocking Time %e\n", t0);
        
        gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads);
        }
        gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            #pragma omp parallel num_threads(arg_nt)
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                alltoall(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads);
            }
            gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Threads Custom Pairwise Time %e\n", arg_nt, t0);
        
        gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall_nonblocking_bench(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads, &(reqs[thread_id * 2 * num_procs]));
        }
        gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            gpuMemcpy(send_data.data(), send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            #pragma omp parallel num_threads(arg_nt)
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                alltoall_nonblocking_bench(send_data.data(), recv_data.data(), s, thread_id, num_procs, num_threads, &(reqs[thread_id * 2 * num_procs]));
            }
            gpuMemcpy(recv_data_d, recv_data.data(), s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Threads Custom Nonblocking Time %e\n", arg_nt, t0);
    }
    free((void *)reqs);
    cudaFree(send_data_d);
    cudaFree(recv_data_d);

    MPIX_Comm_free(locality_comm);

    MPI_Finalize();
    return 0;
}
