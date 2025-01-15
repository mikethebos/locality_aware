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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*num_procs);
    std::vector<double> std_alltoall(max_s*num_procs);
    std::vector<double> new_alltoall(max_s*num_procs);
    for (int j = 0; j < max_s*num_procs; j++)
        send_data[j] = rand();

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    // gpuSetDevice(local_rank);

    double* send_data_d;
    double* recv_data_d;
    gpuMalloc((void**)(&send_data_d), max_s*num_procs*sizeof(double));
    gpuMalloc((void**)(&recv_data_d), max_s*num_procs*sizeof(double));
    gpuMemcpy(send_data_d, send_data.data(), max_s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
    double* send_data_h;
    double* recv_data_h;
    gpuMallocHost((void**)(&send_data_h), max_s*num_procs*sizeof(double));
    gpuMallocHost((void**)(&recv_data_h), max_s*num_procs*sizeof(double));
        
    int arg_nt = atoi(argv[1]);

#pragma omp parallel num_threads(arg_nt)
{
    MPI_Request *reqs = (MPI_Request *)malloc(2 * num_procs * sizeof(MPI_Request));
    
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0 && thread_id == 0) printf("Testing Size %d\n", s);

        int n_iter = max_n_iter;
        if (s > 4096) n_iter /= 10;
        else n_iter *= 10;

        // GPU-Aware PMPI Implementation
        if (thread_id == 0)
        {
            PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, MPI_COMM_WORLD);
            gpuMemcpy(std_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }

        // Copy-to-CPU PMPI Implementation
        if (thread_id == 0)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, MPI_COMM_WORLD);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*num_procs);
            if (err >= 0)
            {
                printf("C2C PMPI Error at IDX %d, rank %d\n", err, rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                // return 1;
            }
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }

        // Copy-to-CPU Alltoall
        if (thread_id == 0)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            alltoall(send_data_h, recv_data_h, s, 0, num_procs, 1);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*num_procs);
            if (err >= 0)
            {
                printf("C2C MPIX Error at IDX %d, rank %d\n", err, rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                // return 1;
            }
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }
        
       // Copy-to-CPU 10Thread Alltoall
        if (thread_id == 0) 
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
#pragma omp barrier
        if (thread_id < num_threads)
            alltoall(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads);
#pragma omp barrier
        if (thread_id == 0)
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);

        if (thread_id == 0)
        {   
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*num_procs);
            if (err >= 0)
            {   
                printf("%dThreads MPIX Error at IDX %d, rank %d\n", num_threads, err, rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                // return 1;
            }
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }
        
        // Copy-to-CPU Alltoall
        if (thread_id == 0)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            alltoall_nonblocking_bench(send_data_h, recv_data_h, s, 0, num_procs, 1, reqs);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*num_procs);
            if (err >= 0)
            {
                printf("C2C MPIX Nonblocking Error at IDX %d, rank %d\n", err, rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                // return 1;
            }
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }
        
       // Copy-to-CPU 10Thread Alltoall
        if (thread_id == 0) 
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
#pragma omp barrier
        if (thread_id < num_threads)
            alltoall_nonblocking_bench(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads, reqs);
#pragma omp barrier
        if (thread_id == 0)
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);

        if (thread_id == 0)
        {   
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*num_procs);
            if (err >= 0)
            {   
                printf("%dThreads MPIX Nonblocking Error at IDX %d, rank %d\n", num_threads, err, rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                // return 1;
            }
            gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        }
   
        // Time Methods!

        // GPU-Aware PMPI Implementation
        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
#pragma omp barrier
        if (thread_id == 0) t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (thread_id == 0)
            {
                PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        if (thread_id == 0) tfinal = (MPI_Wtime() - t0) / n_iter;
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("GPU-Aware PMPI Time %e\n", t0);

        // Copy-to-CPU PMPI Implementation
        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
#pragma omp barrier
        if (thread_id == 0) t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (thread_id == 0)
            {
                gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
                PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, MPI_COMM_WORLD);
                gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            }
        }
        if (thread_id == 0) tfinal = (MPI_Wtime() - t0) / n_iter;
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("Copy-to-CPU PMPI Time %e\n", t0);
  
        // Copy-to-CPU Alltoall
        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
#pragma omp barrier
        if (thread_id == 0) t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (thread_id == 0)
            {
                gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
                alltoall(send_data_h, recv_data_h, s, 0, num_procs, 1);
                gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            }
        }
        if (thread_id == 0) tfinal = (MPI_Wtime() - t0) / n_iter;
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("Copy-to-CPU Pairwise Time %e\n", t0);

        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
        if (thread_id == 0) tfinal = -1.0;
#pragma omp barrier
        double t0_thread = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            if (thread_id == 0) 
                gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
#pragma omp barrier
            if (thread_id < num_threads)
                alltoall(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads);
#pragma omp barrier
            if (thread_id == 0)
                gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        double tfinal_thread = (MPI_Wtime() - t0_thread) / n_iter;
#pragma omp critical
{
        if (tfinal_thread > tfinal)
        {
            tfinal = tfinal_thread;
        }
}
#pragma omp barrier
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("%d Threads Pairwise Time %e\n", num_threads, t0);
        
        // Copy-to-CPU Alltoall
        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
#pragma omp barrier
        if (thread_id == 0) t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (thread_id == 0)
            {
                gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
                alltoall_nonblocking_bench(send_data_h, recv_data_h, s, 0, num_procs, 1, reqs);
                gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
            }
        }
        if (thread_id == 0) tfinal = (MPI_Wtime() - t0) / n_iter;
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("Copy-to-CPU Nonblocking Time %e\n", t0);

        if (thread_id == 0) MPI_Barrier(MPI_COMM_WORLD);
        if (thread_id == 0) tfinal = -1.0;
#pragma omp barrier
        t0_thread = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            if (thread_id == 0) 
                gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
#pragma omp barrier
            if (thread_id < num_threads)
                alltoall_nonblocking_bench(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads, reqs);
#pragma omp barrier
            if (thread_id == 0)
                gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal_thread = (MPI_Wtime() - t0_thread) / n_iter;
#pragma omp critical
{
        if (tfinal_thread > tfinal)
        {
            tfinal = tfinal_thread;
        }
}
#pragma omp barrier
        if (thread_id == 0) MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && thread_id == 0) printf("%d Threads Nonblocking Time %e\n", num_threads, t0);
    }
    free((void *)reqs);
}
    MPIX_Comm_free(xcomm);

    gpuFree(send_data_d);
    gpuFree(recv_data_d);
    gpuFreeHost(send_data_h);
    gpuFreeHost(recv_data_h);

    MPI_Finalize();
    return 0;
}
