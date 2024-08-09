#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include <omp.h>

void alltoall(double* send_data, double* recv_data, int n, int start, int stop, int step, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

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
        
        MPI_Sendrecv(send_data + send_pos, n, MPI_DOUBLE, dest, 0, recv_data + recv_pos, n, MPI_DOUBLE, src, 0, comm, MPI_STATUS_IGNORE);
    }
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
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    int ranks_per_gpu = ppn / num_gpus;
    int rank_gpu = local_rank / ranks_per_gpu;
    int gpu_rank = local_rank % ranks_per_gpu;
    gpuSetDevice(rank_gpu);

    MPI_Comm gpu_comm;
    MPI_Comm_split(xcomm->local_comm, rank_gpu, gpu_rank, &gpu_comm);
    
    MPI_Comm all_masters_comm;
    int master_color = MPI_UNDEFINED;
    if (gpu_rank == 0)
    {
        master_color = 0;
    }
    MPI_Comm_split(MPI_COMM_WORLD, master_color, rank_gpu, &all_masters_comm);
    int master_count;
    if (all_masters_comm != MPI_COMM_NULL)
        MPI_Comm_size(all_masters_comm, &master_count);
    MPI_Bcast(&master_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Comm one_per_gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank_gpu, &one_per_gpu_comm);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*master_count);
    std::vector<double> std_alltoall(max_s*master_count);
    std::vector<double> new_alltoall(max_s*master_count);
    for (int j = 0; j < max_s*master_count; j++)
        send_data[j] = rand();

    double* send_data_d;
    double* recv_data_d;
    double* send_data_h;
    double* recv_data_h;

    if (gpu_rank == 0)
    {
        gpuMalloc((void**)(&send_data_d), max_s*master_count*sizeof(double));
        gpuMalloc((void**)(&recv_data_d), max_s*master_count*sizeof(double));
        gpuMemcpy(send_data_d, send_data.data(), max_s*master_count*sizeof(double), gpuMemcpyHostToDevice);
        gpuMallocHost((void**)(&send_data_h), max_s*master_count*sizeof(double));
        gpuMallocHost((void**)(&recv_data_h), max_s*master_count*sizeof(double));
    }

    double* send_data_shared;
    double* recv_data_shared;
    MPI_Win send_win, recv_win;
    int win_size = 0;
    if (gpu_rank == 0)
    {
        win_size = max_s*master_count;
    }
    MPI_Aint remote_win_s, remote_win_r;
    int disp_unit_s, disp_unit_r;
    MPI_Win_allocate_shared(win_size*sizeof(double), sizeof(double),  
        MPI_INFO_NULL, gpu_comm, &send_data_shared, &send_win);
    if (gpu_rank != 0)
    {
        MPI_Win_shared_query(send_win, 0, &remote_win_s, &disp_unit_s, &send_data_shared);
    }
    
    MPI_Win_allocate_shared(win_size*sizeof(double), sizeof(double), 
        MPI_INFO_NULL, gpu_comm, &recv_data_shared, &recv_win);
    if (gpu_rank != 0)
    {
        MPI_Win_shared_query(recv_win, 0, &remote_win_r, &disp_unit_r, &recv_data_shared);
    }

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        int n_iter = max_n_iter;
        if (s > 4096) n_iter /= 10;

        // GPU-Aware PMPI Implementation
        if (gpu_rank == 0)
        {
            PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, all_masters_comm);
            gpuMemcpy(std_alltoall.data(), recv_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            gpuMemset(recv_data_d, 0, s*master_count*sizeof(double));
        }

        // Copy-to-CPU PMPI Implementation
        if (gpu_rank == 0)
        {
            gpuMemcpy(send_data_h, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, all_masters_comm);
            gpuMemcpy(recv_data_d, recv_data_h, s*master_count*sizeof(double), gpuMemcpyHostToDevice);
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*master_count);
            if (err >= 0)
            {
                printf("C2C PMPI Error at IDX %d, rank %d\n", err, rank);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            gpuMemset(recv_data_d, 0, s*master_count*sizeof(double));
        }

        // Copy-to-CPU Alltoall
        if (gpu_rank == 0)
        {
            gpuMemcpy(send_data_h, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            alltoall(send_data_h, recv_data_h, s, 0, master_count, 1, all_masters_comm);
            gpuMemcpy(recv_data_d, recv_data_h, s*master_count*sizeof(double), gpuMemcpyHostToDevice);
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*master_count);
            if (err >= 0)
            {
                printf("C2C MPIX Pairwise Error at IDX %d, rank %d\n", err, rank);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            gpuMemset(recv_data_d, 0, s*master_count*sizeof(double));
        }

        // Copy-to-CPU 2Thread Alltoall
        MPI_Win_lock_all(MPI_MODE_NOCHECK, send_win);
        if (gpu_rank == 0)
        {
            gpuMemcpy(send_data_shared, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
        }
        MPI_Win_sync(send_win);
        MPI_Barrier(gpu_comm);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, recv_win);
        alltoall(send_data_shared, recv_data_shared, s, gpu_rank, master_count, ranks_per_gpu, one_per_gpu_comm);
        MPI_Win_sync(recv_win);
        MPI_Win_unlock_all(send_win);
        MPI_Barrier(gpu_comm); // needed
        if (gpu_rank == 0)
        {
            cudaMemcpy(recv_data_d, recv_data_shared, s*master_count*sizeof(double), cudaMemcpyHostToDevice);
        }
        MPI_Win_unlock_all(recv_win);
        if (gpu_rank == 0)
        {
            gpuMemcpy(new_alltoall.data(), recv_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            int err = compare(std_alltoall, new_alltoall, s*master_count);
            if (err >= 0)
            {
                printf("C2C %d Processes MPIX Pairwise Error at IDX %d, rank %d\n", ranks_per_gpu, err, rank);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            gpuMemset(recv_data_d, 0, s*master_count*sizeof(double));
        }

        // Time Methods!

        // GPU-Aware PMPI Implementation
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (gpu_rank == 0)
            {
                PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, all_masters_comm);
            }
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware PMPI Time %e\n", t0);

        // Copy-to-CPU PMPI Implementation
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (gpu_rank == 0)
            {
                gpuMemcpy(send_data_h, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
                PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, all_masters_comm);
                gpuMemcpy(recv_data_d, recv_data_h, s*master_count*sizeof(double), gpuMemcpyHostToDevice);
            }
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU PMPI Time %e\n", t0);
  
        // Copy-to-CPU Alltoall
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            if (gpu_rank == 0)
            {
                gpuMemcpy(send_data_h, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
                alltoall(send_data_h, recv_data_h, s, 0, master_count, 1, all_masters_comm);
                gpuMemcpy(recv_data_d, recv_data_h, s*master_count*sizeof(double), gpuMemcpyHostToDevice);
            }
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU Pairwise Time %e\n", t0);
  

        // Copy-to-CPU 2Thread Alltoall
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            MPI_Win_lock_all(MPI_MODE_NOCHECK, send_win);
            if (gpu_rank == 0)
            {
                gpuMemcpy(send_data_shared, send_data_d, s*master_count*sizeof(double), gpuMemcpyDeviceToHost);
            }
            MPI_Win_sync(send_win);
            MPI_Barrier(gpu_comm);
            MPI_Win_lock_all(MPI_MODE_NOCHECK, recv_win);
            alltoall(send_data_shared, recv_data_shared, s, gpu_rank, master_count, ranks_per_gpu, one_per_gpu_comm);
            MPI_Win_sync(recv_win);
            MPI_Win_unlock_all(send_win);
            MPI_Barrier(gpu_comm); // needed
            if (gpu_rank == 0)
            {
                cudaMemcpy(recv_data_d, recv_data_shared, s*master_count*sizeof(double), cudaMemcpyHostToDevice);
            }
            MPI_Win_unlock_all(recv_win);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Processes Pairwise Time %e\n", ranks_per_gpu, t0);
    }
    MPI_Win_free(&send_win);
    MPI_Win_free(&recv_win);

    MPI_Comm_free(&gpu_comm);
    if (gpu_rank == 0)
    {
        MPI_Comm_free(&all_masters_comm);
    }
    MPI_Comm_free(&one_per_gpu_comm);

    MPIX_Comm_free(xcomm);

    if (gpu_rank == 0)
    {
        gpuFree(send_data_d);
        gpuFree(recv_data_d);
        gpuFreeHost(send_data_h);
        gpuFreeHost(recv_data_h);
    }

    MPI_Finalize();
    return 0;
}
