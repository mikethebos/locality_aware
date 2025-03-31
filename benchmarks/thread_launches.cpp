#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include <omp.h>

int neighbor_gpu_copy_cpu_threaded_start(MPIX_Request* request)
{
#ifdef GPU
    int ret = 0;

    cudaMemcpy(request->cpu_sendbuf, request->sendbuf, request->cpu_sendbuf_bytes, cudaMemcpyDeviceToHost);

    // copy recvbuf in case of extra data
    // needed if noncontiguous displs (or custom packing)
    cudaMemcpy(request->cpu_recvbuf, request->recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyDeviceToHost);

    return ret;
#endif
}

// no locality support
int neighbor_gpu_copy_cpu_threaded_wait(MPIX_Request* request, MPI_Status* status)
{    
#ifdef GPU    
    int ret = 0;
    int n_msgs = request->sub_request->global_n_msgs;

    if (n_msgs)
    {
    MPIX_Request *inner_request = request->sub_request;
    const char* send_buffer = (const char*) request->cpu_sendbuf;
    char* recv_buffer = (char*) request->cpu_recvbuf;
    int nthreads = request->num_threads;
    int n_msgs_s_per_thread = request->n_msgs_s_per_thread;
    int extra_msgs_s = request->extra_msgs_s;
    int n_msgs_r_per_thread = request->n_msgs_r_per_thread;
    int extra_msgs_r = request->extra_msgs_r;
    const int *sdispls = request->sdispls;
    const int *rdispls = request->rdispls;
    const int send_bytes = request->send_bytes;
    const int recv_bytes = request->recv_bytes;
    const int *sendcounts = request->sendcounts;
    const int *recvcounts = request->recvcounts;
    MPI_Datatype sendtype = request->sendtype;
    MPI_Datatype recvtype = request->recvtype;
    const int *destinations = request->destinations;
    const int *sources = request->sources;
    
    int tag = 102944;
    
#pragma omp parallel reduction(+:ret)
    {
        int thread_id = omp_get_thread_num();
        int thread_n_msgs_s = n_msgs_s_per_thread;
        int thread_n_msgs_r = n_msgs_r_per_thread;
        if (extra_msgs_s > thread_id)
            thread_n_msgs_s++;
        if (extra_msgs_r > thread_id)
            thread_n_msgs_r++;
            
        int request_idx = (thread_n_msgs_s + thread_n_msgs_r) * thread_id;
        if (extra_msgs_s <= thread_id)
        {
            request_idx += extra_msgs_s;
        }
        if (extra_msgs_r <= thread_id)
        {
            request_idx += extra_msgs_r;
        }
        
        int start_offset = request_idx;
        int count_th = 0;
            
        if (thread_n_msgs_s)
        {
            int baseIdx = thread_n_msgs_s * thread_id;
            if (extra_msgs_s <= thread_id)
            {
                baseIdx += extra_msgs_s;
            }
            for (int idx = baseIdx; idx < baseIdx + thread_n_msgs_s; ++idx)
            {
                ret += MPI_Isend(&(send_buffer[sdispls[idx] * send_bytes]), 
                        sendcounts[idx], 
                        sendtype, 
                        destinations[idx], 
                        tag, 
                        MPI_COMM_WORLD, 
                        &(inner_request->global_requests[request_idx]));
                ++request_idx;
                ++count_th;
            }
        }
        
        if (thread_n_msgs_r)
        {
            int baseIdx = thread_n_msgs_r * thread_id;
            if (extra_msgs_r <= thread_id)
            {
                baseIdx += extra_msgs_r;
            }
            for (int idx = baseIdx; idx < baseIdx + thread_n_msgs_r; ++idx)
            {
                ret += MPI_Irecv(&(recv_buffer[rdispls[idx] * recv_bytes]), 
                        recvcounts[idx], 
                        recvtype, 
                        sources[idx], 
                        tag, 
                        MPI_COMM_WORLD, 
                        &(inner_request->global_requests[request_idx]));
                ++request_idx;
                ++count_th;
            }
        }
        
        ret += MPI_Waitall(count_th, &(inner_request->global_requests[start_offset]), MPI_STATUSES_IGNORE);
    }
    }
    // only copy recvbuf after wait
    cudaMemcpy(request->recvbuf, request->cpu_recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyHostToDevice);

    return ret;
#endif
}

void init_neighbor_gpu_copy_cpu_request_threaded(MPIX_Request** request_ptr, const void* sendbuf, int sendbuf_bytes,
                                                                             void* recvbuf, int recvbuf_bytes, int num_threads,
                                                                             int n_msgs_s_per_thread,
                                                                             int n_msgs_r_per_thread,
                                                                             int extra_msgs_s,
                                                                             int extra_msgs_r,
                                                                             const int *sdispls,
                                                                             const int send_bytes,
                                                                             MPI_Datatype sendtype,
                                                                             const int *sendcounts,
                                                                             const int *destinations,
                                                                             const int *rdispls,
                                                                             const int recv_bytes,
                                                                             MPI_Datatype recvtype,
                                                                             const int *recvcounts,
                                                                             const int *sources)
{
    init_request(request_ptr);
    MPIX_Request* request = *request_ptr;

    request->start_function = (void*) neighbor_gpu_copy_cpu_threaded_start;
    request->wait_function = (void*) neighbor_gpu_copy_cpu_threaded_wait;
    
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    
#ifdef GPU
    request->cpu_sendbuf_bytes = sendbuf_bytes;
    request->cpu_recvbuf_bytes = recvbuf_bytes;
        
    request->num_threads = num_threads;
    request->n_msgs_s_per_thread = n_msgs_s_per_thread;
    request->n_msgs_r_per_thread = n_msgs_r_per_thread;
    request->extra_msgs_s = extra_msgs_s;
    request->extra_msgs_r = extra_msgs_r;
    request->sdispls = (int *) sdispls;
    request->send_bytes = (int) send_bytes;
    request->sendtype = sendtype;
    request->sendcounts = (int *) sendcounts;
    request->destinations = (int *) destinations;
    request->rdispls = (int *) rdispls;
    request->recv_bytes = (int) recv_bytes;
    request->recvtype = recvtype;
    request->recvcounts = (int *) recvcounts;
    request->sources = (int *) sources;
    
    cudaMallocHost((void **)(&(request->cpu_sendbuf)), sendbuf_bytes);
    cudaMallocHost((void **)(&(request->cpu_recvbuf)), recvbuf_bytes);
#endif // free handled in MPIX_Request_free in persistent/persistent.c, must be cudaMallocHost
}

void set_sub_request_in_neighbor_gpu_copy_cpu_request(MPIX_Request* outer_request, MPIX_Request* inner_request)
{
#ifdef GPU
    outer_request->sub_request = inner_request;
#endif
}

void from_nbr_init(double *send_data, double *recv_data, int n, MPIX_Request **request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int outdegree = num_procs;
    int indegree = num_procs;
    
    int *sdispls = (int *)malloc(num_procs * sizeof(int));
    int *rdispls = (int *)malloc(num_procs * sizeof(int));
    int *rcounts = (int *)malloc(num_procs * sizeof(int));
    int *scounts = (int *)malloc(num_procs * sizeof(int));
    int *sources = (int *)malloc(num_procs * sizeof(int));
    int *destinations = (int *)malloc(num_procs * sizeof(int));
    int d = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sdispls[i] = d;
        rdispls[i] = d;
        d += n;
        scounts[i] = n;
        rcounts[i] = n;
        sources[i] = i;
        destinations[i] = i;
    }
    
    // no communication occuring here, so no need for openmp
    int tag = 102944;
    int n_msgs_s = outdegree;
    int n_msgs_r = indegree;
    int num_threads = omp_get_max_threads(); // assume max number of threads always launched

    int n_msgs_s_per_thread = n_msgs_s / num_threads;
    int n_msgs_r_per_thread = n_msgs_r / num_threads;
    int extra_msgs_s = n_msgs_s % num_threads;
    int extra_msgs_r = n_msgs_r % num_threads;
        
    MPIX_Request* inner_request;
    init_neighbor_request(&inner_request);

    inner_request->global_n_msgs = (num_threads * n_msgs_s_per_thread) + extra_msgs_s + (num_threads * n_msgs_r_per_thread) + extra_msgs_r;
    allocate_requests(inner_request->global_n_msgs, &(inner_request->global_requests));
    
    MPIX_Request* outer_request;
    init_neighbor_gpu_copy_cpu_request_threaded(&outer_request, send_data, n*num_procs*sizeof(double),
                                                                recv_data, n*num_procs*sizeof(double), num_threads,
                                                                n_msgs_s_per_thread,
                                                                n_msgs_r_per_thread,
                                                                extra_msgs_s,
                                                                extra_msgs_r,
                                                                sdispls,
                                                                sizeof(double),
                                                                MPI_DOUBLE,
                                                                scounts,
                                                                destinations,
                                                                rdispls,
                                                                sizeof(double),
                                                                MPI_DOUBLE,
                                                                rcounts,
                                                                sources);
    
    int request_idx = 0;

    set_sub_request_in_neighbor_gpu_copy_cpu_request(outer_request, inner_request);
    
#ifdef GPU
    outer_request->not_gpu_neighbor_alltoallv = 0;
    inner_request->not_gpu_neighbor_alltoallv = 0;
#endif
    
    *request_ptr = outer_request;
}

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

    int max_i = 19;
    int max_s = pow(2, max_i);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*num_procs);
    std::vector<double> recv_data(max_s*num_procs);
    std::vector<double> std_alltoall(max_s*num_procs);
    std::vector<double> new_alltoall(max_s*num_procs);
    for (int j = 0; j < max_s*num_procs; j++)
        send_data[j] = rand();

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    gpuSetDevice(local_rank);

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

    MPI_Request *reqs = (MPI_Request *)malloc(arg_nt * 2 * num_procs * sizeof(MPI_Request));

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        int n_iter = max_n_iter;
        if (s > 4096) n_iter /= 10;
        else n_iter *= 10;

        // GPU-Aware PMPI Implementation
        PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, MPI_COMM_WORLD);
        gpuMemcpy(std_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Copy-to-CPU PMPI Implementation
        gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, MPI_COMM_WORLD);
        gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        int err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {
            printf("C2C PMPI Error at IDX %d, rank %d\n", err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Copy-to-CPU Alltoall
        gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        alltoall(send_data_h, recv_data_h, s, 0, num_procs, 1);
        gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {
            printf("C2C MPIX Error at IDX %d, rank %d\n", err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // GPU-Aware Alltoall
        alltoall(send_data_d, recv_data_d, s, 0, num_procs, 1);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {
            printf("GPU MPIX Error at IDX %d, rank %d\n", err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Copy-to-CPU 10Thread Alltoall
        gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads);
        }
        gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {   
            printf("%dThreads MPIX Error at IDX %d, rank %d\n", arg_nt, err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        
        // Copy-to-CPU Alltoall
        gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        alltoall_nonblocking_bench(send_data_h, recv_data_h, s, 0, num_procs, 1, reqs);
        gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {
            printf("C2C MPIX Nonblocking Error at IDX %d, rank %d\n", err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // GPU-Aware Alltoall
        alltoall_nonblocking_bench(send_data_d, recv_data_d, s, 0, num_procs, 1, reqs);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {
            printf("GPU MPIX Nonblocking Error at IDX %d, rank %d\n", err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        // Copy-to-CPU 10Thread Alltoall
        gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        #pragma omp parallel num_threads(arg_nt)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            alltoall_nonblocking_bench(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads, &(reqs[thread_id * 2 * num_procs]));
        }
        gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {   
            printf("%dThreads MPIX Nonblocking Error at IDX %d, rank %d\n", arg_nt, err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        
        // Copy-to-CPU 10Thread Alltoall (neighbor)
        MPIX_Request *neighreq;
        from_nbr_init(send_data_d, recv_data_d, s, &neighreq);
        neighbor_gpu_copy_cpu_threaded_start(neighreq);
        neighbor_gpu_copy_cpu_threaded_wait(neighreq, MPI_STATUS_IGNORE);
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
        err = compare(std_alltoall, new_alltoall, s*num_procs);
        if (err >= 0)
        {   
            printf("%dThreads MPIX Nonblocking (Neighbor) Error at IDX %d, rank %d\n", arg_nt, err, rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(double));
        MPIX_Request_free(neighreq);
   
        // MPI Advance : Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s*num_procs; j++)
	{
            if (fabs(std_alltoall[j] - new_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-PE %e\n", 
                         rank, j, std_alltoall[j], new_alltoall[j]);
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
                xcomm,
                (char *)send_data.data(),
                (char *)recv_data.data());
        gpuMemcpy(new_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s*num_procs; j++)
	{
            if (fabs(std_alltoall[j] - new_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-NB %e\n", 
                         rank, j, std_alltoall[j], new_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
   
        // Time Methods!

        // GPU-Aware PMPI Implementation
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            PMPI_Alltoall(send_data_d, s, MPI_DOUBLE, recv_data_d, s, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware PMPI Time %e\n", t0);

        // Copy-to-CPU PMPI Implementation
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            PMPI_Alltoall(send_data_h, s, MPI_DOUBLE, recv_data_h, s, MPI_DOUBLE, MPI_COMM_WORLD);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU PMPI Time %e\n", t0);
  
        // Copy-to-CPU Alltoall
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            alltoall(send_data_h, recv_data_h, s, 0, num_procs, 1);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU Pairwise Time %e\n", t0);

        // GPU-Aware Alltoall
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            alltoall(send_data_d, recv_data_d, s, 0, num_procs, 1);
        }
    	tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware Pairwise Time %e\n", t0);  

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            #pragma omp parallel num_threads(arg_nt)
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                alltoall(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads);
            }
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Threads Pairwise Time %e\n", arg_nt, t0);
        
        // Copy-to-CPU Alltoall
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            alltoall_nonblocking_bench(send_data_h, recv_data_h, s, 0, num_procs, 1, reqs);
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU Nonblocking Time %e\n", t0);

        // GPU-Aware Alltoall
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {
            alltoall_nonblocking_bench(send_data_d, recv_data_d, s, 0, num_procs, 1, reqs);
        }
    	tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware Nonblocking Time %e\n", t0);  

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            gpuMemcpy(send_data_h, send_data_d, s*num_procs*sizeof(double), gpuMemcpyDeviceToHost);
            #pragma omp parallel num_threads(arg_nt)
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                alltoall_nonblocking_bench(send_data_h, recv_data_h, s, thread_id, num_procs, num_threads, &(reqs[thread_id * 2 * num_procs]));
            }
            gpuMemcpy(recv_data_d, recv_data_h, s*num_procs*sizeof(double), gpuMemcpyHostToDevice);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Threads Nonblocking Time %e\n", arg_nt, t0);
        
        from_nbr_init(send_data_d, recv_data_d, s, &neighreq);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
        {   
            neighbor_gpu_copy_cpu_threaded_start(neighreq);
            neighbor_gpu_copy_cpu_threaded_wait(neighreq, MPI_STATUS_IGNORE);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d Threads Nonblocking (Neighbor) Time %e\n", arg_nt, t0);
        MPIX_Request_free(neighreq);
        
        // Time Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm,
                (char *)send_data.data(),
                (char *)recv_data.data());
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
                    xcomm,
                    (char *)send_data.data(),
                    (char *)recv_data.data());
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX Threaded Pairwise Exchange Time %e\n", t0);

        // Time Threaded Nonblocking
        threaded_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm,
                (char *)send_data.data(),
                (char *)recv_data.data());
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
                    xcomm,
                    (char *)send_data.data(),
                    (char *)recv_data.data());
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX Threaded Nonblocking Time %e\n", t0);
    }
    free((void *)reqs);
    
    MPIX_Comm_free(xcomm);

    gpuFree(send_data_d);
    gpuFree(recv_data_d);
    gpuFreeHost(send_data_h);
    gpuFreeHost(recv_data_h);

    MPI_Finalize();
    return 0;
}
