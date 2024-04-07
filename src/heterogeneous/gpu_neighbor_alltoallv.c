#include "gpu_neighbor_alltoallv.h"
#include "neighborhood/neighbor_persistent.h"
#include "persistent_gpu.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_neighbor_alltoallv_init(neighbor_alltoallv_ftn f,
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    return f(sendbuffer, sendcounts, sdispls, sendtype, recvbuffer, recvcounts, rdispls, recvtype,
            comm, info, request_ptr);
}

int gpu_aware_neighbor_alltoallv_nonblocking_init(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int ierr = gpu_aware_neighbor_alltoallv_init(MPIX_Neighbor_alltoallv_init,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm,
        MPI_INFO_NULL, 
        request_ptr);

    return ierr;
}

int copy_to_cpu_neighbor_alltoallv_init(neighbor_alltoallv_ftn f,
        const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int ierr = 0;

    int indegree, outdegree, weighted;
    ierr += MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = 0;
    int total_bytes_r = 0;
    
    if (outdegree > 0)
    {
        total_bytes_s = (sdispls[outdegree - 1] + sendcounts[outdegree - 1]) * send_bytes;
    }
    if (indegree > 0)
    {
        total_bytes_r = (rdispls[indegree - 1] + recvcounts[indegree - 1]) * recv_bytes;
    }

    MPIX_Request* outer_request = NULL;
    init_neighbor_gpu_copy_cpu_request(&outer_request, sendbuf, total_bytes_s, recvbuf, total_bytes_r);

    MPIX_Request* inner_request = NULL;
    // Collective Among CPUs
#ifdef GPU
    ierr += f(outer_request->cpu_sendbuf, sendcounts, sdispls, sendtype, 
            outer_request->cpu_recvbuf, recvcounts, rdispls, recvtype, comm, info, &inner_request);
#endif

    set_sub_request_in_neighbor_gpu_copy_cpu_request(outer_request, inner_request);
    
    *request_ptr = outer_request;

    return ierr;
}

int copy_to_cpu_neighbor_alltoallv_nonblocking_init(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    return copy_to_cpu_neighbor_alltoallv_init(MPIX_Neighbor_alltoallv_init,
        sendbuf, 
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm,
        info,
        request_ptr);
}

int threaded_neighbor_alltoallv_nonblocking_init(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int ierr = 0;

    int indegree, outdegree, weighted;
    ierr += MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int sources[indegree];
    int sourceweights[indegree];
    int destinations[outdegree];
    int destweights[outdegree];
    ierr += MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    
    int total_bytes_s = 0;
    int total_bytes_r = 0;
    
    if (outdegree > 0)
    {
        total_bytes_s = (sdispls[outdegree - 1] + sendcounts[outdegree - 1]) * send_bytes;
    }
    if (indegree > 0)
    {
        total_bytes_r = (rdispls[indegree - 1] + recvcounts[indegree - 1]) * recv_bytes;
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
    init_neighbor_gpu_copy_cpu_request_threaded(&outer_request, sendbuf, total_bytes_s,
                                                                recvbuf, total_bytes_r, num_threads);
    
    int request_idx = 0;

#ifdef GPU
    const char* send_buffer = (const char*) outer_request->cpu_sendbuf;
    char* recv_buffer = (char*) outer_request->cpu_recvbuf;
#endif
    
    for (int thread_id = 0; thread_id < num_threads; ++thread_id)
    {
        int thread_n_msgs_s = n_msgs_s_per_thread;
        int thread_n_msgs_r = n_msgs_r_per_thread;
        if (extra_msgs_s > thread_id)
            thread_n_msgs_s++;
        if (extra_msgs_r > thread_id)
            thread_n_msgs_r++;
            
        if (thread_n_msgs_s)
        {
            int baseIdx = thread_n_msgs_s * thread_id;
            if (extra_msgs_s <= thread_id)
            {
                baseIdx += extra_msgs_s;
            }
            for (int idx = baseIdx; idx < baseIdx + thread_n_msgs_s; ++idx)
            {
#ifdef GPU
                MPI_Send_init(&(send_buffer[sdispls[idx] * send_bytes]), 
                        sendcounts[idx], 
                        sendtype, 
                        destinations[idx], 
                        tag, 
                        comm->neighbor_comm, 
                        &(inner_request->global_requests[request_idx]));
#endif
                ++request_idx;
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
#ifdef GPU
                MPI_Recv_init(&(recv_buffer[rdispls[idx] * recv_bytes]), 
                        recvcounts[idx], 
                        recvtype, 
                        sources[idx], 
                        tag, 
                        comm->neighbor_comm, 
                        &(inner_request->global_requests[request_idx]));
#endif
                ++request_idx;
            }
        }
    }

    set_sub_request_in_neighbor_gpu_copy_cpu_request(outer_request, inner_request);
    
    *request_ptr = outer_request;
    
    return ierr;
}
