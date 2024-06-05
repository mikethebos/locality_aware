#include "persistent_gpu.h"

#include "neighborhood/neighbor_persistent.h"
#include "collective/collective.h"

// no locality support
int neighbor_gpu_copy_cpu_start(MPIX_Request* request)
{
#ifdef GPU
    cudaMemcpy(request->cpu_sendbuf, request->sendbuf, request->cpu_sendbuf_bytes, cudaMemcpyDeviceToHost);

    // copy recvbuf in case of extra data
    // needed if noncontiguous displs (or custom packing)
    // cudaMemcpy(request->cpu_recvbuf, request->recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyDeviceToHost);
   
    return neighbor_start(request->sub_request);
#endif
}

// no locality support
int neighbor_gpu_copy_cpu_wait(MPIX_Request* request, MPI_Status* status)
{    
#ifdef GPU    
    int ret = neighbor_wait(request->sub_request, status);

    // only copy recvbuf after wait
    cudaMemcpy(request->recvbuf, request->cpu_recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyHostToDevice);
    
    return ret;
#endif
}

// no locality support
int neighbor_gpu_copy_cpu_threaded_start(MPIX_Request* request)
{
#ifdef GPU
    int ret = 0;

    cudaMemcpy(request->cpu_sendbuf, request->sendbuf, request->cpu_sendbuf_bytes, cudaMemcpyDeviceToHost);

    // copy recvbuf in case of extra data
    // needed if noncontiguous displs (or custom packing)
    // cudaMemcpy(request->cpu_recvbuf, request->recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyDeviceToHost);

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
    MPIX_Comm *comm = request->comm;
    
    int tag = 102944;
    
#pragma omp parallel num_threads(nthreads) reduction(+:ret)
    {
        int thread_id = omp_get_thread_num();
        int thread_n_msgs_s = n_msgs_s_per_thread;
        int thread_n_msgs_r = n_msgs_r_per_thread;
        if (extra_msgs_s > thread_id)
            thread_n_msgs_s++;
        if (extra_msgs_r > thread_id)
            thread_n_msgs_r++;
            
        int request_idx = 0;
            
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
                        comm->neighbor_comm, 
                        &(inner_request->neighbor_gpu_reqs[thread_id][request_idx]));
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
                ret += MPI_Irecv(&(recv_buffer[rdispls[idx] * recv_bytes]), 
                        recvcounts[idx], 
                        recvtype, 
                        sources[idx], 
                        tag, 
                        comm->neighbor_comm, 
                        &(inner_request->neighbor_gpu_reqs[thread_id][request_idx]));
                ++request_idx;
            }
        }
        
        ret += MPI_Waitall(request_idx, inner_request->neighbor_gpu_reqs[thread_id], MPI_STATUSES_IGNORE);
    }
    }
    // only copy recvbuf after wait
    cudaMemcpy(request->recvbuf, request->cpu_recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyHostToDevice);

    return ret;
#endif
}

void init_neighbor_gpu_copy_cpu_request(MPIX_Request** request_ptr, const void* sendbuf, int sendbuf_bytes, void* recvbuf, int recvbuf_bytes)
{
    init_request(request_ptr);
    MPIX_Request* request = *request_ptr;

    request->start_function = (void*) neighbor_gpu_copy_cpu_start;
    request->wait_function = (void*) neighbor_gpu_copy_cpu_wait;
    
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    
#ifdef GPU
    request->cpu_sendbuf_bytes = sendbuf_bytes;
    request->cpu_recvbuf_bytes = recvbuf_bytes;
    
    cudaMallocHost((void **)(&(request->cpu_sendbuf)), sendbuf_bytes);
    cudaMallocHost((void **)(&(request->cpu_recvbuf)), recvbuf_bytes);
#endif // free handled in MPIX_Request_free in persistent/persistent.c, must be cudaMallocHost
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
                                                                             const int *sources,
                                                                             MPIX_Comm *comm)
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
    request->comm = comm;
    
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
