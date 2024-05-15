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
   
    int n_msgs = request->sub_request->global_n_msgs;
    
    if (n_msgs)
    {
        int n_threads = request->num_threads;
        int n_msgs_per_thread = n_msgs / n_threads;
        int extra_msgs = n_msgs % n_threads;
        MPI_Request* requests_arr = request->sub_request->global_requests;
#pragma omp parallel num_threads(n_threads) shared(requests_arr) reduction(+:ret)
{
    int thread_id = omp_get_thread_num();
    int thread_n_msgs = n_msgs_per_thread;
    if (extra_msgs > thread_id)
        thread_n_msgs++;
        
    if (thread_n_msgs)
    {
        int baseIdx = thread_n_msgs * thread_id;
        if (extra_msgs <= thread_id)
        {
            baseIdx += extra_msgs;
        }
        for (int idx = baseIdx; idx < baseIdx + thread_n_msgs; ++idx)
        {
            ret += MPI_Start(&(requests_arr[idx]));
        }
    }
}

    }
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
        int n_threads = request->num_threads;
        int n_msgs_per_thread = n_msgs / n_threads;
        int extra_msgs = n_msgs % n_threads;
        MPI_Request* requests_arr = request->sub_request->global_requests;
#pragma omp parallel num_threads(n_threads) shared(requests_arr) reduction(+:ret)
{
    int thread_id = omp_get_thread_num();
    int thread_n_msgs = n_msgs_per_thread;
    if (extra_msgs > thread_id)
        thread_n_msgs++;
        
    if (thread_n_msgs)
    {
        int baseIdx = thread_n_msgs * thread_id;
        if (extra_msgs <= thread_id)
        {
            baseIdx += extra_msgs;
        }
        for (int idx = baseIdx; idx < baseIdx + thread_n_msgs; ++idx)
        {
            ret += MPI_Wait(&(requests_arr[idx]), MPI_STATUS_IGNORE);
        }
    }
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
                                                                             void* recvbuf, int recvbuf_bytes, int num_threads)
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
