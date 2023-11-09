#include "persistent_gpu.h"

#include "neighborhood/neighbor_persistent.h"
#include "collective/collective.h"

// no locality support
int neighbor_gpu_copy_cpu_start(MPIX_Request* request)
{
#ifdef GPU
    // only copy sendbuf on start
    cudaMemcpy(request->cpu_sendbuf, request->sendbuf, request->cpu_sendbuf_bytes, cudaMemcpyDeviceToHost);
   
    neighbor_start(request->sub_request);
#endif
}

// no locality support
int neighbor_gpu_copy_cpu_wait(MPIX_Request* request, MPI_Status* status)
{    
#ifdef GPU    
    neighbor_wait(request->sub_request, status);

    // only copy recvbuf after wait
    cudaMemcpy(request->recvbuf, request->cpu_recvbuf, request->cpu_recvbuf_bytes, cudaMemcpyHostToDevice);
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

void set_sub_request_in_neighbor_gpu_copy_cpu_request(MPIX_Request* outer_request, MPIX_Request* inner_request)
{
#ifdef GPU
    outer_request->sub_request = inner_request;
#endif
}