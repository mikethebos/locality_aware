#ifndef MPI_ADVANCE_PERSISTENT_GPU_H
#define MPI_ADVANCE_PERSISTENT_GPU_H

#include "neighborhood/neighbor_persistent.h"

int neighbor_gpu_copy_cpu_start(MPIX_Request* request);

int neighbor_gpu_copy_cpu_wait(MPIX_Request* request, MPI_Status* status);

int neighbor_gpu_copy_cpu_threaded_start(MPIX_Request* request);

int neighbor_gpu_copy_cpu_threaded_wait(MPIX_Request* request, MPI_Status* status);

void init_neighbor_gpu_copy_cpu_request(MPIX_Request** request_ptr, const void* sendbuf, int sendbuf_bytes, void* recvbuf, int recvbuf_bytes);

void init_neighbor_gpu_copy_cpu_request_threaded(MPIX_Request** request_ptr, const void* sendbuf, int sendbuf_bytes,
                                                                             void* recvbuf, int recvbuf_bytes, int num_threads);

void set_sub_request_in_neighbor_gpu_copy_cpu_request(MPIX_Request* outer_request, MPIX_Request* inner_request);

#endif