#ifndef MPI_ADVANCE_PERSISTENT_GPU_H
#define MPI_ADVANCE_PERSISTENT_GPU_H

#include "neighborhood/neighbor_persistent.h"

int neighbor_gpu_copy_cpu_start(MPIX_Request* request);

int neighbor_gpu_copy_cpu_wait(MPIX_Request* request, MPI_Status* status);

int neighbor_gpu_copy_cpu_threaded_start(MPIX_Request* request);

int neighbor_gpu_copy_cpu_threaded_wait(MPIX_Request* request, MPI_Status* status);

void init_neighbor_gpu_copy_cpu_request(MPIX_Request** request_ptr, const void* sendbuf, int sendbuf_bytes, void* recvbuf, int recvbuf_bytes);

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
                                                                             MPIX_Comm *comm);

void set_sub_request_in_neighbor_gpu_copy_cpu_request(MPIX_Request* outer_request, MPIX_Request* inner_request);

#endif