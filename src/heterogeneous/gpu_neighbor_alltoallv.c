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
    int ierr = gpu_aware_neighbor_alltoallv(MPIX_Neighbor_alltoallv_init,
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
        MPI_Request** request_ptr)
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

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < outdegree; i++)
    {
        sendcount += sendcounts[i];
    }
    for (int i = 0; i < indegree; i++)
    {
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

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

// int threaded_neighbor_alltoallv_nonblocking_init(const void* sendbuf,
//         const int sendcounts[],
//         const int sdispls[],
//         MPI_Datatype sendtype,
//         void* recvbuf,
//         const int recvcounts[],
//         const int rdispls[],
//         MPI_Datatype recvtype,
//         MPIX_Comm* comm,
//         MPI_Info info,
//         MPIX_Request** request_ptr)
// {
//     int ierr = 0;

//     int indegree, outdegree, weighted;
//     ierr += MPI_Dist_graph_neighbors_count(
//             comm->neighbor_comm, 
//             &indegree, 
//             &outdegree, 
//             &weighted);

//     int sources[indegree];
//     int sourceweights[indegree];
//     int destinations[outdegree];
//     int destweights[outdegree];
//     ierr += MPI_Dist_graph_neighbors(
//             comm->neighbor_comm, 
//             indegree, 
//             sources, 
//             sourceweights,
//             outdegree, 
//             destinations, 
//             destweights);

//     MPIX_Request* request;
//     init_neighbor_request(&request);

//     int send_bytes, recv_bytes;
//     MPI_Type_size(sendtype, &send_bytes);
//     MPI_Type_size(recvtype, &recv_bytes);

//     int sendcount = 0;
//     int recvcount = 0;
//     for (int i = 0; i < outdegree; i++)
//     {
//         sendcount += sendcounts[i];
//     }
//     for (int i = 0; i < indegree; i++)
//     {
//         recvcount += recvcounts[i];
//     }

//     int total_bytes_s = sendcount * send_bytes;
//     int total_bytes_r = recvcount * recv_bytes;

//     char* cpu_sendbuf;
//     char* cpu_recvbuf;
//     cudaMallocHost((void**)&cpu_sendbuf, total_bytes_s);
//     cudaMallocHost((void**)&cpu_recvbuf, total_bytes_r);

//     // Copy from GPU to CPU
//     ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

//     memcpy(cpu_recvbuf + (rdispls[rank] * recv_bytes),
//         cpu_sendbuf + (sdispls[rank] * send_bytes),
//         sendcounts[rank] * send_bytes);
 
// /*
//     int* ordered_sends = (int*)malloc(num_procs*sizeof(int));
//     int* ordered_recvs = (int*)malloc(num_procs*sizeof(int));
//     sort(num_procs, ordered_sends, sendcounts);
//     sort(num_procs, ordered_recvs, recvcounts);
// */
// #pragma omp parallel shared(cpu_sendbuf, cpu_recvbuf)
// {
//     int tag = 102944;
//     int send_proc, recv_proc;
//     int send_pos, recv_pos;

//     int n_msgs = num_procs - 1;
//     int thread_id = omp_get_thread_num();
//     int num_threads = omp_get_num_threads();

//     int n_msgs_per_thread = n_msgs / num_threads;
//     int extra_msgs = n_msgs % num_threads;
//     int thread_n_msgs = n_msgs_per_thread;
//     if (extra_msgs > thread_id)
//         thread_n_msgs++;

//     if (thread_n_msgs)
//     {
//         MPI_Request* requests = (MPI_Request*)malloc(2*thread_n_msgs*sizeof(MPI_Request));

//         int idx = thread_id + 1;
//         for (int i = 0; i < thread_n_msgs; i++)
//         {
//             send_proc = rank + idx;
//             if (send_proc >= num_procs)
//                 send_proc -= num_procs;
//             recv_proc = rank - idx;
//             if (recv_proc < 0)
//                 recv_proc += num_procs;
//             send_pos = sdispls[send_proc] * send_bytes;
//             recv_pos = rdispls[recv_proc] * recv_bytes;

//             MPI_Isend(cpu_sendbuf + send_pos,
//                     sendcounts[send_proc],
//                     sendtype,
//                     send_proc,
//                     tag,
//                     comm->global_comm,
//                     &(requests[i]));
//             MPI_Irecv(cpu_recvbuf + recv_pos,
//                     recvcounts[recv_proc],
//                     recvtype,
//                     recv_proc,
//                     tag,
//                     comm->global_comm,
//                     &(requests[thread_n_msgs + i]));
//             idx += num_threads;
//         }

//         MPI_Waitall(2*thread_n_msgs, requests, MPI_STATUSES_IGNORE);

//         free(requests);
//     }
// } 

// /*
//     free(ordered_sends);
//     free(ordered_recvs);
// */
//     ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

//     cudaFreeHost(cpu_sendbuf);
//     cudaFreeHost(cpu_recvbuf);

//     return ierr;
// }


