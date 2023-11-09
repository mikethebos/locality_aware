#ifndef MPI_ADVANCE_GPU_ALLTOALLV_H
#define MPI_ADVANCE_GPU_ALLTOALLV_H

#include "collective/alltoallv.h"
#include "collective/collective.h"
#include "persistent/persistent.h"

typedef int (*neighbor_alltoallv_ftn)(const void*, const int*, const int*, MPI_Datatype, 
                                     void*, const int*, const int*, MPI_Datatype, MPIX_Comm*, MPI_Info, MPIX_Request**);

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
        MPIX_Request** request_ptr);

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
        MPI_Request** request_ptr);

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
        MPIX_Request** request_ptr);

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
        MPIX_Request** request_ptr);

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
//         MPIX_Request** request_ptr);

#endif
