#include "neighbor.h"
#include "neighbor_persistent.h"

int MPIX_Neighbor_alltoallw(
        const void* sendbuf,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        MPI_Datatype* sendtypes,
        void* recvbuf,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        MPI_Datatype* recvtypes,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    int ierr = MPIX_Neighbor_alltoallw_init(
            sendbuf,
            sendcounts,
            sdispls,
            sendtypes,
            recvbuf,
            recvcounts,
            rdispls,
            recvtypes,
            comm,
            MPI_INFO_NULL,
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

    return ierr;
}

int MPIX_Neighbor_alltoallv(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    int ierr = MPIX_Neighbor_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            recvtype,
            comm,
            MPI_INFO_NULL, 
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

    return ierr;
}

int MPIX_Neighbor_part_locality_alltoallv(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    int ierr = MPIX_Neighbor_part_locality_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            recvtype,
            comm,
            MPI_INFO_NULL, 
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

    return ierr;
}

int MPIX_Neighbor_locality_alltoallv(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        const long global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        const long global_rindices[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    int ierr = MPIX_Neighbor_locality_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            global_sindices,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            global_rindices,
            recvtype,
            comm,
            MPI_INFO_NULL, 
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

    return ierr;
}

// assume completely contiguous recvs and if multiple recvs from one process,
// order in *recvbufferPtr not deterministic by newSources
// free *recvbufferPtr after use
int neighbor_alltoallv_unk_anyorder_probe_nonblocking_send(const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void** recvbufferPtr,
        int* recvcounts,
        int* rdispls,
        MPI_Datatype recvtype,
        int* newSources,  // new sources in the order of msg received
        MPIX_Comm* comm)
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
    
    if (outdegree > 0)
    {
        total_bytes_s = (sdispls[outdegree - 1] + sendcounts[outdegree - 1]) * send_bytes;
    }

    int tag = 102944;
    int n_msgs_s = outdegree;
    int n_msgs_r = indegree;
    
    char *initIntRecvBuffer = NULL;
    int intRecvBufferSize = 1000 * recv_bytes; // good initial size to avoid reallocs?
    int cuda = 0;
#ifdef GPU
    cudaMemoryType ptr_send_type;
    cudaPointerAttributes mem;
    cudaPointerGetAttributes(&mem, sendbuffer);
    int cerr = cudaGetLastError();
    if (cerr == cudaErrorInvalidValue)
        ptr_send_type = cudaMemoryTypeHost;
#ifdef CUDART_VERSION
#if (CUDART_VERSION >= 11000)
    else if (mem.type == cudaMemoryTypeUnregistered)
        ptr_send_type = cudaMemoryTypeHost;
#endif
#endif
    else
        ptr_send_type = mem.type;
        
    if (ptr_send_type != cudaMemoryTypeHost)
    {
        cuda = 1;
        cudaMalloc((void **)&initIntRecvBuffer, intRecvBufferSize);
    }
    else
    {
        cudaMallocHost((void **)&initIntRecvBuffer, intRecvBufferSize);
    }
#else
    initIntRecvBuffer = (char *)malloc(intRecvBufferSize);
#endif

    char *sendBufferC = (char *)sendbuffer;

    MPI_Request send_reqs[n_msgs_s];
    for (int i = 0; i < n_msgs_s; i++)
    {
        ierr += MPI_Isend(&(sendBufferC[sdispls[i] * send_bytes]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(send_reqs[i]));
    }
    
    char *currIntRecvBuffer = initIntRecvBuffer;
    int currDispl = 0;
    for (int i = 0; i < n_msgs_r; i++)
    {
        MPI_Status status;
        ierr += MPI_Probe(MPI_ANY_SOURCE, tag, comm->neighbor_comm, &status);
        int proc = status.MPI_SOURCE;
        int count;
        ierr += MPI_Get_count(&status, recvtype, &count);
        
        if ((currDispl + count) * recv_bytes > intRecvBufferSize)
        {
            int newIntRecvBufferSize = (currDispl + count) * recv_bytes;  // should this be bigger to avoid reallocs? this is minimum
#ifdef GPU
            char *newInitIntRecvBuffer;
            if (cuda)
            {
                cudaMalloc((void **)&newInitIntRecvBuffer, newIntRecvBufferSize);
                cudaMemcpy((void *)newInitIntRecvBuffer, (void *)initIntRecvBuffer, intRecvBufferSize, cudaMemcpyDeviceToDevice);
                cudaFree((void *)initIntRecvBuffer);
            }
            else
            {
                cudaMallocHost((void **)&newInitIntRecvBuffer, newIntRecvBufferSize);
                cudaMemcpy((void *)newInitIntRecvBuffer, (void *)initIntRecvBuffer, intRecvBufferSize, cudaMemcpyHostToHost);
                cudaFreeHost((void *)initIntRecvBuffer);
            }
            initIntRecvBuffer = newInitIntRecvBuffer;
#else
            initIntRecvBuffer = (char *)realloc((void *)initIntRecvBuffer, newIntRecvBufferSize);
#endif
            intRecvBufferSize = newIntRecvBufferSize;
            currIntRecvBuffer = &(initIntRecvBuffer[currDispl * recv_bytes]);
        }
        
        recvcounts[i] = count;
        rdispls[i] = currDispl;
        newSources[i] = proc;
        
        ierr += MPI_Recv(currIntRecvBuffer,
                count,
                recvtype,
                proc,
                tag,
                comm->neighbor_comm,
                MPI_STATUS_IGNORE);
        
        currDispl += count;
        currIntRecvBuffer += count * recv_bytes;
    }
    
    ierr += MPI_Waitall(n_msgs_s, send_reqs, MPI_STATUSES_IGNORE);
    
    *recvbufferPtr = initIntRecvBuffer;
    
    return ierr;
}