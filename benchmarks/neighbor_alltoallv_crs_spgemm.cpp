#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

#include <numeric>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    
    int n_iter = 10;
    if(num_procs > 1000)
	    n_iter = 100;

    if (argc == 1)
    {
        if (rank == 0) printf("Pass Matrix Filename as Command Line Arg!\n");
        MPI_Finalize();
        return 0;
    }
    char* filename = argv[1];

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);

    // Form Communication Package (A.send_comm, A.recv_comm)
    form_comm(A);

    MPIX_Comm* xcomm;

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    // Form MPIX_Comm initial communicator (should be cheap)
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int n_recvs, s_recvs, proc;
    std::vector<int> src(A.send_comm.n_msgs+1);
    std::vector<int> rdispls(A.send_comm.n_msgs+1);
    std::vector<int> recvcounts(A.send_comm.n_msgs+1);
    std::vector<long> recvvals(A.send_comm.size_msgs+1);

    std::vector<int> proc_count(num_procs, -1);
    std::vector<int> proc_displs(num_procs, -1);
    std::vector<long> orig_indices(A.send_comm.size_msgs+1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        proc_count[proc] = A.send_comm.counts[i];
        proc_displs[proc] = A.send_comm.ptr[i];
    }
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        orig_indices[i] = A.send_comm.idx[i] + A.first_col;

    s_recvs = -1;
    alltoallv_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
            A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            A.off_proc_columns.data(),
            &n_recvs, &s_recvs, src.data(), recvcounts.data(),
            rdispls.data(), MPI_LONG, recvvals.data(), xinfo, xcomm);
            
    // setup for SpMM - send all nonzero entries for row
    std::vector<int> send_row_sizes(A.send_comm.n_msgs);
    std::vector<int *> colIndicesPtrs(A.send_comm.n_msgs);
    int *sendColIndPtr = A.on_proc.col_idx.data();   // Amanda:  are these globall?
    std::vector<double *> sendDataVals(A.send_comm.n_msgs);
    double *sendDataPtr = A.on_proc.data.data();
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        if (i < A.send_comm.n_msgs - 1)
        {
            send_row_sizes[i] = A.send_comm.ptr[i + 1] - A.send_comm.ptr[i];
        }
        else
        {
            send_row_sizes[i] = A.on_proc.nnz - A.send_comm.ptr[i];  // Amanda:  is A.on_proc.nnz the local number of nonzeros?
        }
        
        colIndicesPtrs[i] = sendColIndPtr;
        sendColIndPtr += send_row_sizes[i];
        
        sendDataVals[i] = sendDataPtr;
        sendDataPtr += send_row_sizes[i];
    }

    // communicate  TODO
    // sends
    // char *sendbuf = (char *)malloc();  // buffer to pack
    // MPI_Request* send_reqs[A.send_comm.n_msgs];
    // int j = 0;
    // for (int i = 0; i < A.send_comm.n_msgs + 1; i++)
    // {
    //     if (A.send_comm.procs[i] != rank)
    //     {
    //         MPI_Isend();
    //         j++;
    //     }
    // }
    // recvs
    

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(xcomm);

    MPI_Finalize();
    return 0;
}
