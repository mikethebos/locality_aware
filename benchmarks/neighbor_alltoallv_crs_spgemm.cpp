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
    std::vector<std::vector<int> *> globalColIndices(A.send_comm.n_msgs);
    std::vector<double *> sendDataVals(A.send_comm.n_msgs);
    std::vector<int> number_of_rows_to_send_to_proc(A.send_comm.n_msgs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        int proc = A.send_comm.procs[i];
        number_of_rows_to_send_to_proc[i] = A.send_comm.ptr[i + 1] - A.send_comm.ptr[i];
        for (int j = A.send_comm.ptr[i]; j < A.send_comm.ptr[i+1]; j++)
        {
            int row = A.send_comm.idx[j];
            // Send row to proc
            int on_proc_rowsize =  A.on_proc.rowptr[row+1] - A.on_proc.rowptr[row];
            int off_proc_rowsize = A.off_proc.rowptr[row+1] - A.off_proc.rowptr[row];
            int rowsize = on_proc_rowsize + off_proc_rowsize;

            // 1. Pack rowsize  MPI_Pack(row_size...)
            // 2. Pack on_proc_rowsize values of A.on_proc.col_idx starting at position A.on_proc.rowptr[row]
            for (int k = A.on_proc.rowptr[row]; k < A.on_proc.rowptr[row+1]; k++)
            {
                int col_idx = A.on_proc.col_idx[k];
                int global_col_idx = col_idx + A.first_col;
                // MPI_Pack(global_col_idx...)
            }
            // 3. Pack off_proc_rowsize values of A.off_proc.col_idx starting at position A.off_proc.rowptr[row]
            for (int k = A.off_proc.rowptr[row]; k < A.off_proc.rowptr[row+1]; k++)
            {
                // col_idx = A.off_proc.col_idx[k];
                // global_col_idx = A.off_proc_columns[col_idx];
                // MPI_Pack(global_col_idx, ...)
            }
            // 4. Pack on proc data
            // MPI_Pack(A.on_proc.data[A.on_proc.rowptr[row]], number of doubles = on_proc_rowsize)

            // 5. Pack off proc data
            // MPI_Pack(A.off_proc.data[A.off_proc.rowptr[row]], number of doubles = off_proc_rowsize)
        }


        int ptr = A.send_comm.idx[i];

	    
        int *localColIndicesPtr = &(A.on_proc.col_idx[ptr]);
        std::vector<int> *globalColIndicesVec = new std::vector<int>(send_row_sizes[i]);
        // compute global column indices
        for (int j = 0; j < send_row_sizes[i]; j++)
        {
            (*globalColIndicesVec)[j] = A.first_col + localColIndicesPtr[j];
        }
        globalColIndices[i] = globalColIndicesVec;
        
        sendDataVals[i] = &(A.on_proc.data[ptr]);
    }
    int num_send_vals = std::accumulate(send_row_sizes.begin(), send_row_sizes.end(), 0);

    // communicate
    // sends
    std::vector<char> packBuf(A.send_comm.n_msgs * sizeof(int) + num_send_vals * sizeof(int) + num_send_vals * sizeof(double));  // buffer to pack
    MPI_Request* send_reqs = (MPI_Request *)malloc(A.send_comm.n_msgs * sizeof(MPI_Request));
    char *packBufPtr = packBuf.data();
    int num_send_reqs = 0;
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        int proc = A.send_comm.procs[i];
        if (proc == rank) continue;
        int pos = 0;
        MPI_Pack(&(send_row_sizes[i]), 1, MPI_INT, packBufPtr, sizeof(int) + send_row_sizes[i] * sizeof(int) + send_row_sizes[i] * sizeof(double), &pos, MPI_COMM_WORLD);
        MPI_Pack(globalColIndices[i]->data(), send_row_sizes[i], MPI_INT, packBufPtr, sizeof(int) + send_row_sizes[i] * sizeof(int) + send_row_sizes[i] * sizeof(double), &pos, MPI_COMM_WORLD);
        MPI_Pack(sendDataVals.data(), send_row_sizes[i], MPI_DOUBLE, packBufPtr, sizeof(int) + send_row_sizes[i] * sizeof(int) + send_row_sizes[i] * sizeof(double), &pos, MPI_COMM_WORLD);
        MPI_Isend(packBufPtr, pos, MPI_PACKED, proc, xinfo->tag, MPI_COMM_WORLD, &(send_reqs[i]));
        packBufPtr += pos;
        num_send_reqs++;
    }
    
    // recvs  TODO
    
    
    MPI_Waitall(num_send_reqs, send_reqs, MPI_STATUSES_IGNORE);
    
    free(send_reqs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        delete globalColIndices[i];
    }

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(xcomm);

    MPI_Finalize();
    return 0;
}
