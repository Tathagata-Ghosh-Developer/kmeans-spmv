#include "spmv_mpi.h"
#include <mpi.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace std;

// Helper to perform local computation
void spmv_local_compute(const LocalCSRMatrix& A, const std::vector<double>& x,
                        std::vector<double>& y_local) {
    // Resize local result vector to match the number of rows this process owns
    y_local.assign(A.local_n, 0.0);
    
    for (int i = 0; i < A.local_n; i++) {
        double sum = 0.0;
        // A.row_ptr is already adjusted to be 0-based for the local partition
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            sum += A.values[j] * x[A.col_idx[j]];
        }
        y_local[i] = sum;
    }
}

double spmv_mpi(const LocalCSRMatrix& local_A, const std::vector<double>& x_global,
              std::vector<double>& y_global, int rank, int size, MPI_Comm comm) {

    // 1. Broadcast the input vector x. 
    // (Cast away const because MPI_Bcast conceptually modifies the buffer on receivers)
    MPI_Bcast(const_cast<double*>(x_global.data()), (int)x_global.size(), MPI_DOUBLE, 0, comm);

    // 2. Perform local computation (Timed)
    // We strictly measure the dot-product loop time for load imbalance analysis
    std::vector<double> y_local;
    double t_start = MPI_Wtime();
    spmv_local_compute(local_A, x_global, y_local);
    double t_end = MPI_Wtime();
    double local_comp_time = (t_end - t_start) * 1000.0; // ms

    // 3. Gather results.
    // We need to know how many elements each process sends to rank 0.
    // In a real app, these counts would be cached, but here we gather them.
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int local_n = local_A.local_n;
    MPI_Gather(&local_n, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
        displs = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Gather sub-vectors into the global y vector on rank 0
    MPI_Gatherv(y_local.data(), local_n, MPI_DOUBLE,
                y_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    return local_comp_time;
}


void partition_and_scatter_matrix(const CSRMatrix& A_global, LocalCSRMatrix& A_local,
                                  int rank, int size, MPI_Comm comm) {

    // Arrays to hold the partition scheme (valid only on Rank 0)
    std::vector<int> send_row_counts(size);
    std::vector<int> send_row_starts(size);
    std::vector<int> send_nnz_counts(size);

    if (rank == 0) {
        // --- Rank 0: Load Balancing Logic ---
        // We want to divide NNZ evenly, not Rows.
        long long total_nnz = A_global.nnz;
        long long target_nnz = total_nnz / size;
        
        int current_rank = 0;
        int current_row_start = 0;
        long long current_rank_nnz = 0;

        for (int row = 0; row < A_global.n; ++row) {
            int row_nnz = A_global.row_ptr[row + 1] - A_global.row_ptr[row];
            current_rank_nnz += row_nnz;

            // Check if we have filled the current rank's quota
            // We move to next rank if:
            // 1. We are not the last rank
            // 2. AND we have reached/exceeded the target NNZ
            if (current_rank < size - 1 && current_rank_nnz >= target_nnz) {
                // Finish current rank
                send_row_counts[current_rank] = (row + 1) - current_row_start;
                send_row_starts[current_rank] = current_row_start;
                send_nnz_counts[current_rank] = current_rank_nnz;

                // Prepare for next rank
                current_rank++;
                current_row_start = row + 1;
                current_rank_nnz = 0; 
            }
        }
        
        // Assign whatever is left to the last rank
        send_row_counts[size - 1] = A_global.n - current_row_start;
        send_row_starts[size - 1] = current_row_start;
        send_nnz_counts[size - 1] = A_global.nnz - A_global.row_ptr[current_row_start];

        std::cout << "Rank 0: Partitioned matrix by non-zeros for load balancing." << std::endl;
    }

    // --- Distribute Partition Info ---
    // 1. Distribute row counts
    MPI_Scatter(send_row_counts.data(), 1, MPI_INT, 
                &A_local.local_n, 1, MPI_INT, 0, comm);
    
    // 2. Distribute global start row indices
    MPI_Scatter(send_row_starts.data(), 1, MPI_INT, 
                &A_local.row_start, 1, MPI_INT, 0, comm);

    // 3. Distribute NNZ counts
    MPI_Scatter(send_nnz_counts.data(), 1, MPI_INT, 
                &A_local.local_nnz, 1, MPI_INT, 0, comm);

    // --- Allocate Local Memory ---
    A_local.row_ptr.resize(A_local.local_n + 1);
    A_local.col_idx.resize(A_local.local_nnz);
    A_local.values.resize(A_local.local_nnz);

    // --- Distribute Matrix Data ---
    // We use MPI_Scatterv for values and col_idx (usually large)
    // We use MPI_Send/Recv loop for row_ptr because it needs re-indexing (offset adjustment)
    
    std::vector<int> sc_counts;
    std::vector<int> sc_displs;
    
    if (rank == 0) {
        sc_counts = send_nnz_counts;
        sc_displs.resize(size);
        sc_displs = 0;
        for(int i=1; i<size; ++i) 
            sc_displs[i] = sc_displs[i-1] + sc_counts[i-1];
    }

    // Scatter Values
    MPI_Scatterv(A_global.values.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE,
                 A_local.values.data(), A_local.local_nnz, MPI_DOUBLE, 0, comm);

    // Scatter Column Indices
    MPI_Scatterv(A_global.col_idx.data(), sc_counts.data(), sc_displs.data(), MPI_INT,
                 A_local.col_idx.data(), A_local.local_nnz, MPI_INT, 0, comm);

    // Distribute Row Pointers (manual send to adjust offsets)
    if (rank == 0) {
        // Rank 0 sets its own
        int r0_start = send_row_starts;
        int r0_offset = A_global.row_ptr[r0_start];
        for(int i=0; i <= send_row_counts; ++i) {
            A_local.row_ptr[i] = A_global.row_ptr[r0_start + i] - r0_offset;
        }

        // Send to others
        for (int r = 1; r < size; ++r) {
            int r_rows = send_row_counts[r];
            int r_start = send_row_starts[r];
            int r_offset = A_global.row_ptr[r_start]; // The start index of NNZ for this chunk
            
            std::vector<int> temp_ptr(r_rows + 1);
            for(int i=0; i <= r_rows; ++i) {
                // Re-index so the local row_ptr starts at 0
                temp_ptr[i] = A_global.row_ptr[r_start + i] - r_offset;
            }
            MPI_Send(temp_ptr.data(), r_rows + 1, MPI_INT, r, 77, comm);
        }
    } else {
        // Receive Row Pointers
        MPI_Recv(A_local.row_ptr.data(), A_local.local_n + 1, MPI_INT, 0, 77, comm, MPI_STATUS_IGNORE);
    }
}