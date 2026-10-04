/*
 * MPI Sparse Matrix-Vector Multiplication (SpMV)
 * Uses Balanced NNZ decomposition for Load Balancing.
 */

#include "sparse_matrix.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

// Helper to load only the specific part of the matrix required by this rank
bool load_partial_matrix(const std::string& filename, 
                         int start_row, int end_row, 
                         CSRMatrix& local_mat, 
                         long long total_nnz_offset, 
                         int total_rows_global) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    // Read Headers (Skip first 2 integers: n, nnz)
    int n_global, nnz_global;
    file.read(reinterpret_cast<char*>(&n_global), sizeof(int));
    file.read(reinterpret_cast<char*>(&nnz_global), sizeof(int));

    local_mat.n = end_row - start_row; // Local number of rows

    // 1. Read Row Pointers
    // We need to read row_ptr[start_row] to row_ptr[end_row + 1]
    int num_ptrs_to_read = local_mat.n + 1;
    local_mat.row_ptr.resize(num_ptrs_to_read);

    // Seek to the start_row's pointer location
    // Header (2 ints) + start_row * sizeof(int)
    std::streampos row_ptr_offset = 2 * sizeof(int) + (std::streampos)start_row * sizeof(int);
    file.seekg(row_ptr_offset);
    file.read(reinterpret_cast<char*>(local_mat.row_ptr.data()), num_ptrs_to_read * sizeof(int));

    // Determine how many NNZs this rank owns based on the pointers read
    // Note: The values in row_ptr are GLOBAL indices. 
    // We need to normalize them for the local values/col_idx arrays.
    int start_nnz_index = local_mat.row_ptr[0];
    int end_nnz_index = local_mat.row_ptr[local_mat.n];
    local_mat.nnz = end_nnz_index - start_nnz_index;

    // Normalize row pointers so they start at 0 for the local arrays
    for (int i = 0; i < num_ptrs_to_read; ++i) {
        local_mat.row_ptr[i] -= start_nnz_index;
    }

    // 2. Read Column Indices
    // Offset: Header (2 ints) + row_ptr (n+1 ints) + start_nnz_index (ints)
    local_mat.col_idx.resize(local_mat.nnz);
    std::streampos col_idx_offset = 2 * sizeof(int) + (std::streampos)(total_rows_global + 1) * sizeof(int) 
                                    + (std::streampos)start_nnz_index * sizeof(int);
    file.seekg(col_idx_offset);
    file.read(reinterpret_cast<char*>(local_mat.col_idx.data()), local_mat.nnz * sizeof(int));

    // 3. Read Values
    // Offset: Header + row_ptr + col_idx (all NNZ) + start_nnz_index (doubles)
    local_mat.values.resize(local_mat.nnz);
    std::streampos values_offset = 2 * sizeof(int) + (std::streampos)(total_rows_global + 1) * sizeof(int) 
                                   + (std::streampos)nnz_global * sizeof(int) 
                                   + (std::streampos)start_nnz_index * sizeof(double);
    file.seekg(values_offset);
    file.read(reinterpret_cast<char*>(local_mat.values.data()), local_mat.nnz * sizeof(double));

    file.close();
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string matrix_file = "data/nlpkkt240_matrix.bin";
    const std::string vector_file = "data/nlpkkt240_vector.bin";

    int n_global = 0;
    int nnz_global = 0;
    
    // --- PART 1: PARTITIONING (Load Balancing) ---
    // Only Rank 0 reads the row pointers to decide how to split the work
    
    std::vector<int> partition_bounds(size + 1); // Stores start row for each rank + end row

    if (rank == 0) {
        std::ifstream file(matrix_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file on Rank 0" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        file.read(reinterpret_cast<char*>(&n_global), sizeof(int));
        file.read(reinterpret_cast<char*>(&nnz_global), sizeof(int));

        // Read all row pointers to calculate load balance
        // This is ~112MB, manageable by Rank 0
        std::vector<int> global_row_ptr(n_global + 1);
        file.read(reinterpret_cast<char*>(global_row_ptr.data()), (n_global + 1) * sizeof(int));
        file.close();

        // Calculate split points based on NNZ
        long long total_nnz_long = nnz_global;
        long long target_nnz_per_rank = total_nnz_long / size;

        partition_bounds[0] = 0;
        int current_rank = 1;
        long long current_nnz_sum = 0;

        for (int i = 0; i < n_global; ++i) {
            int row_nnz = global_row_ptr[i+1] - global_row_ptr[i];
            current_nnz_sum += row_nnz;

            // If we exceeded target and haven't assigned all ranks yet
            if (current_nnz_sum >= target_nnz_per_rank && current_rank < size) {
                partition_bounds[current_rank] = i + 1;
                current_nnz_sum = 0; // Reset for next rank
                current_rank++;
            }
        }
        partition_bounds[size] = n_global; // Last rank takes the rest
    }

    // Broadcast metadata and partition info to all ranks
    MPI_Bcast(&n_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(partition_bounds.data(), size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine local range
    int my_start_row = partition_bounds[rank];
    int my_end_row = partition_bounds[rank + 1];
    int my_num_rows = my_end_row - my_start_row;

    // --- PART 2: PARALLEL I/O ---
    // Each rank loads only its assigned chunk
    
    CSRMatrix local_A;
    // Note: We pass n_global to help calculate file offsets
    if (!load_partial_matrix(matrix_file, my_start_row, my_end_row, local_A, 0, n_global)) {
        std::cerr << "Rank " << rank << " failed to load matrix chunk." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Load the input vector X (Replicated on all ranks)
    // X is ~224MB, fits in RAM.
    std::vector<double> x;
    if (!load_vector(vector_file, x)) { // Uses function from sparse_matrix.cpp
        if (rank == 0) std::cerr << "Failed to load vector." << std::endl;
        MPI_Finalize();
        return 1;
    }

    // --- PART 3: COMPUTATION ---
    
    std::vector<double> local_y(my_num_rows, 0.0);

    // Sync before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // The actual SpMV Kernel
    for (int i = 0; i < local_A.n; i++) {
        double sum = 0.0;
        // local_A.row_ptr is already normalized to start at 0
        for (int j = local_A.row_ptr[i]; j < local_A.row_ptr[i + 1]; j++) {
            sum += local_A.values[j] * x[local_A.col_idx[j]];
        }
        local_y[i] = sum;
    }

    double end_time = MPI_Wtime();
    double computation_time = end_time - start_time;

    // --- PART 4: METRICS & GATHER ---

    // Calculate Load Imbalance Metrics
    double min_time, max_time, avg_time;
    MPI_Reduce(&computation_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&computation_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double sum_time = 0;
    MPI_Reduce(&computation_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time = sum_time / size;

    // Gather results to Rank 0 for verification (Optional, but good for correctness)
    // Since rows are uneven, we need Gatherv
    std::vector<double> global_y;
    std::vector<int> recv_counts;
    std::vector<int> displs;

    if (rank == 0) {
        global_y.resize(n_global);
        recv_counts.resize(size);
        displs.resize(size);

        for(int i=0; i<size; i++) {
            recv_counts[i] = partition_bounds[i+1] - partition_bounds[i];
            displs[i] = partition_bounds[i];
        }
    }

    // Gatherv syntax: sendbuf, sendcount, type, recvbuf, recvcounts[], displs[], type, root, comm
    MPI_Gatherv(local_y.data(), my_num_rows, MPI_DOUBLE,
                global_y.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // --- PART 5: OUTPUT ---

    if (rank == 0) {
        double imbalance = (max_time - min_time) / min_time;
        
        std::cout << "Processors: " << size << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(6) << max_time << " s" << std::endl;
        std::cout << "Min Time: " << min_time << " s" << std::endl;
        std::cout << "Avg Time: " << avg_time << " s" << std::endl;
        std::cout << "Load Imbalance: " << imbalance << std::endl;

        // Verification Check (Sample)
        std::cout << "y[0] = " << global_y[0] << std::endl;
        std::cout << "y[mid] = " << global_y[n_global/2] << std::endl;
        std::cout << "y[end] = " << global_y[n_global-1] << std::endl;
    }

    // For the 64-process plot requirement: Print individual times
    // We allow every rank to print securely using ordered printing or gathering times
    // To keep output clean, we gather times to rank 0 and print there.
    std::vector<double> all_times;
    if (rank == 0) all_times.resize(size);
    MPI_Gather(&computation_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0 && size == 64) {
        std::cout << "\n--- Per-Process Times (for plotting) ---" << std::endl;
        for(int i=0; i<size; i++) {
            std::cout << "Rank " << i << ": " << all_times[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}