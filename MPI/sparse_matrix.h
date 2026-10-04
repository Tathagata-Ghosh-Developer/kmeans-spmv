#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <string>
#include <mpi.h>

// CSR matrix structure
struct CSRMatrix {
    int n;                      // number of rows (and columns)
    int nnz;                    // number of non-zero entries
    std::vector<int> row_ptr;   // row_ptr size = n+1
    std::vector<int> col_idx;   // column indices of non-zeros
    std::vector<double> values; // non-zero values
};

// Local CSR matrix structure for each MPI process
struct LocalCSRMatrix {
    int row_start;              // global index of first row in this partition
    int local_n;                // number of rows in this partition
    int local_nnz;              // number of non-zeros in this partition
    std::vector<int> row_ptr;   // local row pointers (size = local_n+1)
    std::vector<int> col_idx;   // local column indices
    std::vector<double> values; // local non-zero values
};

// Load a CSR matrix from a binary file (format: n, nnz, row_ptr, col_idx, values)
bool load_matrix(const std::string& filename, CSRMatrix& matrix);

// Load a vector from a binary file (format: n, values)
bool load_vector(const std::string& filename, std::vector<double>& vec);

// Print basic information about a CSR matrix
void print_matrix_info(const CSRMatrix& matrix);

// Partition the full matrix (on rank 0) into balanced row blocks by non-zeros and scatter to all ranks
void partition_and_scatter_matrix(const CSRMatrix& A_global, LocalCSRMatrix& A_local,
                                  int rank, int size, MPI_Comm comm);

// Perform parallel SpMV: broadcast x, compute local y, and gather full y on rank 0
// Returns the local computation time (in milliseconds) on this process (excluding communication).
double spmv_mpi(const LocalCSRMatrix& A_local, const std::vector<double>& x_global,
                std::vector<double>& y_global, int rank, int size, MPI_Comm comm);

#endif // SPARSE_MATRIX_H
