#ifndef SPMV_MPI_H
#define SPMV_MPI_H

#include <vector>
#include "sparse_matrix.h"

// Partition the global CSR matrix across MPI processes (balanced by non-zeros)
void partition_and_scatter_matrix(const CSRMatrix& A_global, LocalCSRMatrix& A_local,
                                  int rank, int size, MPI_Comm comm);

// Perform one parallel SpMV and return the local compute time (ms)
double spmv_mpi(const LocalCSRMatrix& A_local, const std::vector<double>& x_global,
                std::vector<double>& y_global, int rank, int size, MPI_Comm comm);

#endif // SPMV_MPI_H
