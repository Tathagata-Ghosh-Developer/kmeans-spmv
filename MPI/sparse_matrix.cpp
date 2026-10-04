#include "sparse_matrix.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <bits/stdc++.h>

using namespace std;

// Load a CSR matrix from a binary file
bool load_matrix(const string& filename, CSRMatrix& matrix) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open matrix file " << filename << endl;
        return false;
    }
    // Read dimensions: n, nnz
    file.read(reinterpret_cast<char*>(&matrix.n), sizeof(int));
    file.read(reinterpret_cast<char*>(&matrix.nnz), sizeof(int));
    // Resize CSR arrays
    matrix.row_ptr.resize(matrix.n + 1);
    matrix.col_idx.resize(matrix.nnz);
    matrix.values.resize(matrix.nnz);
    // Read row_ptr, col_idx, values
    file.read(reinterpret_cast<char*>(matrix.row_ptr.data()), (matrix.n + 1) * sizeof(int));
    file.read(reinterpret_cast<char*>(matrix.col_idx.data()), matrix.nnz * sizeof(int));
    file.read(reinterpret_cast<char*>(matrix.values.data()), matrix.nnz * sizeof(double));
    file.close();
    cout << "Matrix loaded: " << matrix.n << "x" << matrix.n
              << ", nnz = " << matrix.nnz << endl;
    return true;
}

// Load a vector from a binary file
bool load_vector(const string& filename, vector<double>& vec) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open vector file " << filename << endl;
        return false;
    }
    int n;
    file.read(reinterpret_cast<char*>(&n), sizeof(int)); // read vector length
    vec.resize(n);
    file.read(reinterpret_cast<char*>(vec.data()), n * sizeof(double));
    file.close();
    cout << "Vector loaded: size = " << n << endl;
    return true;
}

// Print basic matrix information (dimensions, nonzeros, sparsity, memory)
void print_matrix_info(const CSRMatrix& matrix) {
    cout << "\n=== Matrix Information ===" << endl;
    cout << "Dimensions: " << matrix.n << " x " << matrix.n << endl;
    cout << "Non-zeros: " << matrix.nnz << endl;
    double density = (double)matrix.nnz / ((double)matrix.n * matrix.n);
    cout << "Sparsity: " << fixed << setprecision(2)
              << (1.0 - density) * 100.0 << "% non-zero" << endl;
    // Rough memory usage estimate
    double mem_bytes = matrix.row_ptr.size()*sizeof(int)
                     + matrix.col_idx.size()*sizeof(int)
                     + matrix.values.size()*sizeof(double);
    cout << "Memory usage (approx): " << fixed << setprecision(2)
              << mem_bytes / (1024.0 * 1024.0) << " MB" << endl;
}
