"""Convert a Matrix Market file to the binary CSR layout read by spmv_mpi.

Matrix file layout (native little-endian, as read by MPI/main_mpi.cpp):
    int32 n, int32 nnz, int32 row_ptr[n+1], int32 col_idx[nnz], float64 values[nnz]
Vector file layout:
    int32 n, float64 x[n]

Symmetric inputs (SuiteSparse stores only one triangle) are expanded to
the full matrix. Explicitly stored zeros are kept. The vector is all
ones by default, so y = A x is the row-sum vector; the script prints
y[0], y[n/2] and y[n-1] so they can be compared with the values that
spmv_mpi prints for verification.

Usage:
    python mtx_to_bin.py nlpkkt240.mtx data/nlpkkt240_matrix.bin data/nlpkkt240_vector.bin
    python mtx_to_bin.py A.mtx A.bin x.bin --vector random --seed 0

Memory: nlpkkt240 has 27,993,600 rows and ~774M stored entries after
symmetric expansion; the conversion needs tens of GB of RAM, so run it
on a large-memory node.
"""
import argparse
import sys

import numpy as np


def read_mtx(path):
    try:  # much faster on large files when installed
        import fast_matrix_market as fmm
        coo = fmm.mmread(path)
    except ImportError:
        import warnings
        from scipy.io import mmread
        with warnings.catch_warnings():  # spmatrix/sparray default change
            warnings.simplefilter("ignore", DeprecationWarning)
            coo = mmread(path)
    return coo.tocsr()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mtx")
    ap.add_argument("matrix_out")
    ap.add_argument("vector_out")
    ap.add_argument("--vector", choices=["ones", "random"], default="ones")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = read_mtx(args.mtx)
    A.sort_indices()
    n, m = A.shape
    if n != m:
        sys.exit(f"matrix must be square, got {n} x {m}")
    if A.nnz >= 2**31:
        sys.exit("nnz does not fit in int32; spmv_mpi uses 32-bit indices")

    with open(args.matrix_out, "wb") as fh:
        np.array([n, A.nnz], dtype="<i4").tofile(fh)
        A.indptr.astype("<i4").tofile(fh)
        A.indices.astype("<i4").tofile(fh)
        A.data.astype("<f8").tofile(fh)

    if args.vector == "ones":
        x = np.ones(n)
    else:
        x = np.random.default_rng(args.seed).standard_normal(n)
    with open(args.vector_out, "wb") as fh:
        np.array([n], dtype="<i4").tofile(fh)
        x.astype("<f8").tofile(fh)

    y = A @ x
    print(f"n = {n}, nnz = {A.nnz}")
    print(f"reference y[0] = {y[0]:.6f}")
    print(f"reference y[mid] = {y[n // 2]:.6f}")
    print(f"reference y[end] = {y[n - 1]:.6f}")


if __name__ == "__main__":
    main()
