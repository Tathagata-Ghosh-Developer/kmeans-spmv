#!/bin/bash
# Download nlpkkt240 from the SuiteSparse Matrix Collection and convert it
# to the binary CSR files that MPI/spmv_mpi reads.
#
#   Matrix page: https://sparse.tamu.edu/Schenk/nlpkkt240
#   27,993,600 x 27,993,600, symmetric indefinite KKT matrix,
#   760,648,352 nonzeros (774,472,352 stored entries incl. explicit zeros).
#   The .tar.gz is about 1.3 GB; the converted binary is about 9 GB.
#
# Usage: scripts/fetch_matrix.sh [output dir]   (default: MPI/data)
# Then:  cd MPI && mpiexec -np 4 ./spmv_mpi data/nlpkkt240_matrix.bin data/nlpkkt240_vector.bin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:-"$SCRIPT_DIR/../MPI/data"}
URL="https://sparse.tamu.edu/MM/Schenk/nlpkkt240.tar.gz"
mkdir -p "$OUT"
cd "$OUT"

if [ ! -f nlpkkt240.tar.gz ]; then
    curl -L --fail -o nlpkkt240.tar.gz "$URL"
fi
if [ ! -f nlpkkt240/nlpkkt240.mtx ]; then
    tar -xzf nlpkkt240.tar.gz
fi

python3 "$SCRIPT_DIR/mtx_to_bin.py" nlpkkt240/nlpkkt240.mtx \
    nlpkkt240_matrix.bin nlpkkt240_vector.bin
