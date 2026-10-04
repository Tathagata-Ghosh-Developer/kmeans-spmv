#!/bin/bash
#SBATCH --job-name=spmv_mpi_exp
#SBATCH --output=spmv_experiment_%j.log
#SBATCH --error=spmv_experiment_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
# Partition and node exclusions are site-specific; pass them at submit time,
# e.g. sbatch -p <partition> <script>.sh

# --- CONFIGURATION FIXES ---
# Force standard TCP usage and strictly disable the crashing 'ofi' (libfabric) driver
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_base_exclude=openib,ofi,usnic

# Input files: override with SPMV_MATRIX / SPMV_VECTOR (see scripts/fetch_matrix.sh)
SPMV_MATRIX=${SPMV_MATRIX:-data/nlpkkt240_matrix.bin}
SPMV_VECTOR=${SPMV_VECTOR:-data/nlpkkt240_vector.bin}

# Ensure clean build
make clean
make

echo "Starting Experiments with TCP enforcement..."
echo "============================================"

# Loop through process counts: 1 (seq), then powers of 2
for p in 1 2 4 8 16 32 64 128; do
    echo "Running with $p processes..."
    echo "---------------------------------" >> results_P${p}.txt
    
    # Run 5 times
    for i in {1..5}; do
        echo "  Run $i/5 for $p procs"
        # The environment variables exported above will automatically apply to mpiexec
        mpiexec -np $p ./spmv_mpi "$SPMV_MATRIX" "$SPMV_VECTOR" >> results_P${p}.txt 2>&1
    done
    echo "Finished $p processes."
    echo "-----------------------"
done

echo "All experiments completed."