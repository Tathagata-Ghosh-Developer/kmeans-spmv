#!/bin/bash
#SBATCH --job-name=spmv_extreme
#SBATCH --output=spmv_extreme_%j.log
#SBATCH --error=spmv_extreme_%j.err
#SBATCH --nodes=6                   # Use ALL 8 compute nodes
#SBATCH --ntasks=288                # Request all available physical cores
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
# Partition and node exclusions are site-specific; pass them at submit time,
# e.g. sbatch -p <partition> <script>.sh

# --- SPECIAL CONFIGURATION FOR HIGH CONCURRENCY ---
# 1. Force standard Point-to-Point communication (ob1)
export OMPI_MCA_pml=ob1

# 2. Enable Shared Memory (vader) + TCP
#    Crucial for 128+ ranks to prevent socket exhaustion on the nodes.
export OMPI_MCA_btl=self,vader,tcp

# 3. Ban broken high-speed drivers
export OMPI_MCA_btl_base_exclude=openib,ofi,usnic

# 4. Ensure pure Ethernet interfaces are used
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0

echo "=== Starting Extreme Scaling Experiments (128, 256, 512, 1024) ==="
echo "Physical Cores Available: 288"
echo "Configuration: --oversubscribe enabled for >288 processes"

# Input files: override with SPMV_MATRIX / SPMV_VECTOR (see scripts/fetch_matrix.sh)
SPMV_MATRIX=${SPMV_MATRIX:-data/nlpkkt240_matrix.bin}
SPMV_VECTOR=${SPMV_VECTOR:-data/nlpkkt240_vector.bin}

make clean
make

for p in 128 256 512 1024; do
    echo "Running with $p processes..."
    echo "---------------------------------" >> results_P${p}.txt
    
    for i in {1..5}; do
        echo "  Run $i/5 for $p ranks..."
        
        # --oversubscribe: Allows >288 ranks on 288 cores
        # --map-by node: Distributes ranks round-robin across nodes to spread memory load
        mpiexec --oversubscribe --map-by node -np $p ./spmv_mpi "$SPMV_MATRIX" "$SPMV_VECTOR" >> results_P${p}.txt 2>&1
    done
    echo "Finished $p processes."
    echo "-----------------------"
done

echo "Extreme experiments completed."
