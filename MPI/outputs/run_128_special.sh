#!/bin/bash
#SBATCH --job-name=spmv_128_fix
#SBATCH --output=results_P128.txt
#SBATCH --error=spmv_128_error.txt
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
# Partition and node exclusions are site-specific; pass them at submit time,
# e.g. sbatch -p <partition> <script>.sh

# 1. Force standard Point-to-Point communication
export OMPI_MCA_pml=ob1

# 2. CRITICAL FIX: Add 'vader' for high-speed Shared Memory within nodes
#    'self'  = Process talks to itself
#    'vader' = Processes on the same node talk via RAM (Fast, no sockets)
#    'tcp'   = Processes on different nodes talk via Ethernet
export OMPI_MCA_btl=self,vader,tcp

# 3. Explicitly ban the broken high-speed drivers to prevent Bus Errors
export OMPI_MCA_btl_base_exclude=openib,ofi,usnic

# 4. Exclude virtual network interfaces to ensure real Ethernet is used
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0

echo "=== Running 128-Process Special Configuration ==="
echo "Configuration: PML=ob1, BTL=self,vader,tcp"

# Run 5 times as required
for i in {1..5}; do
    echo "Run $i/5..."
    mpiexec -np 128 ./spmv_mpi
done
