#!/bin/bash
#SBATCH --job-name=kmeans_openmp
#SBATCH --output=kmeans_output_%j.txt
#SBATCH --error=kmeans_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

make clean
make all

chmod +x run_experiments.sh

# Running the experiments (with K=20 clusters)
./run_experiments.sh 20
