#!/bin/bash
#SBATCH --job-name run_newsuitesparsembv_specific1_4nodes_map_rev
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --partition gpuA100x4
#SBATCH --time 01:00:00
#SBATCH --account=bebi-delta-gpu

module load cuda/12.4.0
module load openmpi/5.0.5+cuda

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $HOME/locality_aware-mikethebos/build/benchmarks

SPECIFIC_MATS="$HOME/test_matrices/CurlCurl_4.pm $HOME/test_matrices/nlpkkt80.pm $HOME/test_matrices/nd24k.pm $HOME/test_matrices/dgreen.pm $HOME/test_matrices/adaptive.pm"

echo "Running Unthreaded Neighbor_Alltoallv Test:"
echo "Running Unthreaded Neighbor_Alltoallv Test:" 1>&2
export OMP_NUM_THREADS=1
for MAT in $SPECIFIC_MATS; do
    mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_neighbor_alltoallv_matblockvec r $MAT
done

echo "Running Threaded Neighbor_Alltoallv Test with 2 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 2 threads:" 1>&2
export OMP_NUM_THREADS=2
for MAT in $SPECIFIC_MATS; do
    mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r $MAT
done

echo "Running Threaded Neighbor_Alltoallv Test with 4 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 4 threads:" 1>&2
export OMP_NUM_THREADS=4
for MAT in $SPECIFIC_MATS; do
    mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r $MAT
done

echo "Running Threaded Neighbor_Alltoallv Test with 8 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 8 threads:" 1>&2
export OMP_NUM_THREADS=8
for MAT in $SPECIFIC_MATS; do
    mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r $MAT
done

echo "Running Threaded Neighbor_Alltoallv Test with 16 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 16 threads:" 1>&2
export OMP_NUM_THREADS=16
for MAT in $SPECIFIC_MATS; do
    mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r $MAT
done
