#!/bin/bash
#SBATCH --job-name run_newsuitesparsembv_8nodes_map_rev
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH -N 8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --partition gpuA100x4
#SBATCH --time 00:40:00
#SBATCH --account=bebi-delta-gpu

module load cuda/12.4.0
module load openmpi/5.0.5+cuda

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $HOME/locality_aware-mikethebos/build/benchmarks

echo "Running Unthreaded Neighbor_Alltoallv Test:"
echo "Running Unthreaded Neighbor_Alltoallv Test:" 1>&2
export OMP_NUM_THREADS=1
mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_neighbor_alltoallv_matblockvec r

echo "Running Threaded Neighbor_Alltoallv Test with 2 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 2 threads:" 1>&2
export OMP_NUM_THREADS=2
mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r

echo "Running Threaded Neighbor_Alltoallv Test with 4 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 4 threads:" 1>&2
export OMP_NUM_THREADS=4
mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r

echo "Running Threaded Neighbor_Alltoallv Test with 8 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 8 threads:" 1>&2
export OMP_NUM_THREADS=8
mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r

echo "Running Threaded Neighbor_Alltoallv Test with 16 threads:"
echo "Running Threaded Neighbor_Alltoallv Test with 16 threads:" 1>&2
export OMP_NUM_THREADS=16
mpirun --map-by ppr:1:numa --bind-to numa --rank-by slot --display-map --display-allocation --report-bindings ./new_gpuaware_gpu_suitesparse_threaded_neighbor_alltoallv_matblockvec r
