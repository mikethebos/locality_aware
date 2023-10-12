#!/bin/bash
#SBATCH --job-name gpu_alltoall4
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus 4
#SBATCH --gpu-bind none
#SBATCH --mem 12G
#SBATCH --time 00:10:00
#SBATCH --partition singleGPU

source /users/mcadams/cs499/switch-env_tf.sh
cd /users/mcadams/locality_aware-mikethebos/build/benchmarks

export OMP_NUM_THREADS=1
srun --ntasks 4 --nodes 4 --ntasks-per-node 1 --cpus-per-task 1 --gpus 4 --gpu-bind=none --cpu-bind=none ./gpu_alltoall

export OMP_NUM_THREADS=8
srun --ntasks 4 --nodes 4 --ntasks-per-node 1 --cpus-per-task 8 --gpus 4 --gpu-bind=none --cpu-bind=none ./gpu_threaded_alltoall
