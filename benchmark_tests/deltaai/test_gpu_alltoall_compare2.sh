#!/bin/bash
#SBATCH -J gpu_alltoall_compare2
#SBATCH -e gpu_alltoall_compare2.%j.err
#SBATCH -o gpu_alltoall_compare2.%j.out
#SBATCH -N 2
#SBATCH -G 8
#SBATCH --exclusive
#SBATCH -p ghx4
#SBATCH -t 01:45:00
#SBATCH --account=bebi-dtai-gh

module load craype-accel-nvidia90
module unload gcc-native
module load gcc-native/12
export MPICH_GPU_SUPPORT_ENABLED=1

cd $HOME/locality_aware-mikethebos/build/benchmarks

export OMP_NUM_THREADS=1
echo "Running Unthreaded Alltoall Test:"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=1 --gpus-per-task=1 ./gpu_alltoall_quick

export OMP_NUM_THREADS=2
echo "Running Threaded Alltoall Test with launches, 2 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=2 --gpus-per-task=1 ./thread_launches 2

export OMP_NUM_THREADS=2
echo "Running Threaded Alltoall Test without launches, 2 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=2 --gpus-per-task=1 ./no_thread_launches 2

export OMP_NUM_THREADS=4
echo "Running Threaded Alltoall Test with launches, 4 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 ./thread_launches 4

export OMP_NUM_THREADS=4
echo "Running Threaded Alltoall Test without launches, 4 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 ./no_thread_launches 4

export OMP_NUM_THREADS=8
echo "Running Threaded Alltoall Test with launches, 8 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=8 --gpus-per-task=1 ./thread_launches 8

export OMP_NUM_THREADS=8
echo "Running Threaded Alltoall Test without launches, 8 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=8 --gpus-per-task=1 ./no_thread_launches 8

export OMP_NUM_THREADS=10
echo "Running Threaded Alltoall Test with launches, 10 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=10 --gpus-per-task=1 ./thread_launches 10

export OMP_NUM_THREADS=10
echo "Running Threaded Alltoall Test without launches, 10 threads"
srun -N 2 --ntasks-per-node=4 --cpus-per-task=10 --gpus-per-task=1 ./no_thread_launches 10

#export OMP_NUM_THREADS=1

#echo "Running Multiproc Alltoall Test with 2 procs:"
#jsrun -n2 -r1 -a8 -c8 -g4 -dpacked -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed --print_placement=1 ./gpu_extraproc_alltoall

#echo "Running Multiproc Alltoall Test with 4 procs:"
#jsrun -n2 -r1 -a16 -c16 -g4 -dpacked -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed --print_placement=1 ./gpu_extraproc_alltoall

#echo "Running Multiproc Alltoall Test with 8 procs:"
#jsrun -n2 -r1 -a32 -c32 -g4 -dpacked -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed --print_placement=1 ./gpu_extraproc_alltoall

#echo "Running Multiproc Alltoall Test with 10 procs:"
#jsrun -n2 -r1 -a40 -c40 -g4 -dpacked -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed --print_placement=1 ./gpu_extraproc_alltoall
