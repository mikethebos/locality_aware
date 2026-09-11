#!/bin/sh
#Submit using flux batch <filename>

#flux: --job-name=allreduce_N64
#flux: --output='allreduce_N64.{{id}}.out'
#flux: --error='allreduce_N64.{{id}}.err'
#flux: -N 64
#flux: -l # Add task rank prefixes to each line of output.
#flux: --setattr=thp=always # Enable Transparent Huge Pages (THP)
#flux: -t 20
#flux: -q pbatch # other available queues: pdebug
#flux: -x
#flux: --setattr=gpumode=CPX
#flux: --conf=resource.rediscover=true

module load rocmcc/7.2.1-magic
module load cray-mpich/9.1.0

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

cd $HOME/locality_aware-mikethebos/build-with-hip-and-apu/benchmarks

flux run -N 64 --verbose --setopt=mpibind=verbose --tasks-per-node=24 --gpus-per-node=24 ./gpu_allreduce



