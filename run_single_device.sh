#!/bin/bash
#SBATCH -A bif146
#SBATCH -o qdt_vit_196_128.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --mail-user=dongbochen1218@icloud.com
#SBATCH --mail-type=END

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

source export_ddp_envs.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0
# exec
srun -N 1 -n 8 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Main.py \
        --batch_size 256