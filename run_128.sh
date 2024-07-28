#!/bin/bash
#SBATCH -A bif146
#SBATCH -o qdt_vit_196_128.o%j
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_CUSTOM_CACHE_DIR=$(pwd)
export HOME="/tmp/srun"

module load PrgEnv-cray/8.3.3
module load cce/15.0.0
module load rocm/5.7.0

srun python Main.py --batch_size 128
