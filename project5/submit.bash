#!/bin/bash
#SBATCH -J monteCarlo
#SBATCH -A CS475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o monteCarlo.out
#SBATCH -e monteCarlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joeparallel@oregonstate.edu
/usr/local/apps/cuda/cuda-10.1/bin/nvcc -o monteCarlo monteCarlo.cu
./monteCarlo