#!/bin/bash
#SBATCH -J third
#SBATCH -A CS475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o third.out
#SBATCH -e third.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joeparallel@oregonstate.edu
# number of threads
for gs in 1024 128*1024 256*1024 512*1024 1024*1024 2*1024*1024 4*1024*1024 8*1024*1024  
do
    g++ -DNUM_ELEMENTS=$gs -o third third.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
    ./third
done
rm third
