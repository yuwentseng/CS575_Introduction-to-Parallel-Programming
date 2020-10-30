#!/bin/bash
#!/bin/bash
#SBATCH -J first
#SBATCH -A CS475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o first.out
#SBATCH -e first.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joeparallel@oregonstate.edu
# number of threads
for gs in 1024 256*1024 512*1024 1024*1024 4*1024*1024 8*1024*1024
do
    # number of subdivisions:
    for ls in 8 16 32 64 128 512 1024
    do
	let ng=$gs/$ls
        g++ -DNUM_ELEMENTS=$gs -DLOCAL_SIZE=$ls -DNUM_WORK_GROUPS=$ng -o first first.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
        ./first
    done
done
rm first
