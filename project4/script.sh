#!/bin/bash

# number of threads
for size in 1000 2500 5000 7500 10000 25000 50000 75000 100000 250000 500000 750000 1000000 2500000 5000000 7500000 8000000
do
    echo ARRAYSIZE = $size
    g++  -c simd.cpp -o simd.o
    g++  -DARRAYSIZE=$size -o p4 p4.cpp simd.o  -lm  -fopenmp
    ./p4
done

rm simd.o
rm p4