#!/bin/bash

module load openmpi/1.8.3/gnu/4.9.2

rm -rf stream.txt

# Makefile

OUT=stream.txt

echo "n. thds  			 MB/s socket 0					MB/s socket 1" >>$OUT

for i in $(seq 1 10); do

	echo -n $i "				" >>$OUT

	OMP_NUM_THREADS=$i numactl --cpunodebind 0 --membind 0 ./stream_omp.x \
	| grep "Triad:" | awk '{printf $2}' >>$OUT

	echo -n "				" >>$OUT	

	OMP_NUM_THREADS=$i numactl --cpunodebind 0 --membind 1 ./stream_omp.x \
	| grep "Triad:" | awk '{print $2}' >>$OUT

done