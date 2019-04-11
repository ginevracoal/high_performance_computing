#!/bin/bash

rm -rf cache_misses/*.txt

gcc -std=c99 faster_transpose.c -o faster_transpose.x

for MATRIXSIZE in 1024 2048 4096 8192; do

	BLOCKSIZE=1
	while [ $BLOCKSIZE -le $MATRIXSIZE ]; do
		echo -n $BLOCKSIZE " " >>cache_misses/cache_misses_$MATRIXSIZE.txt

		#echo perf stat -e L1-dcache-load-misses ./faster_transpose.x $MATRIXSIZE $BLOCKSIZE 2>> cache_misses/cache_misses_$MATRIXSIZE.txt
		perf stat -e  L1-dcache-load-misses ./faster_transpose.x $MATRIXSIZE $BLOCKSIZE 2>&1 \
		| awk '/L1-dcache-load-misses/ {print($1)}' | awk -F "." '{print $1 $2 $3}' >>cache_misses/cache_misses_$MATRIXSIZE.txt
		
		BLOCKSIZE=$(($BLOCKSIZE * 2))
	done

done
