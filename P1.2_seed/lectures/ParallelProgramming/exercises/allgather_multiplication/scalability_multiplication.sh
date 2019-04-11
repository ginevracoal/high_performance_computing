#!/bin/bash

module load openmpi/1.8.3/gnu/4.8.3

#MATRIXSIZE=1024

module load openmpi/1.8.3/gnu/4.8.3

rm -rf times/*.txt

gcc serial_multiplication.c -o serial_multiplication.x
mpicc parallel_multiplication.c -o parallel_multiplication.x

for MATRIXSIZE in 512 1024 2048; do

	SER_OUT=times/serial_time.txt
	CLEAN_OUT=times/times_$MATRIXSIZE.txt

	#calculate serial execution time
	/usr/bin/time ./serial_multiplication.x $MATRIXSIZE 2>&1 | grep "elap" | awk '{print($3)}' \
	| awk -F ":" '{print $2}' | awk -F "elapsed" '{printf $1}' >$SER_OUT

	echo "process / serial ex. time / parallel ex. time" >>$CLEAN_OUT

	procs=1;
	while [ $procs -le $MATRIXSIZE ]; do
	# for procs in 1 2 4 8 16 20 ; do 

		#print the number of processes
		echo -n $procs " " >>$CLEAN_OUT

		cat $SER_OUT>>$CLEAN_OUT

		echo -n " " >>$CLEAN_OUT

		#calculate parallel execution	time
		/usr/bin/time  mpirun -np $procs ./parallel_multiplication.x $MATRIXSIZE 2>&1 | grep "elap" \
		| awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}' >>$CLEAN_OUT

		procs=$(($procs * 2))

	done

done


