#!/bin/bash

rm -rf *.txt

# gcc ../jacobi.c -o jacobi.x
gcc -fopenmp openmp_jacobi.c -o openmp_jacobi.x

for MATRIXDIM in 1200 12000 ; do

	OUT=times/times_$MATRIXDIM.txt

	echo "n. of threads / parallel execution time" >>$OUT	
	echo " " >>$OUT
	
	for threads in 1 5 10 ; do 

		#print number of threads
		echo -n $threads "   " >>$OUT

		#calculate serial execution time
		# echo -n "$(./jacobi.x $MATRIXDIM 10 3 3)" | grep "elap" | awk '{printf($4)}' >>$OUT 
		#printf removes \n from output

		# echo -n "    " >>$OUT	

		#calculate parallel execution	time
		./openmp_jacobi.x $MATRIXDIM 10 3 3 $threads | grep "elap" \
		| awk '{print($4)}' >>$OUT

	done
done


