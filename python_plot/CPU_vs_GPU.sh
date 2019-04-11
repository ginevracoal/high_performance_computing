#!/bin/bash
# I chose a default position (3,3) inside the matrix
 
TRIALS=5
ITERATIONS=10

rm -rf times/

module load openmpi/1.10.2/gnu/4.8.3 
module load cuda/7.5

# gcc ../jacobi.c -o jacobi.x
gcc -fopenmp ../OpenMP/openmp_jacobi.c -o openmp_jacobi.x
nvcc ../CUDA/cuda_jacobi.cu -o cuda_jacobi.x

for MATRIXSIZE in 1200 12000; do
	
	#=======================================
	# OpenMP code

	for threads in 1 5 10; do 

		OPENMP_OUT=times/times_openmp_$MATRIXSIZE_$threads.txt
		CUDA_OUT=times/times_cuda_$MATRIXSIZE_1024.txt

		# AVG_TIMES_OUT=times/times_$MATRIXSIZE.txt

		echo Running OpenMP code $TRIALS "times" with $threads threads and matrix size $MATRIXSIZE

		i=1
		while [ $i -le $TRIALS ]; do
			/usr/bin/time ./openmp_jacobi.x $MATRIXSIZE $ITERATIONS 3 3 $threads 2>&1 | grep "elap" \
										| awk '{print($3)}' | awk -F ":" '{print $2}'\
										| awk -F "elapsed" '{printf $1}' >>$OPENMP_OUT
			echo " ">>$OPENMP_OUT
			((i++))
		done

		# # number of threads
		# echo -n $threads " " >>$AVG_TIMES_OUT
		
		# # average over trials
		# cat $OPENMP_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; }\
		# 		END{print ("%.6f ", sum/N)}' >>$AVG_TIMES_OUT

	done


	#=======================================
	# CUDA code

	echo Running CUDA code $TRIALS "times" with matrix size $MATRIXSIZE

	i=1
	while [ $i -le $TRIALS ]; do
		/usr/bin/time ./cuda_jacobi.x $MATRIXSIZE $ITERATIONS 3 3 2>&1 | grep "elap" \
									 | awk '{print($3)}' | awk -F ":" '{print $2}' \
									 | awk -F "elapsed" '{printf $1}' >>$CUDA_OUT
		echo " ">>$CUDA_OUT
		((i++))
	done

	# # number of threads
	# echo -n "1024 " >>$AVG_TIMES_OUT
	
	# # average over trials
	# cat $CUDA_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; }\
	# 		END{print ("%.6f ", sum/N)}' >>$AVG_TIMES_OUT

done



