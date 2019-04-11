#!/bin/bash
# I chose a default position (3,3) inside the matrix
 
TRIALS=1
ITERATIONS=10

rm -rf times/*
# rm solution.dat

module load openmpi/1.10.2/gnu/4.8.3
module load cuda/7.5

# gcc ../jacobi.c -o jacobi.x

gcc -fopenmp ../OpenMP/openmp_jacobi.c -o openmp_jacobi.x
nvcc ../CUDA/cuda_jacobi.cu -o cuda_jacobi.x
nvcc ../CUDA/cuda_jacobi_init.cu -o cuda_jacobi_init.x
nvcc ../CUDA/cuda_jacobi_shared.cu -o cuda_jacobi_shared.x

for MATRIXSIZE in 1200 12000; do

	AVG_TIMES_OUT=times/avg_times_$MATRIXSIZE.txt

	echo "n. of threads / avg. ex. time " >>$AVG_TIMES_OUT
	
	for threads in 1 5 10; do 

		OPENMP_OUT=times/times_openmp_${MATRIXSIZE}_${threads}.txt
		CUDA_OUT=times/times_cuda_${MATRIXSIZE}.txt
		CUDA_INIT_OUT=times/times_cuda_init_${MATRIXSIZE}.txt	
		CUDA_S_OUT=times/times_cuda_shared_${MATRIXSIZE}.txt	

		#=======================================
		# OpenMP code

		echo Running OpenMP code $TRIALS "times" with $threads threads and matrix size $MATRIXSIZE

		i=1
		while [ $i -le $TRIALS ]; do
			./openmp_jacobi.x $MATRIXSIZE $ITERATIONS 3 3 $threads | grep "elap" \
										| awk '{print($4)}' >>$OPENMP_OUT

			# echo " ">>$OPENMP_OUT
			((i++))
		done

		# number of threads
		echo -n "\"" $threads " cores\"		" >>$AVG_TIMES_OUT
		
		# average over trials
		cat $OPENMP_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; } END {print(sum/N)}' >>$AVG_TIMES_OUT

	done

	#=======================================
	# CUDA codes

	echo Running CUDA codes $TRIALS "times" with matrix size $MATRIXSIZE

	i=1
	while [ $i -le $TRIALS ]; do
		./cuda_jacobi.x $MATRIXSIZE $ITERATIONS 3 3 2>&1 | grep "elap" \
									 | awk '{print($4)}' >>$CUDA_OUT
		./cuda_jacobi_init.x $MATRIXSIZE $ITERATIONS 3 3 2>&1 | grep "elap" \
									 | awk '{print($4)}' >>$CUDA_INIT_OUT
 		./cuda_jacobi_shared.x $MATRIXSIZE $ITERATIONS 3 3 2>&1 | grep "elap" \
									 | awk '{print($4)}' >>$CUDA_S_OUT
		# echo " ">>$CUDA_OUT
		((i++))
	done

	echo -n "\"CUDA\"		" >>$AVG_TIMES_OUT 
	# average over trials
	cat $CUDA_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; } END {print(sum/N)}' >>$AVG_TIMES_OUT	

	echo -n "\"CUDA v.2\"		" >>$AVG_TIMES_OUT 
	# average over trials
	cat $CUDA_INIT_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; } END {print(sum/N)}' >>$AVG_TIMES_OUT

	echo -n "\"CUDA v.3\"		" >>$AVG_TIMES_OUT 
	# average over trials
	cat $CUDA_S_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; } END {print(sum/N)}' >>$AVG_TIMES_OUT

done



