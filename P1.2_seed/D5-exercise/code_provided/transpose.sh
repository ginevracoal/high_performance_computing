#!/bin/bash

TRIALS=10

rm -rf times/*.txt

gcc -std=c11 transpose.c -o transpose.x
gcc -std=c11 faster_transpose.c -o faster_transpose.x


for MATRIXSIZE in 1024 2048 4096 8192; do
	k=1

	FIRST_OUT=times/time_$MATRIXSIZE.txt	
	# rm transpose_$MATRIXSIZE.txt	

	./transpose.x $MATRIXSIZE > $FIRST_OUT	

	BLOCKSIZE=1
	while [ $BLOCKSIZE -le $MATRIXSIZE ]; do

		BLOCK_OUT=times/block_time_${MATRIXSIZE}_${BLOCKSIZE}.txt
		CLEAN_OUT=times/clean_times_${MATRIXSIZE}.txt

		echo -n $k " " >>$CLEAN_OUT
		echo -n $BLOCKSIZE " " >>$CLEAN_OUT
		((k++))

		#esegui $trials volte 
		i=1
		while [ $i -le $TRIALS ]; do
			./faster_transpose.x $MATRIXSIZE $BLOCKSIZE >>$BLOCK_OUT
			((i++))
		done

		#calcola la media sui trials
		cat $BLOCK_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$6; }END{printf ("%.6f ", sum/N)}' >>$CLEAN_OUT

#		./transpose.x $MATRIXSIZE >$FIRST_OUT
		cat $FIRST_OUT | awk -F ' ' '{print $6}' >>$CLEAN_OUT

		#echo " ">>faster_$OUT
	
		BLOCKSIZE=$(($BLOCKSIZE * 2))
	done
	
#	./transpose.x $MATRIXSIZE >$FIRST_OUT
#	cat $FIRST_OUT | awk -F ' ' '{print $6}' >prova_$MATRIXSIZE.txt

	echo " " >>$CLEAN_OUT
done



