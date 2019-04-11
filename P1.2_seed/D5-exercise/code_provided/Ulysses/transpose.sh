#!/bin/bash

TRIALS=10

rm -rf times/*.txt

gcc  -std=c99 transpose.c -o transpose.x
gcc  -std=c99 faster_transpose.c -o faster_transpose.x


for MATRIXSIZE in 1024 2048 4096 8192; do
	k=0

	#echo "FAST TRANSPOSE AVERAGE SPEEDUP">>faster_transpose_$MATRIXSIZE.txt
	#echo "n. blocks   Matrix dimension" >>faster_transpose_$MATRIXSIZE.txt

	rm transpose_$MATRIXSIZE.txt	

	./transpose.x $MATRIXSIZE >times/old_time_$MATRIXSIZE.txt	

	BLOCKSIZE=1
	while [ $BLOCKSIZE -le $MATRIXSIZE ]; do
		echo -n $k " " >>transpose_$MATRIXSIZE.txt
		echo -n $BLOCKSIZE " " >>transpose_$MATRIXSIZE.txt
		((k++))

		#esegui $trials volte 
		i=1
		while [ $i -le $TRIALS ]; do
			./faster_transpose.x $MATRIXSIZE $BLOCKSIZE >>times/new_times_$MATRIXSIZE$BLOCKSIZE.txt
			((i++))
		done

		#calcola la media sui trials
		cat times/new_times_$MATRIXSIZE$BLOCKSIZE.txt | awk -F ' ' -v N=$TRIALS '{ sum+=$6; }\
END{printf ("%.6f ", sum/N)}' >>transpose_$MATRIXSIZE.txt

#		./transpose.x $MATRIXSIZE >times/old_time_$MATRIXSIZE.txt
		cat times/old_time_$MATRIXSIZE.txt | awk -F ' ' '{print $6}' >>transpose_$MATRIXSIZE.txt

		#echo " ">>faster_transpose_$MATRIXSIZE.txt
	
		BLOCKSIZE=$(($BLOCKSIZE * 2))
	done
	
#	./transpose.x $MATRIXSIZE >times/old_time_$MATRIXSIZE.txt
#	cat times/old_time_$MATRIXSIZE.txt | awk -F ' ' '{print $6}' >prova_$MATRIXSIZE.txt

	echo " " >>transpose_$MATRIXSIZE.txt
done



