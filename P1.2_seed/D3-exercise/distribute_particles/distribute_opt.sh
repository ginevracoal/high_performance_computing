#!/bin/bash


# 2 arguments are requested:
#  (1) the number of particles to be used
#  (2) the number of grid points to be used

# PARTICLES=10000
# GRIDPTS=100

TRIALS=10

rm -rf times/*.txt

for PARTICLES in 2000 5000 10000; do 

	GRIDPTS=$(($PARTICLES/1000))

	CLEAN_OUT=times/avg_times_${PARTICLES}_${GRIDPTS}.txt

	echo "loop     avg ex time 			speedup" >> $CLEAN_OUT
	echo " " >> $CLEAN_OUT

	i=0
	for i in $(seq 0 5); do

		# COMPILE
		gcc distribute_v${i}.c -o distribute_v${i}.x -lm

		OPT_OUT=times/distribute_v${i}_times.txt

		echo "ex. times" >> $OPT_OUT

		# EXECUTE TRIALS
		k=1
		while [ $k -le $TRIALS ]; do

			# echo -n $PARTICLES " " $GRIDPTS " " >> report/$OPT_OUT

			# calculate times
			./distribute_v${i}.x $PARTICLES $GRIDPTS 1 | awk '{print($3)}' >>$OPT_OUT	
		
			((k++))
		done
			
		# calculate average execution times
		echo -n " " $i "         " >> $CLEAN_OUT
		cat $OPT_OUT | awk -v N=$TRIALS '{ sum+=$1; }END{printf("%.6f ", sum/N)}' >>$CLEAN_OUT
		echo " " >> $CLEAN_OUT						

	done

done




