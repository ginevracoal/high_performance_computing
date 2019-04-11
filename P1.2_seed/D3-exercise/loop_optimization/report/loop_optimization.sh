#!/bin/bash


# 2 arguments are requested:
#  (1) the number of particles to be used
#  (2) the number of grid points to be used

# PARTICLES=10000
# GRIDPTS=10

TRIALS=10

rm -rf times/*.txt

for PARTICLES in 2000 5000 10000; do 

	GRIDPTS=$(($PARTICLES/1000))

	CLEAN_OUT=times/avg_times_${PARTICLES}_${GRIDPTS}.txt

	echo "loop     avg ex time" >> $CLEAN_OUT
	echo " " >> $CLEAN_OUT

	i=0
	for i in $(seq 0 6); do

		# COMPILE
		gcc ../avoid_avoidable_loop${i}.c -o avoid_avoidable_loop${i}.x -lm

		OPT_OUT=times/avoid_avoidable_v${i}_times.txt
		
		echo "ex. times" >> $OPT_OUT

		# EXECUTE TRIALS
		k=1
		while [ $k -le $TRIALS ]; do

			# echo -n $PARTICLES " " $GRIDPTS " " >> report/times/loop_${i}_times.txt

			# calculate times
			./avoid_avoidable_loop${i}.x $PARTICLES $GRIDPTS | awk '{print($4)}' >> $OPT_OUT
		
			((k++))
		done
			

		# calculate average execution times

		echo -n " " $i "         " >> $CLEAN_OUT

		cat $OPT_OUT | awk -v N=$TRIALS '{ sum+=$1; }END{printf("%.6f ", sum/N)}' >> $CLEAN_OUT

		echo " " >> $CLEAN_OUT				

	done

done



