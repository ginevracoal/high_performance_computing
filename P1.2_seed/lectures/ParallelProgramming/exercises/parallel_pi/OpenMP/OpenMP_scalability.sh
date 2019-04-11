#!/bin/bash

TRIALS=10

module purge
module load openmpi

rm -rf times
mkdir times

gcc ../ser_pi.c -o ser_pi.x
gcc -fopenmp openmp_pi.c  -o openmp_pi.x

SER_OUT=times/ser_times.txt
AVG_SER_OUT=times/avg_ser_time.txt
CLEAN_OUT=times/clean_times.txt

echo serial execution

for i in $(seq 1 $TRIALS); do
	/usr/bin/time ./ser_pi.x 2>&1 | grep "elap" | awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}'>>$SER_OUT
done

cat $SER_OUT | awk -v N=$TRIALS '{ sum+=$1; }END{printf("%f ", sum/N)}' >$AVG_SER_OUT

echo parallel execution
echo "n. thds 		serial time 		parallel time">>$CLEAN_OUT

for threads in $(seq 1 24); do 

	PAR_OUT=times/times_$threads.txt;

	echo $threads "threads"
	echo -n $threads "			" >>$CLEAN_OUT	

	cat $AVG_SER_OUT >>$CLEAN_OUT

	echo -n "			" >>$CLEAN_OUT	

	export OMP_NUM_THREADS=$threads
	for i in $(seq 1 $TRIALS); do
		/usr/bin/time ./openmp_pi.x 2>&1 | grep "elap" | awk '{print($7)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print "		" $1}' >>$PAR_OUT
	done

	cat $PAR_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; }END{printf("%f ", sum/N)}' >>$CLEAN_OUT

	echo " " >>$CLEAN_OUT

done




