#!/bin/bash

TRIALS=10

module purge
module load openmpi

rm -rf times
mkdir times

gcc ../ser_pi.c -o ser_pi.x
mpicc mpi_pi.c -o mpi_pi.x

SER_OUT=times/ser_times.txt
AVG_SER_OUT=times/avg_ser_time.txt
CLEAN_OUT=times/clean_times.txt

echo serial execution

for i in $(seq 1 $TRIALS); do
	/usr/bin/time ./ser_pi.x 2>&1 | grep "elap" | awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}' >>$SER_OUT
done

cat $SER_OUT | awk -v N=$TRIALS '{ sum+=$1; }END{printf("%f ", sum/N)}' >$AVG_SER_OUT

echo parallel execution
echo "n. procs 		serial time 		parallel time">>$CLEAN_OUT

for procs in $(seq 1 24); do 

	PAR_OUT=times/times_$procs;

	echo $procs "procs"
	echo -n $procs "				" >>$CLEAN_OUT	

	cat $AVG_SER_OUT >>$CLEAN_OUT

	echo -n "			" >>$CLEAN_OUT
	
	for i in $(seq 1 $TRIALS); do
		/usr/bin/time mpirun -np $procs ./mpi_pi.x 2>&1 | grep "elap" | awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}' >>$PAR_OUT
	done 

	cat $PAR_OUT | awk -F ' ' -v N=$TRIALS '{ sum+=$1; }END{printf("%f ", sum/N)}' >>$CLEAN_OUT

	echo " " >>$CLEAN_OUT
done



