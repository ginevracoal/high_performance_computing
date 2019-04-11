#!/bin/bash

module purge
module load openmpi
module load mkl

rm -rf weak/*.txt strong/*.txt

N=65536

WEAK_OUT_CLEAN=weak/weak_scaling_$N.txt
STRONG_OUT_CLEAN=strong/strong_scaling_$N.txt


#======================================================================
echo Performing strong scaling

echo "n. procs 		|		avg ex. time " >>$STRONG_OUT_CLEAN

# serial execution
STRONG_OUT_SER=times/strong_scaling_${N}.txt

echo -n "1				  $N 				" >>$STRONG_OUT_CLEAN

echo -n 1 " " >>$STRONG_OUT_SER
/usr/bin/time ./pi.x $N 2>&1 | grep "elap" | awk '{print($3)}' \
		| awk -F ":" '{print $2}' | awk -F "elapsed" '{printf $1}' >> $STRONG_OUT_SER
echo " " >>$STRONG_OUT_SER


# avg execution time
cat $STRONG_OUT_SER | awk -F ' ' -v n=$TRIALS '{ sum+=$2; }\
		END{printf ("%.6f ", sum/n)}' >> $STRONG_OUT_CLEAN

echo " " >> $STRONG_OUT_CLEAN

# parallel execution
for procs in 2 4 8 16 20; do  

	STRONG_OUT_PAR=times/strong_scaling_${N}_${procs}.txt

	# number of processors and number of points per proc
	echo -n $procs "				 $(($N/$procs)) 				" >>$STRONG_OUT_CLEAN

	for i in $(seq 1 $TRIALS); do

		echo -n $procs " " >>$STRONG_OUT_PAR
		/usr/bin/time mpirun -np $procs ./mpi_pi.x $(($N/$procs)) 2>&1 | grep "elap" \
					| awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}' \
					>>$STRONG_OUT_PAR
	done

	# avg execution time
	cat $STRONG_OUT_PAR | awk -F ' ' -v n=$TRIALS '{ sum+=$2; }\
			END{printf ("%.6f ", sum/n)}' >> $STRONG_OUT_CLEAN

	echo " " >> $STRONG_OUT_CLEAN
done

#======================================================================
echo Performing weak scaling

echo "n. procs / n. pts per proc / avg exec time " >>$WEAK_OUT_CLEAN

# serial execution
WEAK_OUT_SER=times/weak_scaling_${N}_1.txt

echo -n "1				  $N 				" >>$WEAK_OUT_CLEAN

for i in $(seq 1 $TRIALS); do

	echo -n 1 " " >>$WEAK_OUT_SER
	/usr/bin/time ./pi.x $N 2>&1 | grep "elap" | awk '{print($3)}' \
			| awk -F ":" '{print $2}' | awk -F "elapsed" '{printf $1}' >> $WEAK_OUT_SER
	echo " " >>$WEAK_OUT_SER
done

# avg execution time
cat $WEAK_OUT_SER | awk -F ' ' -v n=$TRIALS '{ sum+=$2; }\
		END{printf ("%.6f ", sum/n)}' >> $WEAK_OUT_CLEAN

echo " " >> $WEAK_OUT_CLEAN

# parallel execution
for procs in 2 4 8 16 20; do

	WEAK_OUT_PAR=times/weak_scaling_${N}_${procs}.txt

	# number of processors and number of points per proc
	echo -n $procs "				$N				" >>$WEAK_OUT_CLEAN

	for i in $(seq 1 $TRIALS); do

		echo -n $procs " " >>$WEAK_OUT_PAR
		/usr/bin/time mpirun -np $procs ./mpi_pi.x $N 2>&1 | grep "elap"\
				| awk '{print($3)}' | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}'\
				>> $WEAK_OUT_PAR
	done

	# avg execution time
	cat $WEAK_OUT_PAR | awk -F ' ' -v n=$TRIALS '{ sum+=$2; }\
			END{printf ("%f ", sum/n)}' >> $WEAK_OUT_CLEAN

	echo " " >> $WEAK_OUT_CLEAN

done

echo " "

done
