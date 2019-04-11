#!/bin/bash

rm -f serial_times.txt

gcc ../pi.c -o pi.x

N=1000

echo Serial implementation on increasing throws

echo "n. of throws / serial execution time" >> serial_times.txt
echo " " >> serial_times.txt

while  [ $N -le 1000000000 ]; do

	echo -n $N "				" >> serial_times.txt

	/usr/bin/time ./pi.x $N 2>&1 | grep "elap" | awk '{print($3)}'\
			 | awk -F ":" '{print $2}' | awk -F "elapsed" '{print $1}' >>serial_times.txt	
	
	N=$(($N*10))

done
