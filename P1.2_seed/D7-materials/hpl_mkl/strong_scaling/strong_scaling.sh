#!/bin/bash

N=16348

OUT_clean=strong_clean_$N.txt
OUT_plot=strong_plot_$N.txt
rm $OUT_clean
rm $OUT_plot

echo "		N		NB		P		Q		Time		Gflops">>$OUT_clean
echo "nprocs		ex. time 		speedup">>$OUT_plot

for nprocs in 1 2 4 6 8 16 24; do

	IN=hpl_${N}_${nprocs}.txt

	cat $IN | awk '$1=="T/V" {for(i=1; i<=2; i++){getline;for(j=2; j<=7; j++)printf($j "  ")};print(" ")}' | awk 'NR>=2{print}' >>$OUT_clean 


done

cat $OUT_clean | awk 'NR>=2{print($3*$4 "		" $5 "		" 114.26/$5)}' >>$OUT_plot



