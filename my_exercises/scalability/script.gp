#! /usr/bin/gnuplot

set title 'strong scalability'
set xlabel 'number of cores'
set ylabel 'speedup'
set xrange [1:20]
set yrange [0:10]
set autoscale

plot 'weak_scalability_clean.txt' 

