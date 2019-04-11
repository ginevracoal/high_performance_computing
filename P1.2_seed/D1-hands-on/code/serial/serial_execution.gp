set terminal png size 1200,800 enhanced font 'Helvetica,15' 
set grid
set key bottom right

set style line 1 lc rgb "green" lw 2
set style line 2 lc rgb "blue" lw 2
set style line 3 lc rgb "yellow" lw 2
set style line 4 lc rgb "purple" lw 2

set output "serial_execution.png"

set title 'Run time on Ulysses cluster'
set xlabel 'Number of points'
set ylabel 'Execution time (s)'

set format x "%.0sx10^%T"

set xrange [0:1000000000]

unset logscale y
set yrange [0:19]

plot "serial_times.txt" using 1:2 ls 1 with linespoints \
title " ",