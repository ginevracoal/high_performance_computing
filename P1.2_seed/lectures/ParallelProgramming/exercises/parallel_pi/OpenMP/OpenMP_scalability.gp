set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green"
set style line 2 lc rgb "blue"
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "Scalability_using_Openmp.png"

set title 'Speedup of pi using OpenMP on Ulysses'
set xlabel 'Number of threads'
set ylabel 'Speedup'

set xrange [1:20]

set yrange [0.00001:]

plot 'times.txt' using 1:($2/$3) notitle with linespoints lw 2

