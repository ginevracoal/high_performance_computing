set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green" lw 2
set style line 2 lc rgb "blue" lw 2
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "Scalability_using_MPI.png"

set title 'Speedup of pi using MPI on my laptop'
set xlabel 'Number of processes'
set ylabel 'Speedup'

set xrange [1:20]

set yrange [0.7:]

plot 'mpi_times.txt' using 1:($2/$3) notitle with linespoints ls 2

