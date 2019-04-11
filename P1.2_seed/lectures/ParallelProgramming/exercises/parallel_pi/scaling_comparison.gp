set terminal png size 1200,800 enhanced font 'Helvetica,15' 
set grid
set key top left

set style line 1 lc rgb "green"
set style line 2 lc rgb "blue"
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "OpenMP_vs_MPI_scaling.png"

set title 'parallel pi approximations'
set xlabel 'Number of threads'
set ylabel 'Speedup'

set xrange [1:24]
set yrange [1:]

set xtics 1

plot "OpenMP/times/clean_times.txt" using 1:($2/$3) ls 1 lw 2\
		title	"OpenMP" with linespoints, \
		 "MPI/times/clean_times.txt" using 1:($2/$3) ls 2 lw 2\
		title	"MPI" with linespoints, \

