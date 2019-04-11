set terminal png size 900,600 enhanced font 'Verdana,12' 
set grid
set key top right

set style line 1 lc rgb "green"
set style line 2 lc rgb "blue"
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "opemp_jacobi.png"

set title 'Execution time using OpenMP on Cosilt'
set xlabel 'Number of threads'
set ylabel 'Execution time (min)'

set xrange [1:10]
set logscale y
set yrange [0.01:]

plot "times/times_1200.txt" using 1:2 ls 1 lw 2\
	title	"Matrix size 1200" with linespoints, \
		"times/times_12000.txt" using 1:2 ls 2 lw 2\
	title	"Matrix size 12000" with linespoints, \