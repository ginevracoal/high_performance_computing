set terminal png size 1200,800 enhanced font 'Helvetica, 15' 
set grid
set key top left

set style line 1 lc rgb "green" 
set style line 2 lc rgb "blue" 
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "loop_optimization.png"

set xlabel 'loop optimization number'
set ylabel 'Speedup'

unset logscale y
unset logscale x
set xrange [0:6] 

set yrange [1:4]


# plot "times/avg_execution_times.txt" using 1:2 ls 1 lw 2\
# 	title	"10000 particles, 100 grid points" with linespoints, \

plot "../times/avg_times_2000_2.txt" using 1:(0.000690/$2) ls 1 lw 2\
	title	"2000 particles, 2 grid points" with linespoints, \
		 "../times/avg_times_5000_5.txt" using 1:(0.027844/$2) ls 2 lw 2\
	title	"5000 particles, 5 grid points" with linespoints, \
		 "../times/avg_times_10000_10.txt" using 1:(0.469080/$2) ls 3 lw 2\
	title	"10000 particles, 10 grid points" with linespoints, \




