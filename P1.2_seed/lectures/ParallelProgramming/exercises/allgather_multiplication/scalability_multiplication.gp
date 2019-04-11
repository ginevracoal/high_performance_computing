set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green" lw 2
set style line 2 lc rgb "blue" lw 2
set style line 3 lc rgb "yellow" lw 2
set style line 4 lc rgb "purple" lw 2

set output "matrix_multiplication_speedup_laptop.png"

set title 'Serial matrix multiplication vs parallel matrix multiplication'
set xlabel 'Block size'
set ylabel 'Speedup'

set logscale xy
set xrange [1:2048]
set yrange [0.5:1000]

plot "times/times_512.txt" using 1:($2/$3) ls 1\
	title	"Matrix size 512" with linespoints, \
		 "times/times_1024.txt" using 1:($2/$3) ls 2\
	title	"Matrix size 1024" with linespoints, \
		 "times/times_2048.txt" using 1:($2/$3) ls 3\
	title	"Matrix size 2048" with linespoints, \



