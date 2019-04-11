set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green" 
set style line 2 lc rgb "blue" 
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "Transpose_execution_time_laptop.png"

set title 'Basic transpose vs fast transpose execution time on my laptop'
set xlabel 'Block size'
set ylabel 'Elapsed time (s)'

set logscale xy
set xrange [1:8192]

set yrange [0.005:3]


plot "times/clean_times_1024.txt" using 2:3 ls 1 lw 2\
	title	"Fast on size 1024" with linespoints, \
		 "times/clean_times_1024.txt" using 2:4 ls 1 lw 1\
	title "Basic on size 1024" with lines, \
		 "times/clean_times_2048.txt" using 2:3 ls 2 lw 2\
	title	"Fast on size 2048" with linespoints, \
		 "times/clean_times_2048.txt" using 2:4 ls 2 lw 1\
	title "Basic on size 2048" with lines, \
		 "times/clean_times_4096.txt" using 2:3 ls 3 lw 2\
	title	"Fast on size 4096" with linespoints, \
		 "times/clean_times_4096.txt" using 2:4 ls 3 lw 1\
	title "Basic on size 4096" with lines, \
		 "times/clean_times_8192.txt" using 2:3 ls 4 lw 2\
	title	"Fast on size 8192" with linespoints, \
		 "times/clean_times_8192.txt" using 2:4 ls 4 lw 1\
	title "Basic on size 8192" with lines, \




