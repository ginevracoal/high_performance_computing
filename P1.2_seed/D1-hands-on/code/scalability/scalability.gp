set terminal png size 1200,800 enhanced font 'Helvetica,15' 
set grid

set style line 1 lc rgb "green" lw 2
set style line 2 lc rgb "blue" lw 2
set style line 3 lc rgb "purple" lw 2

#=========================================================
set output "strong_scaling.png"

set title 'Strong scaling'
set xlabel 'Number of processors'
set ylabel 'Execution time (s)'
set key top right

unset logscale y
set xrange [0:20]
set yrange [0.1:21]

plot "times/strong_scaling_10000000.txt" using 1:3 ls 1\
	title	"10^7 points" with linespoints, \
		 "times/strong_scaling_100000000.txt" using 1:3 ls 2\
	title "10^8 points" with linespoints, \
		 "times/strong_scaling_1000000000.txt" using 1:3 ls 3\
	title "10^9 points" with linespoints,

#=========================================================
set output "weak_scaling.png"

set title 'Weak scaling'
set xlabel 'Number of processors'
set ylabel 'Execution time (s)'
set key center right

set xrange [0:20]
set yrange [0.1:26]

plot "times/weak_scaling_10000000.txt" using 1:3 ls 1\
	title	"10^7 points" with linespoints, \
		 "times/weak_scaling_100000000.txt" using 1:3 ls 2\
	title "10^8 points" with linespoints, \
		 "times/weak_scaling_1000000000.txt" using 1:3 ls 3\
	title "10^9 points" with linespoints,

#=========================================================
set output "speedup.png"

set title 'Strong scaling speedup'
set xlabel 'Number of processors'
set ylabel 'Speedup'

set key top left

set xrange [0:20]
set yrange [0.1:15]

plot "times/strong_scaling_10000000.txt" using 1:(00.20/$3) ls 1\
	title	"10^7 points" with linespoints, \
		 "times/strong_scaling_100000000.txt" using 1:(01.90/$3) ls 2\
	title "10^8 points" with linespoints, \
		 "times/strong_scaling_1000000000.txt" using 1:(18.59/$3) ls 3\
	title "10^9 points" with linespoints,