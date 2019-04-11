set terminal png size 1200,800 enhanced font 'Helvetica,15' 
set grid

set style line 1 lc rgb "green" 
set style line 2 lc rgb "blue" 
set style line 3 lc rgb "purple" 

#=========================================================
set output "strong_scaling.png"

set title 'Strong scaling'
set xlabel 'Number of processors'
set ylabel 'Execution time (s)'
set key top right

set logscale y
set xrange [0:24]
set yrange [7:120]

set xtics ("1" 1, "2" 2, "4" 4, "6" 6, "8" 8, "16" 16, "24" 24)

plot "strong_plot_16348.txt" using 1:2 ls 1 lw 2 title	"matrix dim 16348" with linespoints, \

#=========================================================
set output "speedup.png"

set title 'Strong scaling speedup'
set xlabel 'Number of processors'
set ylabel 'Speedup'

set key top left

set logscale y
set xrange [0:24]
set yrange [1:20]

set xtics ("1" 1, "2" 2, "4" 4, "6" 6, "8" 8, "16" 16, "24" 24)

plot "strong_plot_16348.txt" using 1:3 ls 2 lw 2 title	"matrix dim 16348" with linespoints, \
			x w line ls 1 title "theoretical peak"