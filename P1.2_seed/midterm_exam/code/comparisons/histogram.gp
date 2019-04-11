set terminal png size 1200,800 enhanced font 'Helvetica,15' 
set grid
set key top right

# set style fill transparent solid 0.5
set style histogram rowstacked

set style line 1 lc rgb "green"
set style line 2 lc rgb "blue"

set title 'Comparison between OpenMP and CUDA jacobi codes'
set ylabel 'Execution time (s)'

set yrange [0.0001:]

set logscale y

set style data histogram
set boxwidth 0.5
set style fill solid #border
set style histogram clustered

set output "jacobi_histogram.png"
plot "times/avg_times_12000.txt" using 2:xtic(1) with boxes ls 2\
		title "Matrix size 12000",\
		 "times/avg_times_1200.txt" using 2:xtic(1) with boxes ls 1 \
		title "Matrix size 1200",\

# set output "jacobi_histogram_12000.png"
# plot "times/avg_times_12000.txt" using 2:xtic(1) with boxes ls 2\
# 		title "Matrix size 12000",