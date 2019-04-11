set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green"
set style line 2 lc rgb "blue"
set style line 3 lc rgb "yellow"
set style line 4 lc rgb "purple"

set output "stream.png"

set title 'Stream benchmark results on Ulysses'
set xlabel 'Number of threads'
set ylabel 'Bandwidth (GB/s)'

set xrange [1:10]
set logscale y
set yrange [10000:25000]
set format y "%.2s "
set ytics (10000,15000,20000,25000)

plot "stream.txt" using 1:2 ls 1 lw 2\
	title	"Same node" with linespoints, \
		"stream.txt" using 1:3 ls 2 lw 2\
	title	"External node" with linespoints, \