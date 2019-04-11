set terminal png size 1200,800 enhanced font 'Verdana,12' 
set grid
set key bottom right

set style line 1 lc rgb "green" lw 2
set style line 2 lc rgb "blue" lw 2
set style line 3 lc rgb "yellow" lw 2
set style line 4 lc rgb "purple" lw 2

set output "Cache_misses.png"

set title 'L1 cache misses per block size on my laptop'
set xlabel 'Block size'
set ylabel 'L1 cache misses'

set format y "%.0sx10^%T"

set logscale xy
set xrange [1:8192]
set yrange [:]


plot "cache_misses/cache_misses_1024.txt" using 1:2 ls 1\
	title	"Matrix size 1024" with linespoints, \
		 "cache_misses/cache_misses_2048.txt" using 1:2 ls 2\
	title	"Matrix size 2048" with linespoints, \
		 "cache_misses/cache_misses_4096.txt" using 1:2 ls 3\
	title	"Matrix size 4096" with linespoints, \
		 "cache_misses/cache_misses_8192.txt" using 1:2 ls 4\
	title	"Matrix size 8192" with linespoints, \



