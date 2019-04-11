# Nodeperf
One of the main problems of HPL benchmark is that it is highly dependent on the choice of input parameters, so Intel also provides a program called ***nodeperf.c*** in order to test single nodes performances.

I worked on Cosilt cluster, using nodes `b22` and `b23` with processors `Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz` and 12 cores each.

I loaded modules
```
gnu/4.8.3                  
openmpi/1.10.2/gnu/4.8.3   
mkl/intel
```
then compiled and ran the code with the following commands:
```
$ mpicc -fopenmp -O3 -g nodeperf.c  -m64 -I${MKLROOT}/include -o nodeperf.x -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
$ export OMP_NUM_THREADS=24
$ export OMP_PLACES=cores
$ mpirun ./nodeperf.x
```

The output I got is 
```
No multi-threaded MPI detected (Hybrid may or may not work)

The time/date of the run...  at Wed Feb 14 14:50:26 2018

This driver was compiled with:
	-DITER=4 -DLINUX -DNOACCUR -DPREC=double 
Malloc done.  Used 1846080096 bytes
(0 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23029.313 b22
(1 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23018.520 b22
(2 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23030.189 b22
(3 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23009.561 b22
(4 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23008.764 b22
(5 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23017.965 b22
(6 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23016.139 b22
(7 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23014.458 b22
(8 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23020.953 b22
(9 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23008.856 b22
(10 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23002.089 b22
(11 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23014.282 b22
(12 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23030.674 b23
(13 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23008.074 b23
(14 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23020.855 b23
(15 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 22999.401 b23
(16 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23019.759 b23
(17 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23022.310 b23
(18 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23027.629 b23
(19 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23029.051 b23
(20 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 22996.680 b23
(21 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23006.432 b23
(22 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23024.311 b23
(23 of 24): NN lda=15000 ldb= 192 ldc=15000 0 0 0 23004.737 b23

```


In this case the theoretical peak is $(2.7GHz) x (12 cores/CPU) x (8 flops/cycle) x (2 CPUs/node) = 518.4 Gflop/s$, so node `b22` is slightly slower with *23030.189 Mflops* and node `b23` faster with *23030.674 Mflops*. In both cases I only got 4.4% of the theorerical peak and it is probably due to the fact that multi-threading is not working.
