# Benchmark exercises
 
It this exercise I tested *HPL* and *HPCG* benchmarks on Cosilt cluster, also using multithreading libraries in order to optimize math routines. The goal was to find the best parameters in order to improve the performances and to get as close as possible to the theoretical peak performance.

The codes have been run using two sockets in two different nodes of Cosilt cluster with processors `Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz` and 12 cores each. The associated theoretical peak performance is 

$$
(2.7GHz) x (12 cores/CPU) x (8 flops/cycle) x (2 CPUs/node) = 518.4 Gflop/s.
$$

The loaded modules were:
```
module purge; module load intel/15.0 mkl/intel openmpi/1.8.8/gnu/4.8.3
```

## Precompiled HPL

I collected the following results using matrix size *N=16348*, number of blocks *Nb=128* and *4x6* cells:

<center>

|  executable | 	  Time 			| 	Gflops 		|
|:----------------:|:------------:|:-----------:|
|  xhpl.plasma-gnu   |     80.83    |    3.628e+01	|
|  xhpl.openblas   |      8.20    |     3.575e+02   |
|   xhpl.netlib    |    78.25    |     3.748e+01   |
|  xhpl.mkl-gnu   |			 80.83    |     3.628e+01   |
|   xhpl.atlas  	|		 11.28     |    2.600e+02   |
| xhpl.plasma-mkl |   8.14      |  3.602e+02    |

</center>

In this case, the best one in terms of performances was `xhpl.plasma-mkl`. In order to tune it I run the command `mpirun -np 24 xhpl.plasma-mkl` using the following ***HPL.dat***:

```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any) 
6            device out (6=stdout,7=stderr,file)
3            # of problems sizes (N)
16384 32768 65536         Ns
3            # of NBs
128 181 256          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
2            # of process grids (P x Q)
4 6           Ps
6 4           Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
##### This line (no. 32) is ignored (it serves as a separator). ######
0                               Number of additional problem sizes for PTRANS
1200 10000 30000                values of N
0                               number of additional blocking sizes for PTRANS
40 9 8 13 13 20 16 32 64        values of NB
```

and getting

|		N	|	NB	|	P  	|	Q		| Time	|	Gflops  |
|:---:|:---:|:---:|:--:|:-----:|:------:|
|16384 | 128 | 4 | 6 | 7.89  |3.717e+02  | 
|16384  |181 | 4 | 6 | 8.22  |3.566e+02  | 
|16384  |256 | 4 | 6 | 8.27  |3.545e+02  | 
|32768  |128 | 4 | 6 | 53.81 | 4.360e+02 |  
|32768  |181 | 4 | 6 | 56.49 | 4.152e+02 |  
|32768  |256 | 4 | 6 | 56.17 | 4.176e+02 |  
|65536  |128 | 4 | 6 | 401.35|  4.676e+02|   
|65536  |181 | 4 | 6 | 414.45|  4.528e+02|   
|65536  |256 | 4 | 6 | 406.35|  4.618e+02|   
|16384  |128 | 6 | 4 | 7.97  |3.681e+02  | 
|16384  |181 | 6 | 4 | 8.37  |3.502e+02  | 
|16384  |256 | 6 | 4 | 8.85  |3.314e+02  | 
|32768  |128 | 6 | 4 | 54.61 | 4.296e+02 |  
|32768  |181 | 6 | 4 | 56.56 | 4.148e+02 |  
|32768  |256 | 6 | 4 | 58.49 | 4.010e+02 |  
|65536  |128 | 6 | 4 | 402.44|  4.663e+02|   
|65536  |181 | 6 | 4 | 412.16|  4.553e+02|   
|65536  |256 | 6 | 4 | 408.57|  4.593e+02| 

The best results were given by *N=65536*, *Nb=128* and grid *4x6*, reaching 90% of peak performance.

## HPCG

After getting the HPCG package I compiled it against MKL library using 

```
CXXFLAGS     = $(HPCG_DEFS) -O3 -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=0 -m64 -I${MKLROOT}/include

LINKFLAGS    = $(CXXFLAGS) -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

```

and the output I got is `HPCG result is VALID with a GFLOP/s rating of: 7.82077`, corresponging to *0.01%* of peak performance. I suggest to intepret this result knowing that the highest fraction of peak until November 2017 has been reached from the 10-th TOP500 Computer and corresponds to *5.3%*.
