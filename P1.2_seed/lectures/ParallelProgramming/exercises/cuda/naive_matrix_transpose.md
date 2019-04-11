# Naive CUDA matrix transpose

Naive version of Parallel Transpose of a NxN matrix in CUDA, based on the use of block of threads.

***transpose.cu***:
```
#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>

#include<assert.h>

// #define NUM_BLOCKS 8192
#define NUM_THREADS 512

double cclock()
  /* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

// print the matrix

void print_matrix(int size, double * M)
{
  int i, j;
  for (i=0; i<size; i++)
  {
      for(j=0; j<size; j++)
          {
           fprintf(stdout, "%f     ", M[ i*size + j ]);
          }
      fprintf(stdout, "\n");
   }
   fprintf(stdout, "\n");
}

// transpose the matrix

__global__  void trasp_mat(int MATRIXDIM, double * d_A, double * d_AT)
{	
	int idx = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  // here I am writing i and j as functions of idx, 
  // knowing that idx = i*MATRIXDIM+j
  int i = idx / MATRIXDIM; 
  int j = idx % MATRIXDIM; 
  
	if ( idx < MATRIXDIM * MATRIXDIM )
  {
  	d_AT[ ( j * MATRIXDIM ) + i ] = d_A[ ( i * MATRIXDIM ) + j ];      
	}
	
}


int main( int argc, char * argv [] )
{
  double * h_A, * h_AT; // host pointers
  double * d_A, * d_AT; // device pointers
  int i;
  int MATRIXDIM;
  int size_in_bytes;
  double t_start, t_end;

  if( argc < 2 ){
    fprintf( stderr, "Error. The program runs as following: %s [MATRIXDIM].\nProgram exit ...\n", argv[0]);
    exit(1);
  }

  MATRIXDIM = atoi(argv[1]);
	size_in_bytes =  MATRIXDIM * MATRIXDIM * sizeof( double );

  if( MATRIXDIM < 1 ){
    fprintf( stderr, "Error. Inconsistent parameters.\nProgram exit ...\n", argv[0]);
    exit(1);
  }

  // allocate the pointers

  h_A = ( double * ) malloc( size_in_bytes );
  h_AT = ( double * ) malloc( size_in_bytes );

	//cudaMalloc( (void **) &my_ptr, sizeinbytes );
  cudaMalloc( (void**) &d_A, size_in_bytes );
  cudaMalloc( (void**) &d_AT, size_in_bytes );

  // initialize the matrix A

  for( i = 0; i < MATRIXDIM * MATRIXDIM; i++ ){
    h_A[i] = (double) i;
  }
  
  print_matrix( MATRIXDIM, h_A);

	// copy from cpu to gpu

	//cudaMemcpy( dest, source, sizeinbytes, cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost );
	cudaMemcpy( d_A, h_A, size_in_bytes, cudaMemcpyHostToDevice );

	// (MATRIXDIM * MATRIXDIM + NUM_THREADS) makes sure that we create enough threads
	t_start=cclock();
	trasp_mat<<< (MATRIXDIM * MATRIXDIM + NUM_THREADS) / NUM_THREADS, NUM_THREADS >>>( MATRIXDIM, d_A, d_AT );
  t_end=cclock(); 

	// copying from gpu to cpu

	cudaMemcpy( h_AT, d_AT, size_in_bytes, cudaMemcpyDeviceToHost );

  print_matrix(MATRIXDIM, h_AT);

	fprintf( stdout, " Matrix transpose executed. Time Elapsed %9.4f secs\n", t_end-t_start );

  // free the memory

  free( h_A );
  free( h_AT );

  cudaFree( d_A );
  cudaFree( d_AT );
  
  return 0;
}

```