// Matrix multiplication between square matrices using rows of A shared among threads of the same block. 
// The grid of blocks is made by unidimensional blocks of legth MATRIXSIZE.
// Each block contains NUM_THREADS cells of size CELL_SIZE.
// CUDA kernels are asynchronous, so in order to perform time measurements it is necessary to call 
// cudaDeviceSynchronize() after each kernel launch. 

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>

#define MATRIXSIZE 8 //2048
#define NUM_THREADS 4 //31 // number of threads per block
#define CELL_SIZE MATRIXSIZE / NUM_THREADS // number of blocks per row

double cclock()
  /* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

// print the matrix M

void print_matrix(int rows, int cols, double *M) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%d    ", (int)M[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// ===============================================================================

// KERNEL FUNCTIONS

// initialization

__global__ void matrix_init(double * M){
  int idx = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if( idx < MATRIXSIZE*MATRIXSIZE) {
    M[idx] = 1;
  }
}

// matrix x matrix multiplication

__global__ void matrix_mult(double * A, double * B, double * C){
  __shared__ double s_A[MATRIXSIZE];

  for (int i = 0; i < CELL_SIZE; ++i)
  {
    s_A[ threadIdx.x * CELL_SIZE + i ] = A[ blockIdx.x * MATRIXSIZE + (threadIdx.x * CELL_SIZE + i) ];
  }

  __syncthreads();

  for (int i = 0; i < CELL_SIZE; ++i)
  {
    double sum = 0.;

    for (int j = 0; j < MATRIXSIZE; ++j)
    {
      sum += s_A[j] * B[j * MATRIXSIZE + (threadIdx.x * CELL_SIZE + i)];
    }

    // each block computes an entire row of C
    C[ blockIdx.x * MATRIXSIZE + (threadIdx.x * CELL_SIZE + i) ] = sum; 
  }

}

// ===============================================================================

int main(int argc, char *argv[]) 
{
  double * h_A, * h_B, *h_C; // host pointers
  // double * h_C;
  double * d_A, * d_B, *d_C; // device pointers

  int size_in_bytes;
  double t_start, t_end;

  // MATRIXSIZE = atoi(argv[1]);
  int NUM_BLOCKS = CELL_SIZE; 
  size_in_bytes =  MATRIXSIZE * MATRIXSIZE * sizeof( double );

  if( MATRIXSIZE < 1 ){
    fprintf( stderr, "Error. Inconsistent parameters.\nProgram exit ...\n");
    exit(1);
  }

  // allocate the pointers

  h_A = ( double * ) malloc( size_in_bytes );
  h_B = ( double * ) malloc( size_in_bytes );
  h_C = ( double * ) malloc( size_in_bytes );

  cudaMalloc( (void**) &d_A, size_in_bytes );
  cudaMalloc( (void**) &d_B, size_in_bytes );
  cudaMalloc( (void**) &d_C, size_in_bytes );

  // initialize the matrices

  // srand(time(NULL));
  for(int i = 0; i < MATRIXSIZE * MATRIXSIZE; i++ ){
    h_A[i] = (double) 1; //(rand() % 1000 + 1);
    h_B[i] = (double) i; //(rand() % 1000 + 1);
    h_C[i] = 0;
  }

  // copy from CPU to GPU

  //cudaMemcpy( dest, source, sizeinbytes, cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A, h_A, size_in_bytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_B, h_B, size_in_bytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_C, h_C, size_in_bytes, cudaMemcpyHostToDevice );

  // matrix_init<<< NUM_BLOCKS, NUM_THREADS >>>(d_A);
  // matrix_init<<< NUM_BLOCKS, NUM_THREADS >>>(d_B);

  t_start=cclock();

  matrix_mult<<< NUM_BLOCKS, NUM_THREADS >>>(d_A, d_B, d_C);

  cudaDeviceSynchronize(); // blocks until the device has completed all the preceding requested tasks

  t_end=cclock();  

  // copy from GPU to CPU

  cudaMemcpy( h_A, d_A, size_in_bytes, cudaMemcpyDeviceToHost );
  cudaMemcpy( h_B, d_B, size_in_bytes, cudaMemcpyDeviceToHost );
  cudaMemcpy( h_C, d_C, size_in_bytes, cudaMemcpyDeviceToHost );

  print_matrix(MATRIXSIZE, MATRIXSIZE, h_A);
  print_matrix(MATRIXSIZE, MATRIXSIZE, h_B);
  print_matrix(MATRIXSIZE, MATRIXSIZE, h_C);

  fprintf( stdout, "multiplication executed. Time Elapsed %9.4f secs\n", t_end-t_start );

  // free the memory

  free( h_A );
  free( h_B );
  free( h_C );

  cudaFree( d_A );
  cudaFree( d_B );
  cudaFree( d_C );

  return 0;
}
