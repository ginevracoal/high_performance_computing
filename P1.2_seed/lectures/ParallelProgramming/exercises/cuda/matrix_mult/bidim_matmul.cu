// matrix multiplication between square matrices using bidimensional indexes.

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>

// #define MATRIXSIZE 10
// #define MAX_THREADS 5
// #define NUM_BLOCKS MATRIXSIZE / MAX_THREADS

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

// kernel function

__global__ void matrix_init(double * M, int SIZE){
  int col = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  int row = ( blockIdx.y * blockDim.y ) + threadIdx.y;
  int idx = row * SIZE + col;

  if( idx < SIZE*SIZE) {
    M[idx] = 1;
  }
}

__global__ void matrix_mult(double * A, double * B, double * C, int SIZE){
  int k;
  int col = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  int row = ( blockIdx.y * blockDim.y ) + threadIdx.y;

  // every block of threads computes the i-th column of matrix C
  if( row < SIZE && col < SIZE) {
    C[row * SIZE + col] = 0;
    for( k = 0; k < SIZE; k++ ) {
      C[row * SIZE + col] += A[row * SIZE + k] * B[k * SIZE + col];
    }
  }
}

int main(int argc, char *argv[]) {
  // double * h_A, * h_B, 
  double *h_C; // host pointers
  double * d_A, * d_B, *d_C; // device pointers

  // int i;
  int size_in_bytes, MATRIXSIZE;
  double t_start, t_end;

  MATRIXSIZE = atoi(argv[1]);
  size_in_bytes =  MATRIXSIZE * MATRIXSIZE * sizeof( double );

  if( MATRIXSIZE < 1 ){
    fprintf( stderr, "Error. Inconsistent parameters.\nProgram exit ...\n");
    exit(1);
  }

  // allocate the pointers

  // h_A = ( double * ) malloc( size_in_bytes );
  // h_B = ( double * ) malloc( size_in_bytes );
  h_C = ( double * ) malloc( size_in_bytes );

  cudaMalloc( (void**) &d_A, size_in_bytes );
  cudaMalloc( (void**) &d_B, size_in_bytes );
  cudaMalloc( (void**) &d_C, size_in_bytes );

  dim3 dimBlock(4,4); //4 threads per block
  dim3 dimGrid(MATRIXSIZE/dimBlock.x, MATRIXSIZE/dimBlock.y); // MATRIXSIZE/dimBlock blocks per grid

  // // initialize the matrices

  // srand(time(NULL));
  // for( i = 0; i < MATRIXSIZE * MATRIXSIZE; i++ ){
  //   h_A[i] = (rand() % 1000 + 1); 
  //   h_B[i] = (rand() % 1000 + 1);
  //   h_C[i] = 0;
  // }

  // copy from CPU to GPU

  //cudaMemcpy( dest, source, sizeinbytes, cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost );
  // cudaMemcpy( d_A, h_A, size_in_bytes, cudaMemcpyHostToDevice );
  // cudaMemcpy( d_B, h_B, size_in_bytes, cudaMemcpyHostToDevice );
  // cudaMemcpy( d_C, h_C, size_in_bytes, cudaMemcpyHostToDevice );

  matrix_init<<< dimGrid, dimBlock >>>(d_A, MATRIXSIZE);
  matrix_init<<< dimGrid, dimBlock >>>(d_B, MATRIXSIZE);

  t_start=cclock();
  matrix_mult<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, MATRIXSIZE);
  t_end=cclock(); 

  // copy from GPU to CPU

  cudaMemcpy( h_C, d_C, size_in_bytes, cudaMemcpyDeviceToHost );

  print_matrix(MATRIXSIZE, MATRIXSIZE, h_C);

  fprintf( stdout, "multiplication executed. Time Elapsed %9.4f secs\n", t_end-t_start );

  // free the memory

  // free( h_A );
  // free( h_B );
  free( h_C );

  cudaFree( d_A );
  cudaFree( d_B );
  cudaFree( d_C );

  return 0;
}
