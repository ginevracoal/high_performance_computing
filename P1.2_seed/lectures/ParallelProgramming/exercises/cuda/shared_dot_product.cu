// matrix multiplication between square matrices using bidimensional indexes.

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>

#define SIZE 4 //2048
#define NUM_THREADS 2 //512
#define NUM_BLOCKS SIZE / NUM_THREADS

double cclock()
  /* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

// print the vector

void print_vector(int size, int *v) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%d    ", v[i]);
  }
  printf("\n");
}

// kernel function

__global__ void dot( int *a, int *b, int *c ) {
  __shared__ int temp[NUM_THREADS];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  temp[threadIdx.x] = a[idx] * b[idx];
  __syncthreads();
  if( 0 == threadIdx.x ) {
    int sum = 0;
    for( int i = 0; i < NUM_THREADS; i++ ) sum += temp[i];
    atomicAdd( c , sum );
  }
} 

int main(int argc, char *argv[]) 
{
  int * h_a, * h_b, *h_c; // host pointers
  int * d_a, * d_b, *d_c; // device pointers

  int i;
  int size_in_bytes;
  int t_start, t_end;

  // SIZE = atoi(argv[1]);
  size_in_bytes =  SIZE * sizeof( int );

  if( SIZE < 1 ){
    fprintf( stderr, "Error. Inconsistent parameters.\nProgram exit ...\n");
    exit(1);
  }

  // allocate the pointers

  h_a = ( int * ) malloc( size_in_bytes );
  h_b = ( int * ) malloc( size_in_bytes );
  h_c = ( int * ) malloc( sizeof( int ) );


  cudaMalloc( (void**) &d_a, size_in_bytes );
  cudaMalloc( (void**) &d_b, size_in_bytes );
  cudaMalloc( (void**) &d_c, sizeof( int ) );

  // initialize the vectors

  // srand(time(NULL));
  for( i = 0; i < SIZE; i++ ){
    h_a[i] = (int) 1; //(rand() % 1000 + 1);
    h_b[i] = (int) i; //(rand() % 1000 + 1);
  }

  h_c[0] = 0;

  print_vector(SIZE, h_a);
  print_vector(SIZE, h_b);

  // copy from CPU to GPU

  //cudaMemcpy( dest, source, sizeinbytes, cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost );
  cudaMemcpy( d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, h_b, size_in_bytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_c, h_c, sizeof( int ), cudaMemcpyHostToDevice );

  t_start=cclock();
  dot<<< NUM_BLOCKS, NUM_THREADS >>>(d_a, d_b, d_c);
  t_end=cclock();  

  // copy from GPU to CPU

  cudaMemcpy( h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost );
  cudaMemcpy( h_b, d_b, size_in_bytes, cudaMemcpyDeviceToHost );
  cudaMemcpy( h_c, d_c, sizeof( int ), cudaMemcpyDeviceToHost );

  printf("%d\n", h_c[0]);

  fprintf( stdout, "multiplication executed. Time Elapsed %9.4f secs\n", t_end-t_start );

  // free the memory

  free( h_a );
  free( h_b );
  free( h_c );

  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );

  return 0;
}
