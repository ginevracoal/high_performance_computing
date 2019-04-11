//  the code initializes and computes jacobi inside the kernel function

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define NUM_THREADS 5

// save matrix to file 
void save_gnuplot( double *M, size_t dim );

// return the elapsed time
double seconds( void );

// print the matrix M
void print_matrix(int rows, int cols, double *M);

// ===============================================================================

// KERNEL FUNCTIONS

__global__ void matrix_init(int iterations, int dimension, double * d_matrix, double * d_matrix_new)
{
  int idx = ( blockIdx.x * blockDim.x ) + threadIdx.x; // local index for each thread

  // global indexes for the matrices (idx = i*dimension+j)
  int i = idx / dimension; 
  int j = idx % dimension; 

  double increment; 

  //fill initial values  
  for( i = 1; i <= dimension; ++i ){
    for( j = 1; j <= dimension; ++j ){
      d_matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
    }
  }

  __syncthreads();
        
  // set up borders 
  increment = 100.0 / ( dimension + 1 );
  
  for( i=1; i <= dimension+1; ++i ){
    d_matrix[ i * ( dimension + 2 ) ] = i * increment; //setting left border
    d_matrix[ ( ( dimension + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment; //setting bottom border
    d_matrix_new[ i * ( dimension + 2 ) ] = i * increment;
    d_matrix_new[ ( ( dimension + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
  }

  __syncthreads();

}

// jacobi method

__global__ void jacobi(int iterations, int dimension, double * d_matrix, double * d_matrix_new)
{
  double * tmp_matrix; 
  int idx = ( blockIdx.x * blockDim.x ) + threadIdx.x; // local index for each thread

  // global indexes for the matrices (idx = i*dimension+j)
  int i = idx / dimension; 
  int j = idx % dimension; 

  if( i > 0 && i < (dimension+1) && j > 0 && j < (dimension+2)){
  // if ( (idx < (dimension + 2) * (dimension + 2)) && i > 0 && j > 0){
    for(int it = 0; it < iterations; ++it ){
      // This is a row dominant program.
      d_matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
      ( d_matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
      d_matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] +     
      d_matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
      d_matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 

      // swap the pointers
      tmp_matrix = d_matrix;
      d_matrix = d_matrix_new;
      d_matrix_new = tmp_matrix;

      __syncthreads(); 
    }
  }
}
  
// ===============================================================================

int main(int argc, char* argv[]){

  // timing variables
  double t_start, t_end;
  
  double *h_matrix; // host pointer
  double *d_matrix, *d_matrix_new; // device pointers

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  if(argc != 5) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  row_peek = atoi(argv[3]);
  col_peek = atoi(argv[4]);

  printf("matrix size = %zu\n", dimension);
  printf("number of iterations = %zu\n", iterations);
  printf("element for checking = Mat[%zu,%zu]\n",row_peek, col_peek);

  if((row_peek > dimension) || (col_peek > dimension)){
    fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
    fprintf(stderr, "Arguments n and m must be smaller than %zu\n", dimension);
    return 1;
  }

#ifdef DEBUG
  if(dimension>10){
    printf("Choose a smaller dimension for debug.\n");
    return 2;
  }
#endif

  byte_dimension = sizeof(double*) * ( dimension + 2 ) * ( dimension + 2 );
  
  h_matrix = ( double* )malloc( byte_dimension );
  // h_matrix_new = ( double* )malloc( byte_dimension );

  cudaMalloc( (void **) &d_matrix, byte_dimension); // allocates memory on the GPU
  cudaMalloc( (void **) &d_matrix_new, byte_dimension);

  // memset( h_matrix, 0, byte_dimension ); // sets initial values to zero
  // memset( h_matrix_new, 0, byte_dimension );
  
  t_start = seconds();

  // call kernel functions
  matrix_init<<< ((dimension+2) * (dimension+2))/NUM_THREADS, NUM_THREADS >>>(iterations, dimension, d_matrix, d_matrix_new);
  jacobi<<< ((dimension+2) * (dimension+2))/NUM_THREADS, NUM_THREADS >>>(iterations, dimension, d_matrix, d_matrix_new);

  // copy from gpu to cpu
  cudaMemcpy( h_matrix, d_matrix, byte_dimension, cudaMemcpyDeviceToHost );
 
  t_end = seconds();

#ifdef DEBUG

  print_matrix(dimension+2, dimension+2, h_matrix);
  // free( h_matrix_new );
#endif

  printf( "\nelapsed time = %f seconds\n", t_end - t_start );
  printf( "\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, h_matrix[ ( row_peek + 1 ) * ( dimension + 2 ) + ( col_peek + 1 ) ] );

  // save_gnuplot( h_matrix, dimension );
  
  // free the memory

  free( h_matrix );

  cudaFree( d_matrix );
  cudaFree( d_matrix_new );

  return 0;
}

// ===============================================================================

void print_matrix(int rows, int cols, double *M) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.1f   ", (double)M[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void save_gnuplot( double *M, size_t dimension ){
  
  size_t i , j;
  const double h = 0.1;
  FILE *file;

  file = fopen( "solution.dat", "w" );

  for( i = 0; i < dimension + 2; ++i )
    for( j = 0; j < dimension + 2; ++j )
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( dimension + 2 ) ) + j ] );

  fclose( file );
}

// A Simple timer for measuring the walltime
double seconds(){

    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}
