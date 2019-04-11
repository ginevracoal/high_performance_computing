// first use of cuda, 17/11/17

#include<stdio.h>
#include<assert.h>

#define SIZE 12 //12000000
#define NUM_BLOCKS 8192
#define NUM_THREADS 5012

__global__ void add( int * d_a, int * d_b, int * d_c )
{

	// we have multiple blocks with multiple threads,
	// so that's how we access a thread:
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if ( idx < SIZE )
	{
		d_c[idx] = d_a[idx] + d_b[idx];
	}
}


int main()
{
	int i;
	int * h_a, * h_b, * h_c; // h means host pointer
	int * d_a, * d_b, * d_c; // d means device pointer
	size_t size_in_bytes = SIZE * sizeof(int);

	// allocate the pointers

	h_a = (int *) malloc( size_in_bytes);
	h_b = (int *) malloc( size_in_bytes);
	h_c = (int *) malloc( size_in_bytes);

	cudaMalloc( (void**) &d_a, size_in_bytes );
	cudaMalloc( (void**) &d_b, size_in_bytes );
	cudaMalloc( (void**) &d_c, size_in_bytes );

	// initialize the arrays before copying

	for (i = 0; i < SIZE; ++i)
	{
		h_a[i] = 1;
		h_b[i] = 2;
	}

	// copying from cpu to gpu

	cudaMemcpy( d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, size_in_bytes, cudaMemcpyHostToDevice );

	// SIZE + NUM_THREADS makes sure that we create enough threads
	add<<< (SIZE + NUM_THREADS) / NUM_THREADS, NUM_THREADS >>>( d_a, d_b, d_c );

	// copying from gpu to cpu

	cudaMemcpy( h_c, d_c, size_in_bytes, cudaMemcpyDeviceToHost );

	if ( SIZE < 100 )
	{
		for (i = 0; i < SIZE; ++i)
		{
			fprintf( stdout, "%d", h_c[i] );
		}

	}

	free( h_a );
	free( h_b );
	free( h_c );

	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );

  return 0;
}
