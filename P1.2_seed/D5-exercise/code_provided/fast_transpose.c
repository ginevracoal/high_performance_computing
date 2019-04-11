/*Optimized version of transpose.c code 

The code only works with these requirements:
- square matrix
- number of blocks divides the size of the matrix
*/

#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>


double cclock()
 /* Returns elepsed seconds past from the last call to timer rest */
{

    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

/*calculates the transpose of a square matrix B*/
/*void transpose(double * B, double * BT, int size){*/
/*  for(int i=0; i<size; i++){*/
/*    for(int j=0; j<size; j++)*/
/*      BT[ (j*size) + i ] = B[ (i*size) + j ];*/
/*  }*/
/*}*/

/*prints a square matrix A*/
void print(double * A, int size) {
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      printf("%6.0f\t", A[ (i*size) + j]);
    }
    printf("\n");
  }
	printf("\n");
}

int main( int argc, char * argv [] ) {

  double * A, * AT;
  int i, j;
  double t_start, t_end;
  int BLOCKSIZE, MATRIXSIZE;

  if( argc < 2 ){
    fprintf( stderr, "Error. The program runs as following: %s [MATRIXSIZE].\nProgram exit ...\n", argv[0]);
    exit(1);
  }

  MATRIXSIZE = atoi(argv[1]);
	BLOCKSIZE = atoi(argv[2]);

	int MATRIXDIM = MATRIXSIZE * MATRIXSIZE;
	int BLOCKDIM = BLOCKSIZE * BLOCKSIZE;

  if( MATRIXSIZE < 1 ){
    fprintf( stderr, "Error. Inconsistent parameters.\nProgram exit ...\n");
    exit(1);
  }

  A = ( double * ) malloc( MATRIXDIM * sizeof( double ) );
  AT = ( double * ) malloc( MATRIXDIM * sizeof( double ) );

/*initializes the matrix A	*/
  for( i = 0; i < MATRIXDIM; i++ ){
    A[i] = (double) i;
  }
  
/*	print(A, MATRIXSIZE);*/

	int n_blocks = MATRIXSIZE / BLOCKSIZE; 
	double * B = ( double * ) malloc ( BLOCKDIM * sizeof( double ) );
	double * BT = ( double * ) malloc ( BLOCKDIM * sizeof( double ) );

	int h_B, k_B;

	t_start=cclock();

/*	running on all the blocks*/
	for(int h = 0; h < n_blocks; h++){
		for(int k = 0; k < n_blocks; k++){

			h_B = h * BLOCKSIZE;
			k_B = k * BLOCKSIZE;

/*B is the (h,k) block of A*/
			for (i=0; i < BLOCKSIZE; i++){
				for (j=0; j < BLOCKSIZE; j++){
					B[ (i * BLOCKSIZE) + j ] = A[ ((i+h_B) * MATRIXSIZE) + (j+k_B) ];
				}
			}

/*	print(B, BLOCKSIZE);*/

			//transpose(B, BT, BLOCKSIZE);
			for(int i=0; i<BLOCKSIZE; i++){
					for(int j=0; j<BLOCKSIZE; j++)
						BT[ (j*BLOCKSIZE) + i ] = B[ (i*BLOCKSIZE) + j ];
				}

/*	print(BT, BLOCKSIZE);*/
			
/*	BT is the (k,h) block of AT*/
			for (i=0; i < BLOCKSIZE; i++){
				for (j=0; j < BLOCKSIZE; j++){
					AT[ ((i+k_B) * MATRIXSIZE) + (j+h_B)] = BT[ i*BLOCKSIZE + j];
				}
			}

		}
	}
	
/*	print(AT, MATRIXSIZE);*/
	
  t_end=cclock(); 

  free(A);
  free(AT);
	free(B);
	free(BT);
  
  fprintf( stdout, " Matrix transpose executed. Time Elapsed %9.4f secs\n", t_end-t_start );
  
  return 0;
}
