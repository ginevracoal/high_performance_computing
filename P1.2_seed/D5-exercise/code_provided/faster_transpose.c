/*Optimized version of transpose.c code

The code only works with these requirements:
- square matrix
- number of blocks divides the size of the matrix
*/

#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>

double cclock()
/* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday(&tmp, (struct timezone *)0);
  sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;
  return sec;
}

/*prints a square matrix A*/
void print(double *A, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%6.0f\t", A[(i * size) + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  double *A, *AT;
  int i, j;
  double t_start, t_end;
  int BLOCKSIZE, MATRIXSIZE;

  if (argc < 2) {
    fprintf(stderr,
            "Error. The program runs as following: %s [MATRIXSIZE].\nProgram "
            "exit ...\n",
            argv[0]);
    exit(1);
  }

  MATRIXSIZE = atoi(argv[1]);
  BLOCKSIZE = atoi(argv[2]);

  if (MATRIXSIZE % BLOCKSIZE != 0) MATRIXSIZE += MATRIXSIZE % BLOCKSIZE;

  int MATRIXDIM = MATRIXSIZE * MATRIXSIZE;
  int BLOCKDIM = BLOCKSIZE * BLOCKSIZE;

  if (MATRIXSIZE < 1) {
    fprintf(stderr, "Error. Inconsistent parameters.\nProgram exit ...\n");
    exit(1);
  }

  A = (double *)malloc(MATRIXDIM * sizeof(double));
  AT = (double *)malloc(MATRIXDIM * sizeof(double));

  /*initializes the matrix A	*/
  for (i = 0; i < MATRIXDIM; i++) {
    A[i] = (double)i;
  }

  int n_blocks = MATRIXSIZE / BLOCKSIZE;

  // print(A, MATRIXSIZE);

  int h_A, k_A;

  t_start = cclock();

  /*	running on all the blocks*/
  for (int h = 0; h < n_blocks; h++) {
    for (int k = 0; k < n_blocks; k++) {
      h_A = h * BLOCKSIZE;
      k_A = k * BLOCKSIZE;

      /*	calculating the transpose */
      for (i = 0; i < BLOCKSIZE; i++) {
        for (j = 0; j < BLOCKSIZE; j++) {
          AT[(i + h_A) * MATRIXSIZE + (j + k_A)] =
              A[(j + k_A) * MATRIXSIZE + (i + h_A)];
        }
      }
    }
  }

  t_end = cclock();

  // print(AT, MATRIXSIZE);

  free(A);
  free(AT);

  printf(" Matrix transpose executed. Time Elapsed %9.4f secs\n",
         t_end - t_start);
  /*  fprintf(stderr, "%f\n", t_end-t_start);*/
  return 0;
}
