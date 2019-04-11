// matrix per matrix serial multiplication
// on square matrices
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define SIZE 4096

// PRINT THE MATRIX M

void print_matrix(int rows, int cols, double *M) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%d ", (int)M[i * cols + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  double *A, *B, *C;
  int i, h, k, z;
  int SIZE = atoi(argv[1]);

  // Allocate distributed matrices

  A = (double *)malloc(SIZE * SIZE * sizeof(double));
  B = (double *)malloc(SIZE * SIZE * sizeof(double));
  C = (double *)malloc(SIZE * SIZE * sizeof(double));

  // Initialize matrices

  // srand(time(NULL));

  for (i = 0; i < SIZE * SIZE; i++) {
    A[i] = i;  //(rand() % 1000 + 1);
    B[i] = i;  //(rand() % 1000 + 1);
    C[i] = 0;
  }

  // Calculate C
  for (h = 0; h < SIZE; ++h) {
    for (k = 0; k < SIZE; ++k) {
      for (z = 0; z < SIZE; ++z) {
        C[h * SIZE + z] += A[h * SIZE + k] * B[k * SIZE + z];
      }
    }
  }

  print_matrix(SIZE, SIZE, C);

  free(A);
  free(B);
  free(C);

  return 0;
}
