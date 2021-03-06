#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <omp.h>

// #define nthreads 5

// ===============================================================
// FUNCTION DECLARATIONS

// save matrix to file
void save_gnuplot(double *M, size_t dim);

// evolve Jacobi
void evolve(double *matrix, double *matrix_new, size_t dimension);

// return the elapsed time
double seconds(void);

// print the matrix M
void print_matrix(int rows, int cols, double *M);

// ===============================================================
// MAIN

int main(int argc, char *argv[]) {
  // timing variables
  double t_start, t_end, increment;

  // indexes for loops
  size_t i, j, it;

  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;
  int nthreads;

  // check on input parameters
  // if (argc != 5) {
  //   fprintf(stderr, "\nwrong number of arguments. Usage: ./a.out dim it n
  //   m\n");
  //   return 1;
  // }
  if (argc != 6) {
    fprintf(
        stderr,
        "\nwrong number of arguments. Usage: ./a.out dim it n m nthreads\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  row_peek = atoi(argv[3]);
  col_peek = atoi(argv[4]);
  nthreads = atoi(argv[5]);

#ifdef DEBUG
  if (dimension > 10) {
    printf("Choose a smaller dimension for debug.\n");
    return 2;
  }
#endif

  printf("matrix size = %zu\n", dimension);
  printf("number of iterations = %zu\n", iterations);
  printf("element for checking = Mat[%zu,%zu]\n", row_peek, col_peek);

  if ((row_peek > dimension) || (col_peek > dimension)) {
    fprintf(stderr,
            "Cannot Peek a matrix element outside of the matrix dimension\n");
    fprintf(stderr, "Arguments n and m must be smaller than %zu\n", dimension);
    return 1;
  }

  byte_dimension = sizeof(double *) * (dimension + 2) * (dimension + 2);
  matrix = (double *)malloc(byte_dimension);
  matrix_new = (double *)malloc(byte_dimension);

  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  // fill initial values
  omp_set_num_threads(nthreads);

// i and j will be private for each thread
#pragma omp parallel for private(i, j)
  for (i = 1; i <= dimension; ++i) {
    for (j = 1; j <= dimension; ++j) {
      matrix[(i * (dimension + 2)) + j] = 0.5;
    }
  }

  // set up borders
  increment = 100.0 / (dimension + 1);

#pragma omp parallel for private(i)
  for (i = 1; i <= dimension + 1; ++i) {
    matrix[i * (dimension + 2)] = i * increment;
    matrix[((dimension + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
    matrix_new[i * (dimension + 2)] = i * increment;
    matrix_new[((dimension + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
  }

  // start algorithm
  t_start = seconds();
  for (it = 0; it < iterations; ++it) {
    evolve(matrix, matrix_new, dimension);

    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }
  t_end = seconds();

  printf("\nelapsed time = %f seconds\n", t_end - t_start);
  printf("\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek,
         matrix[(row_peek + 1) * (dimension + 2) + (col_peek + 1)]);

#ifdef DEBUG
  print_matrix(dimension + 2, dimension + 2, matrix);
#endif

  save_gnuplot(matrix, dimension);

  free(matrix);
  free(matrix_new);

  return 0;
}

// ===============================================================
// FUNCTION DEFINITIONS

void evolve(double *matrix, double *matrix_new, size_t dimension) {
  size_t i, j;
// This will be a row dominant program.
#pragma omp parallel for private(i, j)
  for (i = 1; i <= dimension; ++i) {
    for (j = 1; j <= dimension; ++j) {
      matrix_new[(i * (dimension + 2)) + j] =
          (0.25) * (matrix[((i - 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j + 1)] +
                    matrix[((i + 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j - 1)]);
    }
  }
}

void save_gnuplot(double *M, size_t dimension) {
  size_t i, j;
  const double h = 0.1;
  FILE *file;

  file = fopen("solution.dat", "w");

  for (i = 0; i < dimension + 2; ++i)
    for (j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i,
              M[(i * (dimension + 2)) + j]);

  fclose(file);
}

// A Simple timer for measuring the walltime
double seconds() {
  struct timeval tmp;
  double sec;
  gettimeofday(&tmp, (struct timezone *)0);
  sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;
  return sec;
}

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
