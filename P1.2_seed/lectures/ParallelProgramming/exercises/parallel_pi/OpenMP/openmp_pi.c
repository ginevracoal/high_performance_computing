#include <omp.h>
#include <stdio.h>

int main() {
  // int nthreads;
  long n, i;
  double w, x, sum, pi;

  n = 100000000;
  w = 1.0 / n;
  sum = 0.0;

#pragma omp parallel for private(x) reduction(+ : sum)
  for (i = 1; i <= n; i++) {
    x = w * (i - 0.5);
    sum += (4.0 / (1.0 + x * x));
  }

  pi = w * sum;
  printf("Value of pi: %.16g, ", pi);

  // nthreads = omp_get_num_threads();
  // printf("Number of threads: %d\n", nthreads);

  return 0;
}
