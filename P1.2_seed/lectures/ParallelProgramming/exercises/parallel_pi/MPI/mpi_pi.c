#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  // int nprocs = 1, rank = 0;
  int nprocs, rank;
  long n, i, size;
  double w, x, sum, pi;
  double f, start;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);  // number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // labels for the processes

  n = 100000000;  // number of sub-intervals for each process

  size = n / nprocs;
  sum = 0.0;
  f = 1.0 / size;    // I assing one interval to each process, so each one
                     // operates inside an interval of width f
  start = f * rank;  // start is the starting point of the interval relative to
                     // the corresponding process
  w = f / n;         // width of a sub-interval

  for (i = 1; i <= size; i++) {
    x = start + w * (i - 0.5);          // mid point of interval i
    sum = sum + (4.0 / (1.0 + x * x));  // height f(x)
  }

  sum = sum * w;  // area = (sum f(x))*w

  MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  // 1 is the number of arguments
  // 0 is the rank of the process containing the result

  if (rank == 0)  // this only prints the significant value
    printf("Value of pi: %.16g\n", pi);

  MPI_Finalize();

  return 0;
}
