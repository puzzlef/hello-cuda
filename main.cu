#include <random>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




// Fixed config
#ifndef MAX_THREADS
#define MAX_THREADS 8
#endif
#ifndef REPEAT_METHOD
#define REPEAT_METHOD 1
#endif




// PERFORM EXPERIMENT
// ------------------

void runExperiment(int p, int P) {
  random_device dev;
  default_random_engine rnd(dev());
  int repeat = REPEAT_METHOD;
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    printf("P%02d.T%02d: Hello MPI\n", p, t);
  }
}


int main(int argc, char **argv) {
  char pname[MPI_MAX_PROCESSOR_NAME];
  int p = 0, P = 0, lpname = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);
  MPI_Get_processor_name(pname, &lpname);
  omp_set_num_threads(MAX_THREADS);
  printf("P%02d: NAME=%s\n", p, pname);
  printf("P%02d: OMP_NUM_THREADS=%d\n", p, MAX_THREADS);
  MPI_Barrier(MPI_COMM_WORLD);
  runExperiment(p, P);
  MPI_Finalize();
  return 0;
}
