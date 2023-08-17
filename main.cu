#include <random>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD  5
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELLO WORLD
/**
 * Hello world from GPU (CUDA kernel).
 * @details
 * A kernel function in CUDA is defined with __global__. NVCC picks it up and
 * generates (intermediate) GPU code for this function. It also generates a
 * placeholder (CPU code) that can be trigger the execution of this kernel
 * through CUDA runtime. Its return type is always "void".
 *
 * A kernel is called with "kernel<<<blocks, threads>>>(arguments...)" syntax.
 * Each execution of kernel is called a thread. A number of threads are grouped
 * into thread blocks. All thread blocks of a kernel call are grouped into a
 * grid. Threads within a block can communicate and synchronize with each
 * other, but blocks execute independently of each other (though they can still
 * communicate through global GPU memory).
 */
__global__ void sayHelloCuk() {
  DEFINE_CUDA(t, b, B, G);
  // printf() is managed by CUDA driver (how?).
  printf("GPU[B%01d.T%01d]: Hello CUDA\n", b, t);
}


/**
 * Hello world from GPU.
 */
inline void sayHelloCuda() {
  // Call kernel with 4 thread-blocks, and 8 threads per block.
  sayHelloCuk<<<4, 8>>>();
  // Wait for GPU to finish executing the kernel.
  TRY_CUDA( cudaDeviceSynchronize() );
  // Say Hello world from CPU after GPU is done.
  printf("CPU: Hello world!\n");
  printf("\n");
}
#pragma endregion




#pragma region DEVICE PROPERTIES
/**
 * List properties of all CUDA devices.
 */
inline void listDevicePropertiesCuda() {
  // Check how many compute devices are attached.
  int N;
  TRY_CUDA( cudaGetDeviceCount(&N) );
  // List some properties of each device.
  cudaDeviceProp p;
  for (int i=0; i<N; ++i) {
    TRY_CUDA( cudaGetDeviceProperties(&p, i) );
    printf("COMPUTE DEVICE %d:\n", i);
    printf("Name: %s\n",                        p.name);
    printf("Compute capability: %d.%d\n",       p.major, p.minor);
    printf("Multiprocessors: %d\n",             p.multiProcessorCount);
    printf("Clock rate: %d MHz\n",              p.clockRate / 1000);
    printf("Global memory: %zd MB\n",           p.totalGlobalMem / (1024*1024));
    printf("Constant memory: %zd KB\n",         p.totalConstMem / 1024);
    printf("Shared memory per block: %zd KB\n", p.sharedMemPerBlock / 1024);
    printf("Registers per block: %d\n",         p.regsPerBlock);
    printf("Threads per block: %d (max)\n",     p.maxThreadsPerBlock);
    printf("Threads per warp: %d\n",            p.warpSize);
    printf("Block dimension: %dx%dx%d (max)\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("Grid dimension: %dx%dx%d (max)\n",  p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("Device copy overlap: %s\n",         p.deviceOverlap ? "yes" : "no");
    printf("Kernel execution timeout: %s\n",    p.kernelExecTimeoutEnabled ? "yes" : "no");
    printf("\n");
  }
}
#pragma endregion




#pragma region CHOOSE DEVICE
/**
 * Choose a CUDA device with atleast compute capability 1.3.
 */
inline void chooseDeviceCuda() {
  // Get the current device.
  int id;
  TRY_CUDA( cudaGetDevice(&id) );
  printf("Current CUDA device: %d\n", id);
  // Select device with atleast compute capability 1.3.
  cudaDeviceProp p;
  memset(&p, 0, sizeof(p));
  p.major = 1;
  p.minor = 3;
  TRY_CUDA( cudaChooseDevice(&id, &p) );
  printf("CUDA device with atleast compute capability 1.3: %d\n", id);
  printf("Cards that have compute capability 1.3 or higher\n"
         "support double-precision floating-point math.\n");
  TRY_CUDA( cudaSetDevice(id) );
  printf("\n");
}
#pragma endregion




#pragma region MALLOC PERFORMANCE
/**
 * Test malloc performance on host and CUDA device.
 */
inline void testMallocPerformanceCuda() {
  const int repeat = REPEAT_METHOD;
  uint8_t *xH[repeat];
  uint8_t *xD[repeat];
  size_t N = 1024 * 1024 * 1024;
  // Test malloc performance on host.
  float tmallocH = measureDuration([&]() {
    for (int i=0; i<repeat; ++i)
      xH[i] = (uint8_t*) malloc(N);
  });
  printf("Host malloc (1 GB): %.2f ms\n", tmallocH / repeat);
  // Test malloc performance on CUDA device.
  float tmallocD = measureDuration([&]() {
    for (int i=0; i<repeat; ++i)
      TRY_CUDA( cudaMalloc(&xD[i], N) );
  });
  printf("CUDA malloc (1 GB): %.2f ms\n", tmallocD / repeat);
  // Test free performance on host.
  float tfreeH = measureDuration([&]() {
    for (int i=0; i<repeat; ++i)
      free(xH[i]);
  });
  printf("Host free (1 GB): %.2f ms\n", tfreeH / repeat);
  // Test free performance on CUDA device.
  float tfreeD = measureDuration([&]() {
    for (int i=0; i<repeat; ++i)
      TRY_CUDA( cudaFree(xD[i]) );
  });
  printf("CUDA free (1 GB): %.2f ms\n", tfreeD / repeat);
  printf("\n");
}
#pragma endregion




#pragma region MEMCPY PERFORMANCE
/**
 * Test memcpy performance on host and CUDA device.
 */
inline void testMemcpyPerformanceCuda() {
  const int repeat = REPEAT_METHOD;
  uint8_t *aH, *xH;
  uint8_t *aD, *xD;
  size_t N = 1024 * 1024 * 1024;
  // Allocate memory on host and CUDA device.
  aH = (uint8_t*) malloc(N);
  xH = (uint8_t*) malloc(N);
  cudaMalloc(&aD, N);
  cudaMalloc(&xD, N);
  float th2h = measureDuration([&]() {
    memcpy(aH, xH, N);
  }, repeat);
  printf("Host to host (1 GB): %.2f ms\n", th2h);
  float th2d = measureDuration([&]() {
    TRY_CUDA( cudaMemcpy(aD, xH, N, cudaMemcpyHostToDevice) );
  }, repeat);
  printf("Host to device (1 GB): %.2f ms\n", th2d);
  float td2h = measureDuration([&]() {
    TRY_CUDA( cudaMemcpy(aH, xD, N, cudaMemcpyDeviceToHost) );
  }, repeat);
  printf("Device to host (1 GB): %.2f ms\n", td2h);
  float td2d = measureDuration([&]() {
    TRY_CUDA( cudaMemcpy(aD, xD, N, cudaMemcpyDeviceToDevice) );
  }, repeat);
  printf("Device to device (1 GB): %.2f ms\n", td2d);
  // Free memory on host and CUDA device.
  free(aH);
  free(xH);
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(xD) );
  printf("\n");
}
#pragma endregion




#pragma region ADDITION
/**
 * Add two numbers on GPU (CUDA kernel).
 * @details
 * The kernel recieves 3 arguments, the first being global address (GPU) of
 * where it must store the result. This has to be done because the kernel cant
 * return any value. The arguments it recieves are managed by CUDA driver and
 * possibly stored in constant memory (right?). A kernel supports all common
 * operators along with various math functions.
 * @param a result (output)
 * @param x first number
 * @param y second number
 */
template <class T>
__global__ void addNumbersCukW(T *a, T x, T y) {
  *a = x + y;
}


/**
 * Add two numbers on GPU.
 */
inline void addNumbersCuda() {
  // Integers "x", "y" are defined in host memory (CPU).
  int x = 1, y = 2;
  // Memory for storing their sum is allocated on device memory (GPU).
  int aH, *aD;
  TRY_CUDA( cudaMalloc(&aD, sizeof(int)) );
  // Sum is computed by the kernel, with one thread (async).
  addNumbersCukW<<<1, 1>>>(aD, x, y);
  // Wait for kernel to complete, then copy the sum to host memory (aH).
  TRY_CUDA( cudaMemcpy(&aH, aD, sizeof(int), cudaMemcpyDeviceToHost) );
  // Free the space we had occupied (we are good people).
  TRY_CUDA( cudaFree(aD) );
  printf("a = %d, b = %d\n", x, y);
  printf("a + b = %d (GPU)\n", aH);
  printf("\n");
}
#pragma endregion




#pragma region VECTOR ADDITION
/**
 * Add two vectors.
 * @param a result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T>
inline void addVectorsW(T *a, const T *x, const T *y, size_t N) {
  for (size_t i=0; i<N; ++i)
    a[i] = x[i] + y[i];
}


/**
 * Examine if one vector is the sum of two other vectors.
 * @param a result vector
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 * @returns true if a = x + y, false otherwise
 */
template <class T>
inline bool examineAddVectors(const T *a, const T *x, const T *y, size_t N) {
  for (size_t i=0; i<N; ++i)
    if (a[i] != x[i] + y[i])
      return false;
  return true;
}


/**
 * Add two vectors on GPU (CUDA kernel).
 * @details
 * Each thread can compute the sum of multiple components of vectors. Each
 * thread computes the sum of its respective component, and shifts by a
 * stride of the total number of vectors. This is done as long as it does
 * not exceed the length of the vectors.
 * @param a result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T>
__global__ void addVectorsCukW(T *a, const T *x, const T *y, size_t N) {
  DEFINE_CUDA(t, b, B, G);
  // Compute sum at respective index, while within bounds.
  // Shift to the next component, by a stride of total no. of threads.
  for (size_t i=t+B*b; i<N; i+=G*B)
    a[i] = x[i] + y[i];
}


/**
 * Add two vectors on GPU.
 */
inline void addVectorsCuda() {
  const int repeat = REPEAT_METHOD;
  int *aH, *xH, *yH;
  int *aD, *xD, *yD;
  size_t N = 1024 * 1024 * 1024 / sizeof(int);
  // Allocate memory on host and CUDA device.
  aH = (int*) malloc(N * sizeof(int));
  xH = (int*) malloc(N * sizeof(int));
  yH = (int*) malloc(N * sizeof(int));
  TRY_CUDA( cudaMalloc(&aD, N * sizeof(int)) );
  TRY_CUDA( cudaMalloc(&xD, N * sizeof(int)) );
  TRY_CUDA( cudaMalloc(&yD, N * sizeof(int)) );
  // Populate vectors with some values.
  for (int i=0; i<N; ++i) {
    xH[i] = i % 1024;
    yH[i] = i % 1024;
  }
  printf("x = vector of size 1 GB\n");
  printf("y = vector of size 1 GB\n");
  // Copy vectors to CUDA device.
  TRY_CUDA( cudaMemcpy(xD, xH, N * sizeof(int), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(yD, yH, N * sizeof(int), cudaMemcpyHostToDevice) );
  // Add vectors on host.
  float thost = measureDuration([&]() {
    addVectorsW(aH, xH, yH, N);
  }, repeat);
  printf("Vector addition on host (a = x + y): %.2f ms\n", thost);
  // Add vectors on CUDA device.
  for (int blockSize=32; blockSize<=1024; blockSize*=2) {
    int gridSize = 1024 * 1024 / blockSize;
    float tdev = measureDuration([&]() {
      addVectorsCukW<<<gridSize, blockSize>>>(aD, xD, yD, N);
      TRY_CUDA( cudaDeviceSynchronize() );
    }, repeat);
    TRY_CUDA( cudaMemcpy(aH, aD, N * sizeof(int), cudaMemcpyDeviceToHost) );
    assert(examineAddVectors(aH, xH, xH, N));
    printf("Vector addition on device <<<%d, %d>>> (a = x + y): %.2f ms\n", gridSize, blockSize, tdev);
  }
  // Free memory on host and CUDA device.
  free(aH);
  free(xH);
  free(yH);
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(xD) );
  TRY_CUDA( cudaFree(yD) );
  printf("\n");
}
#pragma endregion




#pragma region DOT PRODUCT
/**
 * Find the sum of values in a vector.
 * @param x a vector
 * @param N size of vector
 * @returns sum of values in x
 */
template <class T>
inline T sumValues(const T *x, size_t N) {
  T a = T();
  for (size_t i=0; i<N; ++i)
    a += x[i];
  return a;
}


/**
 * Find dot product of two vectors.
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 * @returns x . y
 */
template <class T>
inline T dotProduct(const T *x, const T *y, size_t N) {
  T a = T();
  for (size_t i=0; i<N; ++i)
    a += x[i] * y[i];
  return a;
}


/**
 * Find sum of values in a vector with a thread block (CUDA device function).
 * @param a vector of values (updated, a[0] is the result)
 * @param N size of vector
 * @param i thread index
 */
template <class T>
__device__ void sumValuesBlockCudU(T *a, size_t N, size_t i) {
  // Reduce the sum in the cache to a single value in binary tree fashion.
  for (; N>1;) {
    size_t DN = (N+1)/2;
    if (i<N/2) a[i] += a[DN+i];
    __syncthreads();
    N = DN;
  }
}


/**
 * Find sum of values in a vector (CUDA device function).
 * @param x vector of values
 * @param N size of vector
 * @param i start index
 * @param DI index stride
 * @returns Σ x[i:DI:N]
 */
template <class T>
__device__ T sumValuesCud(const T *x, size_t N, size_t i, size_t DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


/**
 * Find sum of values in a vector, using reduce approach (CUDA kernel).
 * @tparam CACHE shared memory size
 * @param a partial result (updated)
 * @param x vector of values
 * @param N size of vector
 */
template <class T, int CACHE=BLOCK_LIMIT_REDUCE_CUDA>
__global__ void sumValuesReduceCukW(T *a, const T *x, size_t N) {
  __shared__ T cache[CACHE];
  DEFINE_CUDA(t, b, B, G);
  // Store per-thread sum in shared cache (for further reduction).
  cache[t] = sumValuesCud(x, N, B*b+t, G*B);
  // Wait for all threads within the block to finish.
  __syncthreads();
  // Reduce the sum in the cache to a single value in binary tree fashion.
  sumValuesBlockCudU(cache, B, t);
  // Store this per-block sum into a partial result vector.
  if (t==0) a[b] = cache[0];
}


/**
 * Find sum of values in a vector, using atomic-add approach (CUDA kernel).
 * @param a result (updated)
 * @param x vector of values
 * @param N size of vector
 */
template <class T>
__global__ void sumValuesAtomicCukW(T *a, const T *x, size_t N) {
  DEFINE_CUDA(t, b, B, G);
  for (size_t i=B*b+t; i<N; i+=G*B)
    atomicAdd(a, x[i]);
}


/**
 * Find dot product of two vectors on GPU (CUDA device function).
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 * @param i start index
 * @param DI index stride
 * @returns x[i:DI:N] . y[i:DI:N]
 */
template <class T>
__device__ T dotProductCud(const T *x, const T *y, size_t N, size_t i, size_t DI) {
  // Compute sum of pairwise product at respective index, while within bounds.
  // Shift to the next component, by a stride of total no. of threads.
  T a = T();
  for (; i<N; i+=DI)
    a += x[i] * y[i];
  return a;
}


/**
 * Find dot product of two vectors on GPU (CUDA kernel).
 * @details
 * Each thread computes pairwise product of multiple components of vector.
 * Since there are say 10 components, but only a maximum of say 4 total
 * threads, each thread pairwise product of its component, and shifts by
 * a stride of the total number of threads. This is done as long as it
 * does not exceed the length of the vector. Each thread maintains the
 * sum of the pairwise products it calculates.
 *
 * Once pairwise product calculation completes, the per-thread sum is
 * stored in a cache, and then all threads in a block sync up to calculate
 * the sum for the entire block in a binary tree fashion (in log N steps).
 * The overall sum of each block is then stored in an array, which holds
 * this partial sum. This partial sum is completed on the CPU. Hence, our
 * dot product is complete.
 * @tparam CACHE shared memory size
 * @param a partial result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T, int CACHE=BLOCK_LIMIT_REDUCE_CUDA>
__global__ void dotProductCukW(T *a, const T *x, const T *y, size_t N) {
  __shared__ T cache[CACHE];
  DEFINE_CUDA(t, b, B, G);
  // Store per-thread sum in shared cache (for further reduction).
  cache[t] = dotProductCud(x, y, N, B*b+t, G*B);
  // Wait for all threads within the block to finish.
  __syncthreads();
  // Reduce the sum in the cache to a single value in binary tree fashion.
  sumValuesBlockCudU(cache, B, t);
  // Store this per-block sum into a partial result vector.
  if (t==0) a[b] = cache[0];
}


/**
 * Find dot product of two vectors on GPU, using memcpy approach.
 * @param a partial result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T>
inline void dotProductMemcpyCuW(T *a, const T *x, T *y, size_t N) {
  const int B = blockSizeCu(N,    BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, BLOCK_LIMIT_REDUCE_CUDA);
  dotProductCukW<<<G, B>>>(a, x, y, N);
}


/**
 * Find dot product of two vectors on GPU, using inplace approach.
 * @param a partial result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T>
inline void dotProductInplaceCuW(T *a, const T *x, T *y, size_t N) {
  const int B = blockSizeCu(N,    BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, BLOCK_LIMIT_REDUCE_CUDA);
  dotProductCukW<<<G, B>>>(a, x, y, N);
  TRY_CUDA( cudaDeviceSynchronize() );
  sumValuesReduceCukW<<<1, G>>>(a, a, G);
}


/**
 * Find dot product of two vectors on GPU, using atomic-add approach.
 * @param a result (output)
 * @param b partial result vector (output)
 * @param x first vector
 * @param y second vector
 * @param N size of each vector
 */
template <class T>
inline void dotProductAtomicCuW(T *a, T *b, const T *x, T *y, size_t N) {
  const int B = blockSizeCu(N,    BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, BLOCK_LIMIT_REDUCE_CUDA);
  dotProductCukW<<<G, B>>>(b, x, y, N);
  TRY_CUDA( cudaDeviceSynchronize() );
  sumValuesAtomicCukW<<<1, G>>>(a, b, G);
}


/**
 * Find dot product of two vectors on GPU.
 */
inline void dotProductCuda() {
  const int repeat = REPEAT_METHOD;
  double *aH, *bH, *xH, *yH;
  double *aD, *bD, *xD, *yD;
  double ansH = 0, ansD = 0;
  size_t N = 1024 * 1024 * 1024 / sizeof(double);
  size_t R = reduceSizeCu(N);
  // Allocate memory on host and CUDA device.
  aH = (double*) malloc(R * sizeof(double));
  bH = (double*) malloc(R * sizeof(double));
  xH = (double*) malloc(N * sizeof(double));
  yH = (double*) malloc(N * sizeof(double));
  TRY_CUDA( cudaMalloc(&aD, R * sizeof(double)) );
  TRY_CUDA( cudaMalloc(&bD, R * sizeof(double)) );
  TRY_CUDA( cudaMalloc(&xD, N * sizeof(double)) );
  TRY_CUDA( cudaMalloc(&yD, N * sizeof(double)) );
  // Populate vectors with some values.
  for (int i=0; i<N; ++i) {
    xH[i] = 1.0f / (1 + (i % 1024));
    yH[i] = 1.0f / (1 + (i % 1024));
  }
  printf("x = vector of size 1 GB\n");
  printf("y = vector of size 1 GB\n");
  // Copy vectors to CUDA device.
  TRY_CUDA( cudaMemcpy(xD, xH, N * sizeof(double), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(yD, yH, N * sizeof(double), cudaMemcpyHostToDevice) );
  // Find dot product on host.
  float thost = measureDuration([&]() {
    ansH = dotProduct(xH, yH, N);
  }, repeat);
  printf("Dot product on host (a = x . y): %.2f ms [%e]\n", thost, ansH);
  // Find dot product on CUDA device (memcpy approach).
  float tdev0 = measureDuration([&]() {
    dotProductMemcpyCuW(aD, xD, yD, N);
    TRY_CUDA( cudaMemcpy(aH, aD, R * sizeof(double), cudaMemcpyDeviceToHost) );
    ansD = sumValues(aH, R);
  }, repeat);
  printf("Dot product on device (a = x . y): %.2f ms [%e] (memcpy approach)\n", tdev0, ansD);
  // Find dot product on CUDA device (inplace approach).
  float tdev1 = measureDuration([&]() {
    dotProductInplaceCuW(aD, xD, yD, N);
    TRY_CUDA( cudaMemcpy(aH, aD, sizeof(double), cudaMemcpyDeviceToHost) );
    ansD = *aH;
  }, repeat);
  printf("Dot product on device (a = x . y): %.2f ms [%e] (inplace approach)\n", tdev1, ansD);
  // Find dot product on CUDA device (atomic-add approach).
  float tdev2 = measureDuration([&]() {
    TRY_CUDA( cudaMemset(aD, 0, sizeof(double)) );
    dotProductAtomicCuW(aD, bD, xD, yD, N);
    TRY_CUDA( cudaMemcpy(aH, aD, sizeof(double), cudaMemcpyDeviceToHost) );
    ansD = *aH;
  }, repeat);
  printf("Dot product on device (a = x . y): %.2f ms [%e] (atomic-add approach)\n", tdev2, ansD);
  // Free memory on host and CUDA device.
  free(aH);
  free(bH);
  free(xH);
  free(yH);
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(bD) );
  TRY_CUDA( cudaFree(xD) );
  TRY_CUDA( cudaFree(yD) );
  printf("\n");
}
#pragma endregion




#pragma region HISTOGRAM
/**
 * Fill a buffer with random values.
 * @param buf buffer to fill (output)
 * @param N size of buffer
 */
inline void memsetRandW(uint8_t *buf, size_t N) {
  for (size_t i=0; i<N; ++i)
    buf[i] = rand() & 0xFF;
}


/**
 * Find histogram of values in a buffer.
 * @param hist histogram (updated)
 * @param buf buffer of values
 * @param N size of buffer
 */
inline void histogramU(uint32_t *hist, const uint8_t *buf, size_t N) {
  for (size_t i=0; i<N; ++i)
    ++hist[buf[i]];
}


/**
 * Find the sum of all values in a histogram.
 * @param hist histogram
 * @param H size of histogram
 * @returns sum of all values in histogram
 */
inline size_t histogramSum(const uint32_t *hist, size_t N) {
  size_t a = 0;
  for (size_t i=0; i<N; ++i)
    a += hist[i];
  return a;
}


/**
 * Find histogram of values in a buffer on GPU, with direct atomic operations on global memory (CUDA kernel).
 * @details
 * Each thread atomically increments the bytes in buffer meant for it.
 * This however leads to high contention to the 256 locations in the
 * global memory.
 * @param hist histogram (updated)
 * @param buf buffer of values
 * @param N size of buffer
 */
__global__ void histogramBasicCukU(uint32_t *hist, const uint8_t *buf, size_t N) {
  DEFINE_CUDA(t, b, B, G);
  // Get byte at buffer for this thread.
  // Shift to the next byte, by a stride.
  for (size_t i=B*b+t; i<N; i+=G*B) {
    // Atomically increment appropriate index in histogram.
    atomicAdd(&hist[buf[i]], 1);
  }
}


/**
 * Find histogram of values in a buffer on GPU, with direct atomic operations on global memory.
 * @param hist histogram (updated)
 * @param buf buffer of values
 * @param N size of buffer
 */
inline void histogramBasicCuU(uint32_t *hist, const uint8_t *buf, size_t N) {
  const int B = blockSizeCu(N,    BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, BLOCK_LIMIT_REDUCE_CUDA);
  histogramBasicCukU<<<G, B>>>(hist, buf, N);
}


/**
 * Find histogram of values in a buffer on GPU, with atomic operations on shared memory (CUDA kernel).
 * @details
 * Each thread atomically increments the bytes in buffer meant for it.
 * This is done in the shared thread block memory first, until the
 * buffer is consumed. Then each thread in the block updates the
 * histogram in the global memory atomically. This reduces global
 * memory contention.
 * @param hist histogram (updated)
 * @param buf buffer of values
 * @param N size of buffer
 */
__global__ void histogramSharedCukU(uint32_t *hist, const uint8_t *buf, size_t N) {
  DEFINE_CUDA(t, b, B, G);
  // Initialize shared memory (of size 256).
  const int H = 256;  // Histogram size.
  __shared__ uint32_t cache[H];
  cache[t] = 0;
  __syncthreads();
  // Get byte at buffer for this thread.
  // Shift to the next byte, by a stride.
  for (size_t i=B*b+t; i<N; i+=G*B) {
    // Atomically increment appropriate index in shared memory.
    atomicAdd(&cache[buf[i]], 1);
  }
  // Wait for all threads within the block to finish.
  __syncthreads();
  // Atomically update per-block histogram into global histogram.
  atomicAdd(&hist[t], cache[t]);
}


/**
 * Find histogram of values in a buffer on GPU, with atomic operations on shared memory.
 * @param hist histogram (updated)
 * @param buf buffer of values
 * @param N size of buffer
 */
inline void histogramSharedCuU(uint32_t *hist, const uint8_t *buf, size_t N) {
  const int B = blockSizeCu(N,    BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, BLOCK_LIMIT_REDUCE_CUDA);
  histogramSharedCukU<<<G, B>>>(hist, buf, N);
}


/**
 * Find histogram of values in a buffer on GPU.
 */
inline void histogramCuda() {
  const int repeat = REPEAT_METHOD;
  uint32_t *histH, *histD;
  uint8_t  *bufH,  *bufD;
  size_t H = 256;                 // Histogram size.
  size_t N = 1024 * 1024 * 1024;  // Buffer size.
  size_t sumH = 0,  sumD = 0;     // Histogram sum.
  // Allocate memory on host and CUDA device.
  histH = (uint32_t*) malloc(H * sizeof(uint32_t));
  bufH  = (uint8_t*)  malloc(N * sizeof(uint8_t));
  TRY_CUDA( cudaMalloc(&histD, H * sizeof(uint32_t)) );
  TRY_CUDA( cudaMalloc(&bufD,  N * sizeof(uint8_t)) );
  // Populate buffer with some random values.
  memsetRandW(bufH, N);
  printf("buf = vector of size 1 GB\n");
  // Copy buffer to CUDA device.
  TRY_CUDA( cudaMemcpy(bufD, bufH, N * sizeof(uint8_t), cudaMemcpyHostToDevice) );
  // Find histogram on host.
  float thost = measureDurationMarked([&](auto mark) {
    memset(histH, 0, H * sizeof(uint32_t));
    mark([&]() { histogramU(histH, bufH, N); });
  }, repeat);
  sumH = histogramSum(histH, H);
  printf("Finding histogram of buf on host: %.2f ms\n", thost);
  // Find histogram on CUDA device (basic approach).
  float tdev0 = measureDurationMarked([&](auto mark) {
    TRY_CUDA( cudaMemset(histD, 0, H * sizeof(uint32_t)) );
    TRY_CUDA( cudaDeviceSynchronize() );
    mark([&]() {
      histogramBasicCuU(histD, bufD, N);
      TRY_CUDA( cudaDeviceSynchronize() );
    });
    TRY_CUDA( cudaMemcpy(histH, histD, H * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
  }, repeat);
  sumD = histogramSum(histH, H);
  assert(sumH == sumD);
  printf("Finding histogram of buf on device (basic approach): %.2f ms\n", tdev0);
  // Find histogram on CUDA device (shared approach).
  float tdev1 = measureDurationMarked([&](auto mark) {
    TRY_CUDA( cudaMemset(histD, 0, H * sizeof(uint32_t)) );
    TRY_CUDA( cudaDeviceSynchronize() );
    mark([&]() {
      histogramSharedCuU(histD, bufD, N);
      TRY_CUDA( cudaDeviceSynchronize() );
    });
    TRY_CUDA( cudaMemcpy(histH, histD, H * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
  }, repeat);
  sumD = histogramSum(histH, H);
  assert(sumH == sumD);
  printf("Finding histogram of buf on device (shared approach): %.2f ms\n", tdev1);
  // Free memory on host and CUDA device.
  free(histH);
  free(bufH);
  TRY_CUDA( cudaFree(histD) );
  TRY_CUDA( cudaFree(bufD) );
  printf("\n");
}
#pragma endregion




#pragma region MATRIX MULTIPLICATION
/**
 * Populate a matrix with some values.
 * @param a matrix to populate (output)
 * @param AR number of rows in matrix
 * @param AC number of columns in matrix
 */
template <class T>
inline void populateMatrixW(T *a, size_t AR, size_t AC) {
  for (size_t r=0; r<AR; ++r) {
    for (size_t c=0; c<AC; ++c)
      a[AC*r + c] = T(1) / (1 + (r + c) % 1024);
  }
}


/**
 * Multiply two matrices.
 * @param a result matrix (output)
 * @param x first matrix
 * @param y second matrix
 * @param XR number of rows in first matrix
 * @param XC number of columns in first matrix
 * @param YC number of columns in second matrix
 */
template <class T>
inline void multiplyMatricesW(T *a, const T *x, const T *y, size_t XR, size_t XC, size_t YC) {
  for (size_t r=0; r<XR; ++r) {
    for (size_t c=0; c<YC; ++c) {
      T sum = T();
      for (size_t i=0; i<XC; ++i)
        sum += x[XC*r + i] * y[YC*i + c];
      a[YC*r + c] = sum;
    }
  }
}


/**
 * Multiply two matrices on GPU, with basic approach (CUDA kernel).
 * @param a result matrix (output)
 * @param x first matrix
 * @param y second matrix
 * @param XR number of rows in first matrix
 * @param XC number of columns in first matrix
 * @param YC number of columns in second matrix
 */
template <class T>
__global__ void multiplyMatricesBasicCukW(T *a, const T *x, const T *y, size_t XR, size_t XC, size_t YC) {
  DEFINE2D_CUDA(tx, ty, bx, by, BX, BY, GX, GY);
  size_t r = BY*by + ty;
  size_t c = BX*bx + tx;
  if (r >= XR) return;
  if (c >= YC) return;
  T sum = T();
  for (size_t i=0; i<XC; ++i)
    sum += x[XC*r + i] * y[YC*i + c];
  a[YC*r + c] = sum;
}


/**
 * Multiply two matrices on GPU, with basic approach.
 * @param a result matrix (output)
 * @param x first matrix
 * @param y second matrix
 * @param XR number of rows in first matrix
 * @param XC number of columns in first matrix
 * @param YC number of columns in second matrix
 */
template <class T, int TILEX=32, int TILEY=32>
inline void multiplyMatricesBasicCuW(T *a, const T *x, const T *y, size_t XR, size_t XC, size_t YC) {
  const int BX = TILEX;
  const int BY = TILEY;
  const int GX = (YC + BX - 1) / BX;
  const int GY = (XR + BY - 1) / BY;
  multiplyMatricesBasicCukW<<<dim3(GX, GY), dim3(BX, BY)>>>(a, x, y, XR, XC, YC);
}


/**
 * Multiply two matrices on GPU, with tiled approach (CUDA kernel).
 * @tparam TILEX tile size in X dimension
 * @tparam TILEY tile size in Y dimension
 * @param a result matrix (output)
 * @param x first matrix
 * @param y second matrix
 * @param XR number of rows in first matrix
 * @param XC number of columns in first matrix
 * @param YC number of columns in second matrix
 */
template <class T, int TILEX=32, int TILEY=32>
__global__ void multiplyMatricesTiledCukW(T *a, const T *x, const T *y, size_t XR, size_t XC, size_t YC) {
  DEFINE2D_CUDA(tx, ty, bx, by, BX, BY, GX, GY);
  __shared__ T at[TILEY * TILEX];
  __shared__ T xt[TILEY * TILEX];
  __shared__ T yt[TILEY * TILEX];
  size_t r = BY*by + ty;
  size_t c = BX*bx + tx;
  if (r >= XR) return;
  if (c >= YC) return;
  at[BX*ty + tx] = T();
  for (size_t i=0; i<XC; i+=BX) {
    __syncthreads();
    xt[BX*ty + tx] = x[XC*r + i+tx];
    yt[BX*ty + tx] = y[YC*(i+ty) + c];
    __syncthreads();
    for (size_t j=0; j<BX; ++j)
      at[BX*ty + tx] += xt[BX*ty + j] * yt[BX*j + tx];
  }
  __syncthreads();
  a[YC*r + c] = at[BX*ty + tx];
}


/**
 * Multiply two matrices on GPU, with tiled approach.
 * @tparam TILEX tile size in X dimension
 * @tparam TILEY tile size in Y dimension
 * @param a result matrix (output)
 * @param x first matrix
 * @param y second matrix
 * @param XR number of rows in first matrix
 * @param XC number of columns in first matrix
 * @param YC number of columns in second matrix
 */
template <class T, int TILEX=32, int TILEY=32>
inline void multiplyMatricesTiledCuW(T *a, const T *x, const T *y, size_t XR, size_t XC, size_t YC) {
  const int BX = TILEX;
  const int BY = TILEY;
  const int GX = (YC + BX - 1) / BX;
  const int GY = (XR + BY - 1) / BY;
  multiplyMatricesTiledCukW<T, TILEX, TILEY><<<dim3(GX, GY), dim3(BX, BY)>>>(a, x, y, XR, XC, YC);
}


/**
 * Multiply two matrices on GPU.
 */
inline void multiplyMatricesCuda() {
  const int repeat = REPEAT_METHOD;
  double *aH, *xH, *yH;
  double *aD, *xD, *yD;
  size_t N  = 32 * 1024 * 1024 / sizeof(double);
  size_t XR = sqrt(N), XC = XR, YC = XR;
  size_t sumH = 0, sumD = 0;     // Sum of all values in result matrix.
  // Allocate memory on host and CUDA device.
  aH = (double*) malloc(XR * YC * sizeof(double));
  xH = (double*) malloc(XR * XC * sizeof(double));
  yH = (double*) malloc(XC * YC * sizeof(double));
  TRY_CUDA( cudaMalloc(&aD, XR * YC * sizeof(double)) );
  TRY_CUDA( cudaMalloc(&xD, XR * XC * sizeof(double)) );
  TRY_CUDA( cudaMalloc(&yD, XC * YC * sizeof(double)) );
  // Populate matrices with some values.
  populateMatrixW(xH, XR, XC);
  populateMatrixW(yH, XC, YC);
  printf("x = matrix of size 16 MB\n");
  printf("y = matrix of size 16 MB\n");
  // Copy matrices to CUDA device.
  TRY_CUDA( cudaMemcpy(xD, xH, XR * XC * sizeof(double), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(yD, yH, XC * YC * sizeof(double), cudaMemcpyHostToDevice) );
  // Multiply matrices on host.
  float thost = measureDuration([&]() {
    multiplyMatricesW(aH, xH, yH, XR, XC, YC);
  }, repeat);
  sumH = sumValues(aH, XR * YC);
  printf("Matrix multiplication on host (a = x * y): %.2f ms [%e]\n", thost, sumH);
  // Multiply matrices on CUDA device (basic approach).
  float tdev0 = measureDuration([&]() {
    multiplyMatricesBasicCuW(aD, xD, yD, XR, XC, YC);
    TRY_CUDA( cudaDeviceSynchronize() );
  }, repeat);
  TRY_CUDA( cudaMemcpy(aH, aD, XR * YC * sizeof(double), cudaMemcpyDeviceToHost) );
  sumD = sumValues(aH, XR * YC);
  printf("Matrix multiplication on device (a = x * y): %.2f ms (basic approach) [%e]\n", tdev0, sumD);
  // Multiply matrices on CUDA device (tiled approach).
  float tdev1 = measureDuration([&]() {
    multiplyMatricesTiledCuW(aD, xD, yD, XR, XC, YC);
    TRY_CUDA( cudaDeviceSynchronize() );
  }, repeat);
  sumD = sumValues(aH, XR * YC);
  printf("Matrix multiplication on device (a = x * y): %.2f ms (tiled approach) [%e]\n", tdev1, sumD);
  // Free memory on host and CUDA device.
  free(aH);
  free(xH);
  free(yH);
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(xD) );
  TRY_CUDA( cudaFree(yD) );
  printf("\n");
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 */
void runExperiment() {
  printf("HELLO WORLD:\n");
  sayHelloCuda();
  printf("DEVICE PROPERTIES:\n");
  listDevicePropertiesCuda();
  printf("CHOOSE DEVICE:\n");
  chooseDeviceCuda();
  printf("MALLOC PERFORMANCE:\n");
  testMallocPerformanceCuda();
  printf("MEMCPY PERFORMANCE:\n");
  testMemcpyPerformanceCuda();
  printf("ADDITION:\n");
  addNumbersCuda();
  printf("VECTOR ADDITION:\n");
  addVectorsCuda();
  printf("DOT PRODUCT:\n");
  dotProductCuda();
  printf("HISTOGRAM:\n");
  histogramCuda();
  printf("MATRIX MULTIPLICATION:\n");
  multiplyMatricesCuda();
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  runExperiment();
  return 0;
}
#pragma endregion
#pragma endregion
