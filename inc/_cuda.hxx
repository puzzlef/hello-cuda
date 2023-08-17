#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "_debug.hxx"
#include "_cmath.hxx"

using std::fprintf;
using std::exit;




#pragma region KEYWORDS
#ifndef __global__
/** CUDA kernel function. */
#define __global__

/** CUDA host function. */
#define __host__

/** CUDA device function. */
#define __device__

/** CUDA shared memory. */
#define __shared__

/**
 * Synchronize all threads in a block.
 */
#define __syncthreads()
#endif
#pragma endregion




#pragma region LAUNCH CONFIG
#ifndef BLOCK_LIMIT_CUDA
/** Maximum number of threads per block. */
#define BLOCK_LIMIT_CUDA         1024
/** Maximum number of threads per block, when using a map-like kernel. */
#define BLOCK_LIMIT_MAP_CUDA     256
/** Maximum number of threads per block, when using a reduce-like kernel. */
#define BLOCK_LIMIT_REDUCE_CUDA  256
#endif


#ifndef GRID_LIMIT_CUDA
/** Maximum number of blocks per grid. */
#define GRID_LIMIT_CUDA          2147483647  // 2^31 - 1
/** Maximum number of blocks per grid, when using a map-like kernel. */
#define GRID_LIMIT_MAP_CUDA      65535
/** Maximum number of blocks per grid, when using a reduce-like kernel. */
#define GRID_LIMIT_REDUCE_CUDA   1024
#endif


/**
 * Get the block size for kernel launch, based on number of elements to process.
 * @param N number of elements to process
 * @param BLIM block limit
 * @returns block size
 */
inline int blockSizeCu(size_t N, int BLIM=BLOCK_LIMIT_CUDA) noexcept {
  return int(min(N, size_t(BLIM)));
}


/**
 * Get the grid size for kernel launch, based on number of elements to process.
 * @param N number of elements to process
 * @param B block size
 * @param GLIM grid limit
 * @returns grid size
 */
inline int gridSizeCu(size_t N, int B, int GLIM=GRID_LIMIT_CUDA) noexcept {
  return int(min(ceilDiv(N, size_t(B)), size_t(GLIM)));
}


/**
 * Get the number of elements produced by a reduce-like kernel.
 * @param N number of elements to process
 * @returns number of reduced elements
 */
inline int reduceSizeCu(size_t N) noexcept {
  const int B = blockSizeCu(N,   BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_REDUCE_CUDA);
  return G;
}
#pragma endregion




#pragma region TRY
#ifndef TRY_CUDA
/**
 * Log error on CUDA function call failure.
 * @param err error code
 * @param exp expression string
 * @param func current function name
 * @param line current line number
 * @param file current file name
 */
void tryFailedCuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
    "ERROR: %s: %s\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    cudaGetErrorName(err), cudaGetErrorString(err), exp, func, line, file);
  exit(err);
}

/**
 * Try to execute a CUDA function call.
 * @param exp expression to execute
 */
#define TRY_CUDA(exp)  do { cudaError err = exp; if (err != cudaSuccess) tryFailedCuda(err, #exp, __func__, __LINE__, __FILE__); } while (0)

/**
 * Try to execute a CUDA function call only if build mode is error or higher.
 * @param exp expression to execute
 **/
#define TRY_CUDAE(exp)  PERFORME(TRY_CUDA(exp))

/**
 * Try to execute a CUDA function call only if build mode is warning or higher.
 * @param exp expression to execute
 **/
#define TRY_CUDAW(exp)  PERFORMW(TRY_CUDA(exp))

/**
 * Try to execute a CUDA function call only if build mode is info or higher.
 * @param exp expression to execute
 **/
#define TRY_CUDAI(exp)  PERFORMI(TRY_CUDA(exp))

/**
 * Try to execute a CUDA function call only if build mode is debug or higher.
 * @param exp expression to execute
 **/
#define TRY_CUDAD(exp)  PERFORMD(TRY_CUDA(exp))

/**
 * Try to execute a CUDA function call only if build mode is trace.
 * @param exp expression to execute
 **/
#define TRY_CUDAT(exp)  PERFORMT(TRY_CUDA(exp))
#endif
#pragma endregion




#pragma region UNUSED
/**
 * Mark CUDA variable as unused.
 */
template <class T>
__device__ void unusedCuda(T&&) {}


#ifndef UNUSED_CUDA
/**
 * Mark CUDA variable as unused.
 * @param x variable to mark as unused
 */
#define UNUSED_CUDA(x)  unusedCuda(x)
#endif
#pragma endregion




#pragma region DEFINE
#ifndef DEFINE_CUDA
/**
 * Define thread, block variables for CUDA.
 * @param t thread index
 * @param b block index
 * @param B block size
 * @param G grid size
 */
#define DEFINE_CUDA(t, b, B, G) \
  const int t = threadIdx.x; \
  const int b = blockIdx.x; \
  const int B = blockDim.x; \
  const int G = gridDim.x; \
  UNUSED_CUDA(t); \
  UNUSED_CUDA(b); \
  UNUSED_CUDA(B); \
  UNUSED_CUDA(G);


/**
 * Define 2D thread, block variables for CUDA.
 * @param tx thread x index
 * @param ty thread y index
 * @param bx block x index
 * @param by block y index
 * @param BX block x size
 * @param BY block y size
 * @param GX grid x size
 * @param GY grid y size
 */
#define DEFINE2D_CUDA(tx, ty, bx, by, BX, BY, GX, GY) \
  const int tx = threadIdx.x; \
  const int ty = threadIdx.y; \
  const int bx = blockIdx.x; \
  const int by = blockIdx.y; \
  const int BX = blockDim.x; \
  const int BY = blockDim.y; \
  const int GX = gridDim.x;  \
  const int GY = gridDim.y; \
  UNUSED_CUDA(tx); \
  UNUSED_CUDA(ty); \
  UNUSED_CUDA(bx); \
  UNUSED_CUDA(by); \
  UNUSED_CUDA(BX); \
  UNUSED_CUDA(BY); \
  UNUSED_CUDA(GX); \
  UNUSED_CUDA(GY);
#endif
#pragma endregion
