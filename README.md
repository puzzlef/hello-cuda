A basic "Hello world" or "Hello CUDA" example to perform a number of operations on NVIDIA GPUs using [CUDA].

> **Note**
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cu`.

<br>

```bash
$ bash main.sh

# Cloning into 'hello-cuda'...
# remote: Enumerating objects: 33, done.
# remote: Counting objects: 100% (12/12), done.
# remote: Compressing objects: 100% (11/11), done.
# remote: Total 33 (delta 2), reused 6 (delta 1), pack-reused 21
# Receiving objects: 100% (33/33), 24.58 KiB | 719.00 KiB/s, done.
# Resolving deltas: 100% (9/9), done.
# HELLO WORLD:
# GPU[B1.T0]: Hello CUDA
# GPU[B1.T1]: Hello CUDA
# GPU[B1.T2]: Hello CUDA
# GPU[B1.T3]: Hello CUDA
# GPU[B1.T4]: Hello CUDA
# GPU[B1.T5]: Hello CUDA
# GPU[B1.T6]: Hello CUDA
# GPU[B1.T7]: Hello CUDA
# GPU[B3.T0]: Hello CUDA
# GPU[B3.T1]: Hello CUDA
# GPU[B3.T2]: Hello CUDA
# GPU[B3.T3]: Hello CUDA
# GPU[B3.T4]: Hello CUDA
# GPU[B3.T5]: Hello CUDA
# GPU[B3.T6]: Hello CUDA
# GPU[B3.T7]: Hello CUDA
# GPU[B2.T0]: Hello CUDA
# GPU[B2.T1]: Hello CUDA
# GPU[B2.T2]: Hello CUDA
# GPU[B2.T3]: Hello CUDA
# GPU[B2.T4]: Hello CUDA
# GPU[B2.T5]: Hello CUDA
# GPU[B2.T6]: Hello CUDA
# GPU[B2.T7]: Hello CUDA
# GPU[B0.T0]: Hello CUDA
# GPU[B0.T1]: Hello CUDA
# GPU[B0.T2]: Hello CUDA
# GPU[B0.T3]: Hello CUDA
# GPU[B0.T4]: Hello CUDA
# GPU[B0.T5]: Hello CUDA
# GPU[B0.T6]: Hello CUDA
# GPU[B0.T7]: Hello CUDA
# CPU: Hello world!

# DEVICE PROPERTIES:
# COMPUTE DEVICE 0:
# Name: Tesla V100-PCIE-16GB
# Compute capability: 7.0
# Multiprocessors: 80
# Clock rate: 1380 MHz
# Global memory: 16151 MB
# Constant memory: 64 KB
# Shared memory per block: 48 KB
# Registers per block: 65536
# Threads per block: 1024 (max)
# Threads per warp: 32
# Block dimension: 1024x1024x64 (max)
# Grid dimension: 2147483647x65535x65535 (max)
# Device copy overlap: yes
# Kernel execution timeout: no

# CHOOSE DEVICE:
# Current CUDA device: 0
# CUDA device with atleast compute capability 1.3: 0
# Cards that have compute capability 1.3 or higher
# support double-precision floating-point math.

# MALLOC PERFORMANCE:
# Host malloc (1 GB): 0.00 ms
# CUDA malloc (1 GB): 1.35 ms
# Host free (1 GB): 0.00 ms
# CUDA free (1 GB): 1.51 ms

# MEMCPY PERFORMANCE:
# Host to host (1 GB): 412.59 ms
# Host to device (1 GB): 225.32 ms
# Device to host (1 GB): 246.87 ms
# Device to device (1 GB): 0.04 ms

# ADDITION:
# a = 1, b = 2
# a + b = 3 (GPU)

# VECTOR ADDITION:
# x = vector of size 1 GB
# y = vector of size 1 GB
# Vector addition on host (a = x + y): 438.02 ms
# Vector addition on device <<<32768, 32>>> (a = x + y): 4.33 ms
# Vector addition on device <<<16384, 64>>> (a = x + y): 3.98 ms
# Vector addition on device <<<8192, 128>>> (a = x + y): 4.01 ms
# Vector addition on device <<<4096, 256>>> (a = x + y): 3.97 ms
# Vector addition on device <<<2048, 512>>> (a = x + y): 4.00 ms
# Vector addition on device <<<1024, 1024>>> (a = x + y): 3.97 ms

# DOT PRODUCT:
# x = vector of size 1 GB
# y = vector of size 1 GB
# Dot product on host (a = x . y): 207.39 ms [2.154769e+05]
# Dot product on device (a = x . y): 2.69 ms [2.154769e+05] (memcpy approach)
# Dot product on device (a = x . y): 2.50 ms [2.154769e+05] (inplace approach)
# Dot product on device (a = x . y): 2.50 ms [2.154769e+05] (atomic-add approach)

# HISTOGRAM:
# buf = vector of size 1 GB
# Finding histogram of buf on host: 747.00 ms
# Finding histogram of buf on device (basic approach): 401.06 ms
# Finding histogram of buf on device (shared approach): 6.85 ms

# MATRIX MULTIPLICATION:
# x = matrix of size 16 MB
# y = matrix of size 16 MB
# Matrix multiplication on host (a = x * y): 33307.13 ms [3.287916e+00]
# Matrix multiplication on device (a = x * y): 18.93 ms (basic approach) [3.287916e+00]
# Matrix multiplication on device (a = x * y): 12.20 ms (tiled approach) [3.287916e+00]
```

<br>
<br>


## References

- [CUB Documentation](https://nvlabs.github.io/cub/)
- [moderngpu/moderngpu: Patterns and behaviors for GPU computing](https://github.com/moderngpu/moderngpu)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [CUDA atomicAdd for doubles definition error](https://stackoverflow.com/a/37569519/1413259)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)

<br>
<br>

[![](https://img.youtube.com/vi/8sDg-lD1fZQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=8sDg-lD1fZQ)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/679802777.svg)](https://zenodo.org/doi/10.5281/zenodo.10030458)


[CUDA]: https://docs.nvidia.com/cuda/index.html
