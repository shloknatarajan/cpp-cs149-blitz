# Part 1 — CUDA Basics and Timing

Goal: Get comfortable with kernels, grids/blocks, and timing.

## Hello CUDA Kernel
- Write a kernel that adds a constant c to each element of an array A.
- Launch with <<<numBlocks, threadsPerBlock>>> and compute globalIdx = blockIdx.x * blockDim.x + threadIdx.x.
- Use a grid-stride loop so your kernel works for any N: for (int i = globalIdx; i < N; i += gridDim.x * blockDim.x) { ... }

## Device Memory + Error Checking
- Allocate device buffers with cudaMalloc, copy with cudaMemcpy, free with cudaFree.
- Add a macro to check and fail fast on CUDA errors: #define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(err)); exit(1);} } while(0)
- Use CUDA_CHECK(cudaGetLastError()) after kernel launch.

## Timing: Events vs Synchronize
- Time a kernel two ways:
- 1) End-to-end (H2D + kernel + D2H): surround the cudaMemcpy to device, kernel, and cudaMemcpy back to host.
- 2) Kernel-only: record CUDA events around just the kernel and cudaEventElapsedTime, or call cudaDeviceSynchronize() before stopping a CPU timer. Compare results and explain the difference.