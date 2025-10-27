# Assignment 3 — CUDA Practice Exercises

Goal: Build the CUDA skills needed for asst3: device memory management, kernel launches, timing and synchronization, atomics and avoiding races, prefix-sum and scatter pipelines, and a correct and fast circle renderer using tiling/binning while preserving per-pixel update order.

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

# Part 2 — Races, Atomics, and Safer Patterns

Goal: See races on the GPU and practice safe updates.

## Race Condition Demo
- Allocate a device array of float initialized to 0.
- Launch many threads that do arr[i % K] += 1.0f; without atomics. Observe nondeterministic results.

## Atomic Add on Floats
- Repeat using atomicAdd(&arr[idx], 1.0f); and verify determinism for a single component.
- Discuss limitation: blending a 4-float RGBA requires updating multiple values atomically and in order; per-component atomics alone do not preserve ordering across contributing circles.

## Per-Pixel Ownership (No Locks)
- Implement a simple owner computes pattern: assign one thread per pixel; each pixel-thread reads contributing circles and blends locally, then writes once. This trivially ensures atomicity and per-pixel order without locks. Use this as a mental model for the renderer’s correctness strategy.

# Part 3 — Parallel Prefix-Sum (Scan) and Find Repeats

Goal: Implement Blelloch exclusive scan and use it to compact indices, mirroring the warm-up.

## Blelloch Exclusive Scan (Power-of-2 Size)
- Write device kernels for upsweep and downsweep phases on an array in global memory.
- Launch one kernel per level of the tree (per two_d). For each level, threads process disjoint ranges with stride two_dplus1.
- After upsweep, set the last element to 0, then perform downsweep.
- Provide a host wrapper that rounds allocation to next power of two but only copies back the first N results.

## Tests
- Verify on small arrays by comparing to a CPU reference for random inputs and ones inputs.
- Vary N to include non-powers-of-two (padding logic should make this pass).

## Find Repeats via Mark–Scan–Scatter
- Kernel 1 (mark): for each i, set marks[i] = (A[i] == A[i+1]) ? 1 : 0 and marks[N-1] = 0.
- Run exclusive_scan(marks) -> offsets.
- Kernel 2 (scatter): for each i with marks[i] == 1, write i into out[offsets[i]].
- Validate against a CPU implementation on random data.

# Part 4 — Circle Rendering Correctness

Goal: Ensure atomicity and per-pixel order without locks.

## CPU Reference Sanity
- Study the provided CPU reference and its blending rule. Render small scenes and verify ordering behavior for overlapping circles with alpha.

## Naive CUDA (Circle-per-Thread) and Its Bug
- Implement a kernel that assigns one circle per thread and writes its contributions into the framebuffer. Observe streaking and non-determinism on rand10k and rgb.
- Explain why: multiple circles race to update the same pixel; no guaranteed order between circles for a given pixel; multi-component updates are non-atomic.

## Correctness Strategy: Pixel- or Tile-Ownership
- Option A (pixel-per-thread): each thread owns one pixel, loops through all circles in input order, tests coverage, and blends in sequence. This is correct but slow for many circles.
- Option B (tile-per-block + pixel-per-thread): partition the image into tiles; bin circles into overlapping tiles; inside a tile, each thread owns one pixel and loops over only the tile’s circles in input order. This preserves per-pixel order and avoids races.

# Part 5 — Binning and Tiling Pipeline

Goal: Build the data-parallel pipeline needed for a fast, correct renderer.

## Circle–Tile Binning (Uniform Grid)
- Choose tile size, e.g., 16x16 or 32x32 pixels. Compute grid dims from image size.
- Kernel A (count): for each circle, compute the AABB of covered tiles; atomically increment tileCounts[t] for each overlapped tile.
- Exclusive scan tileCounts to build tileOffsets and totalRefs. Allocate tileRefs[totalRefs].
- Kernel B (scatter): for each circle, write its ID into the corresponding range of tileRefs for each overlapped tile using per-tile atomic index initialized to tileOffsets[t].
- Invariant: within a given tile’s reference list, circle IDs appear in input order if you scatter by increasing circle ID per tile. Enforce stable order by having each thread iterate circle IDs ascending.

## Tile Renderer (Owner-Computes)
- Launch one block per tile; one thread per pixel in the tile (pad for edges).
- Each pixel-thread loops the tile’s circle IDs in order; for each circle, early-reject via circle–box/pixel tests; if covered, blend into a register accumulator; after the loop, write one RGBA to global memory.
- This guarantees atomicity (single writer per pixel) and per-pixel input order.

## Early-Reject and Helpers
- Use the provided circleBoxTest and circlePixelTest helpers to prune work early.
- Consider culling by per-tile AABB during binning to keep reference lists small.

# Part 6 — Optimizations and Measurement

Goal: Make it fast and verify with the checker.

## Memory Layout and Coalescing
- Prefer structure-of-arrays (SoA) for circle data accessed by all threads: separate cx[], cy[], r[], rgba[] to enable coalesced loads.
- For tile loops, prefetch a small batch of circle attributes into registers to reduce repeated global loads.

## Shared Memory Where It Helps
- If a tile references many nearby circles repeatedly, load a chunk of k circle records into shared memory per iteration; synchronize within the block, iterate, then proceed to next chunk. Balance smem size and occupancy.

## Control Divergence and Occupancy
- Choose blockDim to map cleanly to tile size (e.g., 16x16 = 256 threads). Keep register use reasonable to avoid throttling occupancy.
- Use grid-stride loops in kernels that traverse variable amounts of work (e.g., binning) to scale to large inputs.

## Kernel Fusion vs. Clarity
- Keep binning and rendering separate kernels for clarity and profiling. Consider fusing only if it reduces memory traffic without hurting occupancy.

## Timing and Validation
- Use ./render -r cuda --check scenename during development.
- Use ./checker.py in /render to see correctness flags and performance vs reference across scenes.
- Report per-scene timings; focus on rand10k, rand100k, rand1M, micro2M, and the patterned scenes.

# Part 7 — Debugging Tips

Goal: Iterate confidently on correctness issues.

## Determinism Checks
- Render the same scene twice and diff frames to detect nondeterminism. If frames differ, you still have races.

## Reduce to Small Cases
- Use tiny images (e.g., 64x64) and few circles; print or dump intermediate buffers (tile counts/offsets/refs) to validate binning.

## cuda-memcheck and Asserts
- Run with cuda-memcheck to catch out-of-bounds and invalid accesses.
- Use device-side assert to validate invariants (e.g., index bounds) in debug builds.

# Stretch Goals (Optional)

Goal: Explore advanced directions once core requirements are solid.

- Warp-level primitives: use __ballot_sync or __any_sync to short-circuit when no pixels in a warp are covered by a circle iteration.
- Sorting within tiles: if needed, ensure per-tile references are strictly ascending by circle ID via a stable scatter or by radix sort on small lists.
- CPU path: build a parallel CPU renderer using a similar tiling/binning approach and compare to your GPU.

# What to Hand In (Mapping to Asst3)

- Part 1/2 practice supports the SAXPY warm-up and understanding of timing/races.
- Part 3 matches the scan/find_repeats warm-up pipeline.
- Parts 4–6 map directly to the CUDA circle renderer: correctness first (atomicity + per-pixel order), then performance (binning/tiling, culling, memory/layout tuning).

