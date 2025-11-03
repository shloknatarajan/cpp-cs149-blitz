#include <stdio.h>

// This function will be executed on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch the kernel on the GPU
    // <<<1, 1>>> specifies one block and one thread
    helloFromGPU<<<1, 1>>>(); 

    // Synchronize the device to ensure the kernel completes before the CPU continues
    cudaDeviceSynchronize(); 

    printf("Hello World from CPU!\n");
    return 0;
}