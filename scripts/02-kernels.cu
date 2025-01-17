// CUDA Kernels Example: Vector Addition
// Created by Derek Rosenzweig on 12/26/24.

// CUDA Kernels Example: Vector Addition
// Created by Derek Rosenzweig on 12/26/24.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;  // Get unique thread ID
    C[i] = A[i] + B[i];   // Perform element-wise addition
}

int main() {
    const int N = 256;  // Number of elements
    
    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    
    // Initialize vectors A and B
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    
    // Copy the result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("Error at index %d: %f\n", i, h_C[i]);
            break;
        }
    }
    printf("Vector addition completed successfully.\n");
    
    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
