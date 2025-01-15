// Unified Memory Example: Vector Addition
#include <cuda_runtime.h>
#include <cstdio>

// Kernel for vector addition
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // Vector size (1 million elements)
    size_t bytes = N * sizeof(float);

    // Allocate Unified Memory accessible from CPU and GPU
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Initialize vectors A and B on the host
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Define thread hierarchy
    int blockSize = 256; // Threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    VecAdd<<<numBlocks, blockSize>>>(A, B, C, N);

    // Wait for GPU to finish before accessing results on the host
    cudaDeviceSynchronize();

    // Verify results
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Error at index %d: %f\n", i, C[i]);
            return -1;
        }
    }
    printf("Vector addition completed successfully.\n");

    // Free Unified Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}