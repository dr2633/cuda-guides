// Thread Hierarchy Example: Matrix Addition with Synchronization and Benchmarking
// Created by Derek Rosenzweig on 12/26/24

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

const int N = 1024;  // Matrix size (NxN)

// Kernel definition for matrix addition with synchronization
__global__ void MatAdd(float* A, float* B, float* C, int width) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID in x-dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Global thread ID in y-dimension

    // Allocate shared memory for cooperative data sharing within the block
    __shared__ float sharedA[16][16];
    __shared__ float sharedB[16][16];

    // Perform the addition if indices are within matrix bounds
    if (i < width && j < width) {
        int idx = j * width + i;  // Flatten 2D index

        // Load data into shared memory
        sharedA[threadIdx.y][threadIdx.x] = A[idx];
        sharedB[threadIdx.y][threadIdx.x] = B[idx];
        __syncthreads();  // Synchronize all threads in the block

        // Perform addition using shared memory
        C[idx] = sharedA[threadIdx.y][threadIdx.x] + sharedB[threadIdx.y][threadIdx.x];
    }
}

int main() {
    const int matrixSize = N * N;
    const int bytes = matrixSize * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host matrices
    for (int i = 0; i < matrixSize; i++) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define thread hierarchy
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Benchmark kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < matrixSize; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("Error at index %d: %f\n", i, h_C[i]);
            return -1;
        }
    }
    printf("Matrix addition completed successfully.\n");

    // Clean up benchmark resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
