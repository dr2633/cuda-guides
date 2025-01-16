 #include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024; // Matrix size (N x N)
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; // Example initialization
        h_B[i] = 1.0f; // Example initialization
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != N) {
            std::cerr << "Verification failed at index " << i << std::endl;
            break;
        }
    }

    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    std::cout << "Generated PTX file: 07-nvcc-compiler.ptx" << std::endl;
    std::cout << "Generated CUBIN file: 07-nvcc-compiler.cubin" << std::endl;

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


