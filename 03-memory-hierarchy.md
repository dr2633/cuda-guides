# Memory Hierarchy in CUDA

This section explores the memory hierarchy in CUDA, defining key memory spaces and their roles in optimizing GPU computation. Additionally, it includes sample projects and benchmarks to demonstrate efficient memory usage.

---

## **1. Key Concepts and Definitions**

### **1.1 CUDA Memory Spaces**
1. **Private Local Memory**:
    - **Definition**: Memory private to each thread, used for storing thread-specific variables.
    - **Lifetime**: Exists only during the thread’s execution.
    - **Access**: Fast but limited in size.

2. **Shared Memory**:
    - **Definition**: Memory shared among all threads in a block.
    - **Lifetime**: Exists for the duration of the thread block.
    - **Usage**:
        - Enables efficient communication and data sharing among threads in the same block.
        - Reduces global memory access latency by reusing data locally.
    - **Optimization**:
        - Minimize bank conflicts by aligning memory accesses.

3. **Global Memory**:
    - **Definition**: Memory accessible by all threads in all blocks.
    - **Lifetime**: Persistent across kernel launches.
    - **Usage**: Used for storing large datasets shared among all threads.
    - **Optimization**:
        - Use coalesced memory access patterns to maximize throughput.

4. **Constant Memory**:
    - **Definition**: Read-only memory space accessible by all threads, optimized for uniform access patterns.
    - **Lifetime**: Persistent across kernel launches.

5. **Texture Memory**:
    - **Definition**: Specialized memory for read-only access with additional features like filtering and addressing modes.
    - **Usage**:
        - Efficient for 2D/3D data structures.
        - Automatically caches frequently accessed data.

---

## **2. Annotated Code Example**
### **2.1 Shared and Global Memory**
This example demonstrates the use of shared memory for efficient matrix multiplication.

```cpp
// Matrix Multiplication with Shared Memory
#include <cuda_runtime.h>
#include <cstdio>

const int TILE_SIZE = 16; // Tile size for shared memory

// Kernel for matrix multiplication
__global__ void MatMulShared(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0;
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < N && k * TILE_SIZE + tx < N) {
            tileA[ty][tx] = A[row * N + k * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0;
        }
        if (col < N && k * TILE_SIZE + ty < N) {
            tileB[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            value += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 1024; // Matrix size (NxN)
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    MatMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed successfully.\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

---

## **3. Sample Projects and Benchmarks**

### **3.1 Global vs. Shared Memory Performance**
- **Goal**: Compare performance between a naive global memory implementation and an optimized shared memory version for matrix multiplication.
- **Steps**:
    1. Implement matrix multiplication using only global memory.
    2. Implement matrix multiplication using shared memory.
    3. Use CUDA events to measure execution time.
- **Benchmark**:
    - Measure and report speedup achieved with shared memory.

### **3.2 Constant Memory Usage**
- **Goal**: Demonstrate the advantages of constant memory for accessing read-only data.
- **Steps**:
    1. Load a constant matrix into constant memory.
    2. Use it for element-wise addition with another matrix in global memory.
- **Benchmark**:
    - Compare constant memory access time to global memory for the same operation.

### **3.3 Texture Memory for Image Processing**
- **Goal**: Use texture memory to apply a filter to a 2D image.
- **Steps**:
    1. Load an image into texture memory.
    2. Apply a 3x3 convolution filter using texture memory.
- **Benchmark**:
    - Compare texture memory performance to global memory for the same operation.

---

## **4. Takeaways for CUDA Programmers**
1. **Efficient Memory Access**:
    - Use shared memory for fast, reusable data within a block.
    - Optimize global memory access with coalescing.
2. **Specialized Memory**:
    - Utilize constant and texture memory for specific read-only workloads to improve performance.
3. **Performance Tuning**:
    - Benchmark different memory hierarchies to identify bottlenecks and optimize usage.

---
