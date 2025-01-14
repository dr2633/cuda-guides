# CUDA C++ Overview: Key Functions and Syntax for GPU Programming

Math API Reference Manual
https://docs.nvidia.com/cuda/cuda-math-api/index.html

This document provides a detailed overview of key functions, syntax, and concepts in CUDA C++ to help understand GPU programming.

---

## **1. Key CUDA-Specific Keywords and Syntax**

### **1.1 `__global__`**
- **Definition**:
    - A function qualifier that specifies a **kernel** function, which can be called from the host (CPU) and executed on the device (GPU).
- **Syntax**:
  ```cpp
  __global__ void kernelFunctionName(arguments) {
      // Kernel logic here
  }
  ```
- **Key Points**:
    - Must be void-returning.
    - Uses the `<<<...>>>` execution configuration to specify the number of threads and blocks.
- **Example**:
  ```cpp
  __global__ void add(int* a, int* b, int* c) {
      int i = threadIdx.x;
      c[i] = a[i] + b[i];
  }
  ```

---

### **1.2 `__device__`**
- **Definition**:
    - A function qualifier that specifies a function callable **only from the device** (GPU).
- **Syntax**:
  ```cpp
  __device__ returnType functionName(arguments) {
      // Logic here
  }
  ```
- **Key Points**:
    - Cannot be called from the host (CPU).
- **Example**:
  ```cpp
  __device__ float multiply(float x, float y) {
      return x * y;
  }
  ```

---

### **1.3 `__host__`**
- **Definition**:
    - A function qualifier that specifies a function callable **only from the host** (CPU).
- **Key Points**:
    - This is the default for all C++ functions if no qualifier is specified.
- **Example**:
  ```cpp
  __host__ void printMessage() {
      printf("Hello from the host!\n");
  }
  ```

---

### **1.4 `threadIdx`**
- **Definition**:
    - A built-in variable in CUDA that gives the **thread index** within a block.
- **Syntax**:
  ```cpp
  int idx = threadIdx.x;
  ```
- **Key Points**:
    - CUDA threads are grouped into blocks, and `threadIdx` identifies a thread’s position within its block.
    - Components: `threadIdx.x`, `threadIdx.y`, `threadIdx.z`.
- **Example**:
  ```cpp
  int i = threadIdx.x; // Thread index in the x-dimension
  ```

---

### **1.5 `blockIdx`**
- **Definition**:
    - A built-in variable in CUDA that gives the **block index** within a grid.
- **Syntax**:
  ```cpp
  int block = blockIdx.x;
  ```
- **Key Points**:
    - Identifies the block’s position in the grid.
    - Components: `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.
- **Example**:
  ```cpp
  int i = blockIdx.x; // Block index in the x-dimension
  ```

---

### **1.6 `gridDim` and `blockDim`**
- **Definition**:
    - `gridDim`: Built-in variable that provides the dimensions of the grid.
    - `blockDim`: Built-in variable that provides the dimensions of each block.
- **Syntax**:
  ```cpp
  int gridSize = gridDim.x;
  int blockSize = blockDim.x;
  ```
- **Key Points**:
    - Used to calculate the global thread index.
- **Example**:
  ```cpp
  int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
  ```

---

## **2. Execution Configuration Syntax**

### **2.1 `<<<gridDim, blockDim>>>`**
- **Definition**:
    - Specifies the number of blocks and threads per block for kernel execution.
- **Syntax**:
  ```cpp
  kernelFunction<<<numBlocks, threadsPerBlock>>>(arguments);
  ```
- **Key Points**:
    - `numBlocks`: Number of blocks in the grid.
    - `threadsPerBlock`: Number of threads in each block.
    - The execution configuration allows you to leverage massive parallelism on the GPU by breaking a large problem into smaller, parallel tasks.
- **Example**:
  ```cpp
  add<<<4, 256>>>(d_a, d_b, d_c); // 4 blocks, 256 threads per block
  ```
- **Advanced**:
    - Optionally, you can pass a shared memory size and a CUDA stream to the kernel:
      ```cpp
      kernelFunction<<<numBlocks, threadsPerBlock, sharedMemorySize, stream>>>(arguments);
      ```

---

### **2.2 `void` Return Type in Kernels**
- **Definition**:
    - Kernel functions must return `void` because their purpose is to execute operations across many threads rather than return a single value.
- **Key Points**:
    - Output is written directly to device memory, accessible to multiple threads.
    - This design avoids conflicts and leverages GPU memory for parallelism.
- **Example**:
  ```cpp
  __global__ void add(float* a, float* b, float* c) {
      int idx = threadIdx.x;
      c[idx] = a[idx] + b[idx];
  }
  ```

---

## **3. Memory Management Functions**

### **3.1 `cudaMalloc`**
- **Definition**:
    - Allocates memory on the GPU device.
- **Syntax**:
  ```cpp
  cudaMalloc((void**)&pointer, size);
  ```
- **Key Points**:
    - `pointer`: Device pointer to the allocated memory.
    - `size`: Size in bytes to allocate.
- **Example**:
  ```cpp
  float* d_A;
  cudaMalloc((void**)&d_A, N * sizeof(float));
  ```

---

### **3.2 `cudaMemcpy`**
- **Definition**:
    - Copies data between host and device memory.
- **Syntax**:
  ```cpp
  cudaMemcpy(destination, source, size, direction);
  ```
- **Key Points**:
    - `direction`: Specifies the direction of the copy (`cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`).
- **Example**:
  ```cpp
  cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
  ```

---

### **3.3 `cudaFree`**
- **Definition**:
    - Frees allocated memory on the GPU device.
- **Syntax**:
  ```cpp
  cudaFree(pointer);
  ```
- **Key Points**:
    - Ensures efficient memory management by deallocating GPU memory.
- **Example**:
  ```cpp
  cudaFree(d_A);
  ```

---

## **4. Key Takeaways**
1. CUDA C++ introduces keywords like `__global__`, `__device__`, and `__host__` to distinguish between host and device functions.
2. Built-in variables like `threadIdx`, `blockIdx`, `gridDim`, and `blockDim` enable detailed control of thread and block execution.
3. Proper memory management with `cudaMalloc`, `cudaMemcpy`, and `cudaFree` is essential for optimal performance.
4. Execution configuration (`<<<gridDim, blockDim>>>`) is critical for leveraging GPU parallelism efficiently.
5. Kernel functions must return `void`, with output written directly to memory, enabling concurrent computation across thousands of threads.

---