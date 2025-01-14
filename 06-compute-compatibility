# Compute Capability in CUDA

---

## **Key Definitions**
1. **Compute Capability (X.Y)**:
    - A version number identifying the features and instructions supported by a GPU.
    - **Major Revision (X)**: Indicates the core architecture (e.g., Hopper, Ampere, Volta).
    - **Minor Revision (Y)**: Represents incremental improvements to the architecture.

2. **Core Architectures**:
    - **NVIDIA Hopper (9.x)**: Latest architecture for high performance and efficiency.
    - **NVIDIA Ampere (8.x)**: Prior architecture focusing on AI and HPC workloads.
    - **Volta (7.x)**: Introduced Tensor Cores for AI and deep learning.
    - **Pascal (6.x)**, **Maxwell (5.x)**, **Kepler (3.x)**: Earlier architectures with decreasing levels of functionality and efficiency.

3. **Backward Compatibility**:
    - Programs targeting lower compute capabilities will run on higher-capability GPUs, but not vice versa.

---

## **Features by Compute Capability**

| Compute Capability | Architecture | Key Features                              |
|---------------------|--------------|------------------------------------------|
| **9.x**            | Hopper       | Improved Tensor Cores, Asynchronous Data |
| **8.x**            | Ampere       | Third-gen Tensor Cores, Sparsity Support |
| **7.x**            | Volta        | Tensor Cores, NVLink                     |
| **6.x**            | Pascal       | Unified Memory, HBM2                     |
| **5.x**            | Maxwell      | Dynamic Parallelism                      |
| **3.x**            | Kepler       | First-gen CUDA improvements              |

---

## **Practical Example: Querying Compute Capability**

The following script queries the compute capability of the current GPU and adapts program behavior based on the architecture.

---

### **C++ Script: Query Compute Capability**

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-enabled devices found.\n");
        return -1;
    }

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %lu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
        printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);

        // Conditional feature utilization based on compute capability
        if (deviceProp.major >= 8) {
            printf("  This GPU supports advanced Ampere features like sparsity and third-gen Tensor Cores.\n");
        } else if (deviceProp.major >= 7) {
            printf("  This GPU supports Volta architecture features like Tensor Cores.\n");
        } else {
            printf("  Older architecture detected. Some advanced features may not be available.\n");
        }
    }

    return 0;
}
```

### Key Features of the Script

1. **Dynamic Device Query**:
- Queries and prints detailed GPU properties for all CUDA-enabled devices on the system.

2. **Feature-Based Adaptation**:
- Provides program behavior suggestions based on compute capability.

3. **Practical Usage**:
- Use this script to determine the compute capability of a GPU before running feature-dependent code.

### Suggestions for Benchmarks

1. **Feature-Specific Benchmarks**:
- Use Tensor Core operations for GPUs with compute capability 7.0+ (e.g., Hopper, Ampere, Volta).
- Perform warp shuffle operations for older architectures.

2. **Global vs. Shared Memory Usage**:
- Compare performance using global and shared memory for different compute capabilities.



