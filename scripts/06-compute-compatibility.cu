//
// Created by Derek Rosenzweig on 12/29/24.
//

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