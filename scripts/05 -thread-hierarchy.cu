#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

using namespace cooperative_groups;

const int N = 1024; // Array size

// Kernel using asynchronous memcpy
__global__ void ComputeWithAsync(float* src, float* dst, int N) {
    extern __shared__ float sharedData[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    auto g = this_thread_block();

    // Asynchronous copy from global to shared memory
    memcpy_async(g, sharedData, &src[idx], sizeof(float));
    g.sync();

    // Perform computation
    if (idx < N) {
        dst[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_src = (float*)malloc(bytes);
    float *h_dst = (float*)malloc(bytes);

    // Initialize source array
    for (int i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i);
    }

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    ComputeWithAsync<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_src, d_dst, N);

    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        if (h_dst[i] != h_src[i] * 2.0f) {
            printf("Error at index %d: %f\n", i, h_dst[i]);
            return -1;
        }
    }
    printf("Computation with async memcpy completed successfully.\n");

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    free(h_dst);

    return 0;
}
