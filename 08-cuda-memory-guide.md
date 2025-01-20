# CUDA Memory Management Guide for Computer Vision

## CUDA Just-In-Time (JIT) Compilation

### JIT Parameters
1. **Compilation Cache**
```cpp
// Enable JIT caching
nvJitOptions[0].OptType = NVJIT_OPT_CACHE_SIZE;
nvJitOptions[0].OptVal = 1024;  // Cache size in MB
```

2. **Optimization Levels**
```cpp
// Set optimization level
nvJitOptions[1].OptType = NVJIT_OPT_OPTIMIZATION_LEVEL;
nvJitOptions[1].OptVal = 4;  // Maximum optimization
```

3. **Target Architecture**
```cpp
// Specify target architecture
nvJitOptions[2].OptType = NVJIT_OPT_TARGET;
nvJitOptions[2].OptVal = NVJIT_TARGET_SM80;  // For Ampere architecture
```

## Linear Memory Allocation

### cudaMallocPitch()
Optimizes memory access for 2D arrays by ensuring proper alignment.

```cpp
// Declaration
cudaError_t cudaMallocPitch(
    void** devPtr,           // Pointer to allocated memory
    size_t* pitch,           // Pitch of allocation in bytes
    size_t width,            // Width in bytes
    size_t height           // Height in rows
);

// Example for image processing
size_t pitch;
unsigned char* d_image;
// Allocate pitched memory for 1920x1080 grayscale image
cudaMallocPitch(&d_image, &pitch, 1920 * sizeof(unsigned char), 1080);
```

Benefits:
- Ensures coalesced memory access
- Optimizes memory alignment
- Improves cache utilization
- Better performance for 2D operations

### cudaMalloc3D()
Optimizes memory access for 3D arrays, crucial for video processing and volumetric data.

```cpp
// Declaration
cudaError_t cudaMalloc3D(
    cudaPitchedPtr* pitchedDevPtr,  // Pitched pointer to allocated memory
    cudaExtent extent               // Size of allocation
);

// Example for video processing
cudaExtent extent = make_cudaExtent(width, height, depth);
cudaPitchedPtr d_video;
cudaMalloc3D(&d_video, extent);
```

Benefits:
- Optimizes 3D memory layout
- Handles padding automatically
- Improves performance for volumetric data
- Suitable for video frames and 3D convolutions

## Computer Vision Application Example

### Image Convolution with Pitched Memory

```cpp
// Host code
#define IMAGE_WIDTH  1920
#define IMAGE_HEIGHT 1080
#define KERNEL_SIZE 3

__global__ void convolution2D(
    unsigned char* input,
    unsigned char* output,
    size_t pitch,
    float* kernel,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        // Calculate row pitch in elements
        int pitch_elements = pitch / sizeof(unsigned char);
        
        // Convolution operation
        for(int ky = -KERNEL_SIZE/2; ky <= KERNEL_SIZE/2; ky++) {
            for(int kx = -KERNEL_SIZE/2; kx <= KERNEL_SIZE/2; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                
                float val = input[iy * pitch_elements + ix];
                sum += val * kernel[(ky + KERNEL_SIZE/2) * KERNEL_SIZE + 
                                  (kx + KERNEL_SIZE/2)];
            }
        }
        
        output[y * pitch_elements + x] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

// Main function setup
size_t pitch;
unsigned char *d_input, *d_output;
float *d_kernel;

// Allocate pitched memory for input and output images
cudaMallocPitch(&d_input, &pitch, IMAGE_WIDTH * sizeof(unsigned char), 
                IMAGE_HEIGHT);
cudaMallocPitch(&d_output, &pitch, IMAGE_WIDTH * sizeof(unsigned char), 
                IMAGE_HEIGHT);

// Kernel configuration
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(
    (IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
);

// Launch kernel
convolution2D<<<numBlocks, threadsPerBlock>>>(
    d_input, d_output, pitch, d_kernel, IMAGE_WIDTH, IMAGE_HEIGHT
);
```

## Performance Optimization Tips

### Memory Access Patterns
1. **Coalesced Access**
   - Align memory accesses to 128-byte boundaries
   - Use pitched memory for 2D arrays
   - Access consecutive memory locations within warps

2. **Shared Memory Usage**
   - Load frequently accessed data into shared memory
   - Use padding to avoid bank conflicts
   - Consider shared memory size when setting block dimensions

3. **Memory Transfer Optimization**
   - Use pinned memory for host-device transfers
   - Batch small transfers into larger ones
   - Overlap computation with memory transfers using streams

### Common Pitfalls
1. **Unaligned Memory Access**
   - Can reduce memory throughput
   - May cause additional memory transactions
   - Solution: Use cudaMallocPitch() for 2D arrays

2. **Bank Conflicts**
   - Occur in shared memory access
   - Can serialize memory access
   - Solution: Use proper padding and access patterns

3. **Memory Fragmentation**
   - Can occur with frequent malloc/free
   - Reduces available memory
   - Solution: Reuse allocated memory when possible

## Best Practices for Computer Vision

1. **Image Processing**
   - Use pitched memory for image data
   - Process multiple channels in parallel
   - Consider texture memory for read-only image data

2. **Video Processing**
   - Use 3D memory for frame sequences
   - Implement double buffering for real-time processing
   - Use streams for asynchronous frame processing

3. **Feature Detection**
   - Use shared memory for local neighborhoods
   - Implement efficient reduction algorithms
   - Consider using texture memory for gradient calculations

## Error Handling and Debugging

```cpp
// Check for errors after memory allocation
cudaError_t err = cudaMallocPitch(&d_input, &pitch, width, height);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

// Synchronize and check for kernel errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel execution failed (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```
