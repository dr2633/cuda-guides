# Definitions of Key Terms in `cuda_runtime.h`

## Memory Management

### **`cudaMalloc`**
Allocates memory on the GPU device.
- **Usage**: `cudaMalloc(void **devPtr, size_t size)`
- Allocates `size` bytes of memory on the device and returns a pointer to the allocated memory in `devPtr`.

### **`cudaFree`**
Frees previously allocated GPU memory.
- **Usage**: `cudaFree(void *devPtr)`
- Releases the memory pointed to by `devPtr` on the GPU.

### **`cudaMemcpy`**
Transfers data between host and device memory.
- **Usage**: `cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`
- Copies `count` bytes of data between host and device memory, where `kind` specifies the direction (e.g., `cudaMemcpyHostToDevice`).

### **`cudaMemset`**
Initializes or sets device memory to a specific value.
- **Usage**: `cudaMemset(void *devPtr, int value, size_t count)`
- Sets the first `count` bytes of the memory area pointed to by `devPtr` to `value`.

---

## Device Management

### **`cudaGetDevice`**
Returns the currently active GPU device for the host thread.
- **Usage**: `cudaGetDevice(int *device)`
- Stores the device ID in the variable pointed to by `device`.

### **`cudaSetDevice`**
Sets the active GPU device for the host thread.
- **Usage**: `cudaSetDevice(int device)`
- Selects the device with ID `device` for subsequent CUDA operations.

### **`cudaDeviceSynchronize`**
Blocks the host until all preceding tasks on the GPU are complete.
- **Usage**: `cudaDeviceSynchronize()`
- Ensures that all tasks launched on the device are finished.

### **`cudaDeviceReset`**
Resets the GPU device to its default state.
- **Usage**: `cudaDeviceReset()`
- Destroys all allocations and resets the state of the current device.

---

## Kernel Launch Configuration

### **`dim3`**
Specifies thread or block dimensions in 1D, 2D, or 3D.
- **Usage**: `dim3 gridDim(x, y, z)`
- Used to configure the number of blocks or threads in a grid.

### **`<<<...>>>`**
Specifies kernel launch configuration.
- **Usage**: `kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args...)`
- Configures the execution with the number of blocks (`gridDim`), threads per block (`blockDim`), and optional shared memory size or stream.

---

## Error Handling

### **`cudaError_t`**
Enumeration of possible CUDA error codes.
- **Usage**: `cudaError_t err = cudaMalloc(...)`
- Contains error codes such as `cudaSuccess` or `cudaErrorMemoryAllocation`.

### **`cudaGetErrorString`**
Returns a string describing an error code.
- **Usage**: `const char *errStr = cudaGetErrorString(err)`
- Provides a human-readable string for the error code `err`.

### **`cudaGetLastError`**
Retrieves the last error from a CUDA API call.
- **Usage**: `cudaError_t err = cudaGetLastError()`
- Resets the error state and returns the last error.

---

## Stream and Event Management

### **`cudaStreamCreate`**
Creates a stream for asynchronous execution.
- **Usage**: `cudaStreamCreate(cudaStream_t *stream)`
- Initializes a new stream and stores it in `stream`.

### **`cudaStreamSynchronize`**
Blocks until all tasks in a stream are complete.
- **Usage**: `cudaStreamSynchronize(cudaStream_t stream)`
- Ensures all tasks in the specified stream have finished.

### **`cudaEventCreate`**
Creates an event for profiling or synchronization.
- **Usage**: `cudaEventCreate(cudaEvent_t *event)`
- Initializes a new event and stores it in `event`.

---

## Unified Memory

### **`cudaMallocManaged`**
Allocates unified memory accessible by both host and device.
- **Usage**: `cudaMallocManaged(void **devPtr, size_t size)`
- Allocates `size` bytes of managed memory and returns a pointer to it in `devPtr`.

### **`cudaMemPrefetchAsync`**
Prefetches managed memory to a specified device.
- **Usage**: `cudaMemPrefetchAsync(const void *devPtr, size_t count, int device, cudaStream_t stream)`
- Moves `count` bytes of managed memory starting at `devPtr` to the specified device asynchronously.

---

## Device Query and Properties

### **`cudaDeviceProp`**
Structure containing GPU device properties.
- **Fields**: Includes properties like `totalGlobalMem`, `maxThreadsPerBlock`, and `multiProcessorCount`.

### **`cudaGetDeviceProperties`**
Retrieves the properties of a specified device.
- **Usage**: `cudaGetDeviceProperties(cudaDeviceProp *prop, int device)`
- Fills the `cudaDeviceProp` structure with properties of the device specified by `device`.

---

These definitions provide a foundational understanding of key functions and features within `cuda_runtime.h`. Let me know if you need deeper insights or examples for any specific term!
