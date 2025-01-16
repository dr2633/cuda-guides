# CUDA Definitions and Memory Hierarchy

## CUDA Programming Model

### **CUDA**
CUDA is NVIDIA's parallel computing platform and programming model that enables developers to write code that runs on NVIDIA GPUs using extensions to C, C++, and Fortran.

### **Parallel Thread eXecution (PTX)**
PTX is an intermediate representation used in the CUDA programming model allowing portability across GPUs and serving as a lower-level abstraction for CUDA kernels before being compiled into device-specific instructions.

### **Compute Capability**
Compute Capability defines the feature set and performance characteristics of an NVIDIA GPU, determining the hardware's supported CUDA features, capabilities, and instructions.


## Execution Model

### **Thread**
A thread is the smallest unit of execution in CUDA, performing a single sequence of instructions on a GPU, with thousands of threads executing in parallel.

### **Warp**
A warp is a group of 32 threads that execute the same instruction in lockstep on a GPU, forming the fundamental unit of scheduling.

### **Cooperative Thread Array (CTA)**
A Cooperative Thread Array (CTA), also known as a thread block, is a group of threads that execute concurrently and share local resources like shared memory and synchronization mechanisms.

### **Kernel**
A kernel is a GPU function written in CUDA that is executed on the device by multiple threads in parallel, each running a copy of the function.

### **Thread Block**
A thread block is a group of threads organized into a fixed-size three-dimensional array that can communicate through shared memory and synchronize execution within the block.

### **Thread Block Grid**
A thread block grid is an arrangement of thread blocks, organized in one, two, or three dimensions, allowing for scalable execution of parallel workloads on the GPU.


## Memory Hierarchy

### **Registers**
- **Location**: Registers are located on-chip within each Streaming Multiprocessor (SM).
- **Accessibility**: Each thread has its own private set of registers, making them thread-local. Registers are not shared among threads.
- **Latency**: Registers provide the lowest latency and the highest bandwidth of all memory types in CUDA.
- **Usage**: Ideal for storing thread-local variables and temporary data that require rapid access.

### **Shared Memory**
- **Location**: Shared memory is located on-chip within each Streaming Multiprocessor (SM).
- **Accessibility**: Shared by all threads within a thread block. Not accessible by threads from other blocks.
- **Latency**: Low latency, similar to registers, though access may involve bank conflicts that can increase latency.
- **Usage**: Best for data that needs to be shared or reused by threads within a block.

### **L1 Cache**
- **Location**: L1 cache is located on-chip within each Streaming Multiprocessor (SM).
- **Accessibility**: Shared by all threads on an SM, and automatically used by hardware to cache frequently accessed global memory data.
- **Latency**: Faster than shared memory but slower than registers.
- **Usage**: Used to reduce global memory latency for frequently accessed or reused data.

### **L2 Cache**
- **Location**: L2 cache is located on-chip and is shared across all Streaming Multiprocessors (SMs).
- **Accessibility**: Acts as an intermediate cache for global memory accesses and is automatically managed by the hardware.
- **Latency**: Significantly faster than global memory but slower than L1 cache.
- **Usage**: Reduces global memory latency and helps improve memory bandwidth utilization for workloads with frequent global memory access patterns.

### **Global Memory**
- **Location**: Global memory is located off-chip in the GPU's main DRAM (device memory).
- **Accessibility**: Accessible by all threads across all Streaming Multiprocessors (SMs) and by the host (CPU).
- **Latency**: High latency (hundreds of clock cycles), requiring careful management to avoid performance bottlenecks.
- **Usage**: Stores large datasets and data structures that must persist across kernel launches or be shared between the host and the device.

### **Constant Memory**
- **Location**: Constant memory is located in device memory but is cached on-chip for faster access.
- **Accessibility**: Read-only memory accessible by all threads. Only the host can write to constant memory.
- **Latency**: Accessing cached constant memory is as fast as register access; uncached access is as slow as global memory.
- **Usage**: Ideal for storing read-only data that is uniform across threads, such as configuration parameters or lookup tables.

### **Texture and Surface Memory**
- **Location**: Texture and surface memory reside in device memory but are cached in dedicated texture and surface caches.
- **Accessibility**: Accessed through special hardware and APIs for optimized spatial locality and addressing.
- **Latency**: Faster than uncached global memory due to specialized caching but slower than shared or register memory.
- **Usage**: Suited for applications involving 2D or 3D spatial locality, such as image processing and graphics.
