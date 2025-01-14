### 1.1 Benefits of Using GPUs

CPU is designed to excel in executing a **sequence of operations**, called a **thread** as fast as possible and can execute a few tens of threads in parallel. 

GPUs are designed to **execute thousands of threads in parallel**, ammortizing the slower single-thread performande to leverage parallelization and achieve greater throughput. 


More transistors are devoted to data processing rather than data caching and flow control. 

Figure of CPU and GPU architecture 

**Core, L1 cache, L2 cache, L3 cache, DRAM in CPU **
**Transistors, L1 and control on panel, L2 cache DRAM** 

Devoting more transistors to data processing, floating point computations, is beneficial for highly parallel processes. 

- GPUs can hide **memory access latencies** with parallelization instead of relying on large data caches and complex flow control (expensive iin terms of transistors)



