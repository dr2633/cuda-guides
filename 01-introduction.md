# CUDA C++ Tutorial

This repository contains a comprehensive tutorial to help you learn CUDA C++ programming from scratch. The tutorial is structured into well-defined sections, ranging from introductory topics to advanced CUDA features, with practical code examples and exercises.

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Programming Model](#2-programming-model)
3. [Programming Interface](#3-programming-interface)
4. [Hardware Implementation](#4-hardware-implementation)
5. [Performance Guidelines](#5-performance-guidelines)
6. [CUDA-Enabled GPUs](#6-cuda-enabled-gpus)
7. [C++ Language Extensions](#7-c-language-extensions)
8. [Cooperative Groups](#8-cooperative-groups)
9. [CUDA Dynamic Parallelism](#9-cuda-dynamic-parallelism)
10. [Virtual Memory Management](#10-virtual-memory-management)
11. [Stream Ordered Memory Allocator](#11-stream-ordered-memory-allocator)
12. [Graph Memory Nodes](#12-graph-memory-nodes)
13. [Mathematical Functions](#13-mathematical-functions)
14. [C++ Language Support](#14-c-language-support)
15. [Texture Fetching](#15-texture-fetching)
16. [Compute Capabilities](#16-compute-capabilities)
17. [Driver API](#17-driver-api)
18. [CUDA Environment Variables](#18-cuda-environment-variables)
19. [Unified Memory Programming](#19-unified-memory-programming)

## Environment Setup

To set up a CUDA programming environment with C++, follow these steps:

1. **Install CUDA Toolkit**:
    - Download and install the CUDA Toolkit from NVIDIA's [official site](https://developer.nvidia.com/cuda-downloads).

2. **Install Required Packages**:
   Use `pip` to install the following Python tools for development and testing:
   ```bash
   pip install numpy matplotlib cupy
   ```

3. **Set Up Your Compiler**:
    - Ensure `nvcc` is in your system's PATH. You can verify by running:
      ```bash
      nvcc --version
      ```

4. **Clone This Repository**:
   ```bash
   git clone https://github.com/yourusername/cuda-tutorial.git
   cd cuda-tutorial
   ```

5. **Test the Setup**:
    - Navigate to the `01-introduction/` directory and run:
      ```bash
      nvcc kernels.cpp -o kernels
      ./kernels
      ```

6. **Optional: Setup Visual Studio Code for CUDA**:
    - Install the [CUDA extension](https://marketplace.visualstudio.com/items?itemName=nvidia.nsight) for Visual Studio Code.

## Note for macOS Users

Since CUDA is not natively supported on macOS, you can use a Docker container or a third-party cloud service to set up your CUDA environment. Below are the setup instructions for using Docker on macOS:

### Setting Up Docker for CUDA on macOS

1. **Install Docker Desktop**:
    - Download and install Docker Desktop for macOS from [Docker's official site](https://www.docker.com/products/docker-desktop/).
    - Ensure Docker is running after installation.

2. **Pull a CUDA Docker Image**:
    - Pull the official NVIDIA CUDA image:
      ```bash
      docker pull nvidia/cuda:12.0-devel-ubuntu20.04
      ```

3. **Create a Docker Container**:
    - Run the container with GPU support:
      ```bash
      docker run --gpus all -it --rm -v $(pwd):/workspace nvidia/cuda:12.0-devel-ubuntu20.04
      ```
      - `--gpus all`: Enables GPU support in the container.
      - `-v $(pwd):/workspace`: Mounts the current directory into the container for easy file access.

4. **Install Necessary Tools in the Container**:
    - Inside the container, install additional development tools:
      ```bash
      apt-get update && apt-get install -y build-essential cmake python3 python3-pip
      pip3 install numpy matplotlib cupy
      ```

5. **Test the Setup in the Container**:
    - Navigate to your mounted workspace directory:
      ```bash
      cd /workspace/01-introduction
      nvcc hello_world.cpp -o hello_world
      ./hello_world
      ```

### Alternative: Use a Cloud Service
If you prefer not to use Docker, consider using a cloud-based environment such as Google Colab or AWS, which provides access to GPUs with CUDA support.

- **Google Colab**: Offers free GPU access for small-scale development.
- **AWS EC2**: Provides instances with NVIDIA GPUs for larger projects.

Both options allow you to run CUDA code without the need for local GPU support.


### Resources


Hopper Tuning Guide CUDA 
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

H100/H200 User Guide 
https://docs.nvidia.com/dgx/dgxh100-user-guide/dgxh100-user-guide.pdf

Tuning and Deploying on H100 
https://docs.nvidia.com/launchpad/ai/h100-mig/latest/h100-mig-gpu.html

## About CUDA 
https://developer.nvidia.com/about-cuda

## CUDA C++ Best Practices Guide 
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#

## NVIDIA Deep Learning Institute 
https://www.nvidia.com/en-us/training/