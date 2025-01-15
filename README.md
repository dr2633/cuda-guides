# CUDA C++ Tutorial

This repository contains a comprehensive tutorial to help you learn CUDA C++ programming from scratch. The tutorial is structured into well-defined sections, ranging from introductory topics to advanced CUDA features, with practical code examples and exercises.

## About This Guide
If you're a Python developer or AI engineer allergic to C++ (stuffy nose and watery eyes when looking at curly braces), but find yourself tempted to dive deeper into CUDA and GPU programming - you're in the right place! This guide is built to make GPU programming accessible to those comfortable in the safe space of the Python ecosystem, helping you bridge the gap between high-level ML frameworks and hardware-level optimization. Think of it as your friendly prompt to venture beyond `torch.cuda` and write your own GPU kernels! 💪

### Why CUDA C++ Now?
As AI and ML workloads become increasingly complex and computational demands grow, understanding the underlying hardware and how to effectively communicate with it becomes crucial. While frameworks like PyTorch and TensorFlow abstract away much of the complexity, knowledge of CUDA C++ enables you to:
- Write custom kernels for unique model architectures
- Optimize performance for specific hardware configurations
- Prepare for emerging hardware architectures
- Develop efficient solutions for edge deployment and AI inference
- Understand and optimize data center networking patterns

## Table of Contents

1. [Introduction](01-introduction.md)
   - Parallel processing concepts
   - GPU architecture basics
   - CUDA programming model

2. [Kernels](02-kernels.md)
   - Writing your first CUDA kernel
   - Thread hierarchy
   - Synchronization

3. [Memory Hierarchy](03-memory-hierarchy.md)
   - Global, shared, and local memory
   - Memory coalescing
   - Cache optimization

4. [Heterogeneous Computing](04-heterogeneous-computing.md)
   - CPU-GPU interaction
   - Asynchronous operations
   - Multi-GPU systems


### Getting Started
Begin with `docs/introduction.md` for a foundation in parallel processing concepts before moving on to practical CUDA programming.

Remember: You don't need to become a C++ expert - you just need enough understanding to effectively harness GPU power for your ML workloads. Let's begin!

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

## Official Resources

### Hardware-Specific Guides
- [Hopper Tuning Guide CUDA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [H100/H200 User Guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/dgxh100-user-guide.pdf)
- [Tuning and Deploying on H100](https://docs.nvidia.com/launchpad/ai/h100-mig/latest/h100-mig-gpu.html)

### Documentation and Learning
- [About CUDA](https://developer.nvidia.com/about-cuda)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)

## Prerequisites
- Python programming experience
- Basic understanding of ML frameworks (PyTorch + TensorFlow)
- NVIDIA GPU with CUDA support
- Linux/Windows operating system (see macOS section for alternatives)

## Contributing

Contributions are encourages! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests. Feel free to reach out to me at derek.rosenzweig1@gmail.com with any questions or suggestions. 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.