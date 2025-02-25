### Lambda Cloud Instance Setup Guide





##### Instance Connection 






















***

Instance Selection
Hardware Configuration

Instance Type: 1x GH200 (96 GB) ARM64 + H100
Specifications:

64 CPU cores
463.9 GB RAM
4.4 TB SSD
Cost: $1.49/hr


Region Selection
For users based in California, consider these factors when choosing a region:

Network Latency:

Washington DC location will have higher latency (~80-100ms) compared to West Coast locations
For development work, this latency is usually acceptable
For production or latency-sensitive applications, consider waiting for West Coast availability


Filesystem Options:

Create a new filesystem in your selected region
Benefits of creating a new filesystem:

Persistent storage across instance restarts
Better data organization
Easier backup management
Improved performance with local storage



Setup Instructions
1. Creating a New Filesystem

Select "Create a new filesystem" option
Configure filesystem parameters:

Name: [CUDA-Tutorials]-fs
Size: Start with 1TB (can be expanded later)
Type: Performance SSD
Backup frequency: Daily

2. Instance Configuration

SSH Key Setup:

Add your SSH public key to Lambda dashboard
Configure security settings:

Default security group
Allow SSH (port 22)
Configure any additional required ports

3. Network Configuration

Choose networking options:

Default VPC
Public IP address (recommended for development)


Consider setting up VPN if additional security is required

4. Post-Launch Setup
Initial Connection

```bash
# SSH into your instance
ssh -i /path/to/your/private-key ubuntu@your-instance-ip
```

Environment Setup 
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install CUDA development tools
sudo apt install nvidia-cuda-toolkit

# Verify GPU access
nvidia-smi
```

Mount Filesystem


```bash
# Create mount point
sudo mkdir /mnt/data

# Mount filesystem
sudo mount /dev/nvme1n1 /mnt/data

# Add to fstab for persistent mounting
echo '/dev/nvme1n1 /mnt/data ext4 defaults 0 0' | sudo tee -a /etc/fstab

```


Best Practices
Cost Management

Set up billing alerts
Stop instance when not in use
Monitor GPU utilization

Data Management

Use the filesystem for persistent data
Regular backups of critical code and data
Consider using Git for version control

Security

Keep SSH keys secure
Regular security updates
Use strong passwords
Consider setting up CloudFlare Tunnel for secure access

Troubleshooting
Common Issues

SSH Connection Issues:

Verify SSH key permissions (should be 600)
Check security group settings
Confirm instance is running


GPU Access Issues:

Run nvidia-smi to verify GPU visibility
Check CUDA installation
Verify driver compatibility


Filesystem Issues:

Check mount points
Verify filesystem is properly attached
Check disk space usage



Support Resources

Lambda Cloud Documentation
NVIDIA CUDA Documentation
Community Forums

Next Steps

- Set up development environment (IDEs, tools)
- Configure version control
- Install project-specific dependencies
- Set up monitoring and alerts


***

NVIDIA Developer Tools Mac OS 
https://developer.nvidia.com/nvidia-cuda-toolkit-11_7_0-developer-tools-mac-hosts