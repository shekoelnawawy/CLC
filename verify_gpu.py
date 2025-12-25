#!/usr/bin/env python
"""Quick script to verify GPU access in the container."""
import torch
import sys

print("=" * 60)
print("GPU Verification Script")
print("=" * 60)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test GPU computation
    print("\n" + "=" * 60)
    print("Testing GPU computation...")
    try:
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print("✓ GPU computation test PASSED")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
    except Exception as e:
        print(f"✗ GPU computation test FAILED: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ GPU is properly configured and working!")
    print("=" * 60)
    sys.exit(0)
else:
    print("\n" + "=" * 60)
    print("✗ CUDA is NOT available!")
    print("\nTroubleshooting steps:")
    print("1. Check if NVIDIA drivers are installed: nvidia-smi")
    print("2. Verify Docker has GPU access:")
    print("   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi")
    print("3. Ensure NVIDIA Container Toolkit is installed")
    print("4. Check Docker runtime: docker info | grep -i runtime")
    print("=" * 60)
    sys.exit(1)

