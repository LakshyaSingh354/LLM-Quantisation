from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

def get_cuda_arch_flags():
    """Get CUDA architecture flags based on available GPU"""
    if not torch.cuda.is_available():
        return ['-gencode', 'arch=compute_50,code=sm_50']  # fallback
    
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    major, minor = capability
    
    arch_flags = [
        f'-gencode', f'arch=compute_{major}{minor},code=sm_{major}{minor}',
        '-gencode', 'arch=compute_50,code=sm_50',
        '-gencode', 'arch=compute_60,code=sm_60',
        '-gencode', 'arch=compute_70,code=sm_70',
        '-gencode', 'arch=compute_75,code=sm_75',
        '-gencode', 'arch=compute_80,code=sm_80',
        '-gencode', 'arch=compute_86,code=sm_86',
    ]
    
    return arch_flags

setup(
    name='qgemm_cuda',
    version='0.1.0',
    description='Custom quantization CUDA ops for PyTorch',
    ext_modules=[
        CUDAExtension(
            name='qgemm_cuda',
            sources=[
                'qgemm_tiled.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                ] + get_cuda_arch_flags()
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
