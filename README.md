# GPU-Based Galaxy Correlation Function Computation

## Project Overview

This project implements a CUDA-accelerated approach for computing the two-point angular correlation function of galaxies. The two-point correlation function measures how galaxies are spatially distributed compared to a random distribution. This function helps in understanding galaxy clustering, a fundamental aspect of astrophysics and cosmology.

More details about the project can be found in the [GPUProgramming2024_project](GPUProgramming2024_project.pdf) document.

The computation is based on comparing real galaxy data with randomly simulated galaxy distributions and measuring pair separations. These separations are binned into histograms to estimate clustering.

## Approaches Implemented

- **Unified Memory Approach (`cudaMallocManaged`)** – Uses CUDA’s unified memory to manage CPU-GPU memory transfers automatically.
- **Explicit Memory Management Approach** – Allocates separate CPU and GPU memory and manually handles memory transfers.

## System & Execution Details

### 1. Execution Environment

The code was executed on **Dione** ([dione.abo.fi](https://dione.abo.fi)), a batch processing system with **Tesla V100-PCIE-16GB GPUs**.

#### GPU Specifications

- **Compute Capability**: 7.0
- **Total Global Memory**: 16.93 GB
- **L2 Cache Size**: 6.29 MB
- **Multiprocessor Count**: 80
- **Max Threads per Multiprocessor**: 2048
- **Max Threads per Block**: 1024
- **Shared Memory per Block**: 48 KB
- **Clock Speed**: 1380 MHz
- **Concurrent Kernel Execution**: Yes
- **Async Engine Count**: 7

The system has four Tesla V100 GPUs, and our execution was done using one of the GPUs.

### 2. SLURM Batch Execution

The program was run using SLURM batch processing with the following script:

```bash
#!/bin/bash

# Load CUDA module
module load cuda

# Compile the CUDA program
nvcc -O3 -arch=sm_70 -o galaxy gpu_code.cu

# Submit the job using srun
srun --ntasks=1 -c 1 -t 10:00 -p gpu --mem=1G -o out.txt -e err.txt ./galaxy data_100k_arcmin.dat rand_100k_arcmin.dat results.csv
```

For execution on a standalone GPU system without SLURM, run:

```bash
nvcc -O3 -arch=sm_70 -o galaxy gpu_code.cu
./galaxy data_100k_arcmin.dat rand_100k_arcmin.dat results.csv
```

## Performance Comparison

| Approach | Execution Time (Dione) |
|----------|------------------------|
| **Unified Memory** | 0.823799 seconds |
| **Explicit Memory Management** | 0.842649 seconds |

### Observations

- **Unified memory** slightly outperformed explicit memory management, likely due to optimized memory transfers.
- **Explicit memory allocation** could offer better control for larger datasets.
- The execution times are very close, showing that both methods are efficient in GPU-based computing.

## Python-Based Analysis

To visualize the omega values and histogram data, a Jupyter notebook pdf is included for analysis. The graphs generated include:

### **Omega Values Over Index Range**

- Shows galaxy clustering strength based on the correlation function.
- A high omega indicates stronger clustering, while values near zero indicate a random distribution.

### **Histogram Comparison (Real vs Simulated Galaxies)**

- **HistogramDD (blue)** represents galaxy pair separations from real data.
- **HistogramRR (orange dashed line)** represents separations from simulated galaxies.
- Deviations between the two indicate real clustering effects.

## Observations & Learnings

- **Memory Optimization Matters**: Unified memory simplifies memory management, but explicit memory allocation gives finer control.
- **CUDA Synchronization is Crucial**: Ensuring proper execution order via `cudaDeviceSynchronize()` prevents incorrect calculations.
- **Omega Values Reflect Clustering**: High values at lower indices suggest strong galaxy clustering at small scales.
- **Histograms Validate Data Distribution**: Comparing real vs. simulated galaxy separations highlights structural differences.

## Conclusion

This project successfully implemented GPU-accelerated computation of galaxy correlation functions using both memory management techniques. The results show:

- **Unified Memory** provides ease of implementation with negligible performance loss.
- **Explicit Memory** gives control over memory transfers but requires manual handling.
- **CUDA GPUs** significantly speed up computation compared to CPU-based approaches.

If you have any questions or suggestions, feel free to contribute or reach out! 🚀
