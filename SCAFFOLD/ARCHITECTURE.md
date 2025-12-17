# Seticore Architecture

## Overview

Seticore is a high-performance C++/CUDA implementation of the SETI search pipeline. It replaces legacy software with a modernized engine capable of processing high-bandwidth telescope data in real-time.

## Data Flow

The core pipeline processes data in the following stages:

1. **Ingestion**:
    - Reads `.h5` files (HDF5 format) or raw buffers using `H5Reader` / `RawFileGroupReader`.
    - Data is loaded into host memory and then transferred to GPU buffers.

2. **Coarse Channelization (Pre-processing)**:
    - Input data often comes as "coarse channels".
    - `Dedopplerer` processes these channels.

3. **Dedoppler (The Core Search)**:
    - **Goal**: Detect simple signals (narrowband carriers) that are "drifting" in frequency due to the Doppler effect and Earth's rotation.
    - **Algorithm**: The "Taylor Tree" algorithm efficiently sums power along various straight-line paths in the spectrogram.
        - **Kernels**:
            - `sumColumns`: Calculates column sums for normalization.
            - `optimizedTaylorTree`: The heavy lifter. Uses shared memory tiling (`tiledTaylorKernel`) for efficiency on small time-steps, and global memory passes for larger ones.
            - `findTopPathSums`: Extracts the highest power signal for each frequency bin across all drift rates.
    - **Statistics**:
        - Column sums are copied back to CPU.
        - `std::nth_element` is used to estimate the noise floor (median, standard deviation) on the CPU.

4. **Hit Formation**:
    - "Hits" (significant signals) are filtered based on SNR threshold.
    - Hits are de-duplicated (windowing) to avoid reporting the same signal multiple times.
    - Results are stored in `DedopplerHit` structures.

5. **Output**:
    - Hits are serialized to `.capnp` (Cap'n Proto) format or `.dat` files.

## Key Components

### `Dedopplerer` (Class)

- **Role**: Manages GPU memory lifecycles (`cudaMalloc`/`cudaFree`).
- **Files**: `dedoppler.cu`, `dedoppler.h`
- **Key Methods**:
  - `search()`: Orchestrates the search for a block of data.
  - `memoryUsage()`: Reports VRAM usage.

### `Taylor` (Module)

- **Role**: Implements the Taylor Tree summation algorithm.
- **Files**: `taylor.cu`, `taylor.h`
- **Complexity**: High. Contains architecture-specific optimizations (shared memory tiling) that drive performance.

## Portability Analysis

The project is currently hard-linked to NVIDIA CUDA:

- **Headers**: `<cuda.h>`, `cuda_util.h`.
- **Kernels**: `__global__` functions in `.cu` files.
- **Memory**: Direct `cudaMalloc`, `cudaMemcpy`.

### Critical Kernels to Port

1. `sumColumns` (Simple reduction)
2. `oneStepTaylorKernel` (Data shuffle/add)
3. `tiledTaylorKernel` (Shared memory intensive)
4. `findTopPathSums` (Reduction/Max)

## File Formats

- **Input**: HDF5 (`.h5`), Raw (`.raw`).
- **Output**: Cap'n Proto (`.capnp`), Filterbank (`.fil`), Dat (`.dat`).
