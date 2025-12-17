# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Multi-Backend Support**: Added `ComputeBackend` abstract interface to support CUDA, CPU, and SYCL backends seamlessly.
- **CPU Backend**: Implemented `CpuReferenceBackend` using FFTW3 and naive BLAS/Internal implementations for full pipeline execution on CPU.
- **SYCL Backend**: Implemented `SyclBackend` using oneMKL for FFT and BLAS, enabling support for Intel GPUs (Arc, Data Center GPU).
- **Backend-Agnostic Beamformer**: Refactored `Beamformer` to utilize `ComputeBackend`, enabling beamforming on non-NVIDIA hardware.
- **Build System**: Updated Meson build to conditionally compile backends (`-Dcuda=enabled`, `-Dsycl=enabled`).

### Changed
- `BeamformingPipeline` constructor now accepts a `ComputeBackend*`.
- `Beamformer`, `Upchannelizer`, `StampExtractor` constructors updated to accept `ComputeBackend*`.
- `convertRawToComplex`, `shiftFFTOutput` kernels moved to backend implementations.
- `calculatePower` and `incoherentPower` logic moved to backend implementations.
- `main.cpp` refactored to instantiate appropriate backend based on build configuration.

### Deprecated
- Direct usage of CUDA-specific headers in `beamformer.h` (legacy paths preserved but wrapped).
