#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <complex>

// Forward declarations
struct DedopplerHit;

using namespace std;

class FFTPlan {
public:
    virtual ~FFTPlan() = default;
    // Execute C2C FFT. Input and Output are pointers to backend-specific memory (float complex)
    // Forward = true for FFT, false for IFFT
    virtual void execute(void* input, void* output, bool forward) = 0;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Memory Management
    virtual void allocateDevice(void** ptr, size_t size) = 0;
    virtual void allocateHost(void** ptr, size_t size) = 0;
    virtual void allocateManaged(void** ptr, size_t size) = 0;
    virtual void freeDevice(void* ptr) = 0;
    virtual void freeHost(void* ptr) = 0;
    virtual void freeManaged(void* ptr) = 0;
    virtual void verify_call(const char* name) = 0; // equivalent to checkCuda
    virtual void synchronize() = 0;
    
    // Data Transfer
    // In a real generic engine, we might want typed buffers, 
    // but void* matches the current usage.
    virtual void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) = 0;
    virtual void copyHostToDevice(void* dst_device, const void* src_host, size_t size) = 0;
    virtual void zeroDevice(void* ptr, size_t size) = 0;

    // FFT Support
    // create a 1D batched FFT plan
    // size: size of each transform
    // batch_size: number of transforms
    virtual unique_ptr<FFTPlan> createFFTPlan(int size, int batch_size) = 0;

    // Kernels
    // These signatures mirror the specialized operations in dedoppler.cu
    
    virtual void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) = 0;

    // The taylor tree returns a pointer to the result buffer (which might be one of the internal buffers)
    virtual const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                                    int num_timesteps, int num_channels, int drift_block) = 0;

    virtual void findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                                 int drift_block, float* top_path_sums,
                                 int* top_drift_blocks, int* top_path_offsets) = 0;

    // Upchannelizer kernels
    virtual void convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                                     int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) = 0;

    virtual void shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size, std::complex<float>* output, size_t output_size,
                                int fft_size, int num_antennas, int num_polarizations, int num_coarse_channels, int num_timesteps) = 0;

    // GEMM Abstraction
    // C = alpha * op(A) * op(B) + beta * C
    // Batched Strided Matrix Multiplication
    virtual void complexGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        std::complex<float> alpha,
        const std::complex<float>* A, int lda, long long strideA,
        const std::complex<float>* B, int ldb, long long strideB,
        std::complex<float> beta,
        std::complex<float>* C, int ldc, long long strideC,
        int batchCount) = 0;

    virtual void floatGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        float alpha,
        const float* A, int lda, long long strideA,
        const float* B, int ldb, long long strideB,
        float beta,
        float* C, int ldc, long long strideC,
        int batchCount) = 0;

    // Beamformer specific kernels
    // Beamformer specific kernels
    virtual void batchedPower(const std::complex<float>* voltage, float* power,
                              int num_beams, int num_channels, int num_polarizations,
                              int sti, int power_time_offset, int num_output_timesteps) = 0;

    virtual void incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                                 float* output,
                                 int num_beams, int num_coarse_channels, int num_polarizations,
                                 int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) = 0;
};
