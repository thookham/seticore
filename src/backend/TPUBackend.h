#pragma once

#include "ComputeBackend.h"
#include <vector>

class TPUBackend : public ComputeBackend {
public:
    void allocateDevice(void** ptr, size_t size) override;
    void allocateHost(void** ptr, size_t size) override;
    void allocateManaged(void** ptr, size_t size) override;
    void freeDevice(void* ptr) override;
    void freeHost(void* ptr) override;
    void freeManaged(void* ptr) override;
    void verify_call(const char* name) override;
    void synchronize() override;
    
    void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) override;
    void copyHostToDevice(void* dst_device, const void* src_host, size_t size) override;
    void zeroDevice(void* ptr, size_t size) override;

    unique_ptr<FFTPlan> createFFTPlan(int size, int batch_size) override;

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override;

    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override;

    void findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override;

    void convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                             int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) override;

    void shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size, std::complex<float>* output, size_t output_size,
                        int fft_size, int num_antennas, int num_polarizations, int num_coarse_channels, int num_timesteps) override;

    void complexGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        std::complex<float> alpha,
        const std::complex<float>* A, int lda, long long strideA,
        const std::complex<float>* B, int ldb, long long strideB,
        std::complex<float> beta,
        std::complex<float>* C, int ldc, long long strideC,
        int batchCount) override;

    void floatGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        float alpha,
        const float* A, int lda, long long strideA,
        const float* B, int ldb, long long strideB,
        float beta,
        float* C, int ldc, long long strideC,
        int batchCount) override;

    void batchedPower(const std::complex<float>* voltage, float* power,
                      int num_beams, int num_channels, int num_polarizations,
                      int sti, int power_time_offset, int num_output_timesteps) override;

    void incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                                 float* output,
                                 int num_beams, int num_coarse_channels, int num_polarizations,
                                 int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) override;
};
