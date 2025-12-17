#pragma once

#include "ComputeBackend.h"
#include <cuda.h>
#include <vector>

class CudaBackend : public ComputeBackend {
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

    unique_ptr<FFTPlan> createFFTPlan(int size, int batch_size) override {
        // TODO: Implement CUDA FFT plan
        return nullptr;
    }

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override;

    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override;

                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override;

    void convertRawToComplex(const int8_t* input, int input_size,
                             float* buffer, int buffer_size,
                             int num_antennas, int nblocks, int num_coarse_channels,
                             int num_polarizations, int nsamp, int time_per_block) override {
        // TODO: Move implementation from upchannelizer.cu
    }

    void shiftFFTOutput(float* buffer, int buffer_size,
                        float* output, int output_size,
                        int fft_size, int num_antennas, int num_polarizations,
                        int num_coarse_channels, int num_timesteps) override {
        // TODO: Move implementation from upchannelizer.cu
    }
    void batchedPower(const std::complex<float>* voltage, float* power,
                      int num_beams, int num_channels, int num_polarizations,
                      int sti, int power_time_offset, int num_output_timesteps) override;

    void incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                                 float* output,
                                 int num_beams, int num_coarse_channels, int num_polarizations,
                                 int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) override;
};
