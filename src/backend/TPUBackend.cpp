#include "TPUBackend.h"
#include <iostream>
#include <stdexcept>

using namespace std;

void TPUBackend::allocateDevice(void** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return;
    }
    throw runtime_error("TPUBackend::allocateDevice not implemented");
}

void TPUBackend::allocateHost(void** ptr, size_t size) {
    *ptr = malloc(size);
}

void TPUBackend::allocateManaged(void** ptr, size_t size) {
    throw runtime_error("TPUBackend::allocateManaged not implemented");
}

void TPUBackend::freeDevice(void* ptr) {
}

void TPUBackend::freeHost(void* ptr) {
    free(ptr);
}

void TPUBackend::freeManaged(void* ptr) {
}

void TPUBackend::verify_call(const char* name) {
}

void TPUBackend::synchronize() {
}

void TPUBackend::copyDeviceToHost(void* dst_host, const void* src_device, size_t size) {
    throw runtime_error("TPUBackend::copyDeviceToHost not implemented");
}

void TPUBackend::copyHostToDevice(void* dst_device, const void* src_host, size_t size) {
    throw runtime_error("TPUBackend::copyHostToDevice not implemented");
}

void TPUBackend::zeroDevice(void* ptr, size_t size) {
    throw runtime_error("TPUBackend::zeroDevice not implemented");
}

unique_ptr<FFTPlan> TPUBackend::createFFTPlan(int size, int batch_size) {
    throw runtime_error("TPUBackend::createFFTPlan not implemented");
}

void TPUBackend::sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) {
    throw runtime_error("TPUBackend::sumColumns not implemented");
}

const float* TPUBackend::taylorTree(const float* input, float* buffer1, float* buffer2,
                        int num_timesteps, int num_channels, int drift_block) {
    throw runtime_error("TPUBackend::taylorTree not implemented");
}

void TPUBackend::findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                     int drift_block, float* top_path_sums,
                     int* top_drift_blocks, int* top_path_offsets) {
     throw runtime_error("TPUBackend::findTopPathSums not implemented");
}

void TPUBackend::convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                         int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) {
    throw runtime_error("TPUBackend::convertRawToComplex not implemented");
}

void TPUBackend::shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size, std::complex<float>* output, size_t output_size,
                    int fft_size, int num_antennas, int num_polarizations, int num_coarse_channels, int num_timesteps) {
    throw runtime_error("TPUBackend::shiftFFTOutput not implemented");
}

void TPUBackend::complexGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    std::complex<float> alpha,
    const std::complex<float>* A, int lda, long long strideA,
    const std::complex<float>* B, int ldb, long long strideB,
    std::complex<float> beta,
    std::complex<float>* C, int ldc, long long strideC,
    int batchCount) {
    throw runtime_error("TPUBackend::complexGemmStridedBatched not implemented");
}

void TPUBackend::floatGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    throw runtime_error("TPUBackend::floatGemmStridedBatched not implemented");
}

void TPUBackend::batchedPower(const std::complex<float>* voltage, float* power,
                  int num_beams, int num_channels, int num_polarizations,
                  int sti, int power_time_offset, int num_output_timesteps) {
    throw runtime_error("TPUBackend::batchedPower not implemented");
}

void TPUBackend::incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                             float* output,
                             int num_beams, int num_coarse_channels, int num_polarizations,
                             int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) {
    throw runtime_error("TPUBackend::incoherentPower not implemented");
}
