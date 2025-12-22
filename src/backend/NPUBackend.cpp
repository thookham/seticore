#include "NPUBackend.h"
#include <iostream>
#include <stdexcept>

using namespace std;

void NPUBackend::allocateDevice(void** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return;
    }
    // For stub, just allocate host memory or throw
    // *ptr = malloc(size);
    throw runtime_error("NPUBackend::allocateDevice not implemented");
}

void NPUBackend::allocateHost(void** ptr, size_t size) {
    *ptr = malloc(size);
}

void NPUBackend::allocateManaged(void** ptr, size_t size) {
    throw runtime_error("NPUBackend::allocateManaged not implemented");
}

void NPUBackend::freeDevice(void* ptr) {
    // free(ptr);
}

void NPUBackend::freeHost(void* ptr) {
    free(ptr);
}

void NPUBackend::freeManaged(void* ptr) {
}

void NPUBackend::verify_call(const char* name) {
}

void NPUBackend::synchronize() {
}

void NPUBackend::copyDeviceToHost(void* dst_host, const void* src_device, size_t size) {
    throw runtime_error("NPUBackend::copyDeviceToHost not implemented");
}

void NPUBackend::copyHostToDevice(void* dst_device, const void* src_host, size_t size) {
    throw runtime_error("NPUBackend::copyHostToDevice not implemented");
}

void NPUBackend::zeroDevice(void* ptr, size_t size) {
    throw runtime_error("NPUBackend::zeroDevice not implemented");
}

unique_ptr<FFTPlan> NPUBackend::createFFTPlan(int size, int batch_size) {
    throw runtime_error("NPUBackend::createFFTPlan not implemented");
}

void NPUBackend::sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) {
    throw runtime_error("NPUBackend::sumColumns not implemented");
}

const float* NPUBackend::taylorTree(const float* input, float* buffer1, float* buffer2,
                        int num_timesteps, int num_channels, int drift_block) {
    throw runtime_error("NPUBackend::taylorTree not implemented");
}

void NPUBackend::findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                     int drift_block, float* top_path_sums,
                     int* top_drift_blocks, int* top_path_offsets) {
     throw runtime_error("NPUBackend::findTopPathSums not implemented");
}

void NPUBackend::convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                         int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) {
    throw runtime_error("NPUBackend::convertRawToComplex not implemented");
}

void NPUBackend::shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size, std::complex<float>* output, size_t output_size,
                    int fft_size, int num_antennas, int num_polarizations, int num_coarse_channels, int num_timesteps) {
    throw runtime_error("NPUBackend::shiftFFTOutput not implemented");
}

void NPUBackend::complexGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    std::complex<float> alpha,
    const std::complex<float>* A, int lda, long long strideA,
    const std::complex<float>* B, int ldb, long long strideB,
    std::complex<float> beta,
    std::complex<float>* C, int ldc, long long strideC,
    int batchCount) {
    throw runtime_error("NPUBackend::complexGemmStridedBatched not implemented");
}

void NPUBackend::floatGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    throw runtime_error("NPUBackend::floatGemmStridedBatched not implemented");
}

void NPUBackend::batchedPower(const std::complex<float>* voltage, float* power,
                  int num_beams, int num_channels, int num_polarizations,
                  int sti, int power_time_offset, int num_output_timesteps) {
    throw runtime_error("NPUBackend::batchedPower not implemented");
}

void NPUBackend::incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                             float* output,
                             int num_beams, int num_coarse_channels, int num_polarizations,
                             int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) {
    throw runtime_error("NPUBackend::incoherentPower not implemented");
}
