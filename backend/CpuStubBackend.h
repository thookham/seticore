#pragma once

#include "ComputeBackend.h"
#include <cstring>
#include <cstdlib>

/**
 * @brief CPU-only stub backend. 
 * 
 * Provides basic memory management but no actual acceleration.
 * Kernels are either no-ops or simple CPU implementations (tbd).
 * Useful for building/testing on systems without GPUs.
 */
class CpuStubBackend : public ComputeBackend {
public:
    void* allocate(size_t size) override {
        return malloc(size);
    }

    void free(void* ptr) override {
        ::free(ptr);
    }

    void copyHostToDevice(void* dst_device, const void* src_host, size_t size) override {
        memcpy(dst_device, src_host, size);
    }

    void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) override {
        memcpy(dst_host, src_device, size);
    }

    void memset(void* ptr, int value, size_t size) override {
        ::memset(ptr, value, size);
    }

    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override {
        // Warning: No-op / Fallback using CPU logic would go here.
        return buffer1;
    }

    void findTopPathSums(const float* path_sums, int num_timesteps, int num_freqs,
                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override {
        // Warning: No-op
    }

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override {
        // Warning: No-op
    }
};
