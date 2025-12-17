#pragma once

#include "ComputeBackend.h"
#include <cuda_runtime.h>

// Forward declarations of existing kernels/helpers if needed
// In a full refactor, these would be moved or included from a common header.

class CUDABackend : public ComputeBackend {
public:
    void* allocate(size_t size) override {
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    void free(void* ptr) override {
        cudaFree(ptr);
    }

    void copyHostToDevice(void* dst_device, const void* src_host, size_t size) override {
        cudaMemcpy(dst_device, src_host, size, cudaMemcpyHostToDevice);
    }

    void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) override {
        cudaMemcpy(dst_host, src_device, size, cudaMemcpyDeviceToHost);
    }

    void memset(void* ptr, int value, size_t size) override {
        cudaMemset(ptr, value, size);
    }

    // Kernel wrappers would need to call the existing __global__ functions
    // For now, this is a placeholder to show structure.
    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override {
        // Implementation would call optimizedTaylorTree(...)
        return buffer1; 
    }

    void findTopPathSums(const float* path_sums, int num_timesteps, int num_freqs,
                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override {
        // Implementation would call findTopPathSums<<<...>>>(...)
    }

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override {
        // Implementation would call sumColumns<<<...>>>(...)
    }
};
