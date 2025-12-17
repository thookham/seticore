#pragma once

#include <cstddef>
#include <vector>

/**
 * @brief Abstract interface for hardware-accelerated compute backends.
 * 
 * This class defines the primitives required by seticore to run the
 * search pipeline. Implementations can wrap CUDA, HIP, SYCL, or other
 * compute APIs.
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // --- Memory Management ---

    /**
     * @brief Allocate memory on the device.
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Free memory on the device.
     */
    virtual void free(void* ptr) = 0;

    /**
     * @brief Copy data from host to device.
     */
    virtual void copyHostToDevice(void* dst_device, const void* src_host, size_t size) = 0;

    /**
     * @brief Copy data from device to host.
     */
    virtual void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) = 0;

    /**
     * @brief Zero out device memory.
     */
    virtual void memset(void* ptr, int value, size_t size) = 0;

    // --- Core Kernels ---

    /**
     * @brief Run the Taylor Tree algorithm.
     * 
     * @param input Device pointer to input buffer.
     * @param buffer1 Temporary device buffer 1.
     * @param buffer2 Temporary device buffer 2.
     * @param num_timesteps Number of time steps.
     * @param num_channels Number of frequency channels.
     * @param drift_block The drift block index to process.
     * @return const float* Pointer to the result buffer (either buffer1 or buffer2).
     */
    virtual const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                                    int num_timesteps, int num_channels, int drift_block) = 0;

    /**
     * @brief Find the top path sums for each frequency bin.
     */
    virtual void findTopPathSums(const float* path_sums, int num_timesteps, int num_freqs,
                                 int drift_block, float* top_path_sums,
                                 int* top_drift_blocks, int* top_path_offsets) = 0;

    /**
     * @brief Build column sums for normalization.
     */
    virtual void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) = 0;
};
