#pragma once

#include "ComputeBackend.h"

// Check if we are actually compiling with a SYCL compiler
// #include <sycl/sycl.hpp> is the standard, but sometimes it is CL/sycl.hpp
#if defined(__SYCL_COMPILER_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/blas.hpp>
#else
// Fallback or error if not a SYCL compiler, but for now we might want to allow 
// including this header for inspection even if not compiling with dpc++
// But realistically, this file is only used when we have a SYCL compiler.
// We'll assume the compiler provides the headers.
#endif

#include <iostream>
#include <vector>

using namespace std;

// Forward declare or include util for fatal/logging
#include "../../util.h"

#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
using namespace sycl;
#endif

class SyclBackend : public ComputeBackend {
public:
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
    queue q;

    SyclBackend() : q(default_selector_v) {
        cout << "SyclBackend using device: " 
             << q.get_device().get_info<info::device::name>() << endl;
    }
#else
    // Stub for non-SYCL compilation (should not happen if build logic is correct)
    SyclBackend() {
        fatal("SyclBackend compiled without SYCL compiler support!");
    }
#endif

    void allocateDevice(void** ptr, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        *ptr = malloc_device(size, q);
        if (!*ptr && size > 0) {
            fatal("SyclBackend::allocateDevice failed");
        }
#endif
    }

    void allocateHost(void** ptr, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        *ptr = malloc_host(size, q);
#endif
    }

    void allocateManaged(void** ptr, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        *ptr = malloc_shared(size, q);
#endif
    }

    void freeDevice(void* ptr) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        if (ptr) sycl::free(ptr, q);
#endif
    }

    void freeHost(void* ptr) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        if (ptr) sycl::free(ptr, q);
#endif
    }

    void freeManaged(void* ptr) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        if (ptr) sycl::free(ptr, q);
#endif
    }

    void verify_call(const char* name) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        try {
            q.wait_and_throw();
        } catch (exception const& e) {
            fatal(string("SYCL exception in ") + name + ": " + e.what());
        }
#endif
    }

    void synchronize() override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        q.wait();
#endif
    }

    void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        q.memcpy(dst_host, src_device, size).wait();
#endif
    }

    void copyHostToDevice(void* dst_device, const void* src_host, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        q.memcpy(dst_device, src_host, size).wait();
#endif
    }
    
    void zeroDevice(void* ptr, size_t size) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        q.memset(ptr, 0, size).wait();
#endif
    }

    unique_ptr<FFTPlan> createFFTPlan(int size, int batch_size) override;

    // Define SyclFFTPlan inside or outside? 
    // Usually inner class or implementation. 
    // I'll make it a nested class for now or just defined in the cpp if we had one.
    // Since this is a header-only backend (SyclBackend.h), I must define it here.
    // I'll define it *before* createFFTPlan if possible, or forward declare.
    // To fix the previous tool error, I'll replace the whole block I added.
    
    // I'll forward declare SyclFFTPlan here?
    // Actually, C++ allows nested classes.
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
    class SyclFFTPlan : public FFTPlan {
        using Descriptor = oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, 
                                                        oneapi::mkl::dft::domain::COMPLEX>;
        Descriptor descriptor;
        sycl::queue& q;

    public:
        SyclFFTPlan(int size, int batch_size, sycl::queue& queue) 
            : descriptor(size), q(queue) {
            descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, size);
            descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, size);
            descriptor.commit(q);
        }

        void execute(void* input, void* output, bool forward) override {
             auto in_ptr = reinterpret_cast<std::complex<float>*>(input);
             auto out_ptr = reinterpret_cast<std::complex<float>*>(output);
             if (forward) {
                 if (input == output) oneapi::mkl::dft::compute_forward(descriptor, in_ptr).wait();
                 else oneapi::mkl::dft::compute_forward(descriptor, in_ptr, out_ptr).wait();
             } else {
                 if (input == output) oneapi::mkl::dft::compute_backward(descriptor, in_ptr).wait();
                 else oneapi::mkl::dft::compute_backward(descriptor, in_ptr, out_ptr).wait();
             }
        }
    };
#endif
    // End of nested class
    // Now implementation of createFFTPlan
    
    // Note: I need to target the block I previously inserted which had the includes.
    // The previous block started with "#if defined(SYCL... #include ... unique_ptr..."
    // I should replace THAT block.

#include "SyclKernels.h"

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        submitSumColumns(q, input, sums, num_timesteps, num_freqs);
        q.wait(); // Synchronize for simplicity in Phase 2
#endif
    }

    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        const float* source_buffer = input;
        float* target_buffer = buffer1;

        for (int path_length = 2; path_length <= num_timesteps; path_length *= 2) {
            submitTaylorOneStep(q, source_buffer, target_buffer, num_timesteps, num_channels, path_length, drift_block);
            
            // Swap
            if (target_buffer == buffer1) {
                source_buffer = buffer1;
                target_buffer = buffer2;
            } else {
                source_buffer = buffer2;
                target_buffer = buffer1;
            }
        }
        q.wait();
        return source_buffer;
#else
        return nullptr;
#endif
    }

    void findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        submitFindTopPathSums(q, taylor_sums, num_timesteps, num_freqs, drift_block, 
                              top_path_sums, top_drift_blocks, top_path_offsets);
        q.wait();
#endif
        q.wait();
#endif
    }

    void convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                             int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        // nsamp = num_timesteps
        int nsamp = num_timesteps;
        int nblocks = num_timesteps / time_per_block;
        submitConvertRaw(q, raw, complex, num_antennas, nblocks, num_coarse_channels, num_polarizations, nsamp, time_per_block);
        q.wait();
#endif
    }

    void shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size, std::complex<float>* output, size_t output_size,
                        int fft_size, int num_antennas, int num_polarizations, int num_coarse_channels, int num_timesteps) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        submitShift(q, buffer, output, fft_size, num_antennas, num_polarizations, num_coarse_channels, num_timesteps);
        q.wait();
#endif
    }
    
    // Naive / Stub implementations for now
    void complexGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        std::complex<float> alpha,
        const std::complex<float>* A, int lda, long long strideA,
        const std::complex<float>* B, int ldb, long long strideB,
        std::complex<float> beta,
        std::complex<float>* C, int ldc, long long strideC,
        int batchCount) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        oneapi::mkl::blas::column_major::gemm_batch(
            q,
            transA ? oneapi::mkl::transpose::conjtrans : oneapi::mkl::transpose::nontrans,
            transB ? oneapi::mkl::transpose::conjtrans : oneapi::mkl::transpose::nontrans,
            m, n, k,
            alpha,
            A, lda, strideA,
            B, ldb, strideB,
            beta,
            C, ldc, strideC,
            batchCount
        ).wait();
#endif
    }

    void floatGemmStridedBatched(
        bool transA, bool transB,
        int m, int n, int k,
        float alpha,
        const float* A, int lda, long long strideA,
        const float* B, int ldb, long long strideB,
        float beta,
        float* C, int ldc, long long strideC,
        int batchCount) override {
#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
        oneapi::mkl::blas::column_major::gemm_batch(
            q,
            transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
            transB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
            m, n, k,
            alpha,
            A, lda, strideA,
            B, ldb, strideB,
            beta,
            C, ldc, strideC,
            batchCount
        ).wait();
#endif
    }
};
