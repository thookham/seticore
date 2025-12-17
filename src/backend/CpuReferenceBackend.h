#pragma once

#include "ComputeBackend.h"
#include "../../taylor.h"
#include "../../util.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

using namespace std;

#ifdef HAVE_FFTW3
class CpuFFTPlan : public FFTPlan {
    fftwf_plan plan;
    int size;
    int batch_size;
    fftwf_complex* in_ptr;
    fftwf_complex* out_ptr;
    
public:
    CpuFFTPlan(int size, int batch_size) : size(size), batch_size(batch_size) {
        // FFTW planning
        in_ptr = fftwf_alloc_complex(size * batch_size);
        out_ptr = fftwf_alloc_complex(size * batch_size);
        
        int rank = 1;
        int n[] = {size};
        int howmany = batch_size;
        int idist = size;
        int odist = size;
        int istride = 1;
        int ostride = 1;
        int *inembed = n;
        int *onembed = n;

        plan = fftwf_plan_many_dft(rank, n, howmany, 
                                   in_ptr, inembed, istride, idist,
                                   out_ptr, onembed, ostride, odist,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    }
    
    ~CpuFFTPlan() {
        fftwf_destroy_plan(plan);
        fftwf_free(in_ptr);
        fftwf_free(out_ptr);
    }
    
    void execute(void* input, void* output, bool forward) override {
        // Copy input to internal buffer? Or assume pointers are compatible?
        // With generic void*, we should copy to be safe if types differ, 
        // but for high perf we want direct access.
        // float* (interleaved complex) is compatible with fftwf_complex*.
        
        // Execute NEW ARRAY dft
        fftwf_execute_dft(plan, (fftwf_complex*)input, (fftwf_complex*)output);
    }
};
#endif

class CpuReferenceBackend : public ComputeBackend {
public:
    void allocateDevice(void** ptr, size_t size) override {
        *ptr = malloc(size);
        if (!*ptr && size > 0) {
            fatal("CpuReferenceBackend::allocateDevice failed");
        }
    }

    void allocateHost(void** ptr, size_t size) override {
        *ptr = malloc(size);
        if (!*ptr && size > 0) {
            fatal("CpuReferenceBackend::allocateHost failed");
        }
    }

    void allocateManaged(void** ptr, size_t size) override {
        *ptr = malloc(size);
        if (!*ptr && size > 0) {
           fatal("CpuReferenceBackend::allocateManaged failed");
        }
    }

    void freeDevice(void* ptr) override {
        if (ptr) free(ptr);
    }

    void freeHost(void* ptr) override {
        if (ptr) free(ptr);
    }

    void freeManaged(void* ptr) override {
        if (ptr) free(ptr);
    }

    void verify_call(const char* name) override {
        // CPU calls are synchronous and usually don't have async error states like CUDA
    }

    void synchronize() override {
        // CPU execution is synchronous
    }

    unique_ptr<FFTPlan> createFFTPlan(int size, int batch_size) override {
#ifdef HAVE_FFTW3
        return unique_ptr<FFTPlan>(new CpuFFTPlan(size, batch_size));
#else
        cerr << "Warning: FFT requested but seticore compiled without FFTW3." << endl;
        return nullptr;
#endif
    }

    // Kernels
    
    void copyDeviceToHost(void* dst_host, const void* src_device, size_t size) override {
        memcpy(dst_host, src_device, size);
    }

    void copyHostToDevice(void* dst_device, const void* src_host, size_t size) override {
        memcpy(dst_device, src_host, size);
    }
    
    void zeroDevice(void* ptr, size_t size) override {
        memset(ptr, 0, size);
    }

    void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) override {
        // Debug logging
        // cout << "CpuReferenceBackend::sumColumns ts=" << num_timesteps << " freqs=" << num_freqs << endl;
        
        for (int f = 0; f < num_freqs; ++f) {
            float sum = 0.0f;
            for (int t = 0; t < num_timesteps; ++t) {
                sum += input[index2d(t, f, num_freqs)];
            }
            sums[f] = sum;
        }
    }

    const float* taylorTree(const float* input, float* buffer1, float* buffer2,
                            int num_timesteps, int num_channels, int drift_block) override {
        // Logic adapted from basicTaylorTree in taylor.cu
        
        const float* source_buffer = input;
        float* target_buffer = buffer1;

        // Loop path_length from 2 up to num_timesteps
        for (int path_length = 2; path_length <= num_timesteps; path_length *= 2) {
            // For each channel, run one step
            for (int chan = 0; chan < num_channels; ++chan) {
                taylorOneStepOneChannel(source_buffer, target_buffer,
                                        chan, num_timesteps, num_channels, num_channels,
                                        path_length, drift_block);
            }

            // Swap buffers
            if (target_buffer == buffer1) {
                source_buffer = buffer1;
                target_buffer = buffer2;
            } else {
                source_buffer = buffer2;
                target_buffer = buffer1;
            }
        }

        return source_buffer;
    }

    void findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                         int drift_block, float* top_path_sums,
                         int* top_drift_blocks, int* top_path_offsets) override {
        for (int f = 0; f < num_freqs; ++f) {
            for (int step = 0; step < num_timesteps; ++step) {
                long idx = index2d(step, f, num_freqs);
                float val = taylor_sums[idx];
                
                if (val > top_path_sums[f]) {
                    top_path_sums[f] = val;
                    top_drift_blocks[f] = drift_block;
                    top_path_offsets[f] = step;
                }
            }
        }
    }

    void convertRawToComplex(const int8_t* raw, size_t raw_size, std::complex<float>* complex, size_t complex_size,
                             int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) override {
        // complex buffer size should be enough, assumed trusted
        float* buffer = (float*)complex; // Reinterpret cast, std::complex<float> is layout compatible with float[2]
        int nblocks = num_timesteps / time_per_block;
        int nsamp = num_timesteps;
        // previous impl used `convertRaw` style signature logic
        // logic below uses `input` as `raw`, `buffer` as `complex`
        
        for (int pol = 0; pol < num_polarizations; ++pol) {
            for (int antenna = 0; antenna < num_antennas; ++antenna) {
                for (int chan = 0; chan < num_coarse_channels; ++chan) {
                    for (int block = 0; block < nblocks; ++block) {
                       for (int t_in_block = 0; t_in_block < time_per_block; ++t_in_block) {
                           int time = block * time_per_block + t_in_block;
                           if (time >= nsamp) continue;
                           
                           long input_idx = 2 * index5d(block, antenna, num_antennas, chan, num_coarse_channels,
                                                       t_in_block, time_per_block, pol, num_polarizations);
                           long output_idx = 2 * index4d(pol, antenna, num_antennas, chan, num_coarse_channels, time, nsamp);
                           
                           buffer[output_idx] = (float)raw[input_idx];
                           buffer[output_idx + 1] = (float)raw[input_idx + 1];
                       }
                    }
                }
            }
        }
    }

    void batchedPower(const std::complex<float>* voltage, float* power,
                      int num_beams, int num_channels, int num_polarizations,
                      int sti, int power_time_offset, int num_output_timesteps) override {
        // voltage: [time][pol][chan][beam]  (Assuming index4d layout: time outer, beam inner? No.)
        // Original: index4d(time, pol, npol, chan, nchan, beam, nbeam)
        // -> ((time * npol + pol) * nchan + chan) * nbeam + beam
        
        // power: [beam][time][chan]
        // Original: index3d(beam, output_timestep, num_power_timesteps, chan, nchan)
        // -> (beam * num_power_timesteps + output_timestep) * nchan + chan
        
        for (int t = 0; t < num_output_timesteps; ++t) {
            int output_timestep = t + power_time_offset;
            for (int b = 0; b < num_beams; ++b) {
                for (int c = 0; c < num_channels; ++c) {
                    
                    float sum = 0.0f;
                    for (int s = 0; s < sti; ++s) {
                        int input_time = t * sti + s;
                        
                        // Voltage Indices
                        // Pol 0
                        long idx0 = ((long)input_time * num_polarizations * num_channels * num_beams) +
                                    ((long)0 * num_channels * num_beams) +
                                    ((long)c * num_beams) + b;
                                    
                        // Pol 1
                        long idx1 = ((long)input_time * num_polarizations * num_channels * num_beams) +
                                    ((long)1 * num_channels * num_beams) +
                                    ((long)c * num_beams) + b;
                                    
                        std::complex<float> v0 = voltage[idx0];
                        std::complex<float> v1 = voltage[idx1];
                        
                        sum += std::norm(v0) + std::norm(v1); // norm is |z|^2
                    }
                    
                    // Power Index
                    long p_idx = ((long)b * num_output_timesteps * num_channels) +
                                 ((long)output_timestep * num_channels) + c;
                                 
                    // Wait, original power index used num_power_timesteps for dimension, but did it correspond to output_timesteps?
                    // calculatePower args: num_power_timesteps passed as output.num_timesteps.
                    // So yes.
                    
                    // However, we need to verify if `output_timestep` includes offset for ROW calculation?
                    // power_index = index3d(..., output_timestep, num_power_timesteps, ...)
                    // If power is a full buffer, yes.
                    
                    power[p_idx] = sum;
                }
            }
        }
    }

    void incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                         float* output,
                         int num_beams, int num_coarse_channels, int num_polarizations,
                         int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) override {
        // input: prebeam
        // index5d(global_timestep, coarse, num_coarse, fine, fft_size, pol, npol, ant, nant)
        // dims: [time][coarse][fine][pol][ant]
        
        // sq_mags: [coarse][pol][ant]
        // index3d(coarse, pol, npol, ant, nant)
        
        // output: [big_timestep][coarse][fine]
        // index3d(big_timestep, coarse, num_coarse, fine, fft_size)
        // big_timestep relates to output time? Yes.
        
        // Note: num_beams argument seems unused for incoherent (it's 1 beam effectively, separate buffer?)
        // The calling code passes output buffer offset.
        
        for (int t = 0; t < num_output_timesteps; ++t) {
            // int big_timestep = t; // loop local - Not needed, using 't' directly 
            // Original kernel used blockIdx.z for big_timestep.
            // But verify if power_time_offset affects output index?
            // "output.data + data_offset" is passed to kernel. 
            // So output is already offset. generic 't' maps to 0-based in THIS buffer.
            
            // Wait, calculatePower handled offset in index calculation because it wrote to a larger buffer?
            // incoherentPower expects specific output pointer.
            // Let's assume output points to the start of the relevant block.
            // But we need to use 'power_time_offset' for Input calculation?
            // "long global_timestep = index2d(big_timestep, little_timestep, sti);"
            // where big_timestep was from grid.
            // Beamformer.cpp: "output.data + data_offset" passed.
            // "power_time_offset" used to calc data_offset.
            // But inside kernel: "index2d(big_timestep, ..."
            // Wait, does big_timestep implicitly start at 0 or power_time_offset?
            // Kernel: "int big_timestep = blockIdx.z".
            // Grid size: "numOutputTimesteps()". (Local batch size).
            // So big_timestep is 0..N-1 relative to the batch.
            // Global timestep calc uses it directly. 
            // "run" method does NOT apply power_time_offset to input buffer reading?
            // Input buffer is "prebeam".
            // Prebeam is overwritten every batch.
            // So `global_timestep` is relative to start of THIS batch.
            // Correct.
            
            for (int c = 0; c < num_coarse_channels; ++c) {
                for (int f = 0; f < fft_size; ++f) {
                    
                    float sum = 0.0f;
                    
                    // Sum over antennas (reduction)
                     // Using separate reduction logic? No, naive sum is fine.
                    
                    // Sum over STI
                    for (int s = 0; s < sti; ++s) {
                         long global_time = t * sti + s;
                         
                         // Sum over Antennas and Pols
                         for (int a = 0; a < num_antennas; ++a) {
                             // Get weight
                             long sq_idx = ((long)c * num_polarizations * num_antennas) +
                                           ((long)0 * num_antennas) + a; // Pol 0
                             float w0 = square_magnitudes[sq_idx];
                             
                             long sq_idx1 = ((long)c * num_polarizations * num_antennas) +
                                            ((long)1 * num_antennas) + a; // Pol 1
                             float w1 = square_magnitudes[sq_idx1];
                             
                             // Get Input
                             // [time][coarse][fine][pol][ant]
                             long in_base = ((long)global_time * num_coarse_channels * fft_size * num_polarizations * num_antennas) +
                                            ((long)c * fft_size * num_polarizations * num_antennas) +
                                            ((long)f * num_polarizations * num_antennas);
                             
                             long in_idx0 = in_base + ((long)0 * num_antennas) + a;
                             long in_idx1 = in_base + ((long)1 * num_antennas) + a;
                             
                             std::complex<float> v0 = input[in_idx0];
                             std::complex<float> v1 = input[in_idx1];
                             
                             sum += std::norm(v0) * w0 + std::norm(v1) * w1;
                         }
                    }
                    
                    // Output
                    // [big_time][coarse][fine]
                    long out_idx = ((long)t * num_coarse_channels * fft_size) +
                                   ((long)c * fft_size) + f;
                    output[out_idx] = sum;
                }
            }
        }
    }

    void shiftFFTOutput(std::complex<float>* buffer_c, size_t buffer_size,
                        std::complex<float>* output_c, size_t output_size,
                        int fft_size, int num_antennas, int num_polarizations,
                        int num_coarse_channels, int num_timesteps) override {
        float* buffer = (float*)buffer_c;
        float* output = (float*)output_c;
        
        for (int pol = 0; pol < num_polarizations; ++pol) {
            for (int ant = 0; ant < num_antennas; ++ant) {
                for (int c = 0; c < num_coarse_channels; ++c) {
                    for (int t = 0; t < num_timesteps; ++t) {
                        for (int f = 0; f < fft_size; ++f) {
                            int fine_in = f;
                            int fine_out = fine_in ^ (fft_size >> 1);
                            
                            long in_idx = 2 * index5d(pol, ant, num_antennas, c, num_coarse_channels, t, num_timesteps, fine_in, fft_size);
                            long out_idx = 2 * index5d(t, c, num_coarse_channels, fine_out, fft_size, pol, num_polarizations, ant, num_antennas);
                            
                            output[out_idx] = buffer[in_idx];
                            output[out_idx+1] = buffer[in_idx+1];
                        }
                    }
                }
            }
        }
    }

    // Naive CPU GEMM implementation
    void complexGemmStridedBatched(
      bool transA, bool transB,
      int m, int n, int k,
      std::complex<float> alpha,
      const std::complex<float>* A, int lda, long long strideA,
      const std::complex<float>* B, int ldb, long long strideB,
      std::complex<float> beta,
      std::complex<float>* C, int ldc, long long strideC,
      int batchCount) override {
        
        // Parallelize over batches using OpenMP if available, keeping it simple for now.
        // #pragma omp parallel for
        for (int b = 0; b < batchCount; ++b) {
            const std::complex<float>* A_batch = A + b * strideA;
            const std::complex<float>* B_batch = B + b * strideB;
            std::complex<float>* C_batch = C + b * strideC;
            
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    std::complex<float> sum(0.0f, 0.0f);
                    for (int i = 0; i < k; ++i) {
                        std::complex<float> a_val;
                        if (!transA) {
                            a_val = A_batch[row + i * lda]; // Col-major: A[row, i]
                        } else {
                            // ConjTrans
                            a_val = std::conj(A_batch[i + row * lda]); // A[i, row]
                        }
                        
                        std::complex<float> b_val;
                        if (!transB) {
                            b_val = B_batch[i + col * ldb]; // B[i, col]
                        } else {
                            // ConjTrans
                            b_val = std::conj(B_batch[col + i * ldb]); // B[col, i] -> Wait, transB means B^H.
                            // B is k x n. B^H is n x k. 
                            // If transB is true, B input is n x k? No.
                            // Standard BLAS: op(B).
                            // If op(B) is n x k, then B is n x k (if no trans) or k x n (if trans).
                            // Wait. Matrix multiplication dimensions:
                            // C (m x n) = A (m x k) * B (k x n).
                            // If transA, A is k x m. 
                            // If transB, B is n x k. 
                            // So if transB, B buffer stores an n x k matrix, accessed as B[col, i] (if col major).
                            b_val = std::conj(B_batch[col + i * ldb]);
                        }
                        
                        sum += a_val * b_val;
                    }
                    
                    std::complex<float> c_val = C_batch[row + col * ldc];
                    if (beta == std::complex<float>(0.0f, 0.0f)) {
                        C_batch[row + col * ldc] = alpha * sum;
                    } else {
                        C_batch[row + col * ldc] = alpha * sum + beta * c_val;
                    }
                }
            }
        }
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
        // Naive CPU implementation
        for (int b = 0; b < batchCount; ++b) {
            const float* a_ptr = A + b * strideA;
            const float* b_ptr = B + b * strideB;
            float* c_ptr = C + b * strideC;
            
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    float sum = 0.0f;
                    for (int i = 0; i < k; ++i) {
                         float a_val = transA ? a_ptr[i * lda + row] : a_ptr[row * lda + i];
                         float b_val = transB ? b_ptr[col * ldb + i] : b_ptr[i * ldb + col];
                         sum += a_val * b_val;
                    }
                    
                    float c_val = c_ptr[row + col * ldc];
                     if (beta == 0.0f) {
                        c_ptr[row + col * ldc] = alpha * sum;
                    } else {
                        c_ptr[row + col * ldc] = alpha * sum + beta * c_val;
                    }
                }
            }
        }
    }
};
