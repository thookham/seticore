#pragma once

#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_COMPILER_VERSION)
#include <sycl/sycl.hpp>
using namespace sycl;

// Kernel for summing columns
class SumColumnsKernel;
class TaylorOneStepKernel;

void submitSumColumns(queue& q, const float* input, float* sums, int num_timesteps, int num_freqs) {
    q.submit([&](handler& h) {
        h.parallel_for<SumColumnsKernel>(range<1>(num_freqs), [=](id<1> idx) {
            int f = idx[0];
            float sum = 0.0f;
            for (int t = 0; t < num_timesteps; ++t) {
                // Manual index calculation: input is [num_timesteps][num_freqs] (row-major)
                // idx = t * num_freqs + f
                sum += input[t * num_freqs + f];
            }
            sums[f] = sum;
        });
    });
}


// SYCL version of taylorOneStepOneChannel
// We can't include taylor.h easily because it has HOST_DEVICE macros which might conflict or be weird in pure SYCL if not set up right.
// So we re-implement the logic here inside the kernel lambda.
void submitTaylorOneStep(queue& q, const float* source_buffer, float* target_buffer, 
                         int num_timesteps, int num_channels, int path_length, int drift_block) {
    
    // We launch one thread per channel
    q.submit([&](handler& h) {
        h.parallel_for<TaylorOneStepKernel>(range<1>(num_channels), [=](id<1> idx) {
            int chan = idx[0];
            int num_time_blocks = num_timesteps / path_length;
            
            for (int time_block = 0; time_block < num_time_blocks; ++time_block) {
                for (int path_offset = path_length - 1; path_offset >= 0; path_offset--) {
                    int half_offset = path_offset / 2;
                    int chan_shift = (path_offset + 1) / 2 + drift_block * path_length / 2;

                    if (chan + chan_shift < 0 || chan + chan_shift >= num_channels) {
                        continue;
                    }

                    // Indexing logic from taylor.h:
                    // target_buffer[(time_block * path_length + path_offset) * num_target_channels + chan]
                    long target_idx = (long)(time_block * path_length + path_offset) * num_channels + chan;
                    
                    // source_buffer[(time_block * path_length + half_offset) * num_source_channels + chan]
                    long src1_idx = (long)(time_block * path_length + half_offset) * num_channels + chan;
                    
                    // source_buffer[(time_block * path_length + half_offset + path_length / 2) * num_source_channels + chan + chan_shift];
                    long src2_idx = (long)(time_block * path_length + half_offset + path_length / 2) * num_channels + chan + chan_shift;

                    target_buffer[target_idx] = source_buffer[src1_idx] + source_buffer[src2_idx];
                }
            }
        });
    });
}

// Helpers for indexing (inlined to ensure device compatibility)
inline long index4d_sycl(int i, int j, int k, int l, int Nj, int Nk, int Nl) {
    return ((long)i * Nj * Nk * Nl) + ((long)j * Nk * Nl) + ((long)k * Nl) + l;
}

inline long index5d_sycl(int i, int j, int k, int l, int m, int Nj, int Nk, int Nl, int Nm) {
    return ((long)i * Nj * Nk * Nl * Nm) + ((long)j * Nk * Nl * Nm) + ((long)k * Nl * Nm) + ((long)l * Nm) + m;
}

class ConvertRawKernel;
class ShiftFFTKernel;

void submitConvertRaw(queue& q, const int8_t* input, std::complex<float>* buffer, 
                      int num_antennas, int nblocks, int num_coarse_channels,
                      int num_polarizations, int nsamp, int time_per_block) {
    
    // Total work items: nblocks * num_antennas * num_coarse_channels * time_per_block
    // We can use a flat range and reconstruct indices
    size_t total_items = (size_t)nblocks * num_antennas * num_coarse_channels * time_per_block;
    
    q.submit([&](handler& h) {
        h.parallel_for<ConvertRawKernel>(range<1>(total_items), [=](id<1> idx) {
            long linear_id = idx[0];
            
            // Reconstruct indices matching CUDA grid logic:
            // Grid was: (time, block, ant*coarse)
            // But we can iterate naturally:
            
            // logical: block, antenna, chan, time_in_block
            // Dims:
            // N_time_in_block = time_per_block
            // N_chan = num_coarse_channels
            // N_ant = num_antennas
            // N_block = nblocks
            
            // Let's decode from linear_id assuming row-major ordering of loops:
            // block, antenna, chan, time_in_block (outer to inner)
            int t_in_block = linear_id % time_per_block;
            long rem = linear_id / time_per_block;
            int chan = rem % num_coarse_channels;
            rem /= num_coarse_channels;
            int ant = rem % num_antennas;
            int block = rem / num_antennas;
            
            // Now compute input/output indices
            int time = block * time_per_block + t_in_block;
            if (time >= nsamp) return;
            
            for (int pol = 0; pol < num_polarizations; ++pol) {
                // Input index: [block][antenna][chan][t_in_block][pol][real/imag]
                // 2 * index5d(block, antenna, num_antennas, chan, num_coarse_channels, t_in_block, time_per_block, pol, num_polarizations)
                long input_index = 2 * index5d_sycl(block, ant, num_antennas, chan, num_coarse_channels, t_in_block, time_per_block, pol, num_polarizations);
                
                // Output index: [pol][antenna][chan][time]
                // index4d(pol, antenna, num_antennas, chan, num_coarse_channels, time, nsamp)
                long output_index = index4d_sycl(pol, ant, num_antennas, chan, num_coarse_channels, time, nsamp);
                
                // We treat complex<float> as float[2] effectively for assignment
                // But `buffer` is std::complex<float>*.
                // sycl::complex exists but std::complex is often supported.
                // Safest to construct:
                float r = (float)input[input_index];
                float i = (float)input[input_index + 1];
                buffer[output_index] = std::complex<float>(r, i);
            }
        });
    });
}

void submitShift(queue& q, std::complex<float>* buffer, std::complex<float>* output,
                 int fft_size, int num_antennas, int num_polarizations,
                 int num_coarse_channels, int num_timesteps) {
                     
    // Total items: fft_size * coarse * timesteps * ant * pol
    size_t total = (size_t)fft_size * num_coarse_channels * num_timesteps * num_antennas * num_polarizations;
    
    q.submit([&](handler& h) {
        h.parallel_for<ShiftFFTKernel>(range<1>(total), [=](id<1> idx) {
            long linear_id = idx[0];
            
            // Decode logical indices. Order (outer to inner) match CUDA or purely logical?
            // CUDA used:
            // Grid: (fine_chan, coarse_chan, time)
            // Block: (1, ant, pol)
            // So loops: time, coarse, fine, ant, pol
            
            // Let's decode:
            int pol = linear_id % num_polarizations;
            long rem = linear_id / num_polarizations;
            int ant = rem % num_antennas;
            rem /= num_antennas;
            int fine = rem % fft_size;
            rem /= fft_size;
            int coarse = rem % num_coarse_channels;
            int time = rem / num_coarse_channels;
            
            int output_fine_chan = fine ^ (fft_size >> 1);

            // Input: [pol][ant][coarse][time][fine]
            long input_index = index5d_sycl(pol, ant, num_antennas, coarse, num_coarse_channels, 
                                            time, num_timesteps, fine, fft_size);
            
            // Output: [time][coarse][fine_out][pol][ant]
            long output_index = index5d_sycl(time, coarse, num_coarse_channels, 
                                             output_fine_chan, fft_size, 
                                             pol, num_polarizations, ant, num_antennas);
            
            output[output_index] = buffer[input_index];
        });
    });
}

class FindTopPathSumsKernel;

void submitFindTopPathSums(queue& q, const float* taylor_sums, int num_timesteps, int num_freqs,
                           int drift_block, float* top_path_sums,
                           int* top_drift_blocks, int* top_path_offsets) {
    q.submit([&](handler& h) {
        h.parallel_for<FindTopPathSumsKernel>(range<1>(num_freqs), [=](id<1> idx) {
            int f = idx[0];
            
            // Iterate over all time steps for this frequency channel to find the max
            for (int step = 0; step < num_timesteps; ++step) {
                // taylor_sums is [num_timesteps][num_freqs] row major
                // but wait, the taylor tree output might be transposed or indexed differently?
                // The CPU reference logic used: index2d(step, f, num_freqs)
                // Let's assume consistent layout.
                
                long idx_val = (long)step * num_freqs + f;
                float val = taylor_sums[idx_val];

                // Check against current max. 
                // Note: This read/write to global memory might be slow if not cached, 
                // but logic is correct. 
                // Also, we assume initialized top_path_sums by caller (Dedopplerer::search zeroes it).
                
                if (val > top_path_sums[f]) {
                    top_path_sums[f] = val;
                    top_drift_blocks[f] = drift_block;
                    top_path_offsets[f] = step;
                }
            }
        });
    });
}

#endif
