#include <algorithm>
#include <assert.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>
#include <limits.h>

// #include "cuda_util.h" // Removed
#include "dedoppler.h"
// #include "taylor.h" // Removed, now accessed via backend
#include "util.h"
#include "dedoppler_hit.h"

using namespace std;

// Kernel implementations are now in backend

/*
  The Dedopplerer encapsulates the logic of dedoppler search. In particular it manages
  the needed GPU memory so that we can reuse the same memory allocation for different searches.
 */
Dedopplerer::Dedopplerer(int num_timesteps, int num_channels, double foff, double tsamp,
                         bool has_dc_spike, ComputeBackend* backend)
    : num_timesteps(num_timesteps), num_channels(num_channels), foff(foff), tsamp(tsamp),
      has_dc_spike(has_dc_spike), print_hits(false), backend(backend) {
  assert(num_timesteps > 1);
  rounded_num_timesteps = roundUpToPowerOfTwo(num_timesteps);
  drift_timesteps = rounded_num_timesteps - 1;

  drift_rate_resolution = 1e6 * foff / (drift_timesteps * tsamp);
    
  // Allocate everything we need for GPU processing 
  backend->allocateDevice((void**)&buffer1, num_channels * rounded_num_timesteps * sizeof(float));
  backend->verify_call("Dedopplerer buffer1 malloc");

  backend->allocateDevice((void**)&buffer2, num_channels * rounded_num_timesteps * sizeof(float));
  backend->verify_call("Dedopplerer buffer2 malloc");
  
  backend->allocateDevice((void**)&gpu_column_sums, num_channels * sizeof(float));
  backend->allocateHost((void**)&cpu_column_sums, num_channels * sizeof(float));
  backend->verify_call("Dedopplerer column_sums malloc");
  
  backend->allocateDevice((void**)&gpu_top_path_sums, num_channels * sizeof(float));
  backend->allocateHost((void**)&cpu_top_path_sums, num_channels * sizeof(float));
  backend->verify_call("Dedopplerer top_path_sums malloc");
   
  backend->allocateDevice((void**)&gpu_top_drift_blocks, num_channels * sizeof(int));
  backend->allocateHost((void**)&cpu_top_drift_blocks, num_channels * sizeof(int));
  backend->verify_call("Dedopplerer top_drift_blocks malloc");
  
  backend->allocateDevice((void**)&gpu_top_path_offsets, num_channels * sizeof(int));
  backend->allocateHost((void**)&cpu_top_path_offsets, num_channels * sizeof(int));
  backend->verify_call("Dedopplerer top_path_offsets malloc");
}

Dedopplerer::~Dedopplerer() {
  backend->freeDevice(buffer1);
  backend->freeDevice(buffer2);
  backend->freeDevice(gpu_column_sums);
  backend->freeHost(cpu_column_sums);
  backend->freeDevice(gpu_top_path_sums);
  backend->freeHost(cpu_top_path_sums);
  backend->freeDevice(gpu_top_drift_blocks);
  backend->freeHost(cpu_top_drift_blocks);
  backend->freeDevice(gpu_top_path_offsets);
  backend->freeHost(cpu_top_path_offsets);
}

// This implementation is an ugly hack
size_t Dedopplerer::memoryUsage() const {
  return num_channels * rounded_num_timesteps * sizeof(float) * 2
    + num_channels * (2 * sizeof(float) + 2 * sizeof(int));
}



/*
  Takes a bunch of hits that we found for coherent beams, and adds information
  about their incoherent beam

  Input should be the incoherent sum.
  This function re-sorts hits by drift, so be aware that it will change order.
 */
void Dedopplerer::addIncoherentPower(const FilterbankBuffer& input,
                                     vector<DedopplerHit>& hits) {
  assert(input.num_timesteps == rounded_num_timesteps);
  assert(input.num_channels == num_channels);

  sort(hits.begin(), hits.end(), &driftStepsLessThan);
  
  int drift_shift = rounded_num_timesteps - 1;
  
  // The drift block we are currently analyzing
  int current_drift_block = INT_MIN;

  // A pointer for the currently-analyzed drift block
  const float* taylor_sums = nullptr;

  for (DedopplerHit& hit : hits) {
    // Figure out what drift block this hit belongs to
    int drift_block = (int) floor((float) hit.drift_steps / drift_shift);
    int path_offset = hit.drift_steps - drift_block * drift_shift;
    assert(0 <= path_offset && path_offset < drift_shift);

    // We should not go backwards
    assert(drift_block >= current_drift_block);

    if (drift_block > current_drift_block) {
      // We need to analyze a new drift block
      taylor_sums = backend->taylorTree(input.data, buffer1, buffer2,
                                        rounded_num_timesteps, num_channels,
                                        drift_block);
      current_drift_block = drift_block;
    }

    long power_index = index2d(path_offset, hit.index, num_channels);
    assert(taylor_sums != nullptr);
    
    // cudaMemcpy(&hit.incoherent_power, taylor_sums + power_index, sizeof(float), cudaMemcpyDeviceToHost);
    // Since hit.incoherent_power is a float on stack/heap (host), and taylor_sums is on device.
    // backend->copyDeviceToHost expects (void* dst, const void* src, size_t size)
    // Pointer arithmetic on device pointer `taylor_sums` is valid (it's just an address), 
    // but dereferencing is not. We are passing the address.
    backend->copyDeviceToHost(&hit.incoherent_power, taylor_sums + power_index, sizeof(float));
  }
}

/*
  Runs dedoppler search on the input buffer.
  Output is appended to the output vector.
  
  All processing of the input buffer happens on the GPU, so it doesn't need to
  start off with host and device synchronized when search is called, it can still
  have GPU processing pending.
*/
void Dedopplerer::search(const FilterbankBuffer& input,
                         const FilterbankMetadata& metadata,
                         int beam, int coarse_channel,
                         double max_drift, double min_drift, double snr_threshold,
                         vector<DedopplerHit>* output) {
  assert(input.num_timesteps == rounded_num_timesteps);
  assert(input.num_channels == num_channels);

  // Normalize the max drift in units of "horizontal steps per vertical step"
  double diagonal_drift_rate = drift_rate_resolution * drift_timesteps;
  double normalized_max_drift = max_drift / abs(diagonal_drift_rate);
  int min_drift_block = floor(-normalized_max_drift);
  int max_drift_block = floor(normalized_max_drift);

  // Zero out the path sums in between each coarse channel because
  // we pick the top hits separately for each coarse channel
  backend->zeroDevice(gpu_top_path_sums, num_channels * sizeof(float));

  // sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input.data, gpu_column_sums,
  //                                             rounded_num_timesteps, num_channels);
  backend->sumColumns(input.data, gpu_column_sums, rounded_num_timesteps, num_channels);
  backend->verify_call("sumColumns");
  
  int mid = num_channels / 2;

  // Do the Taylor tree algorithm for each drift block
  for (int drift_block = min_drift_block; drift_block <= max_drift_block; ++drift_block) {
    // Calculate Taylor sums
    const float* taylor_sums = backend->taylorTree(input.data, buffer1, buffer2,
                                                   rounded_num_timesteps, num_channels,
                                                   drift_block);

    // Track the best sums
    // findTopPathSums<<<grid_size, CUDA_MAX_THREADS>>>(taylor_sums, rounded_num_timesteps,
    //                                                  num_channels, drift_block,
    //                                                  gpu_top_path_sums,
    //                                                  gpu_top_drift_blocks,
    //                                                  gpu_top_path_offsets);
    backend->findTopPathSums(taylor_sums, rounded_num_timesteps, num_channels, drift_block,
                             gpu_top_path_sums, gpu_top_drift_blocks, gpu_top_path_offsets);
    
    backend->verify_call("findTopPathSums");
  }

  // Now that we have done all the GPU processing for one coarse
  // channel, we can copy the data back to host memory.
  // These copies are not async, so they will synchronize to the default stream.
  
  /*
  cudaMemcpy(cpu_column_sums, gpu_column_sums,
             num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_sums, gpu_top_path_sums,
             num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_drift_blocks, gpu_top_drift_blocks,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_offsets, gpu_top_path_offsets,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  */
  backend->copyDeviceToHost(cpu_column_sums, gpu_column_sums, num_channels * sizeof(float));
  backend->copyDeviceToHost(cpu_top_path_sums, gpu_top_path_sums, num_channels * sizeof(float));
  backend->copyDeviceToHost(cpu_top_drift_blocks, gpu_top_drift_blocks, num_channels * sizeof(int));
  backend->copyDeviceToHost(cpu_top_path_offsets, gpu_top_path_offsets, num_channels * sizeof(int));
  
  backend->verify_call("dedoppler d->h memcpy");
  
  // Use the central 90% of the column sums to calculate standard deviation.
  // We don't need to do a full sort; we can just calculate the 5th,
  // 50th, and 95th percentiles
  auto column_sums_end = cpu_column_sums + num_channels;
  std::nth_element(cpu_column_sums, cpu_column_sums + mid, column_sums_end);
  int first = ceil(0.05 * num_channels);
  int last = floor(0.95 * num_channels);
  std::nth_element(cpu_column_sums, cpu_column_sums + first,
                   cpu_column_sums + mid - 1);
  std::nth_element(cpu_column_sums + mid + 1, cpu_column_sums + last,
                   column_sums_end);
  float median = cpu_column_sums[mid];
    
  float sum = std::accumulate(cpu_column_sums + first, cpu_column_sums + last + 1, 0.0);
  float m = sum / (last + 1 - first);
  float accum = 0.0;
  std::for_each(cpu_column_sums + first, cpu_column_sums + last + 1,
                [&](const float f) {
                  accum += (f - m) * (f - m);
                });
  float std_dev = sqrt(accum / (last + 1 - first));
    
  // We consider two hits to be duplicates if the distance in their
  // frequency indexes is less than window_size. We only want to
  // output the largest representative of any set of duplicates.
  // window_size is chosen just large enough so that a single bright
  // pixel cannot cause multiple hits.
  // First we break up the data into a set of nonoverlapping
  // windows. Any candidate hit must be the largest within this
  // window.
  float path_sum_threshold = snr_threshold * std_dev + median;
  int window_size = 2 * ceil(normalized_max_drift * drift_timesteps);
  for (int i = 0; i * window_size < num_channels; ++i) {
    int candidate_freq = -1;
    float candidate_path_sum = path_sum_threshold;

    for (int j = 0; j < window_size; ++j) {
      int freq = i * window_size + j;
      if (freq >= num_channels) {
        break;
      }
      if (cpu_top_path_sums[freq] > candidate_path_sum) {
        // This is the new best candidate of the window
        candidate_freq = freq;
        candidate_path_sum = cpu_top_path_sums[freq];
      }
    }
    if (candidate_freq < 0) {
      continue;
    }

    // Check every frequency closer than window_size if we have a candidate
    int window_end = min(num_channels, candidate_freq + window_size);
    bool found_larger_path_sum = false;
    for (int freq = max(0, candidate_freq - window_size + 1); freq < window_end; ++freq) {
      if (cpu_top_path_sums[freq] > candidate_path_sum) {
        found_larger_path_sum = true;
        break;
      }
    }
    if (!found_larger_path_sum) {
      // The candidate frequency is the best within its window
      int drift_bins = cpu_top_drift_blocks[candidate_freq] * drift_timesteps +
        cpu_top_path_offsets[candidate_freq];
      double drift_rate = drift_bins * drift_rate_resolution;
      float snr = (candidate_path_sum - median) / std_dev;

      if (abs(drift_rate) >= min_drift) {
        DedopplerHit hit(metadata, candidate_freq, drift_bins, drift_rate,
                         snr, beam, coarse_channel, num_timesteps, candidate_path_sum);
        if (print_hits) {
          cout << "hit: " << hit.toString() << endl;
        }
        output->push_back(hit);
      }
    }
  }
}
