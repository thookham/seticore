#include "multibeam_buffer.h"

#include <assert.h>
#include "util.h"
#ifdef SETICORE_CUDA
#include "cuda_util.h"
#include <cuda_runtime.h>
#endif
#include "filterbank_buffer.h"
#include <iostream>
#include <cstring>

using namespace std;

MultibeamBuffer::MultibeamBuffer(int num_beams, int num_timesteps, int num_channels,
                                 int num_write_timesteps, ComputeBackend* backend)
  : num_beams(num_beams), num_timesteps(num_timesteps), num_channels(num_channels),
    num_write_timesteps(num_write_timesteps), backend(backend) {
  assert(num_write_timesteps <= num_timesteps);
  size_t bytes = sizeof(float) * size();
#ifdef SETICORE_CUDA
  cudaMallocManaged(&data, bytes);
  if (bytes > 2000000) {
    cout << "multibeam buffer memory: " << prettyBytes(bytes) << endl;
  }
  checkCudaMalloc("MultibeamBuffer", bytes);

  cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking);
  checkCuda("MultibeamBuffer stream init");
#else
  backend->allocateManaged((void**)&data, bytes);
#endif
}

MultibeamBuffer::MultibeamBuffer(int num_beams, int num_timesteps, int num_channels, ComputeBackend* backend)
  : MultibeamBuffer(num_beams, num_timesteps, num_channels, num_timesteps, backend) {}

MultibeamBuffer::~MultibeamBuffer() {
#ifdef SETICORE_CUDA
  cudaFree(data);
#else
  backend->freeDevice(data);
#endif
}

long MultibeamBuffer::size() const {
  return num_beams * num_timesteps * num_channels;
}

FilterbankBuffer MultibeamBuffer::getBeam(int beam) {
  assert(0 <= beam && beam < num_beams);
  int beam_size = num_timesteps * num_channels;
  return FilterbankBuffer(num_timesteps, num_channels, data + beam * beam_size, backend);
}

void MultibeamBuffer::set(int beam, int time, int channel, float value) {
  int index = index3d(beam, time, num_timesteps, channel, num_channels);
  data[index] = value;
}

float MultibeamBuffer::get(int beam, int time, int channel) {
#ifdef SETICORE_CUDA
  cudaDeviceSynchronize();
  checkCuda("MultibeamBuffer get");
#endif
  assert(beam < num_beams);
  assert(time < num_timesteps);
  assert(channel < num_channels);
  int index = index3d(beam, time, num_timesteps, channel, num_channels);
  return data[index];
}

void MultibeamBuffer::zeroAsync() {
  size_t size = sizeof(float) * num_beams * num_timesteps * num_channels;
#ifdef SETICORE_CUDA
  cudaMemsetAsync(data, 0, size);
  checkCuda("MultibeamBuffer zeroAsync");
#else
  memset(data, 0, size);
#endif
}

void MultibeamBuffer::copyRegionAsync(int beam, int channel_offset,
                                      FilterbankBuffer* output) {
  float* region_start = data + (beam * num_timesteps * num_channels) + channel_offset;
  size_t source_pitch = sizeof(float) * num_channels;
  size_t width = sizeof(float) * output->num_channels;
  size_t dest_pitch = width;
  
#ifdef SETICORE_CUDA
  cudaMemcpy2DAsync(output->data, dest_pitch,
                    (void*) region_start, source_pitch,
                    width, num_timesteps,
                    cudaMemcpyDefault);
  checkCuda("MultibeamBuffer copyRegionAsync");
#else
  // CPU 2D copy
  for (int i = 0; i < num_timesteps; ++i) {
      void* d = (char*)output->data + i * dest_pitch;
      const void* s = (char*)region_start + i * source_pitch;
      memcpy(d, s, width);
  }
#endif
}

void MultibeamBuffer::hintWritingTime(int time) {
  prefetchStripes(0, 0, time - num_write_timesteps, time + 2 * num_write_timesteps);
}

void MultibeamBuffer::hintReadingBeam(int beam) {
  prefetchStripes(beam - 1, beam + 1, 0, num_write_timesteps);
}

// Truncates [first_time, last_time]
void MultibeamBuffer::prefetchRange(int beam, int first_time, int last_time,
                                    int destination_device) {
#ifdef SETICORE_CUDA
  if (first_time < 0) {
    first_time = 0;
  }
  if (last_time >= num_timesteps) {
    last_time = num_timesteps - 1;
  }
  if (first_time > last_time) {
    return;
  }

  long start_index = index3d(beam, first_time, num_timesteps, 0, num_channels);
  size_t prefetch_size = sizeof(float) * (last_time - first_time + 1) * num_channels;  
  cudaMemPrefetchAsync(data + start_index, prefetch_size, destination_device,
                       prefetch_stream);
  checkCuda("MultibeamBuffer prefetchRange");
#endif
}

void MultibeamBuffer::prefetchStripes(int first_beam, int last_beam,
                                      int first_time, int last_time) {
#ifdef SETICORE_CUDA
  // ... existing implementation ...
  if (first_beam < 0) {
    first_beam = 0;
  }
  if (last_beam >= num_beams) {
    last_beam = num_beams - 1;
  }
  if (first_time < 0) {
    first_time = 0;
  }
  if (last_time >= num_timesteps) {
    last_time = num_timesteps - 1;
  }
  
  const int gpu_id = 0;
  for (int beam = 0; beam < num_beams; ++beam) {
    if (first_beam <= beam && beam <= last_beam) {
      // Prefetch this entire beam to the GPU
      prefetchRange(beam, 0, num_timesteps - 1, gpu_id);
      continue;
    }

    // Prefetch the desired time range to the GPU
    prefetchRange(beam, first_time, last_time, gpu_id);

    // I think this could be shrunk to the CPU page size
    int margin = num_write_timesteps;
    
    // Prefetch earlier times to the CPU, with a margin
    prefetchRange(beam, margin, first_time - margin, cudaCpuDeviceId);

    // Prefetch later times to the CPU, with a margin
    prefetchRange(beam, last_time + margin, num_timesteps - margin, cudaCpuDeviceId);
  }
#endif
  // On CPU, no-op or madvise? No-op is fine.
}
