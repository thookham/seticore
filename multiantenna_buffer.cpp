#include <assert.h>
#include "multiantenna_buffer.h"
#include "util.h"
#ifdef SETICORE_CUDA
#include "cuda_util.h"
#include <cuda_runtime.h>
#endif
#include <cstring>
#include <algorithm>

using namespace std;

MultiantennaBuffer::MultiantennaBuffer(int num_timesteps, int num_channels,
                                       int num_polarizations, int num_antennas,
                                       ComputeBackend* backend)
  : ComplexBuffer(num_timesteps * num_channels * num_polarizations * num_antennas, backend),
    num_timesteps(num_timesteps), num_channels(num_channels),
    num_polarizations(num_polarizations), num_antennas(num_antennas) {
}

#ifdef SETICORE_CUDA
thrust::complex<float> MultiantennaBuffer::get(int time, int channel,
                                               int polarization, int antenna) const {
#else
complex<float> MultiantennaBuffer::get(int time, int channel,
                                       int polarization, int antenna) const {
#endif
  assert(0 <= time && time < num_timesteps);
  assert(0 <= channel && channel < num_channels);
  assert(0 <= polarization && polarization < num_polarizations);
  assert(0 <= antenna && antenna < num_antennas);
  int index = index4d(time, channel, num_channels, polarization, num_polarizations,
                      antenna, num_antennas);
  return get(index);
}
                                       
void MultiantennaBuffer::copyRange(int src_start_channel,
                                   MultiantennaBuffer& dest, int dest_start_time) const {
  assert(src_start_channel >= 0);
  assert(src_start_channel < num_channels);
  assert(src_start_channel + dest.num_channels <= num_channels);
  assert(dest_start_time >= 0);
  assert(dest_start_time + num_timesteps <= dest.num_timesteps);
  assert(num_polarizations == dest.num_polarizations);
  assert(num_antennas == dest.num_antennas);

  int src_index = index4d(0, src_start_channel, num_channels,
                             0, num_polarizations, 0, num_antennas);
  int dest_index = index4d(dest_start_time, 0, dest.num_channels,
                           0, num_polarizations, 0, num_antennas);

#ifdef SETICORE_CUDA
  size_t entry_size = sizeof(thrust::complex<float>) * num_polarizations * num_antennas;
#else
  size_t entry_size = sizeof(complex<float>) * num_polarizations * num_antennas;
#endif

  size_t src_pitch = entry_size * num_channels;
  size_t dest_pitch = entry_size * dest.num_channels;

  auto src_ptr = data + src_index;
  auto dest_ptr = dest.data + dest_index;
  
#ifdef SETICORE_CUDA
  cudaMemcpy2DAsync(dest_ptr, dest_pitch,
                    src_ptr, src_pitch,
                    dest_pitch, num_timesteps,
                    cudaMemcpyDefault);
  checkCuda("MultiantennaBuffer copyRange");
#else
  // CPU Implementation: 2D Copy
  // Each row is dest_pitch bytes wide (contiguous block we want to write)
  // But wait, dest_pitch is the stride of destination logic?
  // cudaMemcpy2D copies 'width' bytes from src to dst.
  // width = dest_pitch here?
  // The call was: (dest_ptr, dest_pitch, src_ptr, src_pitch, width=dest_pitch, height=num_timesteps).
  // So we copy 'dest_pitch' bytes per row.
  
  for (int i = 0; i < num_timesteps; ++i) {
      void* d = (char*)dest_ptr + i * dest_pitch;
      const void* s = (char*)src_ptr + i * src_pitch;
      memcpy(d, s, dest_pitch);
  }
#endif
}
