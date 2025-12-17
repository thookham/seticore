#pragma once

#ifdef SETICORE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "filterbank_buffer.h"
#include "src/backend/ComputeBackend.h"

using namespace std;

/*
  The MultibeamBuffer stores the contents of a filterbank file in unified memory,
  for multiple beams. This can be a single coarse channel, or a number of channels.
*/
class MultibeamBuffer {
 public:
  const long num_beams;
  const long num_timesteps;
  const long num_channels;

  // The number of timesteps that will be written in a batch.
  // Used only to optimize managed memory prefetching, so it can be a guess.
  const long num_write_timesteps;
  
  /*
    Row-major indexed by:
      data[beam][time][freq]
   */
  float* data;
  ComputeBackend* backend; // Added member

  // Create a managed buffer
  MultibeamBuffer(int num_beams, int num_timesteps, int num_channels,
                  int num_write_timesteps, ComputeBackend* backend);
  MultibeamBuffer(int num_beams, int num_timesteps, int num_channels, ComputeBackend* backend);  

  ~MultibeamBuffer();

  long size() const;
  
  FilterbankBuffer getBeam(int beam);

  void set(int beam, int time, int channel, float value);
  
  float get(int beam, int time, int channel);

  // Zero out all the data as an asynchronous GPU operation
  void zeroAsync();

  // Asynchronously copy out some data to a separate buffer.
  // Uses default cuda stream.
  void copyRegionAsync(int beam, int channel_offset, FilterbankBuffer* output);

  // Call this when you are writing this time
  void hintWritingTime(int time);

  // Call this when you are reading this beam
  void hintReadingBeam(int beam);

private:
  // This stream is just for prefetching.
#ifdef SETICORE_CUDA
  cudaStream_t prefetch_stream;
#endif

  void prefetchRange(int beam, int first_time, int last_time, int destinationDevice);
  
  void prefetchStripes(int first_beam, int last_beam, int first_time, int last_time);
};
