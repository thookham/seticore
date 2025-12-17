#include "filterbank_buffer.h"

#include <assert.h>
#include <fmt/core.h>
#include <iostream>
#include <cstring> // for memset

// #include "cuda_util.h" // Removed
#include "util.h"

using namespace std;

// Creates a buffer that owns its own memory.
FilterbankBuffer::FilterbankBuffer(int num_timesteps, int num_channels, ComputeBackend* backend)
  : num_timesteps(num_timesteps), num_channels(num_channels), managed(true),
    size(num_timesteps * num_channels),
    bytes(sizeof(float) * size), backend(backend) {
  backend->allocateManaged((void**)&data, bytes);
  backend->verify_call("FilterbankBuffer malloc");
}

// Creates a buffer that is a view on memory owned by the caller.
FilterbankBuffer::FilterbankBuffer(int num_timesteps, int num_channels, float* data, ComputeBackend* backend)
  : num_timesteps(num_timesteps), num_channels(num_channels), managed(false),
    size(num_timesteps * num_channels),
    bytes(sizeof(float) * size), data(data), backend(backend) {
}

FilterbankBuffer::~FilterbankBuffer() {
  if (managed) {
    backend->freeDevice(data);
  }
}

// Set everything to zero
void FilterbankBuffer::zero() {
  // If managed, we can sometimes use memset from host if on CPU backend.
  // But if Cuda backend and UVM, we can also use host memset?
  // Or we should use backend->zeroDevice?
  // `zeroDevice` usually uses cudaMemset.
  // The original used `memset`. `cudaMallocManaged` memory is accessible on host.
  // But if we want to be safe for device-only buffers (unmanaged?), well the managed flag says it is managed.
  // If we are on CPU backend, memset is fine.
  // If we are on Cuda backend, memset is fine for managed memory (UVM).
  // So keeping memset is "safe" for UVM, but not for pure device memory.
  // Since `managed` implies UVM, let's keep memset for now as it matches original logic.
  // Wait, original logic used `memset`.
  memset(data, 0, sizeof(float) * num_timesteps * num_channels);
}

// Inefficient but useful for testing
void FilterbankBuffer::set(int time, int channel, float value) {
  assert(0 <= time && time < num_timesteps);
  assert(0 <= channel && channel < num_channels);
  int index = time * num_channels + channel;
  data[index] = value;
}

float FilterbankBuffer::get(int time, int channel) const {
  backend->synchronize();
  backend->verify_call("FilterbankBuffer get");
  int index = time * num_channels + channel;
  return data[index];
}

void FilterbankBuffer::assertEqual(const FilterbankBuffer& other, int drift_block) const {
  assert(num_timesteps == other.num_timesteps);
  assert(num_channels == other.num_channels);
  for (int drift = 0; drift < num_timesteps; ++drift) {
    for (int chan = 0; chan < num_channels; ++chan) {
      int last_chan = chan + (num_timesteps - 1) * drift_block + drift;
      if (last_chan < 0 || last_chan >= num_channels) {
        continue;
      }
      assertFloatEq(get(drift, chan), other.get(drift, chan),
                    fmt::format("data[{}][{}]", drift, chan));
    }
  }
}

// Make a filterbank buffer with a bit of deterministic noise so that
// normalization doesn't make everything infinite SNR.
FilterbankBuffer makeNoisyBuffer(int num_timesteps, int num_channels, ComputeBackend* backend) {
  FilterbankBuffer buffer(num_timesteps, num_channels, backend);
  buffer.zero();
  for (int chan = 0; chan < buffer.num_channels; ++chan) {
    buffer.set(0, chan, 0.1 * chan / buffer.num_channels);
  }
  return buffer;
}
