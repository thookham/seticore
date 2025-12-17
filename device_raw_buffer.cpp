#include "device_raw_buffer.h"

#include <assert.h>
#include "util.h"
#ifdef SETICORE_CUDA
#include "cuda_util.h"
#endif
#include <cstring>
#include <iostream>

using namespace std;


DeviceRawBuffer::DeviceRawBuffer(int num_blocks, int num_antennas,
                                 int num_coarse_channels,
                                 int timesteps_per_block, int num_polarizations)
  : num_blocks(num_blocks), num_antennas(num_antennas),
    num_coarse_channels(num_coarse_channels),
    timesteps_per_block(timesteps_per_block), num_polarizations(num_polarizations),
    state(DeviceRawBufferState::unused) {
  size = sizeof(int8_t) * num_blocks * num_antennas * num_coarse_channels *
    timesteps_per_block * num_polarizations * 2;
#ifdef SETICORE_CUDA
  cudaMalloc(&data, size);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  checkCuda("DeviceRawBuffer init");
#else
  data = (int8_t*) malloc(size);
  if (!data && size > 0) fatal("DeviceRawBuffer malloc failed");
#endif
}

DeviceRawBuffer::~DeviceRawBuffer() {
#ifdef SETICORE_CUDA
  cudaFree(data);
  cudaStreamDestroy(stream);
#else
  if (data) free(data);
#endif
}

// Should only be called from the producer thread
void DeviceRawBuffer::copyFromAsync(const RawBuffer& other) {
  assert(size == other.size);
  waitUntilUnused();

  unique_lock<mutex> lock(m);
  assert(state == DeviceRawBufferState::unused);
  state = DeviceRawBufferState::copying;
  lock.unlock();
  // Nobody waits on copying state, so no need to notify
  
#ifdef SETICORE_CUDA
  cudaMemcpyAsync(data, other.data, size, cudaMemcpyHostToDevice, stream);
  cudaStreamAddCallback(stream, DeviceRawBuffer::staticCopyCallback, this, 0);
#else
  // CPU synchronous copy
  memcpy(data, other.data, size);
  copyCallback();
#endif
}

void DeviceRawBuffer::waitUntilReady() {
  unique_lock<mutex> lock(m);
  while (state != DeviceRawBufferState::ready) {
    cv.wait(lock);
  }
}

void DeviceRawBuffer::waitUntilUnused() {
  unique_lock<mutex> lock(m);
  while (state != DeviceRawBufferState::unused) {
    cv.wait(lock);
  }
}

void DeviceRawBuffer::release() {
  unique_lock<mutex> lock(m);
  assert(state == DeviceRawBufferState::ready);
  state = DeviceRawBufferState::unused;
  lock.unlock();
  cv.notify_all();
}

#ifdef SETICORE_CUDA
void CUDART_CB DeviceRawBuffer::staticCopyCallback(cudaStream_t stream,
                                                   cudaError_t status,
                                                   void *device_raw_buffer) {
  assert(status == cudaSuccess);
  DeviceRawBuffer* object = (DeviceRawBuffer*) device_raw_buffer;
  object->copyCallback();
}

void CUDART_CB DeviceRawBuffer::staticRelease(cudaStream_t stream,
                                              cudaError_t status,
                                              void *device_raw_buffer) {
  if (status != cudaSuccess) {
    logErrorTimestamp();
    cerr << "DeviceRawBuffer::staticRelease: " << cudaGetErrorString(status) << endl;
    exit(2);
  }
  DeviceRawBuffer* object = (DeviceRawBuffer*) device_raw_buffer;
  object->release();
}
#endif

void DeviceRawBuffer::copyCallback() {
  // Advance state to "ready"
  unique_lock<mutex> lock(m);
  assert(state == DeviceRawBufferState::copying);
  state = DeviceRawBufferState::ready;
  lock.unlock();
  cv.notify_all();
}
