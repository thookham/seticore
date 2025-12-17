#include <assert.h>
#include "complex_buffer.h"
#include "util.h"
#ifdef SETICORE_CUDA
#include "cuda_util.h"
#include <cuda_runtime.h>
#endif

using namespace std;

ComplexBuffer::ComplexBuffer(size_t size, ComputeBackend* backend)
  : size(size), bytes(size * sizeof(data[0])), backend(backend) {
  backend->allocateManaged((void**)&data, bytes);
}

ComplexBuffer::~ComplexBuffer() {
  backend->freeManaged(data);
}

#ifdef SETICORE_CUDA
thrust::complex<float> ComplexBuffer::get(int index) const {
#else
complex<float> ComplexBuffer::get(int index) const {
#endif
  assert(index >= 0);
  assert((size_t)index < size);
#ifdef SETICORE_CUDA
  cudaDeviceSynchronize();
#endif
  return data[index];
}
