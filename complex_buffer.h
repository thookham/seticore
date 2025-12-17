#pragma once

#ifdef SETICORE_CUDA
#include <thrust/complex.h>
#else
#include <complex>
#endif

using namespace std;

/*
  A buffer that holds complex numbers on the GPU. This is generic so that it
  can be reused by multiple stages of the pipeline.
 */
#include "src/backend/ComputeBackend.h"

class ComplexBuffer {
 public:
  ComplexBuffer(size_t size, ComputeBackend* backend);
  virtual ~ComplexBuffer();

  // No copying
  ComplexBuffer(const ComplexBuffer&) = delete;
  ComplexBuffer& operator=(ComplexBuffer&) = delete;
  
  // The number of complex entries
  const size_t size;

  // The number of bytes allocated
  const size_t bytes;

  ComputeBackend* backend;

  // The data itself
#ifdef SETICORE_CUDA
  thrust::complex<float>* data;
  thrust::complex<float> get(int index) const;
#else
  complex<float>* data;
  complex<float> get(int index) const;
#endif
};
