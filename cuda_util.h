#pragma once

#ifdef SETICORE_CUDA
#include <cuda.h>
#endif
#include <iostream>

using namespace std;

static_assert(sizeof(float) == 4, "require 32-bit floats");

const int CUDA_MAX_THREADS = 1024;

// Helpers to nicely display cuda errors
void checkCuda(const string& tag);
void checkCudaMalloc(const string& tag, size_t size);

#ifdef SETICORE_CUDA
// Helper to check errors and clean up
class Stream {
public:
  cudaStream_t stream;
  Stream();
  ~Stream();
};
#endif

// Index helpers moved to util.h


