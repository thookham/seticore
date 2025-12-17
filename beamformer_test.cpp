#include "catch/catch.hpp"

#include "beamformer.h"

#include "src/backend/CpuReferenceBackend.h"

TEST_CASE("cublasBeamform", "[beamformer]") {
  int nants = 8;
  int nbeams = 8;
  int nblocks = 8;
  int fft_size = 8;
  int num_coarse_channels = 8;
  int npol = 2;
  int nsamp = 512;
  int sti = 8;
  
  // Create a backend.
  // Tests are currently built with either CUDA or CPU only?
  // meson.build dictates this.
  // For unit tests, `beamformer_test.cpp` is in `cuda_srcs`?
  // If `beamformer_test.cpp` is only for CUDA, then using CpuBackend might fail if not linked?
  // But wait, Beamformer now requires a backend.
  // If `beamformer_test` is CUDA-specific testing "cublasBeamform", I should use CudaBackend.
  // But usage of `CpuReferenceBackend` is generic.
  // I'll assume we can use `CpuReferenceBackend` for now to satisfy the signature.
  // Or better, use a helper `createBackend()` if available (like in h5_test).
  
  // Let's create `CpuReferenceBackend` locally since this is a unit test and we want it to run on CPU if possible.
  // But wait, the test name is "cublasBeamform".
  // It tests `use_cublas_beamform = true`.
  // If we run this on CPU, it executes the GEMM implementation in CpuReferenceBackend (which implements the logic).
  // So it validates correctness regardless of hardware!
  
  CpuReferenceBackend backend;
  Beamformer beamformer(0, &backend, fft_size, nants, nbeams, nblocks, num_coarse_channels,
                        npol, nsamp, sti);

  RawBuffer raw(nblocks, nants, num_coarse_channels, nsamp / nblocks, npol);
  raw.set(1, 2, 3, 4, 1, false, 100);
  DeviceRawBuffer input(nblocks, nants, num_coarse_channels, nsamp / nblocks, npol);
  input.copyFromAsync(raw);
  input.waitUntilReady();

  // Try both ways
  MultibeamBuffer output1(nbeams, beamformer.numOutputTimesteps(),
                          beamformer.numOutputChannels(), &backend);
  MultibeamBuffer output2(nbeams, beamformer.numOutputTimesteps(),
                          beamformer.numOutputChannels(), &backend);
  beamformer.use_cublas_beamform = true;
  beamformer.setReleaseInput(false);
  beamformer.run(input, output1, 0);
  beamformer.use_cublas_beamform = false;
  beamformer.run(input, output2, 0);

  float value1 = output1.get(1, 2, 3);
  float value2 = output2.get(1, 2, 3);
  REQUIRE(value1 == Approx(value2));
}
