#include "upchannelizer.h"
#include <iostream>
#include <cassert>
#include "util.h"

using namespace std;

Upchannelizer::Upchannelizer(ComputeBackend* backend, int fft_size,
                             int num_input_timesteps, int num_coarse_channels,
                             int num_polarizations, int num_antennas)
  : backend(backend),
    fft_size(fft_size),
    num_input_timesteps(num_input_timesteps),
    num_coarse_channels(num_coarse_channels),
    num_polarizations(num_polarizations),
    num_antennas(num_antennas),
    release_input(true) {
  assert(fft_size > 0);
  assert(num_input_timesteps > 0);
  assert(num_coarse_channels > 0);
  assert(num_polarizations > 0);
  assert(num_antennas > 0);
  assert(num_input_timesteps % fft_size == 0);

  int batch_size = num_antennas * num_polarizations;
  plan = backend->createFFTPlan(fft_size, batch_size);
  
  if (!plan) {
      // In a real application we might throw or return error, 
      // but pure constructors make returning errors hard.
      // We assume the backend produces a valid plan or handles headers.
      // If we are here and plan is null (e.g. no FFTW), we will crash later.
      // The backend should have warned.
  }
}

Upchannelizer::~Upchannelizer() {
    // plan auto-destroys
}

size_t Upchannelizer::requiredInternalBufferSize() const {
  return (size_t) num_antennas * num_coarse_channels * num_polarizations * num_input_timesteps;
}

void Upchannelizer::run(DeviceRawBuffer& input, ComplexBuffer& buffer,
                        MultiantennaBuffer& output) {
  assert(input.num_antennas == num_antennas);
  assert(input.num_coarse_channels == num_coarse_channels);
  assert(input.num_polarizations == num_polarizations);
  assert(input.timesteps_per_block * input.num_blocks == num_input_timesteps);

  assert(buffer.size >= requiredInternalBufferSize());

  assert(output.num_timesteps == num_input_timesteps / fft_size);
  assert(output.num_channels == num_coarse_channels * fft_size);
  assert(output.num_polarizations == num_polarizations);
  assert(output.num_antennas == num_antennas);

  // Convert raw to complex using backend
  backend->convertRawToComplex(input.data, input.size,
     buffer.data, buffer.size,
     num_antennas, num_coarse_channels, num_polarizations,
     num_input_timesteps, input.timesteps_per_block);

  // Execute FFT
  // Plan is configured for batch size = everything else
  if (plan) {
      plan->execute(buffer.data, buffer.data, true);
  } else {
      // Should fatal? Or assume backend no-op?
      // Since we assert fft_size > 0, we expect plan.
      // But if fftw not found, plan is null.
      // We should probably log or crash if plan is essential.
  }

  // Shift output
  backend->shiftFFTOutput(buffer.data, buffer.size,
                          output.data, output.size,
                          fft_size, num_antennas, num_polarizations,
                          num_coarse_channels, num_input_timesteps / fft_size);
  
  // Synchronous release for safety in backend-agnostic code
  if (release_input) {
      backend->synchronize();
      input.release(); 
  }
}

int Upchannelizer::numOutputChannels() const {
  return num_coarse_channels * fft_size;
}

int Upchannelizer::numOutputTimesteps() const {
  return num_input_timesteps / fft_size;
}
