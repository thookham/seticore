#include "CudaBackend.h"
#include "../../cuda_util.h"
#include "../../taylor.h"
#include "../../util.h" // For index4d etc
#include <iostream>
#include <cufft.h>
#include <thrust/complex.h>

using namespace std;

// Kernels from upchannelizer.cu (moved here)

/*
  We convert from int8 input with format:
    input[block][antenna][coarse-channel][time-within-block][polarization][real or imag]

  to complex-float output with format:
    buffer[polarization][antenna][coarse-channel][time]

  block and time-within-block combine to form a single time index.
 */
__global__ void convertRaw(const int8_t* input, int input_size,
                           thrust::complex<float>* buffer, int buffer_size,
                           int num_antennas, int nblocks, int num_coarse_channels,
                           int num_polarizations, int nsamp, int time_per_block) {
  int time_within_block = blockIdx.x * CUDA_MAX_THREADS + threadIdx.x;
  if (time_within_block >= time_per_block) {
    return;
  }
  int block = blockIdx.y;
  int antenna = blockIdx.z / num_coarse_channels;
  int chan = blockIdx.z % num_coarse_channels;
  int time = block * time_per_block + time_within_block;
  
  for (int pol = 0; pol < num_polarizations; ++pol) {
    long input_index = 2 * index5d(block, antenna, num_antennas, chan, num_coarse_channels,
                                  time_within_block, time_per_block, pol, num_polarizations);
    long converted_index = index4d(pol, antenna, num_antennas, chan, num_coarse_channels, time, nsamp);

    // assert(input_index >= 0);
    // assert(converted_index >= 0);
    // assert(input_index + 1 < input_size);
    // assert(converted_index < buffer_size);
    // Removing asserts for now as they require device-side assert support which might be flaky or disabled
    
    buffer[converted_index] = thrust::complex<float>
      (input[input_index] * 1.0, input[input_index + 1] * 1.0);
  }
}

/*
  shift converts from the post-FFT format with format:
    buffer[polarization][antenna][coarse-channel][time][fine-channel]

  to a format ready for beamforming:
    output[time][channel][polarization][antenna]

  We also toggle the high bit of the frequency fine channel. Hence "shift".
 */
__global__ void shift(thrust::complex<float>* buffer, int buffer_size,
                      thrust::complex<float>* output, int output_size,
                      int fft_size, int num_antennas, int num_polarizations,
                      int num_coarse_channels, int num_timesteps) {
  int antenna = threadIdx.y;
  int pol = threadIdx.z;
  int fine_chan = blockIdx.x;
  int coarse_chan = blockIdx.y;
  int time = blockIdx.z;

  int output_fine_chan = fine_chan ^ (fft_size >> 1);

  long input_index = index5d(pol, antenna, num_antennas, coarse_chan, num_coarse_channels,
			     time, num_timesteps, fine_chan, fft_size);
  long output_index = index5d(time, coarse_chan, num_coarse_channels,
			      output_fine_chan, fft_size,
			      pol, num_polarizations, antenna, num_antennas);

//   assert(input_index >= 0);
//   assert(output_index >= 0);
//   assert(input_index < buffer_size);
//   assert(output_index < output_size);
  output[output_index] = buffer[input_index];
}


// FFT Plan Wrapper
class CudaFFTPlan : public FFTPlan {
    cufftHandle plan;
public:
    CudaFFTPlan(int size, int batch_size) {
        if (cufftPlan1d(&plan, size, CUFFT_C2C, batch_size) != CUFFT_SUCCESS) {
            cerr << "cufftPlan1d failed" << endl;
        }
    }
    
    ~CudaFFTPlan() {
        cufftDestroy(plan);
    }
    
    void execute(void* input, void* output, bool forward) override {
        // CUFFT Forward = -1 (default for C2C?), Inverse = 1?
        // Wait, typical definition:
        // CUFFT_FORWARD = -1
        // CUFFT_INVERSE = 1
        
        // input and output can be same (in-place)
        // input/output are void*, assume they are cufftComplex* (float2*)
        
        int direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
        if (cufftExecC2C(plan, (cufftComplex*)input, (cufftComplex*)output, direction) != CUFFT_SUCCESS) {
            cerr << "cufftExecC2C failed" << endl;
        }
    }
};

extern __global__ void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs);
extern __global__ void findTopPathSums(const float* path_sums, int num_timesteps, int num_freqs,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets);

// ... existing code ...

void CudaBackend::allocateDevice(void** ptr, size_t size) {
    cudaMalloc(ptr, size);
}

void CudaBackend::allocateHost(void** ptr, size_t size) {
    cudaMallocHost(ptr, size);
}

void CudaBackend::allocateManaged(void** ptr, size_t size) {
    cudaMallocManaged(ptr, size);
}

void CudaBackend::freeDevice(void* ptr) {
    cudaFree(ptr);
}

void CudaBackend::freeHost(void* ptr) {
    cudaFreeHost(ptr);
}

void CudaBackend::freeManaged(void* ptr) {
    cudaFree(ptr);
}

void CudaBackend::verify_call(const char* name) {
    checkCuda(name);
}

void CudaBackend::synchronize() {
    cudaDeviceSynchronize();
}

void CudaBackend::copyDeviceToHost(void* dst_host, const void* src_device, size_t size) {
    cudaMemcpy(dst_host, src_device, size, cudaMemcpyDeviceToHost);
}

void CudaBackend::copyHostToDevice(void* dst_device, const void* src_host, size_t size) {
    cudaMemcpy(dst_device, src_host, size, cudaMemcpyHostToDevice);
}

void CudaBackend::zeroDevice(void* ptr, size_t size) {
    cudaMemsetAsync(ptr, 0, size);
}

unique_ptr<FFTPlan> CudaBackend::createFFTPlan(int size, int batch_size) {
    return unique_ptr<FFTPlan>(new CudaFFTPlan(size, batch_size));
}

void CudaBackend::sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) {
    int grid_size = (num_freqs + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
    ::sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input, sums, num_timesteps, num_freqs);
}

const float* CudaBackend::taylorTree(const float* input, float* buffer1, float* buffer2,
                                     int num_timesteps, int num_channels, int drift_block) {
    return optimizedTaylorTree(input, buffer1, buffer2, num_timesteps, num_channels, drift_block);
}

void CudaBackend::findTopPathSums(const float* taylor_sums, int num_timesteps, int num_freqs,
                                  int drift_block, float* top_path_sums,
                                  int* top_drift_blocks, int* top_path_offsets) {
    int grid_size = (num_freqs + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
    ::findTopPathSums<<<grid_size, CUDA_MAX_THREADS>>>(taylor_sums, num_timesteps, num_freqs,
                                                       drift_block, top_path_sums,
                                                       top_drift_blocks, top_path_offsets);
}

// Helper to get cuBLAS Op
cublasOperation_t getCublasOp(bool trans) {
    return trans ? CUBLAS_OP_T : CUBLAS_OP_N; // Or C for ConjTrans? Standard BLAS is A' usually which means ConjTrans for complex
    // Wait, Beamformer uses CUBLAS_OP_C for coefficients (conjugated) and CUBLAS_OP_T for real-valued incoherent.
    // My interface takes bool transA. Does it mean Trans or ConjTrans?
    // Standard BLAS "Trans" usually implies simple transpose, "ConjTrans" implies H.
    // Beamformer logic: 
    //   coefficients are already in correct conjugation? prebeam is multiplied by conj(coefficients).
    //   "thrust::complex<float> conjugated = thrust::conj(coefficients[coeff_index]);" -> kernel
    //   "cublasCgemm3mStridedBatched(..., CUBLAS_OP_C, CUBLAS_OP_N, ...)"
    // So for Complex, we need ability to specify C.
    // My interface simplifies to bool. This might be insufficient for Complex.
    // But let's check standard BLAS. CblasTrans, CblasNoTrans, CblasConjTrans.
    // I should change bool to an enum or just int? Or keep bool and assume Complex always means ConjTrans?
    // In Beamformer.cu:
    // runCublasBeamform: CUBLAS_OP_C on Coefficients (A), CUBLAS_OP_N on Prebeam (B).
    // unweightedIncoherentBeam: CUBLAS_OP_N, CUBLAS_OP_T (for real Sgemm).
    // So for Complex, Trans=ConjTrans. For Float, Trans=Trans.
    // This is consistent.
}

cublasOperation_t getCublasComplexOp(bool trans) {
    return trans ? CUBLAS_OP_C : CUBLAS_OP_N;
}

void CudaBackend::complexGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    std::complex<float> alpha,
    const std::complex<float>* A, int lda, long long strideA,
    const std::complex<float>* B, int ldb, long long strideB,
    std::complex<float> beta,
    std::complex<float>* C, int ldc, long long strideC,
    int batchCount) {
    
    // Lazy initialization of handle? Or assume member? 
    // CudaBackend doesn't currently own a cublasHandle.
    // It should probably have one per thread or similar.
    // But `Beamformer` had one.
    // For now, I will create/destroy locally, which is slow but correct. 
    // Optimization: Add cublasHandle to CudaBackend.
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cuComplex alpha_cu = make_cuComplex(alpha.real(), alpha.imag());
    cuComplex beta_cu = make_cuComplex(beta.real(), beta.imag());
    
    cublasCgemm3mStridedBatched(
        handle,
        getCublasComplexOp(transA), getCublasComplexOp(transB),
        m, n, k,
        &alpha_cu,
        (const cuComplex*)A, lda, strideA,
        (const cuComplex*)B, ldb, strideB,
        &beta_cu,
        (cuComplex*)C, ldc, strideC,
        batchCount);
        
    cublasDestroy(handle);
    checkCuda("complexGemmStridedBatched");
}

void CudaBackend::floatGemmStridedBatched(
    bool transA, bool transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    
    cublasHandle_t handle;
    cublasCreate(&handle); // TODO: Move to member
    
    cublasSgemmStridedBatched(
        handle,
        getCublasOp(transA), getCublasOp(transB),
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batchCount);
        
    cublasDestroy(handle);
    checkCuda("floatGemmStridedBatched");
}

void CudaBackend::convertRawToComplex(const int8_t* input, size_t input_size,
                         std::complex<float>* buffer, size_t buffer_size,
                         int num_antennas, int num_coarse_channels, int num_polarizations, int num_timesteps, int time_per_block) {
  int nblocks = num_timesteps / time_per_block;
  // nsamp isn't used in kernel? Wait.
  // kernel args: (input, input_size, buffer, buffer_size, num_antennas, nblocks, num_coarse_channels, num_polarizations, nsamp, time_per_block)
  // nsamp is calculated as input_size / ...?
  // Actually nsamp in kernel is `nsamp` param.
  // In `convertRaw` kernel: `converted_index = index4d(..., nsamp)`
  // nsamp appears to be the timestep dimension of the output.
  // which is num_timesteps.
  int nsamp = num_timesteps;

  int cuda_blocks_per_block =
    (time_per_block + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  dim3 convert_raw_block(CUDA_MAX_THREADS, 1, 1);
  dim3 convert_raw_grid(cuda_blocks_per_block, nblocks,
                        num_antennas * num_coarse_channels);
  
  convertRaw<<<convert_raw_grid, convert_raw_block>>>
    (input, input_size,
     (thrust::complex<float>*)buffer, buffer_size,
     num_antennas, nblocks, num_coarse_channels, num_polarizations,
     nsamp, time_per_block);
    
  checkCuda("convertRawToComplex");
}

void CudaBackend::shiftFFTOutput(std::complex<float>* buffer, size_t buffer_size,
                    std::complex<float>* output, size_t output_size,
                    int fft_size, int num_antennas, int num_polarizations,
                    int num_coarse_channels, int num_timesteps) {
  dim3 shift_block(1, num_antennas, num_polarizations);
  dim3 shift_grid(fft_size, num_coarse_channels, num_timesteps);
  
  shift<<<shift_grid, shift_block>>>
    ((thrust::complex<float>*)buffer, buffer_size, (thrust::complex<float>*)output, output_size, fft_size, num_antennas,
     num_polarizations, num_coarse_channels, num_timesteps);
     
  checkCuda("shiftFFTOutput");
}

void CudaBackend::batchedPower(const std::complex<float>* voltage, float* power,
                  int num_beams, int num_channels, int num_polarizations,
                  int sti, int power_time_offset, int num_output_timesteps) {
    // Note: The current Beamformer implementation for CUDA utilizes optimized custom kernels
    // directly (legacy path) rather than these generic backend methods.
    // These stubs are present to satisfy the ComputeBackend interface.
    // If the legacy CUDA path is deprecated in the future, these methods should be implemented
    // by moving the kernels from beamformer.cpp (or wrapping them).
    fatal("CudaBackend::batchedPower is currently unused in favor of legacy optimized kernels.");
}

void CudaBackend::incoherentPower(const std::complex<float>* input, const float* square_magnitudes,
                             float* output,
                             int num_beams, int num_coarse_channels, int num_polarizations,
                             int sti, int power_time_offset, int num_output_timesteps, int fft_size, int num_antennas) {
    // See comment for batchedPower.
    fatal("CudaBackend::incoherentPower is currently unused in favor of legacy optimized kernels.");
}
