#pragma once

#ifdef SETICORE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef SETICORE_CUDA
#include <cuda_runtime.h>
#include <thrust/complex.h>
#else
#include <complex>
#endif
#include <string>

using namespace std;

const string VERSION = "1.0.6";

// This is allegedly a SIGPROC standard but the most authoritative source
// I can find is:
//   https://github.com/UCBerkeleySETI/blimpy/blob/master/blimpy/ephemeris/observatory_info.csv
const int NO_TELESCOPE_ID = -1;
const int PARKES = 4;
const int GREEN_BANK = 6;
const int ATA = 9;
const int VLA = 12;
const int MEERKAT = 64;

int roundUpToPowerOfTwo(int n);
bool isPowerOfTwo(int n);
int numDigits(int n);
string zeroPad(int n, int size);
#ifdef SETICORE_CUDA
string cToS(thrust::complex<float> c);
void assertComplexEq(thrust::complex<float> c, float real, float imag);
#else
string cToS(std::complex<float> c);
void assertComplexEq(std::complex<float> c, float real, float imag);
#endif
string stripAnyTrailingSlash(const string& s);

#ifdef SETICORE_CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

static_assert(sizeof(float) == 4, "require 32-bit floats");
void assertFloatEq(float a, float b);
void assertFloatEq(float a, float b, const string& tag);
void assertStringEq(const string& lhs, const string& rhs);
void assertStringEq(const string& lhs, const string& rhs, const string& tag);
string pluralize(int n, const string& noun);
string prettyBytes(size_t n);
long timeInMS();
double unixTimeToMJD(double unix_time);
double hoursToRadians(double hours);
double radiansToHours(double radians);
double degreesToRadians(double degrees);
double radiansToDegrees(double radians);
void logErrorTimestamp();
void logError(const string& message);
void fatal(const string& message);
void fatal(const string& message1, const string& message2);
// Helper to calculate a 2d row-major index, ie for:
//   arr[a][b]
HOST_DEVICE inline long index2d(long a, long b, long b_end) {
  return a * b_end + b;
}

// Helper to calculate a 3d row-major index, ie for:
//   arr[a][b][c]
HOST_DEVICE inline long index3d(long a, long b, long b_end, long c, long c_end) {
  return index2d(a, b, b_end) * c_end + c;
}

// Helper to calculate a 4d row-major index, ie for:
//   arr[a][b][c][d]
HOST_DEVICE inline long index4d(long a, long b, long b_end, long c, long c_end, long d, long d_end) {
  return index3d(a, b, b_end, c, c_end) * d_end + d;
}

// Helper to calculate a 5d row-major index, ie for:
//   arr[a][b][c][d][e]
HOST_DEVICE inline long index5d(long a, long b, long b_end, long c, long c_end, long d, long d_end,
                                long e, long e_end) {
  return index4d(a, b, b_end, c, c_end, d, d_end) * e_end + e;
}

int telescopeID(const string& telescope);
