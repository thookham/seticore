# Multi-Architecture Strategy

## Goal

Transform `seticore` from a CUDA-only application into a hardware-agnostic search engine capable of running on NVIDIA (CUDA), AMD (HIP), Intel (SYCL), and potentially NPUs/TPUs.

## Constraint Analysis

- **Current State**: ~10 custom CUDA kernels with manual memory management.
- **Complexity**: The `tiledTaylorKernel` uses shared memory synchronization (`__syncthreads()`) which is non-trivial to port to high-level graph APIs (like core NPU APIs) but maps well to GPU-compute languages (SYCL, HIP, OpenCL, Vulkan).

## Platform Targets & Technologies

| Target | Primary Tech | Secondary Tech | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA** | **CUDA** (Current) | SYCL / Kokkos | Maximal performance. | Vendor lock-in. |
| **AMD GPU** | **HIP** | OpenCL / SYCL | Easy port from CUDA (HIPify). | Linux-centric tooling. |
| **Intel GPU** | **SYCL** (OneAPI) | OpenCL | Strong modern C++ support. | Runtime size. |
| **NPU / TPU** | **Vulkan Compute** | OpenCL / Custom | Widest reach (mobile/edge). | Verbose API, driver variance. |

## Recommended Strategy: The "Backend" Approach

We should abstract the compute layer into a **Backend Interface**.

### Phase 1: Abstraction Layer

Define a C++ interface for the core operations:

```cpp
class ComputeBackend {
public:
    virtual void allocate(size_t size) = 0;
    virtual void copyToDevice(void* dst, const void* src, size_t size) = 0;
    virtual void taylorTree(const float* input, float* output, ...) = 0;
    virtual void findTopPathSums(...) = 0;
};
```

Refactor `Dedopplerer` to use `ComputeBackend* backend` instead of direct CUDA calls.

### Phase 2: SYCL Implementation (Intel/AMD/NVIDIA)

Implement `SYCLBackend`. SYCL is standard C++ and can target NVIDIA (via PTX backend), AMD, and Intel. It closely matches the "kernel" model of CUDA.

### Phase 3: NPU / Edge (Experimental)

For NPUs (which often lack general purpose "shared memory" thread syncing), the `tiledTaylorKernel` might need a different algorithm or a **Vulkan Compute** shader implementation.

## Immediate Action Items

1. **Isolate Kernels**: Move all `<<<...>>>` calls into a separate `DeviceOps` module.
2. **Feasibility Prototype**: Create a small standalone C++ program that compiles the "Taylor Tree" algorithm using SYCL (using DPC++ compiler) to benchmark performance vs CUDA.
