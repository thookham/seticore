# Notes for seticore

## Initial Observations

- Repository cloned successfully.
- Codebase is C++ / CUDA.
- Build system: Meson.

## Environment Check

- **Windows**: `meson` not found.
- **WSL**: `meson` not found.
- **Action Required**: Install build dependencies in WSL:

  ```bash
  sudo apt-get update
  sudo apt-get install cmake g++ libboost-all-dev libhdf5-dev pkg-config meson ninja-build
  # Note: 'meson' and 'ninja' are often safer to install via apt on modern Ubuntu to avoid PEP 668 issues.
  ```

## Next Steps

- Deep dive into architecture. (Done - see SCAFFOLD/ARCHITECTURE.md)
- Multi-arch strategy. (Done - see SCAFFOLD/MULTI_ARCH_STRATEGY.md)
