# Roadmap for seticore

## Project: High-Performance SETI Core

Seticore is the modern, C++/CUDA implementation of the SETI search algorithm, replacing the legacy SETI@home client infrastructure.

## Status

- **Analysis**: Initial deep dive complete.
- **Sanitization**: Checked and clean.

## Golden Nuggets

These are the files where the actual alien-hunting math happens:

### 1. The Search Kernel (`dedoppler.cu`)

* **Why**: The heart of the project. Implements the "dedoppler" algorithm on the GPU.
- **Key Concepts**:
  - **Taylor Trees**: Referenced in `optimizedTaylorTree`, this is the efficient algorithm for summing paths through the spectrogram to detect drifting signals.
  - `findTopPathSums`: The kernel that aggregates the best signal paths.
  - `Dedopplerer` class: Manages GPU memory lifecycles for the search.

### 2. Event Finding (`find_events.cpp`)

* **Why**: filters the raw "hits" into significant "events".
- **Key Logic**:
  - Post-processing of GPU results.
  - Standard deviation/Noise floor estimation.

### 3. Pipeline Glue (`main.cpp` / `run_dedoppler.cpp`)

* **Why**: Shows how to run the engine.
- **Key Logic**:
  - Loading `.h5` files (HDF5 format from telescopes).
  - Orchestrating the beamforming and dedoppler stages.

## Scaffold

- See `SCAFFOLD/NOTES.md` for personal scratchpad.
