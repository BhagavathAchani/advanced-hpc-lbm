# Lattice Boltzmann Fluid Simulation (CPU & MPI Optimised)

High-performance implementation of a D2Q9-BGK Lattice Boltzmann Method (LBM) fluid simulation, developed as part of an Advanced High Performance Computing course at the University of Bristol.

This repository focuses on CPU-side performance engineering, including serial optimisation, SIMD vectorisation, NUMA-aware OpenMP parallelism, and distributed-memory scaling with MPI. A separate repository contains the CUDA GPU implementation.

## Key Contributions

* Optimised the serial LBM solver using loop fusion, arithmetic simplification, and data-layout refactoring, significantly reducing instruction count and cache misses.
* Applied SIMD vectorisation (AVX) using OpenMP SIMD directives, aligned memory allocation, and `restrict` pointers to maximise single-core throughput.
* Used Roofline modelling and cache-level profiling to identify memory-bandwidth bottlenecks and guide optimisation decisions.
* Implemented NUMA-aware OpenMP parallelisation, achieving strong scaling up to 28 CPU cores.
* Developed a 1D MPI domain decomposition with halo cells and non-blocking communication (MPI_Isend / MPI_Irecv), scaling simulations to 112 CPU cores across multiple nodes.
* Achieved up to 98.37% reduction in compute time for a 2048×2048 grid compared to the unoptimised serial baseline.

## Performance Highlights and Results

### Serial, SIMD, and OpenMP Performance

| Method | 128×128 | 128×256 | 256×256 | 1024×1024 |
|--------|---------|---------|---------|-----------|
| Serial | 12.3 s | 20.2 s | 74.2 s | 345.9 s |
| Serial + Vectorised | 4.7 s | 10.5 s | 36.0 s | 201.5 s |
| OpenMP (28 cores) | 0.64 s | 0.89 s | 2.9 s | 13.1 s |

**Table 1:** Execution times for serial, SIMD-vectorised, and OpenMP parallelised versions of the D2Q9-BGK Lattice Boltzmann simulation.

### MPI Scaling and Performance Improvements

| Grid Size | Compute Time (MPI) | Serial Optimisation (%) | MPI Ballpark (%) |
|-----------|-------------------|------------------------|------------------|
| 128×128 | 0.565 s | -92.04% | -10.31% |
| 128×256 | 0.556 s | -96.14% | -26.81% |
| 256×256 | 1.513 s | -97.37% | -43.95% |
| 1024×1024 | 3.142 s | -98.73% | -43.76% |
| 2048×2048 | 7.927 s | -98.37% | -47.15% |

**Table 2:** MPI compute times and percentage improvements relative to baseline serial performance.

## Build and Run

### Compilation
```bash
make CFLAGS="-O3 -fopenmp"
```

Compiler flags can be modified in the `Makefile` to enable architecture-specific optimisations.

### Execution
```bash
./d2q9-bgk <paramfile> <obstaclefile>
```

Example:
```bash
./d2q9-bgk input_256x256.params obstacles_256x256.dat
```

### MPI Execution

To run the distributed version using MPI:
```bash
mpirun -np <num_ranks> ./d2q9-bgk <paramfile> <obstaclefile>
```

The domain is decomposed along one spatial dimension, with halo exchange performed between neighbouring ranks.

## Correctness and Validation

Simulation results are validated using the provided Python checking scripts:
```bash
make check
```

The checker compares average velocities and final lattice state against reference solutions, ensuring correctness within numerical tolerance.

## Tooling and Platforms

* Languages: C++
* Parallelism: OpenMP, MPI (OpenMPI)
* Vectorisation: SIMD
* Profiling & Analysis: `perf`, Roofline modelling
* Platforms: University of Bristol BlueCrystal Phase 4 (Intel Broadwell CPUs)

## Related Work

* CUDA GPU implementation: see separate repository

## Note

This repository was originally forked from coursework starter code. All performance optimisations, parallelisation strategies, and scalability improvements were implemented independently.
