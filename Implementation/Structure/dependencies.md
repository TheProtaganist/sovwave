# Library Dependency Decisions

## Overview

This document outlines the dependency strategy for Aetheria.jl, explaining which external libraries to use, which functionality to build from scratch, and the rationale behind each decision. The goal is to balance leveraging existing high-quality libraries with maintaining control over wave-based computation primitives.

## Core Philosophy

### Build from Scratch vs. Use Existing Libraries

**Guiding Principles**:

1. **Use existing libraries for**: Well-established, performance-critical operations (FFT, linear algebra)
2. **Build from scratch for**: Wave-based computation primitives unique to Aetheria
3. **Avoid dependencies on**: Python, CUDA, GPU-specific libraries, traditional ML frameworks

**Rationale**: 
- Existing libraries provide optimized, tested implementations of standard operations
- Wave-based computation is novel and requires custom implementation
- Avoiding Python/CUDA maintains the pure Julia, wave-based paradigm

## Dependency Categories

### Essential Dependencies (Must Have)

These libraries are fundamental to Aetheria's operation:

#### 1. FFTW.jl - Fast Fourier Transform

**Purpose**: Frequency domain transformations for harmonic encoding

**Why needed**:
- FFT is essential for converting data to/from frequency domain
- FFTW is the gold standard for FFT performance
- Implementing FFT from scratch would be reinventing the wheel

**Usage in Aetheria**:
- `HarmonicEncoder`: Data to frequency spectrum conversion
- `FrequencySpectrum`: Frequency analysis
- Spectral methods for solving PDEs (optional)

**Version**: FFTW.jl v1.7+

**Alternatives considered**:
- Custom FFT implementation: Too complex, worse performance
- DSP.jl FFT: Less optimized than FFTW

**Decision**: Use FFTW.jl

#### 2. DifferentialEquations.jl - ODE/PDE Solvers

**Purpose**: Time evolution of wave fields (Schrödinger equation)

**Why needed**:
- Provides robust, adaptive time-stepping methods
- Handles stiff equations (important for nonlinear dynamics)
- Extensive testing and optimization

**Usage in Aetheria**:
- `MorphogeneticSimulator`: Wave field evolution
- `TimeEvolution`: Numerical integration
- Adaptive time-stepping for stability

**Version**: DifferentialEquations.jl v7.0+

**Alternatives considered**:
- Custom RK4/Crank-Nicolson: Simpler but less robust
- Manual time-stepping: Reinventing the wheel

**Decision**: Use DifferentialEquations.jl for complex simulations, provide simple custom methods for basic cases

#### 3. StaticArrays.jl - Fixed-Size Arrays

**Purpose**: Efficient small arrays (positions, wave vectors)

**Why needed**:
- Zero-cost abstractions for small arrays
- Stack allocation (no heap overhead)
- Critical for performance in tight loops

**Usage in Aetheria**:
- Position vectors: `SVector{3, Float64}`
- Wave vectors k_j in sacred geometry
- Grid coordinates

**Version**: StaticArrays.jl v1.6+

**Alternatives considered**:
- Regular arrays: Heap allocation overhead
- Tuples: Less convenient API

**Decision**: Use StaticArrays.jl

### Recommended Dependencies (Should Have)

These libraries significantly improve functionality:

#### 4. DSP.jl - Digital Signal Processing

**Purpose**: Signal processing utilities for wave protocols

**Why needed**:
- Windowing functions (Hamming, Hann, etc.)
- Filter design (for signal processing protocols)
- Spectral analysis tools

**Usage in Aetheria**:
- `HarmonicEncoder`: Windowing to reduce spectral leakage
- `WaveProtocols`: Signal processing implementations
- Frequency analysis utilities

**Version**: DSP.jl v0.7+

**Alternatives considered**:
- Custom implementations: Doable but time-consuming
- FFTW.jl alone: Lacks windowing and filtering

**Decision**: Use DSP.jl

#### 5. FileIO.jl + VideoIO.jl - Video I/O

**Purpose**: MP4 model storage format

**Why needed**:
- Reading/writing MP4 files
- Frame extraction and encoding
- Video codec support

**Usage in Aetheria**:
- `MP4Storage`: Save/load models as videos
- Frame-by-frame encoding of weight segments
- GIF extraction

**Version**: FileIO.jl v1.16+, VideoIO.jl v1.0+

**Alternatives considered**:
- Custom video encoding: Extremely complex
- External tools (ffmpeg): Less integrated
- Different format: MP4 is the requirement

**Decision**: Use FileIO.jl + VideoIO.jl

#### 6. Images.jl - Image Processing

**Purpose**: Frame manipulation for MP4 storage

**Why needed**:
- Convert wave fields to RGB images
- Image encoding/decoding
- Color space transformations

**Usage in Aetheria**:
- `MP4Storage`: Encode wave states as visual patterns
- Frequency pattern visualization
- Frame generation

**Version**: Images.jl v0.25+

**Alternatives considered**:
- Manual pixel manipulation: Tedious and error-prone
- External tools: Less integrated

**Decision**: Use Images.jl

### Optional Dependencies (Nice to Have)

These libraries enhance but aren't essential:

#### 7. Plots.jl or Makie.jl - Visualization

**Purpose**: Visualizing wave fields, cymatic patterns, training progress

**Why needed**:
- Debugging and understanding wave dynamics
- Creating figures for documentation
- Interactive exploration

**Usage in Aetheria**:
- Visualize wave fields: amplitude, phase
- Plot training curves
- Animate wave evolution

**Version**: Makie.jl v0.19+ (preferred for interactivity)

**Alternatives considered**:
- Plots.jl: Simpler but less interactive
- PyPlot.jl: Requires Python (violates pure Julia principle)
- No visualization: Harder to debug

**Decision**: Make Makie.jl an optional dependency (not required for core functionality)

#### 8. ProgressMeter.jl - Progress Bars

**Purpose**: Training progress indication

**Why needed**:
- User feedback during long training runs
- Estimated time remaining
- Better UX

**Usage in Aetheria**:
- `TrainingLoop`: Show epoch progress
- `WaveProtocols`: Show testing progress

**Version**: ProgressMeter.jl v1.7+

**Alternatives considered**:
- Custom progress printing: Less polished
- No progress indication: Poor UX

**Decision**: Use ProgressMeter.jl (lightweight dependency)

#### 9. Documenter.jl - Documentation Generation

**Purpose**: Generate documentation website

**Why needed**:
- Professional documentation
- API reference from docstrings
- Searchable docs

**Usage in Aetheria**:
- Build docs website
- API reference
- Tutorial rendering

**Version**: Documenter.jl v0.27+

**Alternatives considered**:
- Manual HTML: Too much work
- Markdown only: Less discoverable

**Decision**: Use Documenter.jl (dev dependency only)

### Testing Dependencies

#### 10. Test.jl - Unit Testing

**Purpose**: Standard Julia testing framework

**Why needed**:
- Built into Julia standard library
- Simple, effective testing

**Usage in Aetheria**:
- All unit tests
- Integration tests

**Version**: Included with Julia

**Decision**: Use Test.jl (no choice, it's standard)

#### 11. PropCheck.jl - Property-Based Testing

**Purpose**: Property-based testing for correctness properties

**Why needed**:
- Test universal properties (round-trip, symmetry, etc.)
- Generate random test cases
- Find edge cases automatically

**Usage in Aetheria**:
- Property tests for all 31 correctness properties
- Fuzzing for robustness

**Version**: PropCheck.jl v0.1+

**Alternatives considered**:
- Manual property testing: Less thorough
- Other PBT libraries: PropCheck.jl is most mature for Julia

**Decision**: Use PropCheck.jl

### Explicitly Avoided Dependencies

These libraries are intentionally NOT used:

#### Python Libraries (PyTorch, TensorFlow, NumPy)

**Why avoided**:
- Violates pure Julia requirement
- Introduces Python dependency
- Incompatible with wave-based paradigm

**Workaround**: Build everything in Julia

#### CUDA.jl / GPU Libraries

**Why avoided**:
- Violates no-CUDA requirement
- Wave-based computation doesn't use GPU matrix operations
- Limits portability

**Workaround**: CPU-based wave computation (potentially use wave protocols for hardware acceleration)

#### Flux.jl / Knet.jl / Other ML Frameworks

**Why avoided**:
- Based on discrete matrix operations
- Incompatible with wave-based paradigm
- Would encourage wrong patterns

**Workaround**: Build custom SOV network framework

#### LinearAlgebra.jl (for core computation)

**Why avoided for core**:
- Matrix operations violate wave-based paradigm
- Discrete tensor calculations not allowed

**Exception**: Can use for utility functions (e.g., matrix rotation for testing symmetry)

**Workaround**: Wave-based operations only in core computation

## Dependency Management Strategy

### Version Pinning

**Strategy**: Use semantic versioning with compatible ranges

```toml
[compat]
julia = "1.9"
FFTW = "1.7"
DifferentialEquations = "7.0"
StaticArrays = "1.6"
```

**Rationale**: 
- Allows patch updates (bug fixes)
- Prevents breaking changes
- Ensures reproducibility

### Minimal Dependencies

**Strategy**: Keep dependency count low

**Current count**: ~6 essential, ~3 recommended, ~3 optional = ~12 total

**Rationale**:
- Fewer dependencies = less maintenance burden
- Faster installation
- Fewer potential conflicts

### Optional Dependencies

**Strategy**: Make visualization and convenience libraries optional

**Implementation**:
```julia
# In Aetheria.jl
function visualize_wave_field(wf::WaveField)
    if !isdefined(Main, :Makie)
        @warn "Makie.jl not loaded. Install and load Makie for visualization."
        return nothing
    end
    # visualization code
end
```

**Rationale**: Core functionality doesn't require visualization

### Dependency Auditing

**Strategy**: Regularly review dependencies

**Process**:
1. Check for security vulnerabilities
2. Update to latest compatible versions
3. Remove unused dependencies
4. Evaluate new alternatives

**Frequency**: Every major release

## Build from Scratch Components

These components are implemented from scratch:

### 1. Wave Field Data Structure

**Why from scratch**:
- Unique to Aetheria
- Combines spatial grid + complex wave function + metadata
- No existing library provides this

**Implementation**: Custom `WaveField` struct

### 2. Sacred Geometry Potentials

**Why from scratch**:
- Novel application of sacred geometry to computation
- Specific mathematical formulations (Flower of Life, Metatron's Cube)
- No existing library

**Implementation**: Custom potential functions

### 3. Cymatic Extractor

**Why from scratch**:
- Novel concept: extracting computation results from standing waves
- Specific to Aetheria's paradigm
- No existing library

**Implementation**: Custom extraction algorithms

### 4. Wave-Based Optimizers

**Why from scratch**:
- Adaptation of optimization to wave states
- Free-energy minimization (not gradient descent)
- No existing library for wave optimization

**Implementation**: Custom `AdamWave` and other optimizers

### 5. Wave-Based Loss Functions

**Why from scratch**:
- Operate on wave states, not discrete values
- Frequency domain metrics
- Phase coherence measures

**Implementation**: Custom loss functions

### 6. Wave-Based Activations

**Why from scratch**:
- Frequency domain nonlinearities
- Wave-specific transformations
- No existing library

**Implementation**: Custom activation functions

### 7. SOV Network Architecture

**Why from scratch**:
- Novel network paradigm
- Wave-based layers
- No existing framework

**Implementation**: Custom network types

### 8. Wave Protocol Testing Framework

**Why from scratch**:
- Unique to Aetheria (Phase 2)
- Tests 144+ wave-based algorithms
- No existing framework

**Implementation**: Custom testing harness

### 9. MP4 Model Storage Format

**Why from scratch (partially)**:
- Novel storage format
- Weight-to-frame encoding is unique
- Uses VideoIO.jl for actual MP4 I/O

**Implementation**: Custom encoding/decoding logic

### 10. GGUF Converter

**Why from scratch**:
- Bridges Aetheria's wave-based format to GGUF
- Unique conversion logic
- No existing converter

**Implementation**: Custom conversion algorithms

## Dependency Decision Matrix

| Functionality | Build from Scratch | Use Library | Library Name |
|---------------|-------------------|-------------|--------------|
| FFT | ❌ | ✅ | FFTW.jl |
| ODE solving | ❌ | ✅ | DifferentialEquations.jl |
| Small arrays | ❌ | ✅ | StaticArrays.jl |
| Signal processing | ❌ | ✅ | DSP.jl |
| Video I/O | ❌ | ✅ | VideoIO.jl |
| Image processing | ❌ | ✅ | Images.jl |
| Wave fields | ✅ | ❌ | - |
| Sacred geometry | ✅ | ❌ | - |
| Cymatic extraction | ✅ | ❌ | - |
| Wave optimizers | ✅ | ❌ | - |
| Wave losses | ✅ | ❌ | - |
| Wave activations | ✅ | ❌ | - |
| SOV networks | ✅ | ❌ | - |
| Wave protocols | ✅ | ❌ | - |
| MP4 encoding logic | ✅ | ❌ | - |
| GGUF conversion | ✅ | ❌ | - |

## Installation and Setup

### User Installation

Users install Aetheria with:

```julia
using Pkg
Pkg.add("Aetheria")
```

This automatically installs all required dependencies.

### Developer Installation

Developers clone and instantiate:

```bash
git clone https://github.com/username/Aetheria.jl.git
cd Aetheria.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs exact versions from `Manifest.toml`.

### Optional Dependencies

For visualization:

```julia
using Pkg
Pkg.add("Makie")
using Makie
using Aetheria
# Now visualization functions work
```

## Future Dependency Considerations

### Potential Additions

1. **BenchmarkTools.jl**: For performance testing
2. **Aqua.jl**: For package quality checks
3. **JET.jl**: For static analysis
4. **Revise.jl**: For development workflow (dev dependency)

### Potential Removals

If functionality can be simplified:
1. **DifferentialEquations.jl**: If simple time-stepping suffices
2. **VideoIO.jl**: If MP4 format is changed

### Monitoring

Watch for:
1. New Julia features that replace dependencies
2. Dependency abandonment or security issues
3. Performance improvements in alternatives

## Summary

Aetheria.jl's dependency strategy:
- **Use libraries for**: Standard operations (FFT, ODE solving, I/O)
- **Build from scratch for**: Wave-based computation primitives
- **Avoid**: Python, CUDA, traditional ML frameworks
- **Keep minimal**: ~12 total dependencies
- **Make optional**: Visualization and convenience features
- **Pin versions**: Ensure reproducibility
- **Regular audits**: Maintain quality and security

This approach balances leveraging the Julia ecosystem with maintaining control over the novel wave-based computation paradigm.
