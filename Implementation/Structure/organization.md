# Project Organization Decisions

## Overview

This document outlines the organizational structure of Aetheria.jl, explaining the rationale behind directory layout, module organization, and file naming conventions. The structure is designed to support wave-based computation while maintaining clarity, modularity, and ease of navigation.

## Directory Structure

```
Aetheria.jl/
├── src/                          # Source code
│   ├── Aetheria.jl              # Main module file
│   ├── Core/                    # Core wave dynamics
│   │   ├── WaveField.jl         # Wave field data structure
│   │   ├── Grid.jl              # Spatial grid management
│   │   └── Constants.jl         # Physical constants
│   ├── Encoding/                # Data to wave conversion
│   │   ├── HarmonicEncoder.jl   # Frequency domain encoding
│   │   └── FrequencySpectrum.jl # Spectrum data structure
│   ├── Simulation/              # Wave evolution
│   │   ├── MorphogeneticSimulator.jl
│   │   ├── TimeEvolution.jl     # Time-stepping methods
│   │   └── EnergyFunctionals.jl # Free energy computation
│   ├── Potentials/              # Sacred geometry potentials
│   │   ├── SacredGeometry.jl    # Abstract types
│   │   ├── FlowerOfLife.jl      # Flower of Life potential
│   │   └── MetatronsCube.jl     # Metatron's Cube potential
│   ├── Extraction/              # Result extraction
│   │   ├── CymaticExtractor.jl  # Standing wave extraction
│   │   └── EigenState.jl        # Eigenstate data structure
│   ├── Optimization/            # Wave-based optimization
│   │   ├── WaveOptimizer.jl     # Abstract optimizer
│   │   ├── AdamWave.jl          # AdamW for waves
│   │   └── GradientFlow.jl      # Basic gradient descent
│   ├── Loss/                    # Loss functions
│   │   ├── WaveLoss.jl          # Abstract loss type
│   │   ├── FrequencyDomainMSE.jl
│   │   ├── PhaseCoherenceLoss.jl
│   │   └── FreeEnergyLoss.jl
│   ├── Activation/              # Activation functions
│   │   ├── WaveActivation.jl    # Abstract activation
│   │   ├── WaveSigmoid.jl
│   │   ├── WaveReLU.jl
│   │   └── WaveTanh.jl
│   ├── Networks/                # SOV network architecture
│   │   ├── SOVLayer.jl          # Abstract layer type
│   │   ├── SOVDense.jl          # Dense SOV layer
│   │   └── SOVNetwork.jl        # Network composition
│   ├── Training/                # Training infrastructure
│   │   ├── TrainingLoop.jl      # Main training loop
│   │   ├── Validation.jl        # Validation utilities
│   │   └── EarlyStopping.jl     # Early stopping logic
│   ├── Inference/               # Inference API
│   │   ├── SingleSample.jl      # Single-sample inference
│   │   └── BatchInference.jl    # Batch inference
│   ├── Data/                    # Dataset support
│   │   ├── DataLoader.jl        # Data loading
│   │   ├── Preprocessing.jl     # Wave-compatible preprocessing
│   │   └── Augmentation.jl      # Frequency domain augmentation
│   ├── Persistence/             # Model storage
│   │   ├── MP4Storage.jl        # MP4/GIF model format
│   │   ├── GGUFConverter.jl     # GGUF conversion
│   │   └── ModelMetadata.jl     # Metadata handling
│   └── WaveProtocols/           # Wave interface testing (Phase 2)
│       ├── WaveProtocol.jl      # Abstract protocol type
│       ├── AudioProtocols.jl    # Audio wave protocols
│       ├── EMFProtocols.jl      # EMF protocols
│       ├── RadioProtocols.jl    # Radio frequency protocols
│       ├── SignalProcessing.jl  # Signal processing protocols
│       ├── TestRunner.jl        # Protocol testing framework
│       └── ProtocolLogger.jl    # Test result logging
├── test/                        # Test suite
│   ├── runtests.jl             # Main test runner
│   ├── core_tests.jl           # Core functionality tests
│   ├── encoding_tests.jl       # Encoding tests
│   ├── simulation_tests.jl     # Simulation tests
│   ├── property_tests.jl       # Property-based tests
│   └── integration_tests.jl    # End-to-end tests
├── docs/                        # Documentation
│   ├── make.jl                 # Documenter.jl build script
│   ├── src/                    # Documentation source
│   │   ├── index.md            # Landing page
│   │   ├── tutorials/          # Tutorial guides
│   │   ├── api/                # API reference
│   │   └── theory/             # Theoretical background
│   └── build/                  # Generated documentation
├── examples/                    # Example scripts
│   ├── basic_training.jl       # Simple training example
│   ├── model_saving.jl         # Save/load example
│   └── gguf_conversion.jl      # GGUF conversion example
├── Implementation/              # Implementation guides (this directory)
│   ├── Mathematics/            # Mathematical foundations
│   ├── Structure/              # Project structure docs
│   ├── Fundamentals/           # Setup and basics
│   ├── Components/             # Component implementation guides
│   ├── Persistence/            # Storage implementation guides
│   ├── Documentation/          # Documentation guides
│   ├── Examples/               # Example code guides
│   ├── Testing/                # Testing guides
│   └── Publishing/             # Publishing guides
├── Project.toml                # Package metadata and dependencies
├── Manifest.toml               # Exact dependency versions
├── README.md                   # Project overview
├── LICENSE                     # License file
└── .gitignore                  # Git ignore rules
```

## Organizational Principles

### 1. Separation by Functionality

Each major functional area has its own directory:
- **Core**: Fundamental data structures (WaveField, Grid)
- **Encoding**: Input data transformation
- **Simulation**: Wave evolution dynamics
- **Potentials**: Sacred geometry landscapes
- **Extraction**: Output decoding
- **Optimization**: Training algorithms
- **Networks**: Architecture definition
- **Persistence**: Model storage

**Rationale**: Clear separation makes the codebase easier to navigate and understand. Each directory has a focused purpose.

### 2. Flat Module Hierarchy

Within each directory, files are relatively flat (not deeply nested):
- `src/Optimization/AdamWave.jl` (good)
- Not: `src/Optimization/Algorithms/Adaptive/AdamWave.jl` (too deep)

**Rationale**: Julia's module system and multiple dispatch reduce the need for deep hierarchies. Flat structures are easier to navigate.

### 3. One Concept Per File

Each file contains one main type or concept:
- `WaveSigmoid.jl` contains `WaveSigmoid` type and related methods
- `FlowerOfLife.jl` contains `FlowerOfLifePotential` type and methods

**Rationale**: Makes it easy to find code. If you need `WaveSigmoid`, you know exactly where to look.

### 4. Abstract Types in Separate Files

Abstract types and their interface definitions get their own files:
- `WaveOptimizer.jl` defines abstract `WaveOptimizer` type
- `AdamWave.jl` defines concrete `AdamWave <: WaveOptimizer`

**Rationale**: Separates interface definition from implementation, making the type hierarchy clear.

### 5. Test Structure Mirrors Source

Test files mirror the source structure:
- `src/Encoding/HarmonicEncoder.jl` → `test/encoding_tests.jl`
- `src/Optimization/AdamWave.jl` → `test/optimization_tests.jl`

**Rationale**: Easy to find tests for any source file. Maintains parallel structure.

### 6. Documentation Separate from Code

Documentation lives in `docs/`, not inline in source files (though docstrings are still used):
- `docs/src/tutorials/` for tutorials
- `docs/src/api/` for API reference
- `docs/src/theory/` for theoretical background

**Rationale**: Keeps source files focused on code. Allows rich documentation with images, equations, and examples.

### 7. Examples as Standalone Scripts

Examples are complete, runnable scripts in `examples/`:
- Each example is self-contained
- Can be run directly: `julia examples/basic_training.jl`

**Rationale**: Users can quickly try examples without navigating complex project structure.

### 8. Implementation Guides Separate

The `Implementation/` directory contains guides for future implementation:
- Not part of the package itself
- Used during development to guide implementation
- Can be removed or moved to docs after implementation

**Rationale**: Keeps planning/design documents separate from actual code.

## Module Organization

### Main Module: Aetheria.jl

The main module file `src/Aetheria.jl` serves as the entry point:

```julia
module Aetheria

# Core exports
export WaveField, Grid, FrequencySpectrum

# Encoding exports
export HarmonicEncoder, encode, decode

# Simulation exports
export MorphogeneticSimulator, evolve!, compute_free_energy

# Potential exports
export FlowerOfLifePotential, MetatronsCubePotential

# Extraction exports
export CymaticExtractor, extract_eigenstates

# Optimization exports
export AdamWave, update!

# Loss exports
export FrequencyDomainMSE, PhaseCoherenceLoss, FreeEnergyLoss

# Activation exports
export WaveSigmoid, WaveReLU, WaveTanh

# Network exports
export SOVDense, SOVNetwork, forward

# Training exports
export train!, validate

# Inference exports
export predict, predict_batch

# Persistence exports
export save_model, load_model, convert_to_gguf

# Include all submodules
include("Core/WaveField.jl")
include("Core/Grid.jl")
# ... (all other includes)

end # module
```

**Rationale**: Single entry point makes it clear what's public API. Users do `using Aetheria` and get everything they need.

### Submodule Organization

Each functional area can optionally be a submodule:

```julia
module Encoding
    export HarmonicEncoder, encode, decode
    include("HarmonicEncoder.jl")
    include("FrequencySpectrum.jl")
end
```

**Decision**: Start without submodules (flat structure), add them later if needed for namespace management.

**Rationale**: Submodules add complexity. Only use them if there are naming conflicts or if the codebase grows very large.

## File Naming Conventions

### Source Files

- **PascalCase** for type definitions: `WaveField.jl`, `HarmonicEncoder.jl`
- **Descriptive names**: File name matches main type name
- **Plural for collections**: `Constants.jl` (contains multiple constants)

### Test Files

- **snake_case** with `_tests` suffix: `encoding_tests.jl`, `simulation_tests.jl`
- **Grouped by functionality**: Not one test file per source file (too many files)

### Documentation Files

- **lowercase with hyphens**: `getting-started.md`, `api-reference.md`
- **Descriptive names**: Clear what the document contains

### Example Files

- **snake_case**: `basic_training.jl`, `model_saving.jl`
- **Descriptive of what example demonstrates**

## Dependency Management

### Project.toml

Contains package metadata and dependencies:

```toml
name = "Aetheria"
uuid = "..."
version = "0.1.0"

[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
# ... other dependencies

[compat]
julia = "1.9"
FFTW = "1.7"
DSP = "0.7"
```

**Rationale**: Explicit dependency versions ensure reproducibility.

### Manifest.toml

Auto-generated, contains exact versions of all dependencies (including transitive):
- Committed to git for reproducibility
- Users get exact same environment

**Rationale**: Ensures everyone uses the same dependency versions.

## Code Organization Patterns

### Type Hierarchy

Use abstract types to define interfaces:

```julia
abstract type WaveOptimizer end
abstract type WaveLoss end
abstract type WaveActivation end
abstract type SacredGeometryPotential end
```

Concrete types implement the interface:

```julia
struct AdamWave <: WaveOptimizer
    # fields
end

struct FrequencyDomainMSE <: WaveLoss
    # fields
end
```

**Rationale**: Enables polymorphism and extensibility. Users can define custom types.

### Multiple Dispatch

Use multiple dispatch for operations:

```julia
# Generic interface
function update!(opt::WaveOptimizer, params, gradients)
    error("Not implemented")
end

# Specific implementation
function update!(opt::AdamWave, params, gradients)
    # AdamWave-specific logic
end
```

**Rationale**: Julia's strength. Allows specialized implementations without inheritance complexity.

### Composition Over Inheritance

Prefer composition:

```julia
struct SOVNetwork
    layers::Vector{SOVLayer}
    encoder::HarmonicEncoder
    extractor::CymaticExtractor
end
```

Rather than deep inheritance hierarchies.

**Rationale**: More flexible, easier to understand and modify.

## Documentation Organization

### API Documentation

Generated from docstrings using Documenter.jl:

```julia
"""
    encode(encoder::HarmonicEncoder, data::AbstractArray)

Convert input data to frequency domain representation.

# Arguments
- `encoder::HarmonicEncoder`: The encoder instance
- `data::AbstractArray`: Input data to encode

# Returns
- `FrequencySpectrum`: Frequency domain representation

# Examples
```julia
encoder = HarmonicEncoder(frequency_range=(0.0, 10.0), resolution=256)
spectrum = encode(encoder, [1.0, 2.0, 3.0])
```
"""
function encode(encoder::HarmonicEncoder, data::AbstractArray)
    # implementation
end
```

**Rationale**: Documentation lives with code, automatically stays in sync.

### Tutorial Documentation

Narrative guides in `docs/src/tutorials/`:
- Getting started
- Basic training
- Advanced features
- Custom components

**Rationale**: Tutorials teach concepts and workflows, complementing API reference.

### Theoretical Documentation

Mathematical background in `docs/src/theory/`:
- Wave dynamics
- Sacred geometry
- Optimization theory

**Rationale**: Helps users understand the "why" behind the implementation.

## Testing Organization

### Test Structure

```julia
# test/runtests.jl
using Test
using Aetheria

@testset "Aetheria.jl" begin
    include("core_tests.jl")
    include("encoding_tests.jl")
    include("simulation_tests.jl")
    include("property_tests.jl")
    include("integration_tests.jl")
end
```

Each test file contains related tests:

```julia
# test/encoding_tests.jl
@testset "Encoding" begin
    @testset "HarmonicEncoder" begin
        # encoder tests
    end
    
    @testset "FrequencySpectrum" begin
        # spectrum tests
    end
end
```

**Rationale**: Hierarchical test organization makes it easy to run specific test groups.

### Property-Based Tests

Separate file for property tests:

```julia
# test/property_tests.jl
using PropCheck

@testset "Property Tests" begin
    @testset "Encoder Round-Trip" begin
        # Property 1: encode-decode round-trip
    end
    
    @testset "Wave Field Continuity" begin
        # Property 6: Lipschitz continuity
    end
end
```

**Rationale**: Property tests are conceptually different from unit tests, deserve separate organization.

## Build and CI Organization

### GitHub Actions (or similar CI)

```yaml
# .github/workflows/CI.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

**Rationale**: Automated testing on every commit ensures code quality.

## Future Considerations

### Scaling

As the project grows:
1. **Introduce submodules** if namespace conflicts arise
2. **Split large files** if they exceed ~500 lines
3. **Add benchmarks/** directory for performance testing
4. **Add scripts/** directory for utility scripts

### Extensibility

Design for extensibility:
- Abstract types allow users to define custom components
- Multiple dispatch enables adding new methods without modifying existing code
- Composition allows mixing and matching components

### Maintenance

Keep organization clean:
- Regular refactoring to maintain structure
- Remove dead code promptly
- Update documentation when structure changes

## Summary

The Aetheria.jl project organization follows these principles:
1. **Functional separation**: Each directory has a clear purpose
2. **Flat hierarchy**: Avoid deep nesting
3. **One concept per file**: Easy to find code
4. **Mirrored test structure**: Tests parallel source
5. **Separate documentation**: Rich docs without cluttering source
6. **Explicit dependencies**: Reproducible environment
7. **Extensible design**: Users can add custom components

This structure supports wave-based computation while maintaining clarity and ease of use.
