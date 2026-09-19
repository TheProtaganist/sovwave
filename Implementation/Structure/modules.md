# Module Structure and Dependencies

## Overview

This document details the module structure of Aetheria.jl, explaining how different components interact, their dependencies, and the data flow through the system. Understanding this structure is essential for implementing and extending the library.

## Module Hierarchy

```
Aetheria (main module)
├── Core
│   ├── WaveField
│   ├── Grid
│   └── Constants
├── Encoding
│   ├── HarmonicEncoder
│   └── FrequencySpectrum
├── Simulation
│   ├── MorphogeneticSimulator
│   ├── TimeEvolution
│   └── EnergyFunctionals
├── Potentials
│   ├── SacredGeometry (abstract)
│   ├── FlowerOfLife
│   └── MetatronsCube
├── Extraction
│   ├── CymaticExtractor
│   └── EigenState
├── Optimization
│   ├── WaveOptimizer (abstract)
│   ├── AdamWave
│   └── GradientFlow
├── Loss
│   ├── WaveLoss (abstract)
│   ├── FrequencyDomainMSE
│   ├── PhaseCoherenceLoss
│   └── FreeEnergyLoss
├── Activation
│   ├── WaveActivation (abstract)
│   ├── WaveSigmoid
│   ├── WaveReLU
│   └── WaveTanh
├── Networks
│   ├── SOVLayer (abstract)
│   ├── SOVDense
│   └── SOVNetwork
├── Training
│   ├── TrainingLoop
│   ├── Validation
│   └── EarlyStopping
├── Inference
│   ├── SingleSample
│   └── BatchInference
├── Data
│   ├── DataLoader
│   ├── Preprocessing
│   └── Augmentation
├── Persistence
│   ├── MP4Storage
│   ├── GGUFConverter
│   └── ModelMetadata
└── WaveProtocols (Phase 2 only)
    ├── WaveProtocol (abstract)
    ├── AudioProtocols
    ├── EMFProtocols
    ├── RadioProtocols
    ├── SignalProcessing
    ├── TestRunner
    └── ProtocolLogger
```

## Module Descriptions

### Core Module

**Purpose**: Fundamental data structures and constants

**Components**:
- `WaveField`: Complex-valued wave function on spatial grid
- `Grid`: Spatial discretization (dimensions, spacing, origin)
- `Constants`: Physical constants (ℏ, mass, etc.)

**Dependencies**: 
- External: StaticArrays.jl (for position vectors)
- Internal: None (foundation layer)

**Exports**:
```julia
export WaveField, Grid
export create_wave_field, create_grid
```

**Key Types**:
```julia
struct Grid
    dimensions::Tuple{Int, Int, Int}
    spacing::Tuple{Float64, Float64, Float64}
    origin::Tuple{Float64, Float64, Float64}
end

struct WaveField
    spatial_grid::Grid
    psi::Array{ComplexF64}
    time::Float64
    hbar::Float64
    mass::Float64
end
```

### Encoding Module

**Purpose**: Convert input data to frequency domain representations

**Components**:
- `HarmonicEncoder`: Performs FFT-based encoding/decoding
- `FrequencySpectrum`: Stores frequency domain data

**Dependencies**:
- External: FFTW.jl, DSP.jl
- Internal: Core (WaveField, Grid)

**Exports**:
```julia
export HarmonicEncoder, FrequencySpectrum
export encode, decode
```

**Key Types**:
```julia
struct HarmonicEncoder
    frequency_range::Tuple{Float64, Float64}
    resolution::Int
    window_function::Function
end

struct FrequencySpectrum
    frequencies::Vector{Float64}
    amplitudes::Vector{Float64}
    phases::Vector{Float64}
    metadata::Dict{Symbol, Any}
end
```

**Data Flow**:
```
Input Data → HarmonicEncoder.encode() → FrequencySpectrum → WaveField
```

### Simulation Module

**Purpose**: Evolve wave fields according to Schrödinger equation

**Components**:
- `MorphogeneticSimulator`: Main simulation engine
- `TimeEvolution`: Time-stepping methods
- `EnergyFunctionals`: Compute free energy

**Dependencies**:
- External: DifferentialEquations.jl (optional)
- Internal: Core (WaveField), Potentials (SacredGeometry)

**Exports**:
```julia
export MorphogeneticSimulator
export evolve!, compute_free_energy, apply_interference
```

**Key Types**:
```julia
struct MorphogeneticSimulator
    wave_field::WaveField
    potential::SacredGeometryPotential
    time_step::Float64
    nonlinearity::Float64
end
```

**Data Flow**:
```
WaveField + Potential → evolve!() → Updated WaveField
```

### Potentials Module

**Purpose**: Sacred geometry potential landscapes

**Components**:
- `SacredGeometry`: Abstract type for potentials
- `FlowerOfLife`: Hexagonal C6 lattice potential
- `MetatronsCube`: 3D octahedral potential

**Dependencies**:
- External: StaticArrays.jl
- Internal: Core (Grid)

**Exports**:
```julia
export SacredGeometryPotential
export FlowerOfLifePotential, MetatronsCubePotential
export evaluate_potential, create_potential_field
```

**Key Types**:
```julia
abstract type SacredGeometryPotential end

struct FlowerOfLifePotential <: SacredGeometryPotential
    lattice_spacing::Float64
    well_depth::Float64
    symmetry::Symbol
end

struct MetatronsCubePotential <: SacredGeometryPotential
    scale::Float64
    well_depth::Float64
end
```

### Extraction Module

**Purpose**: Extract computational results from wave patterns

**Components**:
- `CymaticExtractor`: Identifies standing waves and eigenstates
- `EigenState`: Stores extracted eigenstate information

**Dependencies**:
- External: FFTW.jl (for frequency analysis)
- Internal: Core (WaveField)

**Exports**:
```julia
export CymaticExtractor, EigenState
export extract_eigenstates, compute_resonances
```

**Key Types**:
```julia
struct CymaticExtractor
    resonance_threshold::Float64
    max_modes::Int
end

struct EigenState
    frequency::Float64
    amplitude::Float64
    phase::Float64
    spatial_pattern::Array{ComplexF64}
    energy::Float64
end
```

**Data Flow**:
```
WaveField → extract_eigenstates() → Vector{EigenState} → Output
```

### Optimization Module

**Purpose**: Wave-based optimization algorithms

**Components**:
- `WaveOptimizer`: Abstract optimizer interface
- `AdamWave`: AdamW adapted for wave states
- `GradientFlow`: Basic gradient descent

**Dependencies**:
- External: None
- Internal: Core (WaveField), Simulation (EnergyFunctionals)

**Exports**:
```julia
export WaveOptimizer, AdamWave, GradientFlow
export update!, compute_energy_gradient
```

**Key Types**:
```julia
abstract type WaveOptimizer end

struct AdamWave <: WaveOptimizer
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    weight_decay::Float64
    first_moment::Dict{Symbol, WaveField}
    second_moment::Dict{Symbol, WaveField}
end
```

**Data Flow**:
```
WaveField + Loss → compute_energy_gradient() → Gradient
Gradient + Optimizer → update!() → Updated WaveField
```

### Loss Module

**Purpose**: Measure SOV network performance

**Components**:
- `WaveLoss`: Abstract loss interface
- `FrequencyDomainMSE`: MSE in frequency domain
- `PhaseCoherenceLoss`: Phase coherence metric
- `FreeEnergyLoss`: Free energy functional

**Dependencies**:
- External: None
- Internal: Core (WaveField), Simulation (EnergyFunctionals)

**Exports**:
```julia
export WaveLoss
export FrequencyDomainMSE, PhaseCoherenceLoss, FreeEnergyLoss
export compute_loss
```

**Key Types**:
```julia
abstract type WaveLoss end

struct FrequencyDomainMSE <: WaveLoss
    weight_function::Function
end

struct PhaseCoherenceLoss <: WaveLoss
    coherence_threshold::Float64
end
```

**Data Flow**:
```
Predicted WaveField + Target WaveField → compute_loss() → Loss Value
```

### Activation Module

**Purpose**: Nonlinear transformations for wave states

**Components**:
- `WaveActivation`: Abstract activation interface
- `WaveSigmoid`, `WaveReLU`, `WaveTanh`: Specific activations

**Dependencies**:
- External: FFTW.jl (for frequency domain operations)
- Internal: Core (WaveField)

**Exports**:
```julia
export WaveActivation
export WaveSigmoid, WaveReLU, WaveTanh
export apply_activation
```

**Key Types**:
```julia
abstract type WaveActivation end

struct WaveSigmoid <: WaveActivation
    steepness::Float64
end

struct WaveReLU <: WaveActivation
    threshold_frequency::Float64
end
```

**Data Flow**:
```
WaveField → apply_activation() → Transformed WaveField
```

### Networks Module

**Purpose**: SOV network architecture

**Components**:
- `SOVLayer`: Abstract layer interface
- `SOVDense`: Dense SOV layer
- `SOVNetwork`: Network composition

**Dependencies**:
- External: None
- Internal: Core, Encoding, Simulation, Potentials, Extraction, Activation

**Exports**:
```julia
export SOVLayer, SOVDense, SOVNetwork
export forward, create_sov_network
```

**Key Types**:
```julia
abstract type SOVLayer end

struct SOVDense <: SOVLayer
    input_frequencies::Int
    output_frequencies::Int
    potential::SacredGeometryPotential
    activation::WaveActivation
    wave_params::WaveField
end

struct SOVNetwork
    layers::Vector{SOVLayer}
    encoder::HarmonicEncoder
    extractor::CymaticExtractor
end
```

**Data Flow**:
```
Input Data → Encoder → WaveField → Layer 1 → ... → Layer N → Extractor → Output
```

### Training Module

**Purpose**: Training loop infrastructure

**Components**:
- `TrainingLoop`: Main training orchestration
- `Validation`: Validation utilities
- `EarlyStopping`: Early stopping logic

**Dependencies**:
- External: ProgressMeter.jl (optional)
- Internal: Networks, Optimization, Loss, Data

**Exports**:
```julia
export train!, validate
export EarlyStoppingConfig
```

**Data Flow**:
```
Network + Data + Optimizer + Loss → train!() → Trained Network
```

### Inference Module

**Purpose**: Prediction with trained networks

**Components**:
- `SingleSample`: Single-sample inference
- `BatchInference`: Batch inference

**Dependencies**:
- External: None
- Internal: Networks, Encoding, Extraction

**Exports**:
```julia
export predict, predict_batch
```

**Data Flow**:
```
Trained Network + Input Data → predict() → Predictions
```

### Data Module

**Purpose**: Dataset loading and preprocessing

**Components**:
- `DataLoader`: Load common dataset formats
- `Preprocessing`: Wave-compatible preprocessing
- `Augmentation`: Frequency domain augmentation

**Dependencies**:
- External: FileIO.jl (for data loading)
- Internal: Encoding (for wave compatibility)

**Exports**:
```julia
export load_dataset, preprocess, augment
export DataBatch
```

**Data Flow**:
```
Raw Data → load_dataset() → preprocess() → augment() → Wave-Compatible Data
```

### Persistence Module

**Purpose**: Model storage and conversion

**Components**:
- `MP4Storage`: MP4/GIF model format
- `GGUFConverter`: GGUF format conversion
- `ModelMetadata`: Metadata handling

**Dependencies**:
- External: VideoIO.jl, Images.jl, FileIO.jl
- Internal: Networks, Core

**Exports**:
```julia
export save_model, load_model
export convert_to_gguf, load_from_gguf
export export_to_gifs
```

**Data Flow**:
```
SOVNetwork → save_model() → MP4 File
MP4 File → load_model() → SOVNetwork
MP4 File → convert_to_gguf() → GGUF File
```

### WaveProtocols Module (Phase 2 Only)

**Purpose**: Test wave-based hardware communication protocols

**Components**:
- `WaveProtocol`: Abstract protocol interface
- `AudioProtocols`, `EMFProtocols`, `RadioProtocols`, `SignalProcessing`: Specific protocols
- `TestRunner`: Protocol testing framework
- `ProtocolLogger`: Test result logging

**Dependencies**:
- External: DSP.jl, FFTW.jl
- Internal: Core

**Exports**:
```julia
export WaveProtocol
export AudioWaveProtocol, EMFProtocol, RadioProtocol
export test_protocol, run_protocol_competition
export ProtocolTestResult
```

**Data Flow**:
```
Test Data → test_protocol() → ProtocolTestResult → Logger
All Results → run_protocol_competition() → Winning Protocol
```

## Inter-Module Dependencies

### Dependency Graph

```
Core (foundation)
  ↓
Encoding, Potentials (use Core)
  ↓
Simulation (uses Core, Potentials)
  ↓
Extraction (uses Core, Simulation)
  ↓
Optimization, Loss, Activation (use Core, Simulation)
  ↓
Networks (uses all above)
  ↓
Training, Inference (use Networks)
  ↓
Data, Persistence (use Networks, Encoding)
```

### Dependency Rules

1. **No circular dependencies**: Modules form a DAG (directed acyclic graph)
2. **Core is foundation**: All modules can depend on Core
3. **High-level depends on low-level**: Training depends on Networks, not vice versa
4. **Minimal coupling**: Modules interact through well-defined interfaces

### Interface Contracts

Each module defines clear interfaces:

**Example: WaveOptimizer Interface**
```julia
abstract type WaveOptimizer end

# Required methods for all optimizers
function update!(opt::WaveOptimizer, params, gradients)
    error("update! not implemented for $(typeof(opt))")
end

# Optional methods with default implementations
function reset!(opt::WaveOptimizer)
    # Default: do nothing
end
```

## Data Flow Through System

### Training Pipeline

```
1. Data Loading
   Raw Data → DataLoader → Preprocessed Data

2. Encoding
   Preprocessed Data → HarmonicEncoder → FrequencySpectrum → WaveField

3. Forward Pass
   WaveField → SOVNetwork.forward() → Output WaveField

4. Extraction
   Output WaveField → CymaticExtractor → Predictions

5. Loss Computation
   Predictions + Targets → WaveLoss → Loss Value

6. Optimization
   Loss → compute_energy_gradient() → Gradients
   Gradients + Optimizer → update!() → Updated Network

7. Repeat steps 2-6 for all batches/epochs
```

### Inference Pipeline

```
1. Data Loading
   Raw Data → Preprocessing

2. Encoding
   Preprocessed Data → HarmonicEncoder → WaveField

3. Forward Pass
   WaveField → Trained SOVNetwork → Output WaveField

4. Extraction
   Output WaveField → CymaticExtractor → Predictions

5. Post-processing
   Predictions → Format Conversion → Final Output
```

### Model Persistence Pipeline

```
Saving:
SOVNetwork → extract_wave_params() → Wave States
Wave States → encode_as_frames() → Image Frames
Image Frames → VideoIO → MP4 File

Loading:
MP4 File → VideoIO → Image Frames
Image Frames → decode_to_wave_states() → Wave States
Wave States + Architecture → reconstruct_network() → SOVNetwork
```

## Module Communication Patterns

### 1. Direct Function Calls

Most common pattern:
```julia
# Encoding module calls Core
wave_field = create_wave_field(grid, psi, time, hbar, mass)

# Simulation module calls Potentials
potential_values = evaluate_potential(flower_of_life, positions)
```

### 2. Multiple Dispatch

Polymorphic behavior:
```julia
# Different optimizers implement same interface
update!(adam_wave::AdamWave, params, grads)
update!(gradient_flow::GradientFlow, params, grads)
```

### 3. Callbacks

For extensibility:
```julia
# Training loop accepts callbacks
train!(network, data, optimizer; 
       on_epoch_end = epoch -> println("Epoch complete"))
```

### 4. Composition

Building complex objects:
```julia
# Network composes multiple components
network = SOVNetwork(
    layers = [layer1, layer2, layer3],
    encoder = HarmonicEncoder(...),
    extractor = CymaticExtractor(...)
)
```

## Extension Points

Users can extend Aetheria by:

### 1. Custom Potentials

```julia
struct MyCustomPotential <: SacredGeometryPotential
    # custom fields
end

function evaluate_potential(pot::MyCustomPotential, position)
    # custom implementation
end
```

### 2. Custom Optimizers

```julia
struct MyOptimizer <: WaveOptimizer
    # custom fields
end

function update!(opt::MyOptimizer, params, gradients)
    # custom implementation
end
```

### 3. Custom Loss Functions

```julia
struct MyLoss <: WaveLoss
    # custom fields
end

function compute_loss(loss::MyLoss, predicted, target)
    # custom implementation
end
```

### 4. Custom Layers

```julia
struct MyLayer <: SOVLayer
    # custom fields
end

function forward(layer::MyLayer, input_wave)
    # custom implementation
end
```

## Module Initialization

### Main Module (Aetheria.jl)

```julia
module Aetheria

# Version info
const VERSION = v"0.1.0"

# Include all submodules in dependency order
include("Core/WaveField.jl")
include("Core/Grid.jl")
include("Core/Constants.jl")

include("Encoding/FrequencySpectrum.jl")
include("Encoding/HarmonicEncoder.jl")

include("Potentials/SacredGeometry.jl")
include("Potentials/FlowerOfLife.jl")
include("Potentials/MetatronsCube.jl")

# ... (continue for all modules)

# Export public API
export WaveField, Grid
export HarmonicEncoder, encode, decode
export MorphogeneticSimulator, evolve!
# ... (all public exports)

end # module
```

## Testing Structure

Tests mirror module structure:

```
test/
├── runtests.jl
├── core_tests.jl          # Tests Core module
├── encoding_tests.jl      # Tests Encoding module
├── simulation_tests.jl    # Tests Simulation module
├── potentials_tests.jl    # Tests Potentials module
├── extraction_tests.jl    # Tests Extraction module
├── optimization_tests.jl  # Tests Optimization module
├── loss_tests.jl          # Tests Loss module
├── activation_tests.jl    # Tests Activation module
├── networks_tests.jl      # Tests Networks module
├── training_tests.jl      # Tests Training module
├── inference_tests.jl     # Tests Inference module
├── data_tests.jl          # Tests Data module
├── persistence_tests.jl   # Tests Persistence module
├── property_tests.jl      # Property-based tests
└── integration_tests.jl   # End-to-end tests
```

## Summary

Aetheria.jl's module structure:
- **Layered architecture**: Foundation (Core) → Mid-level (Encoding, Simulation) → High-level (Networks, Training)
- **Clear dependencies**: No circular dependencies, well-defined interfaces
- **Extensible design**: Users can add custom components via abstract types
- **Modular testing**: Each module has corresponding tests
- **Data flow**: Input → Encoding → Simulation → Extraction → Output
- **Communication**: Direct calls, multiple dispatch, composition

This structure supports wave-based computation while maintaining clarity, modularity, and extensibility.
