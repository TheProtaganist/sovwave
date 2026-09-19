# Basic Code Structure Guide

## Overview

This guide provides templates and patterns for writing Aetheria.jl code, including naming conventions, file organization, type definitions, function signatures, and documentation standards. Following these patterns ensures consistency and maintainability across the codebase.

## File Organization Patterns

### One Type Per File

Each major type gets its own file named after the type:

```julia
# src/Core/WaveField.jl
struct WaveField
    # fields
end

# Methods for WaveField
function create_wave_field(...)
    # implementation
end
```

### Abstract Types in Separate Files

Abstract types and their interface definitions:

```julia
# src/Optimization/WaveOptimizer.jl
"""
Abstract base type for wave-based optimizers.

All optimizers must implement:
- `update!(opt, params, gradients)`: Update parameters
"""
abstract type WaveOptimizer end

# Interface methods with error messages
function update!(opt::WaveOptimizer, params, gradients)
    error("update! not implemented for $(typeof(opt))")
end

function reset!(opt::WaveOptimizer)
    # Default implementation: do nothing
end
```

### Concrete Implementations

Concrete types in separate files:

```julia
# src/Optimization/AdamWave.jl
"""
    AdamWave <: WaveOptimizer

AdamW optimizer adapted for wave states.

# Fields
- `learning_rate::Float64`: Learning rate
- `beta1::Float64`: First moment decay rate
- `beta2::Float64`: Second moment decay rate
- `epsilon::Float64`: Numerical stability constant
- `weight_decay::Float64`: Weight decay coefficient
"""
struct AdamWave <: WaveOptimizer
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    weight_decay::Float64
    
    # Internal state
    first_moment::Dict{Symbol, WaveField}
    second_moment::Dict{Symbol, WaveField}
end

# Constructor with defaults
function AdamWave(;
    learning_rate::Float64 = 0.001,
    beta1::Float64 = 0.9,
    beta2::Float64 = 0.999,
    epsilon::Float64 = 1e-8,
    weight_decay::Float64 = 0.01
)
    AdamWave(
        learning_rate, beta1, beta2, epsilon, weight_decay,
        Dict{Symbol, WaveField}(),
        Dict{Symbol, WaveField}()
    )
end

# Implement interface
function update!(opt::AdamWave, params::Dict{Symbol, WaveField}, 
                 gradients::Dict{Symbol, WaveField})
    # Implementation
end
```

## Naming Conventions

### Types

**PascalCase** for all types:

```julia
# Structs
struct WaveField end
struct HarmonicEncoder end
struct FlowerOfLifePotential end

# Abstract types
abstract type WaveOptimizer end
abstract type SacredGeometryPotential end
```

### Functions

**snake_case** for functions:

```julia
function create_wave_field(grid, psi, time, hbar, mass)
    # implementation
end

function compute_free_energy(simulator)
    # implementation
end

function extract_eigenstates(extractor, wave_field)
    # implementation
end
```

### Methods with Mutation

Functions that modify arguments end with `!`:

```julia
function evolve!(simulator::MorphogeneticSimulator, steps::Int)
    # Modifies simulator.wave_field in place
end

function update!(optimizer::WaveOptimizer, params, gradients)
    # Modifies params in place
end
```

### Constants

**SCREAMING_SNAKE_CASE** for constants:

```julia
const DEFAULT_HBAR = 1.0
const DEFAULT_MASS = 1.0
const MAX_ITERATIONS = 1000
```

### Private Functions

Prefix with underscore (convention, not enforced):

```julia
function _internal_helper(x)
    # Private helper function
end
```

## Type Definition Patterns

### Immutable Structs (Default)

Most types should be immutable:

```julia
struct WaveField
    spatial_grid::Grid
    psi::Array{ComplexF64}
    time::Float64
    hbar::Float64
    mass::Float64
end
```

**Rationale**: Immutability prevents accidental modification and enables compiler optimizations.

### Mutable Structs (When Needed)

Use `mutable struct` only when necessary:

```julia
mutable struct MorphogeneticSimulator
    wave_field::WaveField  # Will be replaced during evolution
    potential::SacredGeometryPotential
    time_step::Float64
    nonlinearity::Float64
end
```

**When to use**: When the struct needs to be modified in place (e.g., simulators, optimizers with internal state).

### Parametric Types

Use type parameters for flexibility:

```julia
struct FrequencySpectrum{T<:Real}
    frequencies::Vector{T}
    amplitudes::Vector{T}
    phases::Vector{T}
    metadata::Dict{Symbol, Any}
end
```

**Rationale**: Allows Float32 or Float64 without code duplication.

### Type Aliases

For clarity:

```julia
const Position = SVector{3, Float64}
const WaveVector = SVector{3, Float64}
const ComplexWaveFunction = Array{ComplexF64}
```

## Function Signature Patterns

### Basic Function

```julia
"""
    function_name(arg1::Type1, arg2::Type2) -> ReturnType

Brief description of what the function does.

# Arguments
- `arg1::Type1`: Description of arg1
- `arg2::Type2`: Description of arg2

# Returns
- `ReturnType`: Description of return value

# Examples
```julia
result = function_name(value1, value2)
```
"""
function function_name(arg1::Type1, arg2::Type2)::ReturnType
    # Implementation
    return result
end
```

### Keyword Arguments

```julia
function create_encoder(;
    frequency_range::Tuple{Float64, Float64} = (0.0, 10.0),
    resolution::Int = 256,
    window_function::Function = hamming
)
    HarmonicEncoder(frequency_range, resolution, window_function)
end
```

**Rationale**: Keyword arguments make calls more readable and allow defaults.

### Multiple Dispatch

Specialize on types:

```julia
# Generic fallback
function evaluate_potential(pot::SacredGeometryPotential, position::Vector{Float64})
    error("evaluate_potential not implemented for $(typeof(pot))")
end

# Specific implementation for Flower of Life
function evaluate_potential(pot::FlowerOfLifePotential, position::Vector{Float64})
    # Flower of Life specific calculation
end

# Specific implementation for Metatron's Cube
function evaluate_potential(pot::MetatronsCubePotential, position::Vector{Float64})
    # Metatron's Cube specific calculation
end
```

### Varargs and Splatting

```julia
# Variable number of arguments
function combine_wave_fields(fields::WaveField...)
    # Combine multiple wave fields
end

# Usage
combined = combine_wave_fields(field1, field2, field3)
```

## Documentation Patterns

### Docstring Template

```julia
"""
    TypeOrFunctionName(args...) -> ReturnType

One-line summary of what this does.

More detailed description if needed. Can span multiple paragraphs.
Explain the purpose, behavior, and any important details.

# Arguments
- `arg1::Type1`: Description of first argument
- `arg2::Type2`: Description of second argument
- `kwarg1::Type3=default`: Description of keyword argument

# Returns
- `ReturnType`: Description of what is returned

# Throws
- `ErrorType`: When this error occurs

# Examples
```julia
# Basic usage
result = function_name(arg1, arg2)

# With keyword arguments
result = function_name(arg1, arg2; kwarg1=value)
```

# Notes
Additional notes, warnings, or implementation details.

# See Also
- [`related_function`](@ref): Related functionality
- [`RelatedType`](@ref): Related type
"""
```

### Type Documentation

```julia
"""
    WaveField

Represents a complex-valued wave function on a spatial grid.

The wave field Ψ(r,t) is the fundamental computational state in Aetheria.
It evolves according to the modified Schrödinger equation.

# Fields
- `spatial_grid::Grid`: Spatial discretization
- `psi::Array{ComplexF64}`: Complex wave function values
- `time::Float64`: Current time
- `hbar::Float64`: Reduced Planck constant
- `mass::Float64`: Effective mass parameter

# Examples
```julia
grid = Grid((64, 64, 64), (0.1, 0.1, 0.1), (0.0, 0.0, 0.0))
psi = zeros(ComplexF64, 64, 64, 64)
wave_field = WaveField(grid, psi, 0.0, 1.0, 1.0)
```

# See Also
- [`Grid`](@ref): Spatial grid structure
- [`create_wave_field`](@ref): Constructor function
"""
struct WaveField
    spatial_grid::Grid
    psi::Array{ComplexF64}
    time::Float64
    hbar::Float64
    mass::Float64
end
```

### Module Documentation

```julia
"""
    Aetheria.Encoding

Data encoding and frequency domain transformations.

This module provides functionality for converting input data to frequency
domain representations (harmonic encoding) and back to data domain (decoding).

# Exports
- `HarmonicEncoder`: Main encoder type
- `FrequencySpectrum`: Frequency domain representation
- `encode`: Convert data to frequency domain
- `decode`: Convert frequency domain back to data

# Examples
```julia
using Aetheria.Encoding

encoder = HarmonicEncoder(frequency_range=(0.0, 10.0), resolution=256)
spectrum = encode(encoder, input_data)
reconstructed = decode(encoder, spectrum)
```
"""
module Encoding
    # module contents
end
```

## Code Organization Patterns

### Module Structure

```julia
module Aetheria

# Version
const VERSION = v"0.1.0"

# Exports (grouped by functionality)
# Core
export WaveField, Grid

# Encoding
export HarmonicEncoder, FrequencySpectrum
export encode, decode

# Simulation
export MorphogeneticSimulator
export evolve!, compute_free_energy

# ... (more exports)

# Includes (in dependency order)
include("Core/WaveField.jl")
include("Core/Grid.jl")

include("Encoding/FrequencySpectrum.jl")
include("Encoding/HarmonicEncoder.jl")

# ... (more includes)

end # module
```

### File Header

```julia
# src/Encoding/HarmonicEncoder.jl
#
# Harmonic encoder for converting data to frequency domain.
# Part of Aetheria.jl wave-based computation library.

using FFTW
using DSP

# Type definition
struct HarmonicEncoder
    # ...
end

# Methods
function encode(encoder::HarmonicEncoder, data::AbstractArray)
    # ...
end
```

## Error Handling Patterns

### Argument Validation

```julia
function create_wave_field(grid::Grid, psi::Array{ComplexF64}, 
                           time::Float64, hbar::Float64, mass::Float64)
    # Validate arguments
    size(psi) == grid.dimensions || 
        throw(ArgumentError("psi dimensions must match grid dimensions"))
    
    time >= 0.0 || 
        throw(ArgumentError("time must be non-negative"))
    
    hbar > 0.0 || 
        throw(ArgumentError("hbar must be positive"))
    
    mass > 0.0 || 
        throw(ArgumentError("mass must be positive"))
    
    # Create wave field
    return WaveField(grid, psi, time, hbar, mass)
end
```

### Custom Exceptions

```julia
# Define custom exception types
struct WaveFieldError <: Exception
    msg::String
end

struct ConvergenceError <: Exception
    msg::String
    iterations::Int
end

# Usage
function evolve!(sim::MorphogeneticSimulator, steps::Int)
    if any(isnan, sim.wave_field.psi)
        throw(WaveFieldError("Wave field contains NaN values"))
    end
    
    # Evolution logic
end
```

### Try-Catch Blocks

```julia
function safe_operation(x)
    try
        result = risky_computation(x)
        return result
    catch e
        if e isa DomainError
            @warn "Domain error occurred, using fallback"
            return fallback_value
        else
            rethrow(e)  # Re-throw unexpected errors
        end
    end
end
```

## Testing Patterns

### Unit Test Structure

```julia
# test/encoding_tests.jl
using Test
using Aetheria

@testset "Encoding" begin
    @testset "HarmonicEncoder Construction" begin
        encoder = HarmonicEncoder(
            frequency_range=(0.0, 10.0),
            resolution=256
        )
        
        @test encoder.frequency_range == (0.0, 10.0)
        @test encoder.resolution == 256
    end
    
    @testset "Encode-Decode Round-Trip" begin
        encoder = HarmonicEncoder(
            frequency_range=(0.0, 10.0),
            resolution=256
        )
        
        data = randn(100)
        spectrum = encode(encoder, data)
        reconstructed = decode(encoder, spectrum)
        
        @test isapprox(data, reconstructed, rtol=1e-6)
    end
end
```

### Property-Based Test Structure

```julia
# test/property_tests.jl
using Test
using PropCheck
using Aetheria

@testset "Property Tests" begin
    @testset "Encoder Round-Trip Property" begin
        @check function encoder_roundtrip(data::Vector{Float64})
            # Ensure data is reasonable
            length(data) > 0 || return true
            
            encoder = HarmonicEncoder(
                frequency_range=(0.0, 10.0),
                resolution=256
            )
            
            spectrum = encode(encoder, data)
            reconstructed = decode(encoder, spectrum)
            
            # Property: round-trip should preserve data
            return isapprox(data, reconstructed, rtol=1e-6)
        end
    end
end
```

## Performance Patterns

### Type Stability

Ensure functions return consistent types:

```julia
# Good: Type-stable
function compute_energy(wave_field::WaveField)::Float64
    # Always returns Float64
    return sum(abs2, wave_field.psi)
end

# Bad: Type-unstable
function compute_energy(wave_field::WaveField)
    if some_condition
        return sum(abs2, wave_field.psi)  # Float64
    else
        return 0  # Int
    end
end
```

### Avoid Global Variables

```julia
# Bad: Global variable
global_counter = 0

function increment_counter()
    global global_counter
    global_counter += 1
end

# Good: Pass as argument
function increment_counter(counter::Int)
    return counter + 1
end
```

### Pre-allocate Arrays

```julia
# Good: Pre-allocate
function compute_gradients!(gradients::Array{Float64}, params::Array{Float64})
    # Modify gradients in place
    for i in eachindex(gradients, params)
        gradients[i] = 2 * params[i]  # Example
    end
end

# Usage
gradients = zeros(size(params))
compute_gradients!(gradients, params)
```

### Use Views Instead of Copies

```julia
# Bad: Creates copy
function process_subarray(arr::Array{Float64})
    sub = arr[1:10]  # Copy
    return sum(sub)
end

# Good: Uses view
function process_subarray(arr::Array{Float64})
    sub = @view arr[1:10]  # View (no copy)
    return sum(sub)
end
```

## Code Style Guidelines

### Indentation

- Use 4 spaces (not tabs)
- Align continuation lines

```julia
function long_function_name(arg1::Type1, arg2::Type2,
                            arg3::Type3, arg4::Type4)
    # Function body
end
```

### Line Length

- Aim for 80-100 characters per line
- Break long lines at logical points

```julia
# Good
result = compute_complex_operation(
    argument1,
    argument2,
    argument3
)

# Avoid
result = compute_complex_operation(argument1, argument2, argument3, argument4, argument5)
```

### Whitespace

```julia
# Around operators
x = a + b
y = c * d

# After commas
func(a, b, c)

# Not inside parentheses
func(x)  # Good
func( x )  # Bad

# Blank lines between functions
function func1()
    # ...
end

function func2()
    # ...
end
```

### Comments

```julia
# Single-line comments for brief explanations
x = compute_value()  # Compute the value

# Multi-line comments for longer explanations
#=
This is a longer explanation that spans multiple lines.
It provides context for the following code block.
=#
```

## Summary

Aetheria.jl code structure follows these principles:
- **One type per file**: Easy to find code
- **PascalCase for types**, **snake_case for functions**
- **Immutable by default**: Use `mutable struct` only when needed
- **Comprehensive docstrings**: Document all public API
- **Multiple dispatch**: Specialize on types
- **Type stability**: Ensure consistent return types
- **Pre-allocate**: Avoid unnecessary allocations
- **Test thoroughly**: Unit tests + property tests

Following these patterns ensures consistent, maintainable, and performant code throughout the Aetheria.jl library.
