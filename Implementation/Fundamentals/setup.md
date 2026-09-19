# Initial Setup Guide

## Overview

This guide walks through the initial setup of the Aetheria.jl project, from creating the Julia package structure to configuring dependencies and establishing the development environment. Follow these steps to get started with implementing Aetheria.jl.

## Prerequisites

### Required Software

1. **Julia 1.9 or later**
   - Download from: https://julialang.org/downloads/
   - Verify installation: `julia --version`

2. **Git**
   - For version control
   - Verify installation: `git --version`

3. **Text Editor or IDE**
   - Recommended: VS Code with Julia extension
   - Alternatives: Vim, Emacs, Atom, etc.

### Recommended Tools

1. **Julia VS Code Extension**
   - Provides syntax highlighting, REPL integration, debugging
   - Install from VS Code marketplace

2. **Revise.jl** (for development)
   - Automatically reloads code changes
   - Install: `julia -e 'using Pkg; Pkg.add("Revise")'`

## Step 1: Create Package Structure

### Initialize Julia Package

```bash
# Create project directory
mkdir Aetheria.jl
cd Aetheria.jl

# Initialize as Julia package
julia --project=. -e 'using Pkg; Pkg.generate(".")'
```

This creates:
- `Project.toml`: Package metadata and dependencies
- `src/Aetheria.jl`: Main module file

### Create Directory Structure

```bash
# Source directories
mkdir -p src/Core
mkdir -p src/Encoding
mkdir -p src/Simulation
mkdir -p src/Potentials
mkdir -p src/Extraction
mkdir -p src/Optimization
mkdir -p src/Loss
mkdir -p src/Activation
mkdir -p src/Networks
mkdir -p src/Training
mkdir -p src/Inference
mkdir -p src/Data
mkdir -p src/Persistence
mkdir -p src/WaveProtocols

# Test directory
mkdir -p test

# Documentation directory
mkdir -p docs/src/tutorials
mkdir -p docs/src/api
mkdir -p docs/src/theory

# Examples directory
mkdir -p examples

# Implementation guides (already exists from spec creation)
# mkdir -p Implementation/Mathematics
# mkdir -p Implementation/Structure
# mkdir -p Implementation/Fundamentals
```

### Create Essential Files

```bash
# Test runner
touch test/runtests.jl

# Documentation build script
touch docs/make.jl

# README
touch README.md

# License
touch LICENSE

# Git ignore
touch .gitignore
```

## Step 2: Configure Project.toml

Edit `Project.toml` to include package metadata:

```toml
name = "Aetheria"
uuid = "GENERATE-NEW-UUID"  # Generate using: julia -e 'using UUIDs; println(uuid4())'
authors = ["Your Name <your.email@example.com>"]
version = "0.1.0"

[deps]
# Essential dependencies
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

# Recommended dependencies
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
VideoIO = "d6d074c3-1acf-5d4c-9a43-ef38773959a2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"

# Optional dependencies (for development)
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"

[compat]
julia = "1.9"
FFTW = "1.7"
DifferentialEquations = "7.0"
StaticArrays = "1.6"
DSP = "0.7"
FileIO = "1.16"
VideoIO = "1.0"
Images = "0.25"
ProgressMeter = "1.7"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
PropCheck = "0a2f5f00-7b9f-11e9-2d3d-8d5c7b1e4f4e"

[targets]
test = ["Test", "PropCheck"]
```

**Generate UUID**:
```bash
julia -e 'using UUIDs; println(uuid4())'
```

Copy the generated UUID into `Project.toml`.

## Step 3: Initialize Git Repository

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Julia
*.jl.cov
*.jl.*.cov
*.jl.mem
/Manifest.toml
/docs/build/
/docs/site/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Temporary files
*.tmp
*.log
EOF

# Initial commit
git add .
git commit -m "Initial Aetheria.jl package structure"
```

## Step 4: Install Dependencies

```bash
# Activate project environment
julia --project=.

# In Julia REPL:
using Pkg
Pkg.instantiate()  # Install all dependencies from Project.toml
```

This creates `Manifest.toml` with exact dependency versions.

## Step 5: Create Main Module File

Edit `src/Aetheria.jl`:

```julia
module Aetheria

# Version information
const VERSION = v"0.1.0"

# Welcome message
function __init__()
    @info "Aetheria.jl v$VERSION - Wave-Based Computation Library"
end

# Placeholder exports (will be populated as modules are implemented)
export WaveField, Grid
export HarmonicEncoder, encode, decode
export MorphogeneticSimulator, evolve!
export FlowerOfLifePotential, MetatronsCubePotential
export CymaticExtractor, extract_eigenstates
export AdamWave, update!
export SOVNetwork, forward
export train!, predict

# Module includes (will be added as files are created)
# include("Core/WaveField.jl")
# include("Core/Grid.jl")
# ... (more includes as implementation progresses)

end # module Aetheria
```

## Step 6: Create Basic Test File

Edit `test/runtests.jl`:

```julia
using Test
using Aetheria

@testset "Aetheria.jl" begin
    @testset "Package Loading" begin
        @test isdefined(Aetheria, :VERSION)
        @test Aetheria.VERSION == v"0.1.0"
    end
    
    # More test sets will be added as implementation progresses
    # include("core_tests.jl")
    # include("encoding_tests.jl")
    # ... (more test files)
end
```

## Step 7: Verify Setup

### Test Package Loading

```bash
julia --project=. -e 'using Aetheria'
```

Should output:
```
[ Info: Aetheria.jl v0.1.0 - Wave-Based Computation Library
```

### Run Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Should pass the basic package loading test.

### Check Dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

Should list all installed dependencies.

## Step 8: Setup Documentation

### Install Documenter.jl

```bash
julia --project=. -e 'using Pkg; Pkg.add("Documenter")'
```

### Create docs/make.jl

```julia
using Documenter
using Aetheria

makedocs(
    sitename = "Aetheria.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Aetheria],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/getting-started.md",
            "tutorials/basic-training.md",
        ],
        "API Reference" => [
            "api/core.md",
            "api/encoding.md",
            "api/simulation.md",
        ],
        "Theory" => [
            "theory/wave-dynamics.md",
            "theory/sacred-geometry.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/username/Aetheria.jl.git",
)
```

### Create docs/src/index.md

```markdown
# Aetheria.jl

Wave-Based Computation Library

## Overview

Aetheria.jl is a revolutionary Julia library that implements wave-based computation using harmonic interference, signal processing, and frequency-based operations. Unlike traditional neural networks that rely on discrete matrix calculations, Aetheria.jl leverages Self-Organizing Vacuum (SOV) dynamics, morphogenetic wave interference, and sacred geometry principles.

## Features

- **Wave-Based Computation**: Continuous wave dynamics instead of discrete matrix operations
- **Sacred Geometry Potentials**: Flower of Life and Metatron's Cube patterns
- **Cymatic Extraction**: Results extracted from standing wave patterns
- **Novel Model Storage**: MP4/GIF format for visual frequency patterns
- **Pure Julia**: No Python, CUDA, or GPU dependencies

## Installation

```julia
using Pkg
Pkg.add("Aetheria")
```

## Quick Start

```julia
using Aetheria

# Create a simple SOV network
network = SOVNetwork(...)

# Train the network
train!(network, data, optimizer)

# Make predictions
predictions = predict(network, test_data)
```

## Documentation Structure

- **Tutorials**: Step-by-step guides for common tasks
- **API Reference**: Detailed function and type documentation
- **Theory**: Mathematical foundations and concepts
```

## Step 9: Create README.md

```markdown
# Aetheria.jl

[![Build Status](https://github.com/username/Aetheria.jl/workflows/CI/badge.svg)](https://github.com/username/Aetheria.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/Aetheria.jl/stable)

Wave-Based Computation Library for Julia

## Overview

Aetheria.jl implements Self-Organizing Vacuum (SOV) networks using continuous wave dynamics, harmonic interference, and sacred geometry principles. It replaces traditional discrete matrix operations with wave-based computation.

## Features

- 🌊 Wave-based computation (no CPU/GPU matrix operations)
- 🔷 Sacred geometry potentials (Flower of Life, Metatron's Cube)
- 🎵 Cymatic pattern extraction
- 🎬 Novel MP4/GIF model storage format
- 🔄 GGUF format compatibility
- 💎 Pure Julia implementation

## Installation

```julia
using Pkg
Pkg.add("Aetheria")
```

## Quick Example

```julia
using Aetheria

# Create harmonic encoder
encoder = HarmonicEncoder(frequency_range=(0.0, 10.0), resolution=256)

# Create SOV network
network = SOVNetwork(
    layers = [SOVDense(10, 20), SOVDense(20, 10)],
    encoder = encoder,
    extractor = CymaticExtractor()
)

# Train
train!(network, training_data, AdamWave(learning_rate=0.01))

# Predict
predictions = predict(network, test_data)
```

## Documentation

See the [documentation](https://username.github.io/Aetheria.jl/stable) for detailed guides and API reference.

## Citation

If you use Aetheria.jl in your research, please cite:

```bibtex
@software{aetheria2024,
  title = {Aetheria.jl: Wave-Based Computation Library},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/username/Aetheria.jl}
}
```

## License

MIT License - see LICENSE file for details.
```

## Step 10: Development Workflow Setup

### Configure Revise.jl (Optional but Recommended)

Create `~/.julia/config/startup.jl`:

```julia
# Auto-load Revise in development
try
    using Revise
catch e
    @warn "Revise.jl not available"
end
```

This automatically reloads code changes without restarting Julia.

### VS Code Configuration (Optional)

Create `.vscode/settings.json`:

```json
{
    "julia.environmentPath": "${workspaceFolder}",
    "julia.enableTelemetry": false,
    "files.associations": {
        "*.jl": "julia"
    }
}
```

## Setup Checklist

Use this checklist to verify your setup:

- [ ] Julia 1.9+ installed and working
- [ ] Git initialized with .gitignore
- [ ] Package structure created (src/, test/, docs/, examples/)
- [ ] Project.toml configured with dependencies
- [ ] UUID generated and added to Project.toml
- [ ] Dependencies installed (Pkg.instantiate())
- [ ] Main module file (src/Aetheria.jl) created
- [ ] Basic test file (test/runtests.jl) created
- [ ] Package loads without errors (using Aetheria)
- [ ] Tests pass (Pkg.test())
- [ ] Documentation structure created (docs/)
- [ ] README.md written
- [ ] LICENSE file added
- [ ] Initial git commit made

## Next Steps

After completing setup:

1. **Phase 1 (Current)**: Continue creating foundation documentation
   - Mathematical foundations (already in Implementation/Mathematics/)
   - Project structure (already in Implementation/Structure/)
   - Code structure guide (next task)

2. **Phase 2**: Implement wave protocol testing
   - Create wave protocol testing framework
   - Test 144+ algorithms
   - Identify winning protocol

3. **Phase 3**: Implement SOV components
   - Follow implementation guides in Implementation/Components/
   - Build core functionality
   - Add tests for each component

4. **Phase 4**: Implement persistence layer
   - MP4/GIF storage
   - GGUF conversion

5. **Phase 5**: Documentation and publishing
   - Complete API documentation
   - Write tutorials
   - Publish package

## Troubleshooting

### Package Won't Load

**Problem**: `ERROR: LoadError: ArgumentError: Package Aetheria not found`

**Solution**: Make sure you're in the project directory and using `--project=.`:
```bash
cd Aetheria.jl
julia --project=.
```

### Dependency Installation Fails

**Problem**: `ERROR: Unsatisfiable requirements detected for package X`

**Solution**: Check Julia version compatibility in Project.toml [compat] section.

### Tests Fail

**Problem**: Tests fail with missing dependencies

**Solution**: Install test dependencies:
```bash
julia --project=. -e 'using Pkg; Pkg.add("Test"); Pkg.add("PropCheck")'
```

### Git Issues

**Problem**: Git not tracking files

**Solution**: Check .gitignore isn't too aggressive. Ensure files are added:
```bash
git status
git add <missing-files>
```

## Resources

- **Julia Documentation**: https://docs.julialang.org/
- **Pkg.jl Documentation**: https://pkgdocs.julialang.org/
- **Documenter.jl**: https://documenter.juliadocs.org/
- **Julia Discourse**: https://discourse.julialang.org/

## Summary

You now have a complete Aetheria.jl package structure with:
- Proper Julia package layout
- Configured dependencies
- Git version control
- Basic tests
- Documentation framework
- Development workflow

Ready to start implementing the wave-based computation library!
