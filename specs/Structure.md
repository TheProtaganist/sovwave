# Structure — SOVWave / Aetheria.jl layout + style bible

> Enforceable. Ground truth for WHAT to build is `agenda/agenda.md` §4 (5 components) + `Instructions.md` phases. `Implementation/Structure/*` + `Fundamentals/*` are explanation/reference for HOW — consulted, not copied. If code disagrees with THIS file, code changes.

## 1. Root layout (root-package, no nested Aetheria.jl/)
```
sovwave/
  Project.toml / Manifest.toml
  src/Aetheria.jl
  src/Core/ Grid.jl WaveField.jl Constants.jl
  src/Encoding/ FrequencySpectrum.jl HarmonicEncoder.jl
  src/Simulation/ MorphogeneticSimulator.jl TimeEvolution.jl EnergyFunctionals.jl
  src/Potentials/ SacredGeometry.jl FlowerOfLife.jl MetatronsCube.jl
  src/Extraction/ EigenState.jl CymaticExtractor.jl
  src/Optimization/ WaveOptimizer.jl GradientFlow.jl AdamWave.jl
  src/Loss/ WaveLoss.jl FrequencyDomainMSE.jl PhaseCoherenceLoss.jl FreeEnergyLoss.jl
  src/Activation/ WaveActivation.jl WaveSigmoid.jl WaveReLU.jl WaveTanh.jl
  src/Networks/ SOVLayer.jl SOVDense.jl SOVNetwork.jl
  src/Training/ TrainingLoop.jl Validation.jl EarlyStopping.jl
  src/Inference/ SingleSample.jl BatchInference.jl
  src/Data/ DataLoader.jl Preprocessing.jl Augmentation.jl
  src/Persistence/ ModelMetadata.jl MP4Storage.jl GGUFConverter.jl
  src/WaveProtocols/ WaveProtocol.jl AudioProtocols.jl EMFProtocols.jl RadioProtocols.jl SignalProcessing.jl TestRunner.jl ProtocolLogger.jl (sim only v1)
  test/runtests.jl + *_tests.jl (mirror src)
  web/apps/ index.html _shared/ 01-encoder/ 02-wells/ 03-simulator/ 04-extraction/ 05-hardware-api/ 06-sov-network/ 07-persistence/
  specs/ Requirements.md Structure.md Tasks.md Theory.md Winners.md Guide.md
  docs/ examples/ logs/theory/ logs/wave-protocols/ Implementation/ agenda/
```

## 2. Folder roles + import rules
| Dir | Purpose (agenda §) | May import | Must NOT import |
|---|---|---|---|
| Core | Grid, WaveField, consts (supports §5.1) | StaticArrays only | anything internal |
| Encoding | X→Omega spectra (agenda §4.1) | Core | Simulation, Networks |
| Potentials | FoL/MC attractors (agenda §4.2) | Core | Encoding |
| Simulation | zero-ALU propagation + F relax (agenda §4.3, §5.1/§5.3) | Core, Potentials | Networks, Training |
| Extraction | eigenstate decode (agenda §4.4) | Core, Simulation | Training |
| Optimization/Loss/Activation | free-energy descent, cost, native g|Psi|² (Instructions Phase 3 advanced) | Core | Inference |
| Networks | SOV layers vs MLP/KAN (Instructions Phase 3 basic) | all above exc. Training | Training, Persistence |
| Training/Inference/Data | loops, API, loaders + datasets (Phase 3) | Networks | — |
| Persistence | mp4-of-gifs + gguf + metadata (Instructions Phase 4) | Core, Networks | — |
| WaveProtocols | SIMULATED hw API + winner harness (agenda §4.5 sim, Instructions Phase 2) | Core | — |

No circular deps. `Aetheria.jl` includes in dependency order Core->Encoding->Potentials->Simulation->Extraction->Optimization->Loss->Activation->Networks->Training->Inference->Data->Persistence->WaveProtocols.

## 3. File rules
- One public type per file, filename = TypeName. Abstract types own file (`WaveOptimizer.jl`).
- Every public type/func has docstring: purpose, fields/args, math ref, example.
- Immutable `struct` default; `mutable struct` only with justification comment.

## 4. Coding style
- Types `PascalCase` (`WaveField`), funcs `snake_case` (`create_wave_field`), mutating `!` (`evolve!`).
- 4 spaces, 80-100 cols, `x = a + b`, `func(a,b)` spacing, blank line between funcs.
- Type-stable returns (`::Float64`), no globals, pre-allocate, `@view` not copies.
- Tests: `@testset` hierarchy mirroring src; property tests separate `property_tests.jl`.

## 5. Julia <-> JS contract
- Julia Float64 ground truth tol 1e-6 (C6: 1e-10). JS Float32 vis tol 1e-3.
- Shared constants duplicated in `web/apps/_shared/wave-utils.js` with comment `// MIRROR src/Core/Constants.jl`.
- Each demo: sliders for params (hbar,m,g,V0,a), canvas + expected-vs-actual readout + `console.assert`.

## 6. Dependencies v1
Zero-deps first. Promote only via Theory win: FFTW (FFT), StaticArrays (SVector), DifferentialEquations (stiff evolve), DSP/FileIO/VideoIO/Images/ProgressMeter optional. Never Python/CUDA/Flux/PyTorch.

## 7. Change rule
Structure change => update this file + `Tasks.md` + module docstring same commit.
