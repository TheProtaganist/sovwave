# Requirements — SOVWave / Aetheria.jl v1 (Initial Iteration)

> Source of truth for DONE. Editable. Every task in `Tasks.md` MUST cite one or more REQ IDs. No REQ, no work.
> Ground truth: `agenda/agenda.md` (the IDEA to implement: 5 components §4, theory §5, stack §6, results honesty §7, zero-ALU direction §8) + `Instructions.md` (4-phase roadmap: fundamentals → wave interface → SOV rise → mp4/gguf compat).
> `Implementation/` math/structure/fundamentals docs are EXPLANATIONS — reference only, never the product. Math details live there; product scope lives here.

## 0. Goal (from agenda + Instructions)
Implement our OWN wave-computation idea inspired by the agenda paper — a from-scratch Julia library like Flux but with no CUDA, no discrete CPU matmuls, stripping out transistors for freer wave compute — plus JS demos in `web/apps/` mirroring each agenda component, with spec-based tests and Alpha-Evolve winners.

Project-only languages: Julia. No C++/C# in v1 (agenda §6 trifecta deferred; Instructions says Julia only).

## 1. Functional Requirements (agenda §4 five components + Instructions phases)

### REQ-BOOT — Julia package skeleton
- MUST: root-package layout (`Project.toml` at repo root, `name=Aetheria`, Julia compat 1.9+ tested on 1.10.2), `src/Aetheria.jl` loads with `julia --project=. -e 'using Aetheria'`, `test/runtests.jl` green.

### REQ-ENC — Harmonic Encoder (agenda §4.1)
- MUST: map input vector `X ∈ R^N` to boundary driving frequencies `Omega(r,t)` (spectral densities, NOT tensors). `encode/decode` round-trip.
- MUST: `src/Encoding/{FrequencySpectrum,HarmonicEncoder}.jl`.

### REQ-WELL — Sacred Geometry Potential Wells (agenda §4.2)
- MUST: pre-configured spatial attractors Flower of Life + Metatron's Cube guiding self-organization and dictating interference nodes. C6 symmetry `V(R60*r)==V(r)` to 1e-10.
- MUST: `src/Potentials/{SacredGeometry,FlowerOfLife,MetatronsCube}.jl`.

### REQ-SIM — Morphogenetic Wave Simulator (agenda §4.3, zero-ALU inference)
- MUST: propagate encoder waveforms through wells instead of iterating weight layers. Governed by agenda §5.1 Schrodinger + §5.3 free-energy relaxation (NOT backprop). Equilibrium `dF/dPsi* = 0` = inference done.
- MUST: `src/Simulation/{MorphogeneticSimulator,TimeEvolution,EnergyFunctionals}.jl` + `src/Core/{Grid,WaveField,Constants}.jl`.

### REQ-EXT — Cymatic Eigen-State Extraction (agenda §4.4)
- MUST: decode relaxed standing-wave topologies back into digital inferences (`EigenState`: nodes/antinodes/spectrum/label).
- MUST: `src/Extraction/{EigenState,CymaticExtractor}.jl`.

### REQ-HW — Direct Frequency Hardware API, simulated (agenda §4.5)
- MUST v1: simulated translation layer only (no real SDR/FPGA): `WaveProtocol` abstract + Audio/EMF/Radio/SignalProcessing sim variants + `TestRunner` + `ProtocolLogger` writing `logs/wave-protocols/`. Real broadcast deferred to v2.
- MUST: `src/WaveProtocols/*.jl`. This satisfies Instructions Phase 2 (test many wave algorithms, log results, pick winner protocol).

### REQ-SOV — SOV networks replace MLPs/KANs (Instructions Phase 3)
- MUST: basic network library then advanced settings: `SOVLayer abstract + SOVDense + SOVNetwork`, `WaveOptimizer abstract + GradientFlow + AdamWave`, `WaveLoss abstract + FrequencyDomainMSE + PhaseCoherenceLoss + FreeEnergyLoss`, `WaveActivation + Sigmoid/ReLU/Tanh`, `TrainingLoop + Validation + EarlyStopping`, `SingleSample + BatchInference`, `DataLoader + Preprocessing + Augmentation` + dataset support.
- v1 minimal but every API exists + tested.

### REQ-STORE — Model format: mp4 of gifs + gguf (Instructions Phase 4)
- MUST v1: `ModelMetadata` save/load round-trip; `MP4Storage` gif-sequence↔mp4↔zip design docced, stub throws `not-implemented v2` if deps missing; `GGUFConverter` stub API for mp4→gguf→quant path.
- MUST: `src/Persistence/{ModelMetadata,MP4Storage,GGUFConverter}.jl`.

### REQ-DEMO — web/apps JS mirrors (one per agenda component)
- MUST: static no-build vanilla HTML+Canvas+JS (Node 10 / npm broken on WSL1).
- MUST: `web/apps/index.html` gallery + `01-encoder, 02-wells, 03-simulator, 04-extraction, 05-hardware-api, 06-sov-network, 07-persistence`. Each mirrors Julia constants, shows expected-vs-actual + console asserts.

### REQ-TEST — spec-based tests
- MUST: Julia `Test.jl` per-module `test/*_tests.jl`. Minimum: encode round-trip, C6 symmetry, F-monotonic to equilibrium, eigenstate decode on synthetic modes, protocol log written, metadata round-trip.
- MUST: JS console asserts (no Jest on Node 10).

## 2. Non-functional
- REQ-STYLE: follow `Structure.md`.
- REQ-REPRO: fixed seeds; theory logs `logs/theory/<ALG>/`, protocol logs `logs/wave-protocols/`.
- REQ-NODEPS-v1: zero-deps first; promote FFTW/StaticArrays/DifferentialEquations/DSP/FileIO/VideoIO/Images only on Theory win. Never Python/CUDA/Flux/PyTorch.
- REQ-PAPER-FIDELITY: agenda §7 honesty — label simulated-vs-physical claims; §8 zero-ALU is direction, v1 simulates it.

## 3. Out of scope v1
Real SDR/FPGA broadcast, C++/C# code, full mp4 encode + quantization, 1.5B-param comparisons, ADC hardware.

## 4. Acceptance (finished v1)
`using Aetheria` OK; `Pkg.test()` pass; 7 demos open offline; wave-protocol winner logged; SOV net trains tiny dataset end-to-end (encode→wells→simulate→extract); metadata round-trip OK; `Tasks.md` checked; `Winners.md` ≥1 winner per Theory ALG.

## 5. Edit policy
Append `REQ-XXX`, never renumber. Mark superseded + link task.

