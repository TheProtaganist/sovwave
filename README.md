# 🌊 Sovwave.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blue.svg)](https://julialang.org)
[![Tuning](https://img.shields.io/badge/Tuning-432Hz%20Sacred%20Harmonics-emerald.svg)](https://en.wikipedia.org/wiki/Concert_pitch)
[![Architecture](https://img.shields.io/badge/Architecture-Pure%20Wave%20Computing-purple.svg)](#core-architecture)
[![Serialization](https://img.shields.io/badge/Model%20Format-Dual--Stream%20MKV%20Video-red.svg)](#video-model-serialization)

> **Pure Wave Computing, Quantum-Acoustic Deep Learning, and Morphogenetic Intelligence in Julia.**
>
> Zero Matrix Multiplications. Zero Backpropagation. Zero Markov Chains. Models saved as playable, glowing audio-visual video files.

---

## 📖 Table of Contents
1. [Overview & Philosophy](#-overview--philosophy)
2. [Core Architecture](#-core-architecture)
3. [Emergent Video Model Serialization (.mkv & .mp4)](#-emergent-video-model-serialization)
4. [Trained Lattice Audio Sonification](#-trained-lattice-audio-sonification)
5. [Direct Video Inference](#-direct-video-inference)
6. [Complete Configuration Guide (`model_config.yaml`)](#-complete-configuration-guide)
7. [Quick Start & Code Examples](#-quick-start--code-examples)
8. [Tournament Winners (144 Algorithms)](#-tournament-winners-144-algorithms)
9. [Documentation & GitHub Pages](#-documentation--github-pages)
10. [License](#-license)

---

## 🌌 Overview & Philosophy

Modern artificial intelligence is dominated by discrete matrix multiplications ($\text{GEMM}$), backpropagation gradient descent, and high-power GPUs. **Sovwave.jl** is built on a fundamentally different paradigm: **continuous morphogenetic wave computing** grounded in Self-Organizing Vacuum (SOV) dynamics and sacred geometry.

Instead of binary weights and float matrices:
- Data points exist in a **continuous $d$-dimensional spatial manifold** ($d \in [1, 5+]$) with physical wave properties (mass, charge, energy, spin).
- Layers consist of **quantum oscillator lattice nodes** vibrating with dynamic amplitude $A$, phase $\phi$, relative frequency $f$, and fractal scale $\beta_s$ (Golden Ratio $\Phi \approx 1.6180339887$).
- Information propagates through **continuous harmonic interference** and temporal superposition driven by a quantum frequency $\omega$ (432 Hz).
- Optimization seeks the **thermodynamic minimum ground-state energy** ($\mathcal{H}_{\text{wave}} \to 0$) via multi-island genetic evolution.
- Trained models are serialized into **standard video files (`.mkv` and `.mp4`)** that can be opened in any media player (VLC, Windows Media Player, QuickTime) to visually observe the model's inner dynamics and hear its harmonic oscillations.

---

## ⚛️ Core Architecture

```
                    ┌────────────────────────────────────────┐
                    │    Continuous Wave Field Manifold      │
                    │   Points in d-Dimensional Space (x)    │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       WaveLayer 1 (Lattice Nodes)      │
                    │     Amplitudes A, Phases φ, Freqs f    │
                    └───────────────────┬────────────────────┘
                                        │
                         Harmonic Superposition (ω = 432 Hz)
                         Fractal Scaling (β_s = 1.618)
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       WaveLayer 2..L (Interference)    │
                    │      Ginzburg-Landau Field Energy      │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │        Output Wave Prediction          │
                    │     Ground-State Energy Minimization   │
                    └────────────────────────────────────────┘
```

### Mathematical Foundations
1. **Wave Function Formulation**:
   $$\psi_i(t) = A_i \cdot \cos\left(2\pi f_i \cdot \omega t + \phi_i\right) \cdot \exp\left(-\frac{t}{\beta_s}\right)$$
2. **Fractal Scale Coupling**:
   $$\beta_s = \Phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895$$
3. **Phase Tracking**:
   High-precision Neumaier compensated angle summation maintaining phase coherence across all temporal steps without numerical drift.

---

## 🎬 Emergent Video Model Serialization

Traditional machine learning packages save models as opaque binary blobs (`.pt`, `.safetensors`, `.h5`). **Sovwave.jl** serializes models into standard **Matroska (`.mkv`)** and **MPEG-4 (`.mp4`)** video files.

### Dual-Stream Matroska Architecture:
1. **Stream 0:0 (`VISUAL_BRAIN`) — H.264 Video**:
   - Visualizes post-convergence wave dynamics through the model's internal layers.
   - **NO Black Screen**: Training steps $1$ to $n$ are excluded. Frame 1 represents **Step $n$** (the fully trained, coherent model at rest) and blooms immediately into radiant color, evolving through to **Step $n_{\text{final}}$**.
   - **Grand Champion Algorithm**: Uses `potts_model_q_state_domains` (8.7 ns / pt, zero memory allocations) simulating spontaneous discrete symmetry breaking across 3-state Potts clock domains.
   - **Configurable State Colors (CMY Default)**:
     - **State 0**: `FF00FF` (Magenta: `[1.0, 0.0, 1.0]`)
     - **State 1**: `FFFF00` (Yellow: `[1.0, 1.0, 0.0]`)
     - **State 2**: `00FFFF` (Cyan: `[0.0, 1.0, 1.0]`)
   - **Configurable Pixel Square Size**:
     - `pixel_scale: 1` — literal 1-pixel per node (ultra-fine quantum density).
     - `pixel_scale: 2` — sharp, crisp 2×2 pixel squares.
     - `pixel_scale: "auto"` — automatically scales to target screen height without giant blocky pixels.
2. **Stream 0:1 (`MODEL_WEIGHTS`) — FFV1 Lossless Video**:
   - Exact, bit-for-bit lossless mathematical stream containing raw model weights for reconstruction.
3. **Stream 0:2 (`MODEL_AUDIO`) — AAC 192k Presentation Audio**:
   - High-fidelity stereo audio synthesized directly from the model's trained oscillator lattice.

---

## 🎵 Trained Lattice Audio Sonification

Hear your model think! Sovwave models generate continuous sound based on their learned parameters:

- **Carrier Frequency**: Customizable to **432 Hz** (Verdi / Sacred Tuning), **528 Hz** (Solfeggio Transformation), **440 Hz** (Concert Pitch), or any user-defined float.
- **Synthesized Waveforms**:
  - `:physical` — Spring-mass damped harmonic resonator with golden ratio overtones.
  - `:harmonic` — Fundamental plus proportional overtone harmonics.
  - `:sine` — Pure sinusoidal wave.
  - `:triangle` & `:sawtooth` — Continuous geometric waveforms.
  - `:binaural` — Stereo phase offset for brainwave entrainment (e.g. 10 Hz Alpha, 6 Hz Theta).
- **Envelopes**: Full ADSR (`attack`, `decay`, `sustain`, `release`) and exponential decay.

> [!IMPORTANT]
> **Audio is strictly for human presentation and listening.** Inference does **NOT** use or require the audio track; `infer` decodes purely from video weights.

---

## ⚡ Direct Video Inference

Run predictions directly from the `.mkv` video file without separate model weight files:

```julia
using Sovwave
using Sovwave.Audio

# Load and predict directly from MKV video file!
input_vector = rand(32)
prediction = infer("wave_model_brain.mkv", input_vector)

println("Output activations from video: ", prediction)
```

The inference engine decodes the exact FFV1 weights stream, reconstructs the quantum lattice, and propagates inputs through the wave field in microseconds.

---

## ⚙️ Complete Configuration Guide (`model_config.yaml`)

```yaml
# ============================================================================
# Sovwave Model Configuration
# ============================================================================

# Wave Field: Data points exist on a d-dimensional continuous wave surface
field:
  n_points: 128            # Number of spatial lattice points
  properties:              # Properties measured at each point
    - mass
    - charge
    - energy
    - spin
  dimensions: 3            # Manifold dimensionality (1D, 2D, 3D, 5D+)
  distribution: uniform    # Spatial distribution (:uniform, :fibonacci, :random)

# Model Architecture: Learnable wave parameters across l layers and n quantum nodes
model:
  layers: 4                # Number of wave layers
  embed_dims: 32           # Feature embedding dimensions
  nodes: 64                # Quantum lattice nodes per layer
  omega: 432.0             # Harmonic frequency ω (Hz) driving temporal superposition
  beta_s: 1.618033989      # Fractal scaling parameter (Golden Ratio Φ)
  t_frames: 8              # Temporal superposition time frames

# Training Parameters: Evolution-based optimization seeking ground state energy
train:
  batch_size: 16           # Training batch size
  learning_rate: 0.05      # Mutation amplitude scale
  epochs: 50               # Number of training generations
  population_size: 24      # Evolutionary island population size
  elite_fraction: 0.15     # Top fraction preserved without mutation
  mutation_decay: 0.995    # Cooling decay schedule
  energy_target: 0.001     # Target ground-state energy
  sonify: true             # Generate audio feedback during training
  sonify_realtime: false   # Stream directly to speakers in real-time
  audio_sample_rate: 48000 # Audio sampling rate (Hz)

# User-Defined Audio Sonification:
audio:
  carrier_frequency: 432.0    # Carrier frequency (e.g., 432.0, 528.0, 440.0)
  tuning_standard: 432.0      # Reference base tuning in Hz
  waveform: physical          # Waveform (:physical, :harmonic, :sine, :binaural)
  binaural_beat: 10.0         # Binaural frequency offset in Hz (10 Hz Alpha)
  envelope: exponential_decay # Envelope (:exponential_decay, :adsr, :percussive)
  attack: 0.02                # Attack time in seconds
  decay: 0.15                 # Decay time in seconds
  sustain: 0.60               # Sustain level [0.0, 1.0]
  release: 0.25               # Release time in seconds
  harmonic_richness: 1.2      # Harmonic overtone multiplier
  volume: 0.85                # Master audio volume
  pan: 0.0                    # Stereo pan [-1.0 = left, 0.0 = center, +1.0 = right]
  sample_rate: 48000          # PCM sampling rate
  channels: 2                 # 1 = mono, 2 = stereo

# Video Model Serialization Settings:
video:
  render_mode: potts_model_q_state_domains   # Tournament Grand Champion
  pixel_scale: 2               # Pixel square size (1=1px, 2=2px, or "auto")
  target_height: 480           # Target display height in pixels for auto-scale
  fps: 4                       # Post-convergence playback frame rate (Hz)
  frames: 16                   # Number of frames from step n to step n_final
  state_colors:                # Emergent 3-state Potts clock domain colors in RGB
    - [1.0, 0.0, 1.0]          # State 0: FF00FF (Magenta)
    - [1.0, 1.0, 0.0]          # State 1: FFFF00 (Yellow)
    - [0.0, 1.0, 1.0]          # State 2: 00FFFF (Cyan)
```

---

## 🚀 Quick Start & Code Examples

### 1. Installation
```julia
using Pkg
Pkg.add(url="https://github.com/TheProtaganist/sovwave.git")
```

### 2. End-to-End Training & Video Serialization
```julia
using Sovwave
using Sovwave.Audio

# 1. Load configuration
cfg = load_config("model_config.yaml")

# 2. Build continuous field & model lattice
field = create_field(cfg.field)
model = WaveModel(cfg)

# 3. Generate training data
inputs = [rand(cfg.model.embed_dims) for _ in 1:100]
targets = [rand(cfg.model.embed_dims) for _ in 1:100]

# 4. Train via wave evolution seeking ground state
history = train!(model, inputs, targets, cfg.train; audio_cfg=cfg.audio)

# 5. Save model as video brain with visual & audio tracks
save_model(model, "my_wave_model.mkv"; video_cfg=cfg.video, audio_cfg=cfg.audio)
# Creates both 'my_wave_model.mkv' and companion 'my_wave_model.mp4'!

# 6. Run direct video inference
prediction = infer("my_wave_model.mkv", inputs[1])
println("Prediction: ", prediction)
```

### 3. Run the Demonstration
```bash
julia --project=. examples/waveml_demo.jl
```

---

## 🏆 Tournament Winners (144 Algorithms)

To select the visual emergence and computational engines, a **144-algorithm tournament across 12 elimination rounds** was benchmarked under nanosecond timing and allocation profilers:

| Category | Winner Algorithm | Execution Speed | Memory | Special Characteristics |
|:---|:---|:---|:---|:---|
| **👑 Grand Champion** | `potts_model_q_state_domains` | **8.7 ns / pt** | **0 allocs** | Thermodynamic clock domains with dynamic boundary walls |
| **🎨 Aesthetic Champion** | `fibonacci_phyllotaxis_resonance` | 13.2 ns / pt | 0 allocs | Golden angle spirals ($\Phi_{\text{angle}} \approx 137.5^\circ$) |
| **⚡ Phase Precision** | `neumaier_compensated_angle` | 1.1 ns / pt | 0 allocs | 389x faster than BigFloat; zero phase drift |
| **🌊 Field Propagation** | `coupled_harmonic_stencil` | 4.2 ns / pt | 0 allocs | Continuous $d$-dimensional Laplacian stencil |

*Full tournament specifications and benchmark logs are available in [`specs/Video_Winners.md`](specs/Video_Winners.md) and [`specs/Winners.md`](specs/Winners.md).*

---

## 🌐 Documentation & GitHub Pages

Interactive documentation, API references, mathematical derivations, and video player guides are hosted on GitHub Pages:
👉 **[https://theprotaganist.github.io/sovwave/](https://theprotaganist.github.io/sovwave/)**

The local documentation source is located in [`docs/index.html`](docs/index.html).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 TheProtaganist
