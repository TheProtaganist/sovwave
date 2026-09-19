# 🌊 Sovwave.jl (v0.2.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blue.svg)](https://julialang.org)
[![Tuning](https://img.shields.io/badge/Tuning-432Hz%20Sacred%20Harmonics-emerald.svg)](https://en.wikipedia.org/wiki/Concert_pitch)
[![Architecture](https://img.shields.io/badge/Architecture-Pure%20Wave%20Computing-purple.svg)](#-core-architecture)
[![Model Format](https://img.shields.io/badge/Model%20Format-Dual--Stream%20MKV%20Video-red.svg)](#-emergent-video-model-serialization)
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-cyan.svg)](https://theprotaganist.github.io/sovwave/)

> **Pure Wave Computing, Quantum-Acoustic Deep Learning, and Morphogenetic Intelligence in Julia.**
>
> Zero Matrix Multiplications. Zero Backpropagation. Zero Markov Chains. Continuous physical wave mechanics competing directly with PyTorch, JAX, NumPy, and Flux.jl. Models serialized losslessly as playable, glowing audio-visual video files.

---

## 📖 Table of Contents
1. [Overview & Philosophy](#-overview--philosophy)
2. [Direct Installation via Julia Pkg](#-direct-installation-via-julia-pkg)
3. [Competitor Comparison Matrix (Sovwave vs PyTorch, JAX, NumPy, Flux)](#-competitor-comparison-matrix)
4. [Continuous Harmonic Wave Tokenizer (WaveForm Architecture)](#-continuous-harmonic-wave-tokenizer)
5. [Multi-Modal Dataset Formatting & DataLoaders](#-multi-modal-dataset-formatting--dataloaders)
6. [Native Hugging Face Hub Integration](#-native-hugging-face-hub-integration)
7. [The 5 Native Model Architectures](#-the-5-native-model-architectures)
   - [1. LLM (:llm) — Autoregressive Harmonic Wave Language Model](#1-wave-large-language-model-llm)
   - [2. Image Generation (:image_generation) — 2D Surface De-interference](#2-wave-image-generator-image_generation)
   - [3. Text-to-3D (:text_to_3d) — Volumetric Radiance Fields](#3-wave-text-to-3d-text_to_3d)
   - [4. Jev Decision Engine (:jev) — Hallucination-Immune Decisions](#4-jev-decision-engine-jev)
   - [5. Text-to-Video (:text_to_video) — Spatio-Temporal Wave Dynamics](#5-wave-text-to-video-text_to_video)
8. [Model Reading & Introspection API](#-model-reading--introspection-api)
9. [Emergent Video Model Serialization (.mkv & .mp4)](#-emergent-video-model-serialization)
10. [Trained Lattice Audio Sonification](#-trained-lattice-audio-sonification)
11. [7x144 Evolutionary Tournament Grand Champions](#-7x144-evolutionary-tournament-grand-champions)
12. [Quick Start & Code Examples](#-quick-start--code-examples)
13. [Documentation & GitHub Pages](#-documentation--github-pages)
14. [License](#-license)

---

## 🌌 Overview & Philosophy

Modern artificial intelligence is dominated by discrete matrix multiplications ($\text{GEMM}$), backpropagation gradient descent, and high-power GPUs. **Sovwave.jl** is built on a fundamentally different paradigm: **continuous morphogenetic wave computing** grounded in Self-Organizing Vacuum (SOV) dynamics, quantum lattice resonance, and sacred geometry.

Instead of discrete binary weights and float matrices:
- Data points exist in a **continuous $d$-dimensional spatial manifold** ($d \in [1, 5+]$) with physical wave properties (mass, charge, energy, spin).
- Layers consist of **quantum oscillator lattice nodes** vibrating with dynamic amplitude $A$, phase $\phi$, relative frequency $f$, and fractal scale $\beta_s$ (Golden Ratio $\Phi \approx 1.6180339887$).
- Information propagates through **continuous harmonic interference** and temporal superposition driven by a quantum carrier frequency $\omega$ (432 Hz).
- Optimization seeks the **thermodynamic minimum ground-state energy** ($\mathcal{H}_{\text{wave}} \to 0$) via multi-island genetic evolution.
- Trained models are serialized into **standard video files (`.mkv` and `.mp4`)** that can be opened in any media player (VLC, QuickTime, MPV) to visually observe the model's inner dynamics and hear its harmonic oscillations.

---

## 📦 Direct Installation via Julia Pkg

Install Sovwave directly from Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/TheProtaganist/sovwave.git")
```

Or from the Julia REPL package mode (press `]`):
```text
pkg> add https://github.com/TheProtaganist/sovwave.git
```

Requires **Julia 1.9+**. Pure standard Julia stdlib + YAML dependency — zero heavyweight binary drivers, zero CUDA requirements, 100% headless compatible.

---

## ⚔️ Competitor Comparison Matrix

| Feature | Sovwave.jl | PyTorch | JAX | NumPy | Flux.jl |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Fundamental Primitive** | **Continuous Wave Fields $\psi(t)$** | Discrete Tensors | Discrete Arrays | Discrete Arrays | Discrete Tensors |
| **Optimization Engine** | **Thermodynamic Ground-State Evolution** | Reverse-mode Autograd | VJP / Autodiff | None (Manual) | Zygote.jl Autograd |
| **Hardware Requirement** | **CPU / Zero Graphics Drivers** | NVIDIA CUDA / AMD ROCm | TPU / GPU | CPU Only | CPU / CUDA.jl |
| **Model Tokenizer** | **Acoustic WaveForms (Sound & Frequency)** | Discrete Integer IDs | Discrete Integer IDs | None | Discrete Tokens |
| **Model Serialization** | **Lossless Dual-Stream Video (`.mkv` / `.mp4`)** | `.pt` / `.safetensors` | `.msgpack` / FlatBuffers | `.npy` / `.npz` | `.bson` / JLD2 |
| **Model Sonification** | **Native Acoustic Synthesis (WAV/Live)** | Third-party / None | None | None | None |
| **Hallucination Immunity** | **Built-in (Jev Decision Engine)** | Vulnerable | Vulnerable | N/A | Vulnerable |
| **Extensible Model Types** | **LLM, Image, 3D, Jev, Video + Custom** | Multi-library | Multi-library | Manual | Manual |

---

## 🎵 Continuous Harmonic Wave Tokenizer

Unlike traditional discrete tokenizers that assign arbitrary integer IDs (`"hello" -> 1432`), Sovwave's **WaveTokenizer** projects text into continuous **physical `WaveForm` packets**:

$$\psi_k(t) = \left[ \cos(2\pi f_k t + \phi_k) + \sum_{m} A_m \cos(2\pi f_m t + \phi_m) \right] \cdot \exp(-t / \tau)$$

Every token is an acoustic waveform:
- **Resonant Carrier Frequency** $f_k \in [432 \text{ Hz}, 8 \text{ kHz}]$ computed via golden ratio intervals ($\Phi \approx 1.618$).
- **Circular Phase** $\phi_k \in [0, 2\pi)$ assigned on the unit circle.
- **Continuous Sound Wave Buffer** $\psi(t)$: real audio samples that can be listened to directly.
- **Harmonic Overtones**: 2nd octave, 5th harmonic, and golden overtone.

```julia
using Sovwave

tok = default_tokenizer()

# Tokenize text into physical WaveForms
waveforms = tokenize(tok, "harmonic wave computing")

# Inspect the continuous waveform properties
wf = waveforms[1]
println("Token: ", wf.token)           # "ha"
println("Frequency: ", wf.frequency)   # 999.94 Hz
println("Phase: ", wf.phase)           # 4.228 rad
println("Sound samples: ", wf.samples) # Vector of continuous acoustic values

# Synthesize continuous sound audio buffer
audio_buffer = to_audio(waveforms; sample_rate=48000.0)

# Export as a real playable WAV audio file
sonify_tokens(tok, "harmonic wave computing", path="tokens.wav")

# Resonant decoding back to text
text = decode(tok, waveforms)
```

---

## 📊 Multi-Modal Dataset Formatting & DataLoaders

Sovwave provides a universal data preparation engine converting real-world modalities into continuous harmonic wave fields:

```julia
using Sovwave

# 1. Tabular / Numerical data (samples × features)
X = rand(100, 8)
y = rand(100)
ds_tab = from_tabular(X, y; task=:regression, embed_dim=32)

# 2. Text data
texts = ["quantum harmonic computing", "morphogenetic wave field"]
labels = [1, 2]
ds_txt = from_text(texts, labels; embed_dim=32)

# 3. 2D Image matrices
imgs = [rand(16, 16) for _ in 1:10]
ds_img = from_image(imgs, rand(0:1, 10); embed_dim=32)

# 4. Time-Series streams
stream = sin.(range(0, 20π, length=500))
ds_ts = from_timeseries(stream; window_size=32, horizon=4, embed_dim=32)

# 5. Jev Decision Engine state data
states = [Dict("temp" => 23.5, "alarm" => false), Dict("temp" => 92.0, "alarm" => true)]
questions = ["Is equipment operational?", "Failure probability?"]
decisions = [[1.0, 0.0], [0.0, 1.0]]
ds_jev = from_jev_state(states, questions, decisions; embed_dim=32)

# Batched DataLoader with phase-coherent shuffling
loader = WaveDataLoader(ds_tab; batch_size=16, shuffle=true)
for (batch_x, batch_y) in loader
    # batch_x is Vector{Vector{Float64}} of wave packets
end
```

---

## 🤗 Native Hugging Face Hub Integration

Fetch, cache, authenticate, and convert Hugging Face datasets directly into continuous `WaveDataset`s:

```julia
using Sovwave

# Supports automatic token authentication via:
# 1. Explicit `token` kwarg
# 2. ENV["HF_TOKEN"]
# 3. ENV["HUGGING_FACE_HUB_TOKEN"]
ds = load_hf_dataset(
    "imdb";
    split = "train",
    limit = 200,
    embed_dim = 32,
    token = get(ENV, "HF_TOKEN", nothing)
)

println("Loaded Hugging Face samples: ", num_samples(ds))
```

---

## 🧠 The 5 Native Model Architectures

Sovwave includes 5 default trainable model architectures that operate natively over continuous wave fields:

### 1. Wave Large Language Model (`:llm`)
Autoregressive wave sequence modeling. Context tokens excite continuous wave packets that propagate through quantum lattice layers, and next tokens are decoded via harmonic cosine resonance against vocabulary overtones:
```julia
generated_text = generate_text(model, tok, "The universe is", max_new_tokens=16, temperature=0.7)
```

### 2. Wave Image Generator (`:image_generation`)
Synthesizes continuous 2D spatial pixel manifolds from latent harmonic states via cymatic de-interference:
```julia
img = generate_image(model, "cybernetic spiral wave"; height=32, width=32)
```

### 3. Wave Text-to-3D (`:text_to_3d`)
Synthesizes continuous 3D volumetric radiance fields $(x, y, z) \mapsto (\sigma, R, G, B)$ mapped over tetrahedral harmonic coordinate manifolds:
```julia
vol = generate_3d(model, "sacred crystal"; resolution=8)
# Returns 4D Tensor: (8, 8, 8, 4) where channel 1 = Density, channels 2-4 = RGB
```

### 4. Jev Decision Engine (`:jev`)
**Non-autoregressive, transformer-based AI decision model.**
*Jev reads language but never writes it.* It ingests an application's data state alongside a set of predefined questions and returns **type-safe**, structured probabilistic decisions (`:boolean`, `:category`, `:rubric`). Because it cannot generate open-ended text, it is **completely immune to traditional text hallucinations**:
```julia
state = Dict("oil_pressure" => 4.2, "bearing_temp" => 74.5, "active" => true)
questions = [
    "Should emergency shutdown trigger?",   # :boolean
    "Recommended maintenance schedule?",     # :category
    "Mechanical health score?"              # :rubric [0.0, 1.0]
]

decisions = jev_decide(model, state, questions; question_types=[:boolean, :category, :rubric])
for d in decisions
    println("$(d.question) -> $(d.decision) (Confidence: $(d.confidence))")
end
```

### 5. Wave Text-to-Video (`:text_to_video`)
Evolves spatio-temporal wave dynamics across consecutive temporal frames and serializes directly to `.mkv` and companion `.mp4`:
```julia
generate_video(model, "fluid vortex dynamics", "vortex.mkv"; frames=16, fps=4)
```

### Extensible Model Registry
Add custom model types to Sovwave with `register_model_type!`:
```julia
register_model_type!(:custom_acoustic_radar, Dict(
    "name" => "Acoustic Radar Processor",
    "paradigm" => "Binaural Interference Localization",
    "description" => "Custom wave engine"
))
```

---

## 🔍 Model Reading & Introspection API

Inspect model layers, parameter counts, quantum lattices, ground-state energies, and physical values in-memory or directly from `.mkv` files:

```julia
using Sovwave

# Inspect in-memory model
model = WaveModel(default_config())
println("Layers: ", num_layers(model))
println("Total Parameters: ", parameter_count(model))

# Print beautiful structured summary
model_summary(model)

# Inspect specific layer details
details = layer_details(model, 1)
println("Layer 1 Amplitudes: mean=", details[:amplitudes][:mean])
println("Layer 1 Frequencies: min=", details[:frequencies][:min])

# Inspect model file directly from disk
info = inspect_model("trained_brain.mkv")
```

---

## 🏆 7x144 Evolutionary Tournament Grand Champions

Every core subsystem in Sovwave was developed through a dedicated **144-algorithm tournament** (12 rounds × 12 candidates, with 6 evolutionary mutations of the previous winner + 6 new unrelated candidates per round). 

A total of **1,008 candidate algorithms** were benchmarked under nanosecond timing and allocation profilers:

| Subsystem | Tournament File | Grand Champion Algorithm | Category | Throughput / Latency | Key Physical Metric | Fitness Score |
|---|---|---|---|---|---|---|
| **1. Tokenizer** | [`tournament_tokenizer.jl`](test/audio/tournament_tokenizer.jl) | `CharLevel_ContinuousFourier_FastEnvelope_QuantumManifold` | WinnerVariant | 76.7 µs / token packet | 100.0% accuracy, 1.000 coherence | **7,130.31** |
| **2. Dataset** | [`tournament_dataset.jl`](test/audio/tournament_dataset.jl) | `PottsState_CategoricalProjection_FastPacked16` | WinnerVariant | 53.6 µs / 50 samples | 100.0% fidelity, 0.6662 energy var | **6,852.06** |
| **3. LLM (:llm)** | [`tournament_llm.jl`](test/audio/tournament_llm.jl) | `FibonacciSpiral_ContextMemory_v2_UltraFastBurst1` | WinnerVariant | 8,386.6 tokens/sec (0.6 ms) | 0.7273 diversity, 0 hallucination | **10,375.12** |
| **4. Image Gen** | [`tournament_image_gen.jl`](test/audio/tournament_image_gen.jl) | `ChladniNodal_GridProjection` | Cymatic | 429,456.1 pixels/sec (0.6 ms) | 1.0000 spatial contrast | **11,294.56** |
| **5. Text-to-3D** | [`tournament_text_to_3d.jl`](test/audio/tournament_text_to_3d.jl) | `PlatonicTetrahedral_ResonanceField` | Platonic | 134,928.0 voxels/sec (0.47 ms) | 1.0000 density variance | **12,000.00** |
| **6. Jev Engine** | [`tournament_jev.jl`](test/audio/tournament_jev.jl) | `ChladniNodal_CategorySeparator` | Cymatic | 2,321.7 decisions/sec (1.72 ms) | 100.0% type safety, 0% hallucination | **7,595.38** |
| **7. Video Gen** | [`tournament_text_to_video.jl`](test/audio/tournament_text_to_video.jl) | `LagrangianPhaseFlow_VideoDynamics_v2_HighRes8Frames` | WinnerVariant | 1,538.6 frames/sec (5.2 ms) | 0.9822 temporal continuity | **11,946.46** |

*Complete mathematical derivations and logs: [`specs/Tournament_7x144_Winners.md`](specs/Tournament_7x144_Winners.md).*

---

## 🎬 Emergent Video Model Serialization

Sovwave serializes models into standard **Matroska (`.mkv`)** and **MPEG-4 (`.mp4`)** video files.

### Dual-Stream Matroska Architecture:
1. **Stream 0:0 (`VISUAL_BRAIN`) — H.264 Video**:
   - Visualizes post-convergence wave dynamics through the model's internal layers.
   - **NO Black Screen**: Frame 1 represents **Step $n$** (the fully trained coherent model at rest) and blooms immediately into radiant color.
   - **Grand Champion Algorithm**: `potts_model_q_state_domains` (8.7 ns / pt, 0 allocs) with CMY color coding (State 0: `FF00FF` Magenta, State 1: `FFFF00` Yellow, State 2: `00FFFF` Cyan).
   - **Configurable Pixel Square Size**: `pixel_scale: 1`, `pixel_scale: 2`, or `pixel_scale: "auto"`.
2. **Stream 0:1 (`MODEL_WEIGHTS`) — FFV1 Lossless Video**:
   - 100% bit-for-bit lossless mathematical stream containing raw model weights for reconstruction.

---

## 🔊 Trained Lattice Audio Sonification

The audio track inside the video file allows you to **hear the trained model**. During inference, audio is **not used as an input feature**; it exists solely to sonify the internal state of the neural model:

$$\omega_i(t) = \omega \cdot \left(1 + \frac{1}{2}\sum_{d} \left(A_{i,d} \cos(\phi_{i,d}) + f_{i,d}\right)\right)$$

Hear the model live in real time or export as a WAV file:
```julia
# Export trained model oscillations as a WAV file
save_wav(model, "model_voice.wav"; duration_sec=5.0)
```

---

## 🌐 Documentation & GitHub Pages

Interactive documentation, API references, mathematical derivations, and live Web Audio/Canvas visualizers are hosted on GitHub Pages:
👉 **[https://theprotaganist.github.io/sovwave/](https://theprotaganist.github.io/sovwave/)**

The local documentation source is located in [`docs/index.html`](docs/index.html).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 TheProtaganist
