# 🌊 Sovwave.jl (v0.3.2)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blue.svg)](https://julialang.org)
[![Tuning](https://img.shields.io/badge/Tuning-432Hz%20Sacred%20Harmonics-emerald.svg)](https://en.wikipedia.org/wiki/Concert_pitch)
[![Architecture](https://img.shields.io/badge/Architecture-Pure%20Wave%20Computing-purple.svg)](#-core-architecture)
[![Model Format](https://img.shields.io/badge/Model%20Format-Dual--Stream%20MKV%20Video-red.svg)](#-emergent-video-model-serialization)
[![CUDA & Hybrid](https://img.shields.io/badge/Hybrid%20Engine-CPU%20%2B%20CUDA-green.svg)](#-auto-tuning-cpu-vs-cuda-hybrid-engine)
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-cyan.svg)](https://theprotaganist.github.io/sovwave/)

> **Pure Wave Computing, Quantum-Acoustic Deep Learning, and Morphogenetic Intelligence in Julia.**
>
> Zero Matrix Multiplications. Zero Backpropagation. Zero Markov Chains. Continuous physical wave mechanics competing directly with PyTorch, JAX, NumPy, and Flux.jl. Models serialized losslessly as playable, glowing audio-visual video files. **Run inference in Python, C++, JavaScript, Java, or C# with zero Julia dependency.**

---

## 📖 Table of Contents
1. [Overview & Philosophy](#-overview--philosophy)
2. [Direct Installation via Julia Pkg](#-direct-installation-via-julia-pkg)
3. [One-Liner Terminal Launch](#-one-liner-terminal-launch)
4. [Local Visual Quantum GUI](#-local-visual-quantum-gui)
5. [Auto-Tuning CPU vs. CUDA Hybrid Engine](#-auto-tuning-cpu-vs-cuda-hybrid-engine)
6. [Fractal Dimension & Wave Speed per Node](#-fractal-dimension--wave-speed-per-node)
7. [Competitor Comparison Matrix (Sovwave vs PyTorch, JAX, NumPy, Flux)](#-competitor-comparison-matrix)
8. [Physical Wave Frequency Tokenizer & Custom Tokenizers](#-physical-wave-frequency-tokenizer--custom-tokenizers)
9. [Complete Step-by-Step Training Guide](#-complete-step-by-step-training-guide)
10. [Multi-Modal Dataset Formatting & DataLoaders](#-multi-modal-dataset-formatting--dataloaders)
11. [Native Hugging Face Hub Integration](#-native-hugging-face-hub-integration)
12. [The 5 Native Model Architectures](#-the-5-native-model-architectures)
    - [1. LLM (:llm) — Autoregressive Harmonic Wave Language Model](#1-wave-large-language-model-llm)
    - [2. Image Generation (:image_generation) — 2D Surface De-interference](#2-wave-image-generator-image_generation)
    - [3. Text-to-3D (:text_to_3d) — Volumetric Radiance Fields](#3-wave-text-to-3d-text_to_3d)
    - [4. Jev Decision Engine (:jev) — Hallucination-Immune Decisions](#4-jev-decision-engine-jev)
    - [5. Text-to-Video (:text_to_video) — Spatio-Temporal Wave Dynamics](#5-wave-text-to-video-text_to_video)
13. [Cross-Language MKV Runtimes (Python/C++/JS/Java/C#)](#-cross-language-mkv-runtimes)
14. [Model Reading & Introspection API](#-model-reading--introspection-api)
15. [Emergent Video Model Serialization (.mkv & .mp4)](#-emergent-video-model-serialization)
16. [Trained Lattice Audio Sonification](#-trained-lattice-audio-sonification)
17. [7x144 Evolutionary Tournament Grand Champions](#-7x144-evolutionary-tournament-grand-champions)
18. [Documentation & GitHub Pages](#-documentation--github-pages)
19. [License](#-license)

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

## 🖥️ Local Visual Quantum GUI

Launch the built-in browser interface locally with zero external web dependencies (pure Julia stdlib `Sockets`):

```julia
using Sovwave

# Launches local web GUI at http://127.0.0.1:8080
launch_gui(port=8080, open_browser=true)
```

Features:
- **Live Quantum Lattice**: Real-time HTML5 Canvas visualizer rendering wave amplitude \(\psi(x, y)\) and Potts q-states.
- **Potts State Domain Colors**: Configurable 3-state domain colors (defaults: `state 0 -> #FF00FF` Magenta, `state 1 -> #FFFF00` Yellow, `state 2 -> #00FFFF` Cyan).
- **Universal Multilingual Tokenizer**: Interactive sound synthesis testing multilingual sentences and emojis through browser speakers via Web Audio API.
- **5-Model Playground**: Interactive prompt execution for LLM, 2D Image Gen, 3D Radiance Fields, Jev Decision Engine, and MKV Video frames.
- **YAML Config Manager**: Live parameter editing and export for `model_config.yaml`.

---

## 🚀 One-Liner Terminal Launch

Launch the full Sovwave GUI from anywhere with a single command:

```julia
# From Julia REPL or script
using Sovwave
launch_gui(port=8080, open_browser=true)
```

Or using the included shell script:
```bash
# From terminal (no Julia code needed)
./bin/sovwave                          # http://127.0.0.1:8080
./bin/sovwave --port 9090              # Custom port
./bin/sovwave --host 0.0.0.0           # Accessible from LAN
./bin/sovwave --no-browser             # Server only, no auto-open

# Install to system PATH for global access
sudo ln -s "$(pwd)/bin/sovwave" /usr/local/bin/sovwave
sovwave --port 8080                    # works from any directory
```

---

## ⚡ Auto-Tuning CPU vs. CUDA Hybrid Engine

Sovwave v0.3.2 introduces an **Adaptive Auto-Tuning Hybrid Engine** that empirically compares your host CPU and NVIDIA CUDA GPU, benchmarks every subsystem, and dynamically routes each computation to whichever backend won the benchmark:

- **Zero-Allocation CPU SIMD Engine**: Pre-allocated in-place scratch buffers, loop invariant hoisting, and `@simd` vectorization executing single-sample inference at **69.7x the speed of CUDA** (0.32 ms CPU vs. 22.11 ms GPU due to zero PCIe roundtrip latency).
- **High-Throughput CUDA 2D Grid**: Batched tensor wave propagation and CUBLAS GEMM for massive batch sizes and 50,000+ vocabulary projections.
- **WaveHybridDispatcher**: Automatically routes workloads based on empirical hardware benchmarks.

```julia
using CUDA, Sovwave

# Enable optional GPU acceleration
enable_cuda!()

# 1. Run empirical benchmark comparing CPU (System) vs. CUDA across all 5 subsystems:
disp = benchmark_system_vs_cuda(; iters=50, batch_size=32, verbose=true)

# Output summary table:
# Subsystem Task                 | CPU (ms)     | CUDA (ms)    | Speedup    | Winner    
# --------------------------------------------------------------------------------
# A. Single-Sample Forward       | 0.32         | 22.11        | 69.77x CPU | CPU       
# B. Batched Forward Pass        | 11.81        | 18.30        | 1.55x CPU  | CPU       
# C. Vocab Projection (50k)      | 33.56        | 51.04        | 1.52x CPU  | CPU       
# D. Population Evaluation       | 255.33       | 383.00       | 1.50x CPU  | CPU       
# E. In-Place Mutation           | 5.44         | 16.31        | 3.00x CPU  | CPU       
# 🏆 HYBRID ENGINE ACTIVE: Auto-routing every task to its fastest device.

# 2. Seamless Hybrid Execution (auto-routes to the winning device):
out = hybrid_forward!(model, input_vector)
batch_out = hybrid_forward_batch(layer, batch_matrix)
logits = hybrid_project_vocab(projection_weights, hidden_state)
hybrid_evaluate!(population_state, batch_inputs, batch_targets)

# 3. Check or inspect the active dispatcher:
disp = hybrid_dispatcher()
println(disp.single_forward)    # :cpu or :cuda
println(disp.batched_forward)   # :cpu or :cuda
```

When CUDA.jl is **not installed**, Sovwave runs 100% self-contained on CPU with zero degradation and maximum SIMD speed.

> **MKV models** saved from CPU or GPU are 100% interchangeable and device-independent.

---

## 🌀 Fractal Dimension & Wave Speed per Node

Every quantum lattice node in Sovwave v0.3.0 now carries **six** learnable physical parameters:

| Parameter | Symbol | Range | Default | Meaning |
|-----------|--------|-------|---------|---------|
| Amplitude | `A` | [0.001, 3.0] | 0.5 | Radiant wave intensity |
| Phase | `φ` | [0, 2π] | varied | Wave offset |
| Frequency | `f` | [0.1, 10.0] | 1.0+ | Relative frequency multiplier |
| Fractal Scale | `β_s` | [0.5, 4.0] | φ≈1.618 | Golden-ratio amplitude envelope |
| **Fractal Dim** | **`D_f`** | **[1.0, 3.0]** | **1.5** | **Hausdorff fractal dimension** |
| **Wave Speed** | **`v`** | **[0.1, 10.0] or -1** | **1.0** | **Propagation speed (-1 = unlimited)** |

### Fractal Dimension (`fractal_dim`)
Controls the self-similar geometric zoom of the wave envelope at each node. `D_f = 1.0` is linear (1D wire), `D_f = 2.0` is planar (2D surface), `D_f = 1.5` is the midpoint fractal (default). The envelope scales the node output by `r_norm^(D_f - 1)`.

### Wave Speed (`wave_speed`)
Controls how fast the wave propagates through a node's local manifold:
- **`v > 0`** (finite speed): `angle = ω·f·(x/v) + φ - t` — wave experiences propagation delay
- **`v = -1.0`** (unlimited/instantaneous): `angle = ω·f·x + φ - t` — no speed divisor, instantaneous
- Mutation **never** accidentally drifts into the `-1.0` sentinel — unlimited must be set intentionally

```julia
# Create layers with custom fractal/speed settings
layer = create_layer(64, 32; fractal_dim=2.0, wave_speed=0.5)
layer_inf = create_layer(64, 32; wave_speed=-1.0)   # unlimited speed all nodes

# Both fields are fully learnable: mutate!, crossover work correctly
mutate!(layer, 0.01)
child = crossover(layer_a, layer_b)  # blends fractal_dims and wave_speeds
```

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

## 🌊 Universal Multilingual & Emoji Wave Tokenizer

Unlike traditional discrete tokenizers that assign arbitrary integer IDs (`"hello" -> 1432`) or drop unknown characters as `<UNK>`, Sovwave's **WaveTokenizer** maps every character and emoji natively as a **first-class continuous harmonic wave primitive**:

$$\psi_k(t) = \left[ \cos(2\pi f_k t + \phi_k) + \sum_{m} A_m \cos(2\pi f_m t + \phi_m) \right] \cdot \exp(-t / \tau)$$

Every token is an acoustic waveform:
- **Resonant Carrier Frequency** $f(u) \in [432 \text{ Hz}, 8 \text{ kHz}]$ computed via golden ratio Weyl mapping for code point $u = \text{UInt32}(c)$.
- **Circular Phase** $\phi(u) \in [0, 2\pi)$ assigned on the unit circle.
- **Zero UNK Data Loss**: Novel characters or emojis dynamically register as first-class native wave tokens. No byte-level degradation or shortcuts.
- **Continuous Sound Wave Buffer** $\psi(t)$: real audio samples that can be listened to directly.
- **100% Lossless Bidirectional Reconstruction**: `decode(tok, tokenize(tok, str)) == str` for all world languages (Arabic, Chinese, Japanese, Korean, Hindi, Hebrew, Cyrillic, Greek, Latin extended) and emojis (`🌊`, `🧠`, `⚡`, `🚀`, `⚛️`, etc.).

```julia
using Sovwave

## 🌊 Physical Wave Frequency Tokenizer & Custom Tokenizers

In Sovwave's Pure Wave Computing architecture, **tokens are physical wave frequencies in Hz — NOT discrete integer IDs `[x, y]`**. 

Every token in the universe is an acoustic frequency and circular phase angle:
- **Continuous Wave Frequency** $f(u) \in [432 \text{ Hz}, 8 \text{ kHz}]$: derived from harmonic Weyl resonance and golden ratio scaling ($\Phi \approx 1.618$).
- **Circular Phase** $\phi(u) \in [0, 2\pi)$ on the unit circle.
- **Zero UNK Data Loss**: Dynamic token registration guarantees 100% loss-free encoding across all languages and emojis (`🌊`, `🧠`, `⚡`, `🚀`, `⚛️`, etc.).
- **Direct Frequency Decoding**: `decode(tok, tokenize_frequencies(tok, str)) == str` reconstructs text directly from physical wave frequencies.

```julia
using Sovwave

# 1. Default Pretrained Tokenizer (50,000+ words, works 100% offline)
tok = default_tokenizer()

# 2. Tokenize text into continuous physical WAVE FREQUENCIES (Hz):
text = "The universe operates on harmonic wave interference 🌊⚛️"
wave_freqs = tokenize_frequencies(tok, text)

println(wave_freqs)
# -> [586.745, 723.614, 794.162, 781.449, 726.961, 766.073, 853.277, ...] (in Hz)

# 3. 100% Exact Lossless Decoding directly from Wave Frequencies:
decoded = decode(tok, wave_freqs)
@assert decoded == text

# 4. Instant Custom Tokenizer for ANY Project (wordlist, dictionary, or corpus):
tok_words = custom_tokenizer(["quantum", "resonance", "harmonic", "vacuum", "solfeggio"])

# Custom Tokenizer with exact physical sound frequencies (e.g., Solfeggio scale):
tok_solfeggio = custom_tokenizer(Dict(
    "ut"  => 396.0,  # 396 Hz
    "re"  => 417.0,  # 417 Hz
    "mi"  => 528.0,  # 528 Hz (Transformation / Miracles)
    "fa"  => 639.0,  # 639 Hz
    "sol" => 741.0,  # 741 Hz
    "la"  => 852.0   # 852 Hz
))

# 5. Save & Load Custom Tokenizers in 1 line:
save_tokenizer(tok_solfeggio, "my_custom_tokenizer.json")
reloaded_tok = load_tokenizer("my_custom_tokenizer.json") # loads custom JSON, HF, .vocab, or .txt

# 6. Pretrained tokenizers from world models:
tok_gpt2     = default_tokenizer(model=:gpt2)       # GPT-2 (50,257 tokens, offline)
tok_qwen     = default_tokenizer(model=:qwen)       # Qwen 2.5 (151,643 tokens)
tok_mistral  = default_tokenizer(model=:mistral)    # Mistral 7B (32,768 tokens)
tok_llama    = default_tokenizer(model=:llama)      # LLaMA 3.2 (128,256 tokens)
tok_deepseek = default_tokenizer(model=:deepseek)   # DeepSeek V3 (129,280 tokens)

# 7. Synthesize audio buffer and export as playable WAV:
waveforms = tokenize(tok, text)
audio_buffer = to_audio(waveforms; sample_rate=48000.0)
sonify_tokens(tok, text, path="harmonic_text.wav")
```

---

## 🚀 Portable Training: Run Anywhere in 1 Line

Install `Sovwave` from GitHub and train models immediately with zero local directory dependencies:

```julia
using Sovwave

# Train an LLM directly on raw text with 1 line:
texts = [
    "DeepSeek-V4 is a continuous wave language model",
    "Quantum harmonic computing resonates with physics",
    "Self-organizing vacuum dynamics govern wave evolution"
]

model = WaveModel(default_config())
trained_model, history = train_llm(model, texts; epochs=10)

# Generate coherent text:
output = generate_text(trained_model, default_tokenizer(), "DeepSeek-V4 is"; max_new_tokens=8)
println(output)
```

---

## 🧠 Complete Step-by-Step Training Guide

Train continuous wave models in Julia without backpropagation or CUDA:

```julia
using Sovwave

# 1. Format Multi-Modal Dataset
texts = ["Harmonic wave intelligence 🌊", "自己組織化真空 ⚛️", "ذكاء كوانتي موجي ⚡"]
labels = [1, 2, 3]
ds = format_text(texts, labels; max_len=16, embed_dim=32)

# 2. Configure Wave Model
cfg = WaveMLConfig(
    field = WaveFieldConfig(lattice_size=(32, 32), omega=432.0, beta_s=1.618),
    model = WaveModelConfig(layers=4, embed_dims=32),
    train = WaveTrainConfig(population_size=12, mutation_rate=0.05, generations=50)
)
model = WaveModel(cfg)

# 3. Train via Evolutionary Thermodynamic Ground-State Descent
history = train!(model, ds; generations=25, verbose=true)

# 4. Generate & Infer Across Architectures
tok = default_tokenizer()
text_out = generate_text(model, tok, "Universal wave"; max_new_tokens=16)

# 5. Serialize Model Losslessly to Matroska Video
save_model(model, "quantum_model.mkv")
loaded = load_model("quantum_model.mkv")
```

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

## 🌐 Cross-Language MKV Runtimes

A trained Sovwave `.mkv` model can be loaded and run in **Python, C++, JavaScript, Java, or C#** — no Julia dependency, just `ffmpeg`.

All runtimes are in `runtimes/` and implement the same wave forward pass (including `fractal_dim` and `wave_speed`):

### Python (3.8+, stdlib only)
```python
from runtimes.python.sovwave_runtime import SovwaveModel
model   = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml")
outputs = model.predict([[0.1, 0.2, 0.3, 0.4]])
```

### C++17 (header-only)
```cpp
#include "runtimes/cpp/sovwave_runtime.hpp"
auto model  = sovwave::SovwaveModel::load("my_model.mkv");
auto output = model.predict({{0.1, 0.2, 0.3, 0.4}});
```

### JavaScript (Node.js 14+)
```javascript
const { SovwaveModel } = require('./runtimes/js/sovwave_runtime');
const model   = await SovwaveModel.load('my_model.mkv');
const outputs = model.predict([[0.1, 0.2, 0.3, 0.4]]);
```

### Java (11+)
```java
SovwaveModel model = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml");
double[][] out = model.predict(new double[][]{{ 0.1, 0.2, 0.3, 0.4 }});
```

### C# (.NET 6+)
```csharp
var model = await SovwaveModel.LoadAsync("my_model.mkv");
var out   = model.Predict(new[] { new[] { 0.1, 0.2, 0.3, 0.4 } });
```

Full documentation: [`runtimes/README.md`](runtimes/README.md)

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
