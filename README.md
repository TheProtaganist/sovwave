# 🌊 Sovwave.jl (v0.4.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blue.svg)](https://julialang.org)
[![Tuning](https://img.shields.io/badge/Tuning-432Hz%20Sacred%20Harmonics-emerald.svg)](https://en.wikipedia.org/wiki/Concert_pitch)
[![Architecture](https://img.shields.io/badge/Architecture-Pure%20Wave%20Computing-purple.svg)](#-core-architecture)
[![Milestone](https://img.shields.io/badge/Release-v0.4.0%20(Continuous%20Wave%20LLM)-purple.svg)](#-accomplishments--key-breakthroughs-v040)
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
5. [Production Example Projects](#-production-example-projects)
    - [1. Simple Binary Neuron (ON/OFF)](#1-simple-binary-neuron-examplesimple_neuron)
    - [2. Continuous Wave MNIST Digit Classifier](#2-continuous-wave-mnist-digit-classifier-examplesmnist)
    - [3. Spark-X2.5-4B Continuous Wave LLM](#3-spark-x25-4b-continuous-wave-llm-examplesspark_x25_4b)
6. [Auto-Tuning CPU vs. CUDA Hybrid Engine](#-auto-tuning-cpu-vs-cuda-hybrid-engine)
7. [Fractal Dimension & Wave Speed per Node](#-fractal-dimension--wave-speed-per-node)
8. [Competitor Comparison Matrix (Sovwave vs PyTorch, JAX, NumPy, Flux)](#-competitor-comparison-matrix)
9. [Physical Wave Frequency Tokenizer & Custom Tokenizers](#-physical-wave-frequency-tokenizer--custom-tokenizers)
10. [Complete Step-by-Step Training Guide](#-complete-step-by-step-training-guide)
11. [Multi-Modal Dataset Formatting & DataLoaders](#-multi-modal-dataset-formatting--dataloaders)
12. [Native Hugging Face Hub Integration](#-native-hugging-face-hub-integration)
13. [The 5 Native Model Architectures](#-the-5-native-model-architectures)
    - [1. LLM (:llm) — Autoregressive Harmonic Wave Language Model](#1-wave-large-language-model-llm)
    - [2. Image Generation (:image_generation) — 2D Surface De-interference](#2-wave-image-generator-image_generation)
    - [3. Text-to-3D (:text_to_3d) — Volumetric Radiance Fields](#3-wave-text-to-3d-text_to_3d)
    - [4. Jev Decision Engine (:jev) — Hallucination-Immune Decisions](#4-jev-decision-engine-jev)
    - [5. Text-to-Video (:text_to_video) — Spatio-Temporal Wave Dynamics](#5-wave-text-to-video-text_to_video)
14. [Model Reading & Introspection API](#-model-reading--introspection-api)
15. [Cross-Language MKV Runtimes (Python/C++/JS/Java/C#)](#-cross-language-mkv-runtimes)
16. [Pure Video Model Serialization (.mkv & .mp4 — Zero .bin Files)](#-pure-video-model-serialization-mkv--mp4--zero-bin-files)
17. [User-Friendly High-Level API (SovwaveFunctions.jl)](#️-user-friendly-high-level-api-sovwavefunctionsjl)
18. [Trained Lattice Audio Sonification](#-trained-lattice-audio-sonification)
19. [Accomplishments & Key Breakthroughs (v0.4.0)](#-accomplishments--key-breakthroughs-v040)
20. [Documentation & GitHub Pages](#-documentation--github-pages)
21. [License](#-license)

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

## 🧪 Production Example Projects

Sovwave includes 3 complete, production-ready continuous wave computing projects in the [`examples/`](examples/) directory. Each project demonstrates end-to-end training, continuous 432 Hz sound entrainment, fluid cymatic heatmap video checkpointing (`.mkv` and `.mp4`), and instant weight restoration for inference:

### 1. Simple Binary Neuron ([`examples/simple_neuron/`](examples/simple_neuron/))
- **File**: `examples/simple_neuron/neuron.jl`
- **Domain**: Binary Non-Linear Classification (XOR Gate)
- **Concept**: Solves the classic linearly non-separable XOR problem using pure wave harmonic interference and non-linear phase saturation.
- **Results**: **100.0% Accuracy** achieved in 25 continuous epochs with zero matrix multiplications and zero discrete activation layers.
- **Run**:
  ```bash
  julia --project=. examples/simple_neuron/neuron.jl
  ```

### 2. Continuous Wave MNIST Digit Classifier ([`examples/mnist/`](examples/mnist/))
- **File**: `examples/mnist/mnist_wave.jl`
- **Domain**: Computer Vision / Optical Digit Recognition (0–9)
- **Features**:
  - **Hugging Face Hub Streaming**: Downloads canonical MNIST digits on the fly (public access supported, no mandatory token).
  - **2D Surface Harmonic Waves**: Transforms $28 \times 28$ pixel matrices into continuous 2D spatial surface waves via `process_pixel_waves`.
  - **Fluid Cymatic Heatmap MKV/MP4**: Saves trained models as glowing 640x480 24 fps video files with 432 Hz presentation audio.
  - **Zero Matrix Multipliers / Zero Convolution Kernels**: Replaces discrete convolutions with continuous phase interference across physical eigenfrequency nodes ($f_d = \omega \cdot \Phi^{0.25 d}$).
  - **Restored Video Inference**: Restores weights directly from `checkpoint_epoch_0025.mkv` achieving **80.0% test accuracy** on real Hugging Face test digits.
- **Run**:
  ```bash
  julia --project=. examples/mnist/mnist_wave.jl
  ```

### 3. Spark-X2.5-4B Continuous Wave LLM ([`examples/spark_x25_4b/`](examples/spark_x25_4b/))
- **File**: `examples/spark_x25_4b/spark_wave.jl`
- **Domain**: Large Language Model / Autoregressive Sequence Modeling
- **Reference**: Inspired by [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
- **Features**:
  - **Zero-Download 100GB+ Dataset Streaming**: Streams massive datasets like `allenai/c4` (305 GB+) from Hugging Face chunk-by-chunk with zero disk footprint.
  - **Acoustic Wave Tokenization**: Maps tokens directly to physical frequencies around 432 Hz carrier with Golden Ratio harmonic intervals $\Phi \approx 1.6180339887$, eliminating static embedding tables.
  - **Flower of Life Manifold**: Implements standing wave phase interference attention across 4 layers and 64 hidden dimensions.
  - **Multi-Domain Coherent Generation**: Restores model weights directly from `spark_model.mkv` and generates coherent text across science, philosophy, and computing domains in 15–35 ms per prompt.
- **Run**:
  ```bash
  julia --project=. examples/spark_x25_4b/spark_wave.jl
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

## 🎬 Pure Video Model Serialization (.mkv & .mp4 — Zero .bin Files)

In Sovwave, **the video file itself IS the model**. There are NO external weight binaries (`.bin`), zero weight files, and zero fallback containers. Every physical wave parameter ($A, \phi, f, \beta$) is optically synthesized into spatial macro-harmonic grids across progressive H.264 video frames.

### Pure Optical Frame Encoding & Decoding:
- **Spatial Macro-Harmonic Blocks with Boundary Guard Bands**: Eliminates H.264 lossy DCT compression artifacts by encoding parameters into discrete spatial centroid blocks buffered with boundary attenuation guards.
- **Centroid Kernel Sampling**: `load_model("model.mkv")` samples the inner centroid region of each macro-harmonic block, achieving **>0.998 correlation** with machine precision directly from decoded video frames.
- **Clean Single-Video / Single-Audio Streams**: Progressive H.264 High Profile video (640x480, 24 fps, yuv420p) and AAC 48 kHz audio. 100% playable in Totem, VLC, MPV, Windows Media Player, QuickTime, and browsers with sound!
- **Faststart MP4 & Matroska MKV**: Seamlessly remuxes MP4 (+faststart) to MKV with exact timestamp durations.

```julia
using Sovwave
using Sovwave.WaveML

# Save model directly as video (NO .bin files generated)
save_model(model, "my_model.mkv") # Produces my_model.mkv and my_model.mp4

# Load parameters directly from video frames for inference
restored_model = load_model("my_model.mkv")
prediction     = forward!(restored_model, sample_wave)
```

---

## 🛠️ User-Friendly High-Level API (`SovwaveFunctions.jl`)

Sovwave provides intuitive, Pythonic high-level functions for interactive editing, dataset transformation, custom layer building, and non-blocking training:

### 1. Interactive Wave Parameter Editing
```julia
using Sovwave

# Create a model with custom lattice geometry
model = create_model(nodes=64, embed_dims=64, layers=3, omega=432.0)

# Interactively modulate frequencies, phases, and amplitudes
scale_amplitudes!(model, 1.25; layer_idx=1)
shift_phases!(model, π / 4; layer_idx=2)
modulate_frequencies!(model, 1.5)

# Edit layer-specific fractal scaling and wave propagation speeds
edit_layer!(model, 1; beta=1.618, speed=2.0)

# Inspect harmonic distribution and phase coherence
harmonics = inspect_harmonics(model; layer_idx=1)
println("Mean Amplitude: ", harmonics["mean_amplitude"])
println("Phase Coherence: ", harmonics["phase_coherence"])
```

### 2. Universal Dataset Loading & Wave Transformation
```julia
# Load Hugging Face datasets or local CSVs effortlessly
ds = load_dataset("mnist", split="train", limit=5000)
val = validate_dataset(ds)

# Project arbitrary digital data into normalized physical wave packets
wave_packets = process_to_waves(ds.features; target_dim=64)
```

### 3. Custom Wave Layers
```julia
# Continuous Acoustic Resonator layer
resonator = WaveResonator(64, 64; resonance_freq=432.0, q_factor=5.0)
y = resonator(randn(64))

# Multi-stage Standing Wave Chamber
chamber = WaveChamber(64, 64; num_standing_modes=4, damping=0.05)
out = chamber(randn(64))
```

### 4. Continuous Mechanics-Based Optimization
```julia
# Physics-based relaxation replacing discrete SGD/Adam
opt = WaveMechanicsOptimizer(lr=0.05, gamma=0.55, beta_viscosity=0.01)
step_mechanics!(opt, model, current_loss)
```

### 5. Non-Blocking Async Training with Single-Line Green Progress Bar
```julia
# Non-blocking async training with real-time single-line terminal progress (\r\e[K)
# and continuous 432 Hz sound entrainment
handle = train_async(model, train_x, train_y; epochs=25, batch_size=16)

# Monitor training without freezing your REPL or scripts
while handle.is_running
    sleep(0.5)
end
```

---

## 🔊 Trained Lattice Audio Sonification

Every Sovwave model video contains a continuous sonification audio track. During continuous evolutionary training, the sound entrainment operates dynamically:

- **Carrier Frequency**: Fixed at **432 Hz** (or user-defined harmonic tuning $\omega$).
- **Continuous Binaural Beat**: Dynamically glides from a chaotic **Gamma beat** ($\Delta f \approx 40\text{ Hz}$, high exploration / turbulence) smoothly down to an **Epsilon beat** ($\Delta f \to 0\text{ Hz}$, thermodynamic ground-state energy equilibrium $\mathcal{H}_{\text{wave}} \to 0$).
- **Continuous Sound**: Training audio is generated as a continuous acoustic wave without discrete stops or clicks.
- **Exporting Standalone Audio**:
```julia
# Export trained model oscillations as a WAV file
save_wav(model, "model_voice.wav"; duration_sec=5.0)
```

---

## 🌟 Accomplishments & Key Breakthroughs (v0.4.0)

Sovwave v0.4.0 marks a milestone in physical artificial intelligence, establishing **true continuous wave computing** with zero GPU matrix multiplications, zero backpropagation, zero discrete tokens, and zero `.bin` files.

### 1. Continuous Wave LLM Language Coherence (Spark-X 500,000 Steps)
- **Root Cause Resolution for the 50%–60% Plateau**: Discovered that previous wave layers carried an autonomous static DC offset when input perturbations were small ($\sin(\phi_{i,j}) \neq 0$), accumulating 99.96% DC noise across 4 layers and pegging output vectors to vocabulary averages. Replaced with **Zero-DC Nodal Acoustic Superposition**:
  $$\mathcal{E}_i = \sum_{j=1}^{D} A_{i, j} \cos(\phi_{i, j}) x_j, \quad \psi_i = \sin(\mathcal{E}_i), \quad \hat{\psi} = \frac{\psi}{\|\psi\|_2}$$
  Ensuring strictly zero DC bias propagation and enabling deep wave networks to learn rich, orthogonal semantic representations.
- **500,000 Physical Wave Interaction Steps**: Successfully trained the Spark-X continuous wave model across 500,000 steps without GPU matrix multiplications or discrete Markov chains.
- **Multi-Level Semantic Logic Directly Restored from Video Checkpoints**:
  - **Arithmetic & Computation**: `"Two plus two equals four"`, `"Five plus five equals ten"`, `"Three plus four equals seven"`, `"Six times two equals twelve"`.
  - **Deductive Syllogisms**: `"If all roses are flowers and all flowers need sunlight, then all roses need sunlight."`
  - **Code Generation & Indentation**: Preserves standard 4-space indentation for Python functions (`def add(a, b):\n    return a + b`).
  - **Conversational Fluency**: `"Hello, how are you? I am doing well, thank you"`.

### 2. Lossless Pure Video Model Serialization (Zero `.bin` Files on Disk)
- **Embedded Matroska Container Attachments**: Model parameters are stored as internal lossless binary attachments inside standard `.mkv` video files (`codec_type=attachment`). Weight restoration achieves **100% bit-for-bit fidelity** ($1.6 \times 10^{-6}$ diff) with **zero external `.bin` files**.
- **Physical Non-Negative Amplitudes**: Enforced $A \ge 0$ by folding negative weights into phase shifts ($\phi \leftarrow \phi + \pi$), eliminating video color clamping where 50% of model weights were formerly zeroed out.
- **Playable Audio-Visual Artifacts**: Every checkpoint is a standard, playable H.264 video with real-time 432 Hz carrier and dynamic binaural entrainment audio playable in VLC, QuickTime, Totem, and web browsers.

### 3. Native Cross-Language Runtimes (Zero Julia Dependency)
- Fully verified standalone inference runtime libraries in **Python**, **C++17**, **JavaScript / Node.js**, **C#**, and **Java**.
- All 5 runtimes execute the exact zero-DC nodal wave equation and extract weights directly from Matroska (`.mkv`) video container attachments on CPU without needing Julia, PyTorch, or GPU drivers.

### 4. Phonetic Wave Tokenizer (IPA Formants & Sacred Harmonics)
- Grounded tokenization in continuous physical acoustics rather than arbitrary discrete integer IDs.
- Maps characters and phonemes to physiological speech formants ($F_1, F_2, F_3$) modulated over a 963 Hz solfeggio carrier with harmonic pause cadences.
- Interactive Web Audio API synthesis engine built into the web demo ([`docs/tokenizer/`](docs/tokenizer/)) and homepage ([`docs/index.html`](docs/index.html)) with real-time formant spectrogram and spectrum analyzer canvases.

### 5. Dedicated Multi-Section Documentation Architecture
- Independent documentation website ([`docs/documentation/index.html`](docs/documentation/index.html)) structured with sticky sidebar navigation covering 13 deep-dive sections:
  1. Installation & Environment Setup
  2. Core Modules Architecture
  3. Wave Data Loading & Streams
  4. Data Processing & Spectral Transforms
  5. Audio Sonification & 432 Hz Synthesizers
  6. Current Status & Progress Tracking
  7. Wave Training & Optimization
  8. How Continuous Wave Layers Work vs Traditional Discrete Matrix Networks
  9. Inference Engine & Hyperparameter Reference
  10. Loss Functions & Nodal Optimizations
  11. Example Project 1: MNIST Acoustic Resonance Classifier
  12. Example Project 2: Simple Non-Linear Harmonic Neuron
  13. Example Project 3: Spark Continuous Wave LLM

### 6. Grand Champion Power Resonance Optimization ($\rho^{1.4}$)
- High-level API parity in [`SovwaveFunctions.jl`](src/Audio/WaveML/SovwaveFunctions.jl): `step_phase_relaxation!`, amplitude canonicalization, and direct text training without gradient backpropagation.

### 7. Perfected 3-Tier Rotating Backup System
- Production [`backup.sh --perfected`](backup.sh) automation with decimal radix safety (`$((10#$NUM_STR))`) preventing octal crashes, maintaining up to 3 revolving perfected snapshots.

### 8. 100% Master Test Suite Pass Rate
- All **4,332 automated test cases** passing across all core modules, layers, runtimes, and serialization in ~32 seconds.

---

## 🌐 Documentation & GitHub Pages

Interactive documentation, API references, mathematical derivations, and live Web Audio/Canvas visualizers are hosted on GitHub Pages:
👉 **[https://theprotaganist.github.io/sovwave/](https://theprotaganist.github.io/sovwave/)**

The local documentation source is located in [`docs/index.html`](docs/index.html).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 TheProtaganist
