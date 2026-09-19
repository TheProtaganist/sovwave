# WaveML: Pure Wave Computing Deep Learning System
## User & Developer Guide

WaveML is an alternative deep learning framework designed in Julia. It eliminates traditional binary logic, matrix multiplications, backpropagation, and Markov chains. All computations occur through **continuous wave evolution seeking the lowest possible energy state**.

Models are saved as **lossless MKV video files** (`.mkv`) where each pixel visually reflects the model's evolved brain, and predictions are run directly on the video files. Training can be heard as sound with user-definable carrier frequencies (432 Hz, 440 Hz, 528 Hz, or custom), waveforms, envelopes, and binaural beats.

---

## 1. Quick Start

### Installation & Loading
From within Julia:
```julia
using Pkg
Pkg.activate(".")

using Aetheria
using Aetheria.Audio
```

### 3-Minute Example
```julia
using Aetheria.Audio

# 1. Load config (or customize in Julia)
cfg = load_config("model_config.yaml")

# 2. Create the Wave Model
model = WaveModel(cfg)

# 3. Create training data (inputs in embedding space, targets in node space)
inputs  = [rand(cfg.model.embed_dims) for _ in 1:100]
targets = [zeros(cfg.model.nodes) for _ in 1:100] # Seeks ground-state zero energy

# 4. Train via wave evolution (No backprop, no Markov chains!)
trained_model, history = train!(
    model, inputs, targets, cfg.train;
    audio_cfg = cfg.audio, # User-defined carrier frequency and sound
    audio_save_dir = "audio_output"
)

# 5. Save the model's "brain" as an MKV video
save_model(trained_model, "my_wave_model.mkv")

# 6. Run inference directly from the MKV video!
test_sample = rand(cfg.model.embed_dims)
prediction = infer("my_wave_model.mkv", test_sample)
println("Output from MKV model: ", prediction[1:5])
```

---

## 2. YAML Configuration (`model_config.yaml`)

Every property of the wave field, model lattice, training loop, and audio sonification is configurable via YAML:

```yaml
# ============================================================================
# WaveML Model Configuration (Pure Wave-Based Computing)
# ============================================================================

# Wave Field: Data points exist on a d-dimensional continuous wave surface
field:
  n_points: 128             # Number of data points n (0 to n)
  properties:               # x properties shared by all data points
    - mass
    - charge
    - energy
    - spin
  dimensions: 3             # d dimensions (1D curve, 2D plane, 3D manifold, 4D, 5D+)
  distribution: uniform     # uniform, fibonacci, random, lattice

# Model Architecture: Quantum lattice and temporal superposition
model:
  layers: 4                 # l layers
  embed_dims: 32            # d embedding dimensions across l layers
  nodes: 64                 # n quantum lattice nodes per layer
  omega: 432.0              # Harmonic frequency ω (Hz) for temporal superposition
  beta_s: 1.618033989       # Fractal scaling parameter (Golden Ratio Φ)
  t_frames: 8               # Temporal superposition time frames

# Training Parameters: Evolution-based optimization seeking ground-state energy
train:
  batch_size: 16            # Batch size b
  learning_rate: 0.05       # Mutation amplitude scale lr
  epochs: 50
  population_size: 24       # Number of wave models in evolution population
  elite_fraction: 0.15      # Proportion of elite models preserved per generation
  mutation_decay: 0.995     # 1Cycle harmonic mutation annealing factor
  energy_target: 0.001      # Target ground-state energy for early stopping
  sonify: true              # Output training as sound
  sonify_realtime: false    # Set true to ALSO stream to speakers in real-time
  audio_sample_rate: 48000

# User-Defined Sound & Audio Properties:
# Complete control over carrier frequency, waveform, binaural beats, and tuning.
# Set carrier_frequency to 432.0, 440.0, 528.0 (Solfeggio), or ANY custom value!
audio:
  carrier_frequency: 432.0    # Carrier frequency in Hz (e.g., 432.0, 440.0, 528.0, or custom)
  tuning_standard: 432.0      # Base tuning reference frequency in Hz
  waveform: physical          # Waveform: sine, harmonic, physical, triangle, sawtooth, binaural
  binaural_beat: 10.0         # Stereo beat frequency offset in Hz (10.0 Alpha, 6.0 Theta, 0.0 off)
  envelope: exponential_decay # Envelope: exponential_decay, adsr, percussive, sustain
  attack: 0.02                # ADSR Attack time in seconds
  decay: 0.15                 # ADSR Decay time in seconds
  sustain: 0.60               # ADSR Sustain level (0.0 to 1.0)
  release: 0.25               # ADSR Release time in seconds
  harmonic_richness: 1.2      # Overtone factor (0.0 = pure fundamental, 2.0 = rich harmonics)
  volume: 0.85                # Master audio volume gain (0.0 to 1.0)
  pan: 0.0                    # Stereo pan (-1.0 = left, 0.0 = center, +1.0 = right)
  sample_rate: 48000          # Audio sample rate in Hz (44100, 48000, 96000)
  channels: 2                 # 1 = mono, 2 = stereo (allows binaural spatial depth)
  realtime_player: auto       # Command for live playback ("auto", "paplay", "aplay", "pw-play")
```

---

## 3. Audio & Sound Control (User-Defined Frequencies)

You can control all sound parameters directly in YAML or programmatically in Julia:

### Custom Carrier Frequencies
```julia
# 440 Hz (Concert Pitch A4)
audio_cfg_440 = WaveAudioConfig(carrier_frequency=440.0, waveform=:sine)

# 528 Hz (Solfeggio / DNA Repair Frequency) with Stereo Alpha Entrainment (10 Hz)
audio_cfg_528 = WaveAudioConfig(
    carrier_frequency = 528.0,
    waveform = :harmonic,
    binaural_beat = 10.0,
    channels = 2,
    volume = 0.9
)

# Any arbitrary frequency (e.g. 314.159 Hz or 108.0 Hz)
audio_cfg_custom = WaveAudioConfig(
    carrier_frequency = 108.0,
    envelope = :adsr,
    attack = 0.05,
    decay = 0.20,
    sustain = 0.70,
    release = 0.30
)
```

### Waveform Choices
| Waveform | Symbol | Characteristic |
|---|---|---|
| Pure Sine | `:sine` | Fundamental frequency, pure tone |
| Harmonic | `:harmonic` | Fundamental + overtone cascade controlled by `harmonic_richness` |
| Physical Resonator | `:physical` | Tournament champion: spring-mass resonator with natural damping |
| Triangle | `:triangle` | Warm odd harmonics |
| Sawtooth | `:sawtooth` | Rich, bright full-spectrum harmonics |
| Binaural | `:binaural` | Left ear: $f - \Delta f / 2$, Right ear: $f + \Delta f / 2$ |

### Exporting Sound
```julia
# 1. Render model parameters to audio buffer
buffer = sonify_model(model; audio_cfg=audio_cfg_528, duration=1.0)

# 2. Save to 16-bit PCM WAV (automatically detects mono or stereo)
save_wav(buffer, "output_528hz.wav"; sample_rate=48000)

# 3. Play through system audio in real time
play_realtime!(buffer; sample_rate=48000)
```

---

## 4. MKV Video Model Serialization & Inference

In WaveML, models are saved as actual **MKV video files** (`.mkv`) using lossless `FFV1` encoding:
- **Each frame is a layer** of the model.
- **The pixel grid is the quantum lattice**: width = embedding dimensions, height = nodes.
- **Each pixel's color is a learned evolution**:
  - Red = Amplitude
  - Green = Phase $[0, 2\pi]$
  - Blue = Relative Frequency
- Colors are not artificially constrained: the evolution itself defines what colors and emergent visual structures represent the model's brain.
- You can open the `.mkv` file in **VLC, MPV, or any video player** to directly watch the model's brain!

### Saving Models as MKV
```julia
save_model(model, "wave_brain.mkv"; fps=2)
```

### Loading & Inference directly from MKV
```julia
# Run predictions directly from the video file!
outputs = infer("wave_brain.mkv", test_samples)

# Or load the model object from the video file
loaded_model = load_model("wave_brain.mkv")
outputs = predict(loaded_model, test_samples)
```

---

## 5. Training Mechanics & Speed Metrics

WaveML reports real-time calculation metrics during training:
- **Calculation time**: Time taken per epoch in milliseconds (`calc_time_ms`)
- **Speed per point**: Average nanoseconds per data point evaluation (`ns/pt`)
- **Throughput**: Data points processed per second (`throughput_pts_sec`)
- **Energy**: Total ground-state loss (`loss`)
- **Accuracy**: Activation accuracy (`accuracy`)

Example training output:
```text
==============================================================================
 ⚡ WAVEML PURE WAVE COMPUTING: TRAINING INITIALIZED ⚡
==============================================================================
  Dataset Size: 200 | Batch Size: 16 | Epochs: 25 | Pop Size: 16
  Base Mutation Rate (lr): 0.0800 | Energy Target: 0.005000
  User-Defined Carrier Frequency: 432.00 Hz | Waveform: physical
------------------------------------------------------------------------------
  Epoch   1/ 25 | Energy:  0.31621 | Acc:  59.2% | Calc: 126.10 ms |    2030.1 pts/s | lr: 0.0187
  Epoch   5/ 25 | Energy:  0.31621 | Acc:  57.4% | Calc: 130.20 ms |    1966.2 pts/s | lr: 0.0613
  Epoch  10/ 25 | Energy:  0.29168 | Acc:  57.1% | Calc: 123.97 ms |    2065.1 pts/s | lr: 0.0760
  Epoch  15/ 25 | Energy:  0.21063 | Acc:  61.4% | Calc: 128.90 ms |    1986.0 pts/s | lr: 0.0489
  Epoch  20/ 25 | Energy:  0.21063 | Acc:  60.9% | Calc: 125.28 ms |    2043.5 pts/s | lr: 0.0151
  Epoch  25/ 25 | Energy:  0.21063 | Acc:  59.6% | Calc: 136.76 ms |    1871.8 pts/s | lr: 0.0000
------------------------------------------------------------------------------
 🏆 TRAINING COMPLETE: Best Energy = 0.210634 (Epoch 14) in 4.08 seconds
==============================================================================
```

---

## 6. Running Tests & Demonstrations

### Run Master Test Suite
```bash
julia --project=. test/runtests.jl
```
*(453 / 453 tests pass)*

### Run WaveML Demo
```bash
julia --project=. examples/waveml_demo.jl
```
*(Demonstrates YAML config loading, wave evolution training, 528 Hz stereo binaural audio generation, MKV video brain saving, and direct inference from MKV!)*
