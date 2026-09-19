"""
    Audio

Quantum-acoustic synthesis engine within Aetheria — bridges sacred geometry,
wave mechanics, and real-time audio generation. Unlike traditional discrete
sample-based synthesis, the Audio module treats sound as continuous wave
functions that maintain phase coherence across all buffer boundaries.

# Mathematical Foundation

The Audio module is grounded in three key principles:

1. **Wave-Continuous Processing**: Sound is represented as continuous mathematical
   functions `ψ(t) = A·f(2πft + φ)` rather than discrete sample arrays. Phase state
   `φ` advances coherently across buffers: `φ_new = mod2π(φ_old + 2πfΔt·N_samples)`.

2. **Sacred Tuning (432Hz)**: Uses A4 = 432Hz tuning instead of the standard 440Hz,
   aligning with golden ratio (φ ≈ 1.618) and Fibonacci harmonics. Frequencies are
   computed via equal temperament: `f(n) = 432 · 2^((n-69)/12)` for MIDI note `n`.

3. **Quantum-Inspired Superposition**: Multiple wave functions combine via linear
   superposition `Ψ = Σᵢ ψᵢ`, with probabilistic collapse for quantum-like state
   selection. Fractal generation uses golden ratio scaling: `f_harmonic = f₀ · φⁿ`.

# Architecture

The module is organized following Aetheria conventions (specs/Structure.md §2):

- `Core/`: Fundamental types (WaveFunction, AudioConstants) and phase tracking
- `Generation/`: Wave synthesis (TrigGenerator, SacredTuning, FractalGenerator)
- `Processing/`: Wave manipulation (BinauralEngine, Superposition, QuantumProcessor)
- `Output/`: Real-time audio (RingBuffer, AudioDevice) with lock-free concurrency
- `Native/`: C++/Assembly for SIMD trigonometric generation (4x-8x speedup)

# Performance Targets

- Single wave generation: <1ms for 1024 samples @ 48kHz
- 16-wave superposition: <5ms per buffer
- Fractal generation (depth 10): <10ms per buffer
- Total audio callback latency: <10ms (prevents glitches/underruns)

# Integration with Aetheria

Audio waves are compatible with Aetheria's WaveField type for future quantum
interference experiments. The `QuantumProcessor` module provides a bridge for
applying Aetheria's morphogenetic wave mechanics to audio synthesis.

# References

- Sacred tuning: Horowitz, L. (1998). "Healing Codes for the Biological Apocalypse"
- Golden ratio harmonics: Livio, M. (2002). "The Golden Ratio: The Story of Phi"
- Binaural beats: Oster, G. (1973). "Auditory Beats in the Brain", Scientific American
- Wave superposition: Griffiths, D.J. (2018). "Introduction to Quantum Mechanics"
- Ginzburg-Landau theory: See `specs/Requirements.md` Mathematical Framework §C

# Example Usage

```julia
using Aetheria.Audio

# Generate a 432Hz sine wave
wave = WaveFunction(frequency=432.0, amplitude=0.5, trig_func=:sin)
buffer = zeros(Float64, 1024)
generate_buffer!(buffer, wave, 48000.0)

# Create a theta-wave binaural beat (6Hz) for meditation
binaural = create_brainwave_beat(:theta, carrier=432.0)
stereo_buffer = zeros(Float64, 2, 1024)
generate_binaural!(stereo_buffer, binaural, 48000.0)

# Superpose multiple waves with golden ratio harmonics
waves = [WaveFunction(frequency=432.0 * φ^n) for n in 0:3]
combined = zeros(Float64, 1024)
superpose!(combined, waves, 48000.0)

# Real-time audio output
device = init_device(AudioConfig(sample_rate=48000, buffer_size=512, channels=2))
start!(device)
# ... write samples to device.ring_buffer ...
stop!(device)
close!(device)
```

See `docs/audio/tutorial.md` for comprehensive guides and `examples/audio_*.jl`
for runnable demonstrations of all features.
"""
module Audio

using Base # Explicit dependency clarity (follows Aetheria pattern)

# Module constants
const VERSION = v"0.1.0"

# --- Guarded includes (stub build pattern from Aetheria.jl) ------------------
const _AUDIO_DIR = joinpath(@__DIR__)

# Include order follows dependency graph (specs/Structure.md §2):
# Core → Generation → Processing → Output (Native is called via ccall)
for _rel in (
    "Core/AudioConstants.jl",
    "Core/WaveFunction.jl",
    "Core/WaveDataPoint.jl",
    "Native/ccall_bindings.jl",
    "Generation/SacredTuning.jl",
    "Generation/TrigGenerator.jl",
    "Generation/FractalGenerator.jl",
    "Processing/BinauralEngine.jl",
    "Processing/Superposition.jl",
    "Processing/QuantumProcessor.jl",
    "Processing/WaveComputing.jl",
    "Processing/WaveMath.jl",
    "WaveML/WaveML.jl",
    "Output/RingBuffer.jl",
    "Output/AudioDevice.jl",
)
    _p = joinpath(_AUDIO_DIR, _rel)
    if isfile(_p)
        include(_p)
    else
        @debug "Audio: $(_rel) not implemented yet — excluded from stub build (expected until its section lands)"
    end
end

# Re-export key components for convenient wave-based programming
export WaveFunction, WaveDataPoint, create_data_points, wave_hit!
export evaluate_wave_at_points, evaluate_wave_at_points_timed, reset_data_points!, data_point_summary
export SacredNote, midi_to_freq_432, freq_440_to_432, freq_432_to_440, golden_harmonic, fibonacci_harmonics, generate_chromatic_scale
export get_trig_function, evaluate_wave, generate_buffer!, generate_buffer_fractal!
export apply_fractal, fibonacci_fractal_wave, multiscale_pattern, fractal_dimension_spectrum
export BinauralBeat, generate_binaural!, create_brainwave_beat
export superpose!, normalize_preserve_phase!, interference_pattern
export QuantumWaveState, quantum_superpose, sample_categorical, quantum_interference!, normalize_probabilities!
export LockFreeRingBuffer

# Wave Programming & Tournament Winner Engine
export WaveProgram, WaveInstruction, WaveExecutionStats
export def_point!, def_points!, get_point, set_point!
export emit_wave!, emit_binaural!, emit_interference!, emit_fractal!
export wave_gate!, wave_unary_gate!
export run!, run_step!, metrics, reset_program!
export simd_fma_wave_hit!, simd_fma_wave_hit
export @wave_program

# Mathematical Wave Engine: Calculus, Topology, Trig, Geometry, Algebra
export wave_derivative, wave_second_derivative, wave_integral, wave_laplacian, wave_taylor_approx
export topological_winding_number, count_nodal_domains, phase_vortex_charge, apply_mobius_twist!
export evaluate_12_trig, fourier_harmonic_synthesis, lissajous_coordinate
export phi_logarithmic_spiral, platonic_solid_resonance, flower_of_life_field
export evaluate_algebraic_expression, compile_wave_expression, WaveVariableContext, WaveMathDomain

# WaveML Deep Learning System Exports
using .WaveML
export WaveML
export WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig, WaveMLConfig, parse_color_rgb
export load_config, save_config, default_config
export WaveFieldPoint, WaveField, create_field, propagate_field!, field_energy, reset_field!, evaluate_field
export WaveLayer, create_layer, forward!, layer_energy, mutate!, crossover
export WaveModel, model_energy, clone
export compute_loss, energy_loss, mmd_loss, resonance_loss, interference_loss, wave_accuracy
export EvolutionState, init_population, evaluate_population!, evolve_generation!
export WaveForm, WaveTokenizer, default_tokenizer, build_tokenizer, tokenize, tokenize_ids, decode, to_wave_form, to_audio, sonify_tokens, to_wave_packet, encode_sequence, decode_embedding, decode_sequence_embeddings
export unicode_wave_frequency, unicode_wave_phase, token_wave_frequency, token_wave_phase, register_token!
export WaveDataset, WaveDataLoader, format_tabular, format_text, format_images, format_timeseries, format_jev, format_dataset, from_tabular, from_text, from_image, from_timeseries, from_jev_state, num_samples, batch_size, num_batches
export load_hf_dataset, hf_auth_token, hf_dataset_info
export WaveHead, create_head, apply_head, head_loss, mutate_head!
export generate_text, generate_image, generate_3d, jev_decide, generate_video, register_model_type!, list_model_types
export num_layers, parameter_count, get_layer, layer_details, model_summary, inspect_model
export TrainingMetrics, TrainingHistory, train!
export save_model, load_model, model_to_rgb_frames, rgb_frames_to_model, model_to_visual_frames, emergent_color
export infer, predict
export sonify_model, sonify_step, save_wav, play_realtime!

end # module Audio
