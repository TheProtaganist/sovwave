"""
    WaveML

Pure Wave Computing Deep Learning Framework for Julia.
Replaces discrete binary neural networks and Markov chain approximations with
continuous wave evolution:
- Multi-dimensional wave fields (1D through 5D+) with configurable properties
- Quantum lattice nodes and harmonic temporal superposition driven by ω (Hz)
- Evolution-based optimization seeking lowest possible energy ground state
- Models serialized losslessly as MKV video files (each pixel = evolved wave parameter)
- Inference directly from MKV video files
- Training sonification to hear the learning process as sound in real-time or WAV export
"""
module WaveML

using Base
using Printf
using Random
using Statistics
using LinearAlgebra
using YAML

# Include all sub-modules in dependency order
include("Config.jl")
include("Field.jl")
include("Layer.jl")
include("Model.jl")
include("Loss.jl")
include("Evolution.jl")
include("Sonify.jl")
include("Training.jl")
include("Serialize.jl")
include("Inference.jl")

# Public exports
export WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig, WaveMLConfig, parse_color_rgb
export load_config, save_config, default_config

export WaveFieldPoint, WaveField
export create_field, propagate_field!, field_energy, reset_field!, evaluate_field

export WaveLayer
export create_layer, forward!, layer_energy, mutate!, crossover

export WaveModel
export model_energy, clone

export compute_loss, energy_loss, mmd_loss, resonance_loss, interference_loss, wave_accuracy

export EvolutionState
export init_population, evaluate_population!, evolve_generation!

export TrainingMetrics, TrainingHistory, train!

export save_model, load_model, model_to_rgb_frames, rgb_frames_to_model, model_to_visual_frames, emergent_color

export infer, predict

export sonify_model, sonify_step, save_wav, play_realtime!

end # module WaveML
