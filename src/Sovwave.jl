"""
    Sovwave

From-scratch Julia library for pure wave-based computation, quantum acoustic synthesis,
and continuous morphogenetic wave intelligence (Self-Organizing Vacuum / SOV dynamics).

Zero matrix multiplication (no CUDA, no discrete CPU matmul paradigm), zero Markov chains,
zero backpropagation. Computes via continuous harmonic wave interference in physical manifolds.
"""
module Sovwave

using Base # explicit self-containment intent (empty deps; nothing but stdlib/Base)

const VERSION = v"0.3.5"

# --- guarded includes --------------------------------------------------------
const _SRC_DIR = joinpath(@__DIR__)

for _rel in (
    "Core/Grid.jl",
    "Core/WaveField.jl",
    "Core/Constants.jl",
    "Encoding/FrequencySpectrum.jl",
    "Encoding/HarmonicEncoder.jl",
    "Potentials/SacredGeometry.jl",
    "Potentials/FlowerOfLife.jl",
    "Potentials/MetatronsCube.jl",
    "Simulation/MorphogeneticSimulator.jl",
    "Simulation/TimeEvolution.jl",
    "Simulation/EnergyFunctionals.jl",
    "Extraction/EigenState.jl",
    "Extraction/CymaticExtractor.jl",
    "Optimization/WaveOptimizer.jl",
    "Optimization/GradientFlow.jl",
    "Optimization/AdamWave.jl",
    "Loss/WaveLoss.jl",
    "Loss/FrequencyDomainMSE.jl",
    "Loss/PhaseCoherenceLoss.jl",
    "Loss/FreeEnergyLoss.jl",
    "Activation/WaveActivation.jl",
    "Activation/WaveSigmoid.jl",
    "Activation/WaveReLU.jl",
    "Activation/WaveTanh.jl",
    "Networks/SOVLayer.jl",
    "Networks/SOVDense.jl",
    "Networks/SOVNetwork.jl",
    "Training/TrainingLoop.jl",
    "Training/Validation.jl",
    "Training/EarlyStopping.jl",
    "Inference/SingleSample.jl",
    "Inference/BatchInference.jl",
    "Data/DataLoader.jl",
    "Data/Preprocessing.jl",
    "Data/Augmentation.jl",
    "Persistence/ModelMetadata.jl",
    "Persistence/MP4Storage.jl",
    "Persistence/GGUFConverter.jl",
    "WaveProtocols/WaveProtocol.jl",
    "WaveProtocols/SignalProcessing.jl",
    "WaveProtocols/AudioProtocols.jl",
    "WaveProtocols/EMFProtocols.jl",
    "WaveProtocols/RadioProtocols.jl",
    "WaveProtocols/TestRunner.jl",
    "WaveProtocols/ProtocolLogger.jl",
    "Audio/Audio.jl",
    "GUI/Server.jl",
)
    _p = joinpath(_SRC_DIR, _rel)
    if isfile(_p)
        include(_p)
    else
        @debug "Sovwave: $(_rel) not implemented yet — excluded from stub build"
    end
end

# Re-export Audio and WaveML
using .Audio
export Audio
using .Audio.WaveML
export WaveML

# Re-export WaveML APIs at top-level Sovwave namespace
export WaveModel, WaveLayer, create_layer, WaveField, WaveForm, WaveTokenizer, WaveDataset, WaveDataLoader, WaveHead
export WaveMLConfig, WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig
export default_config, load_config, save_config
export default_tokenizer, build_tokenizer, tokenize, tokenize_frequencies, token_frequency, token_frequencies, tokenize_ids, decode, to_wave_form, to_audio, sonify_tokens, to_wave_packet, encode_sequence, decode_embedding
export unicode_wave_frequency, unicode_wave_phase, token_wave_frequency, token_wave_phase, register_token!
export convert_tokenizer, load_tokenizer_file, load_huggingface_tokenizer, convert_hf_tokenizer
export load_pretrained_tokenizer, gpt2_tokenizer, qwen_tokenizer, mistral_tokenizer, llama_tokenizer, deepseek_tokenizer, deepseek_v4_tokenizer
export save_tokenizer, load_tokenizer, custom_tokenizer
export format_tabular, format_text, format_images, format_timeseries, format_jev, format_dataset
export from_tabular, from_text, from_image, from_timeseries, from_jev_state, num_samples, batch_size, num_batches
export load_hf_dataset, hf_auth_token, hf_dataset_info
export generate_text, generate_image, generate_3d, jev_decide, generate_video, register_model_type!, list_model_types
export WaveHead, create_head, apply_head, head_loss, mutate_head!
export inspect_model, model_summary, parameter_count, num_layers, get_layer, layer_details
export save_model, load_model, infer, predict, train!, train_text!, train_llm, sonify_model, save_wav
export forward!, mutate!, crossover, layer_energy, model_energy, init_population, evaluate_population!, evolve_generation!

# Continuous Physical Wave Computing & Morphogenetic Dynamics (agenda.md)
export FlowerOfLifePotential, flower_of_life_potential, compute_flower_of_life_grid!
export MorphogeneticConfig, MorphogeneticField2D, step_schrodinger!, step_ginzburg_landau!, relax_to_eigenstate!, compute_free_energy, compute_free_energy!
export CymaticExtractorConfig, extract_cymatic_eigenfrequency, compute_phase_coherence, detect_standing_wave_nodes, cymatic_decode_vocab

# Sound-Native Computing Substrate & Fractal Storage
export SoundComputeConfig, SoundAcousticBuffer, sound_native_forward!, sound_native_model_forward!
export FractalLatticeConfig, evaluate_fractal_layer!, compress_layer_to_fractal, get_fractal_param

# Optional GPU acceleration & Hybrid Engine
export enable_cuda!, disable_cuda!, cuda_available, to_gpu, to_cpu
export WaveHybridDispatcher, hybrid_dispatcher, benchmark_system_vs_cuda
export hybrid_forward!, hybrid_forward_batch, hybrid_project_vocab, hybrid_evaluate!

# User-Friendly High-Level API (SovwaveFunctions)
using .Audio.WaveML.SovwaveFunctions
export SovwaveFunctions
export create_model
export edit_layer!, edit_model!, modulate_frequencies!, shift_phases!, scale_amplitudes!, inspect_harmonics
export load_dataset, validate_dataset, process_to_waves
export WaveResonator, WaveChamber
export WaveMechanicsOptimizer, step_mechanics!, apply_wave_mechanics!
export train_async, train_wave, AsyncTrainingHandle

# Visual GUI Server
using .GUI
export launch_gui, stop_gui!

# Backwards compatibility alias
const Aetheria = Sovwave
export Sovwave, Aetheria

end # module Sovwave
