"""
    Sovwave

From-scratch Julia library for pure wave-based computation, quantum acoustic synthesis,
and continuous morphogenetic wave intelligence (Self-Organizing Vacuum / SOV dynamics).

Zero matrix multiplication (no CUDA, no discrete CPU matmul paradigm), zero Markov chains,
zero backpropagation. Computes via continuous harmonic wave interference in physical manifolds.
"""
module Sovwave

using Base # explicit self-containment intent (empty deps; nothing but stdlib/Base)

const VERSION = v"0.3.2"

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
export WaveModel, WaveLayer, WaveField, WaveForm, WaveTokenizer, WaveDataset, WaveDataLoader, WaveHead
export WaveMLConfig, WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig
export default_config, load_config, save_config
export default_tokenizer, build_tokenizer, tokenize, tokenize_frequencies, token_frequency, token_frequencies, tokenize_ids, decode, to_wave_form, to_audio, sonify_tokens, to_wave_packet, encode_sequence, decode_embedding
export unicode_wave_frequency, unicode_wave_phase, token_wave_frequency, token_wave_phase, register_token!
export convert_tokenizer, load_tokenizer_file, load_huggingface_tokenizer, convert_hf_tokenizer
export load_pretrained_tokenizer, gpt2_tokenizer, qwen_tokenizer, mistral_tokenizer, llama_tokenizer
export save_tokenizer, load_tokenizer, custom_tokenizer
export format_tabular, format_text, format_images, format_timeseries, format_jev, format_dataset
export from_tabular, from_text, from_image, from_timeseries, from_jev_state, num_samples, batch_size, num_batches
export load_hf_dataset, hf_auth_token, hf_dataset_info
export generate_text, generate_image, generate_3d, jev_decide, generate_video
export inspect_model, model_summary, parameter_count, num_layers, get_layer, layer_details
export save_model, load_model, infer, predict, train!, train_text!, train_llm, sonify_model, save_wav
export forward!, mutate!, crossover, layer_energy, model_energy, init_population, evaluate_population!, evolve_generation!

# Optional GPU acceleration & Hybrid Engine
export enable_cuda!, disable_cuda!, cuda_available, to_gpu, to_cpu
export WaveHybridDispatcher, hybrid_dispatcher, benchmark_system_vs_cuda
export hybrid_forward!, hybrid_forward_batch, hybrid_project_vocab, hybrid_evaluate!

# Visual GUI Server
using .GUI
export launch_gui, stop_gui!

# Backwards compatibility alias
const Aetheria = Sovwave
export Sovwave, Aetheria

end # module Sovwave
