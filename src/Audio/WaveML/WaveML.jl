"""
    WaveML

Pure Wave Computing Deep Learning Framework for Julia.
Replaces discrete binary neural networks and Markov chain approximations with
continuous wave evolution:
- Multi-dimensional wave fields (1D through 5D+) with configurable properties
- Quantum lattice nodes and harmonic temporal superposition driven by ω (Hz)
- Evolution-based optimization seeking lowest possible energy ground state
- Continuous harmonic wave tokenizer (WaveTokenizer)
- Multi-modal dataset formatting & dataloaders (WaveDataset, WaveDataLoader)
- Native Hugging Face Hub dataset integration with token authentication
- 5 Native Model Architectures: LLM, Image Gen, Text-to-3D, Jev Decision Engine, Text-to-Video
- Task heads (WaveHead) for classification, regression, generation, decision, and embedding
- Model reading & introspection API (inspect_model, model_summary, parameter_count)
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

# Include all sub-modules in proper dependency order
include("Config.jl")
include("Field.jl")
include("Layer.jl")
include("Model.jl")
include("SoundCompute.jl")
include("FractalStorage.jl")
include("Loss.jl")
include("Evolution.jl")
include("HuggingFace.jl")
include("PretrainedVocab.jl")
include("Tokenizer.jl")
include("TokenizerConverter.jl")
include("Dataset.jl")
include("Heads.jl")
include("ModelTypes.jl")
include("Sonify.jl")
include("Training.jl")
include("Serialize.jl")
include("Inference.jl")
include("Introspect.jl")
include("CUDASupport.jl")

# Public exports: Config
export WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig, WaveMLConfig, parse_color_rgb
export load_config, save_config, default_config

# Public exports: Field
export WaveFieldPoint, WaveField
export create_field, propagate_field!, field_energy, reset_field!, evaluate_field

# Public exports: Layer
export WaveLayer
export create_layer, forward!, layer_energy, mutate!, crossover

# Public exports: Sound-Native Computing Substrate
export SoundComputeConfig, SoundAcousticBuffer, sound_native_forward!, sound_native_model_forward!

# Public exports: Fractal Storage & Continuous Compression
export FractalLatticeConfig, evaluate_fractal_layer!, compress_layer_to_fractal, get_fractal_param

# Public exports: Model
export WaveModel
export model_energy, clone

# Public exports: Loss
export compute_loss, energy_loss, mmd_loss, resonance_loss, interference_loss, wave_accuracy

# Public exports: Evolution
export EvolutionState
export init_population, evaluate_population!, evolve_generation!

# Public exports: Tokenizer & Converter
export WaveForm, WaveTokenizer, default_tokenizer, build_tokenizer
export tokenize, tokenize_ids, decode, to_wave_form, to_audio, sonify_tokens
export to_wave_packet, encode_sequence, decode_embedding, decode_sequence_embeddings
export unicode_wave_frequency, unicode_wave_phase, token_wave_frequency, token_wave_phase, register_token!
export convert_tokenizer, load_tokenizer_file, load_huggingface_tokenizer, convert_hf_tokenizer
export load_pretrained_tokenizer, gpt2_tokenizer, qwen_tokenizer, mistral_tokenizer, llama_tokenizer, deepseek_tokenizer, deepseek_v4_tokenizer
export save_tokenizer, load_tokenizer, custom_tokenizer

# Public exports: Dataset
export WaveDataset, WaveDataLoader, WaveDataStreamer
export format_tabular, format_text, format_lm_text, format_images, format_timeseries, format_jev, format_dataset
export process_pixel_waves, process_wave_tokens, process_digital_data, stream_dataset
export num_samples, batch_size, num_batches

# Public exports: HuggingFace
export load_hf_dataset, hf_auth_token, hf_dataset_info

# Public exports: Heads
export WaveHead, create_head, apply_head, head_loss, mutate_head!

# Public exports: Model Types
export generate_text, generate_image, generate_3d, jev_decide, generate_video
export register_model_type!, list_model_types

# Public exports: Introspection
export num_layers, parameter_count, get_layer, layer_details, model_summary, inspect_model

# Public exports: Training
export TrainingMetrics, TrainingHistory, train!, train_text!, train_llm

# Public exports: Serialization
export save_model, load_model, model_to_rgb_frames, rgb_frames_to_model, model_to_visual_frames, emergent_color
export serialize_wave_model_binary, deserialize_wave_model_binary

# Public exports: Inference
export infer, predict

# Public exports: Sonification
export sonify_model, sonify_step, save_wav, play_realtime!

# Public exports: CUDA (optional GPU acceleration)
export enable_cuda!, disable_cuda!, cuda_available, to_gpu, to_cpu
export WaveHybridDispatcher, hybrid_dispatcher, benchmark_system_vs_cuda
export hybrid_forward!, hybrid_forward_batch, hybrid_project_vocab, hybrid_evaluate!

end # module WaveML
