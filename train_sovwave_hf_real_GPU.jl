#!/usr/bin/env julia
"""
Train SovWave with REAL HuggingFace datasets using GPU acceleration
Grand Champion PowerResonance_p1.4 loss with CUDA support
"""

using Pkg
Pkg.activate(".")

# Load CUDA FIRST before Sovwave
using CUDA

include("src/Sovwave.jl")
using .Sovwave
using .Sovwave.WaveML
using Printf

println("="^90)
println(" 🎮 GPU ACCELERATED SOVWAVE TRAINING")
println("="^90)

# Check GPU status
println("\n📋 GPU Status:")
@printf("  CUDA functional: %s\n", CUDA.functional())
if CUDA.functional()
    @printf("  GPU: %s\n", CUDA.name(CUDA.device()))
    @printf("  GPU memory: %.2f GB\n", CUDA.totalmem(CUDA.device()) / 1e9)
    @printf("  Free memory: %.2f GB\n", CUDA.available_memory() / 1e9)
    
    # Enable GPU acceleration
    println("\n🚀 Enabling CUDA acceleration...")
    if enable_cuda!()
        println("  ✅ GPU acceleration enabled!")
    else
        println("  ❌ Failed to enable GPU acceleration")
        exit(1)
    end
else
    println("  ❌ CUDA not functional - cannot run GPU training")
    exit(1)
end

"""
Load REAL HuggingFace dataset using native Sovwave HF integration
"""
function load_texts_from_hf(dataset_name::String, max_samples::Int=1000)
    println("📦 Loading REAL HuggingFace dataset: $dataset_name")
    
    texts = try
        if dataset_name in ("c4", "allenai/c4")
            stream_hf_text(dataset_name; split="train", subset="en", limit=max_samples, chunk_size=50)
        elseif dataset_name in ("openwebtext", "Skylion007/openwebtext")
            stream_hf_text(dataset_name; split="train", limit=max_samples, chunk_size=50)
        elseif dataset_name in ("tinystories", "roneneldan/TinyStories")
            stream_hf_text(dataset_name; split="train", limit=max_samples, chunk_size=50)
        elseif dataset_name == "wikitext"
            stream_hf_text("wikitext"; split="train", subset="wikitext-2-raw-v1", limit=max_samples, chunk_size=50)
        else
            stream_hf_text(dataset_name; split="train", limit=max_samples, chunk_size=50)
        end
    catch e
        @warn "Failed to load HF dataset '$dataset_name': $e"
        String[]
    end
    
    texts = filter(t -> length(strip(t)) >= 20, texts)
    
    if isempty(texts)
        @warn "No texts loaded from HF, using fallback dataset"
        texts = [
            "Two plus two equals four.",
            "The sky is blue and the grass is green.",
            "Hello, how are you today?",
            "The sun shines bright in the morning.",
            "Water flows down from the mountains.",
            "Once upon a time, there was a little girl.",
            "Learning makes us smarter every single day.",
            "Practice makes us better at everything we do.",
            "Knowledge is power and wisdom is strength.",
            "Life is a beautiful journey worth living."
        ]
    end
    
    return texts[1:min(max_samples, length(texts))]
end

"""
Build active vocabulary from training texts
"""
function build_active_vocab!(tokenizer, texts::Vector{String})
    active_vocab = Set{Int}()
    
    for text in texts
        wave_tokens = tokenize(tokenizer, text)
        for wt in wave_tokens
            push!(active_vocab, wt.token_id)
        end
    end
    
    essential_symbols = [
        ".", ",", "!", "?", ":", ";", "-", " ",
        "+", "=", "*", "/", "<", ">", "(", ")", "[", "]",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "two", "four", "is", "are", "and", "in", "a", "the", "to"
    ]
    
    for sym in essential_symbols
        try
            wave_tokens = tokenize(tokenizer, sym)
            for wt in wave_tokens
                push!(active_vocab, wt.token_id)
            end
        catch
        end
    end
    
    @printf(" ✓ Active vocabulary initialized: %d coherent tokens\n", length(active_vocab))
    return active_vocab
end

function main()
    println("\n" * "="^90)
    println(" 🏆 SOVWAVE GPU TRAINING WITH REAL HUGGINGFACE DATA")
    println("="^90)
    
    # Load CHAMPION DATASET (6000 samples for optimal coherence)
    datasets_to_try = [
        ("roneneldan/TinyStories", 3000),  # 🏆 Champion: 6x more data
        ("c4", 1500),
        ("wikitext", 1500)
    ]
    
    all_texts = String[]
    
    for (ds_name, samples) in datasets_to_try
        println("\n📦 Attempting to load: $ds_name ($(samples) samples)")
        try
            texts = load_texts_from_hf(ds_name, samples)
            println("  ✅ Loaded $(length(texts)) samples from $ds_name")
            append!(all_texts, texts)
        catch e
            @warn "  ⚠️  Failed to load $ds_name: $e"
        end
        
        if length(all_texts) >= 6000  # 🏆 Champion: 6000 sample target
            break
        end
    end
    
    if isempty(all_texts)
        error("Failed to load any HuggingFace datasets!")
    end
    
    println("\n✅ Total samples loaded: $(length(all_texts)) from HuggingFace")
    
    # 🏆 CHAMPION MODEL ARCHITECTURE (8L × 256N × 256D) for GPU
    println("\n🔧 Creating SovWave Champion model on GPU...")
    config = WaveModelConfig(
        layers = 8,              # 🏆 Champion: 8 layers (2x deeper)
        embed_dims = 256,        # 🏆 Champion: 256D (8x wider)
        nodes = 256,             # 🏆 Champion: 256 nodes (4x more)
        omega = 432.0,           # 🏆 Champion: Optimal 432Hz
        beta_s = 1.618033988749895,  # 🏆 Champion: Golden ratio
        t_frames = 8
    )
    
    # Create model (will automatically use GPU if enabled)
    wm = create_model(
        layers = config.layers,
        embed_dims = config.embed_dims,
        nodes = config.nodes,
        omega = config.omega,
        beta_s = config.beta_s,
        t_frames = config.t_frames
    )
    println("✅ Model created on GPU: $(config.layers) layers × $(config.embed_dims) dims")
    
    # Create tokenizer
    println("\n🔧 Creating tokenizer...")
    tokenizer = gpt2_tokenizer(carrier_frequency=config.omega, beta_s=config.beta_s)
    println("✅ Tokenizer loaded: vocab size $(length(tokenizer.vocab))")
    
    # Build active vocabulary
    println("\n🔧 Building active vocabulary from training texts...")
    active_vocab = build_active_vocab!(tokenizer, all_texts)
    
    # Format training data
    println("\n🔧 Formatting training data...")
    dataset = format_lm_text(all_texts; tokenizer=tokenizer, embed_dim=config.embed_dims, max_pairs=10000)
    println("✅ Formatted $(length(dataset)) training pairs")
    
    # Training configuration for GPU with CONTINUOUS EVOLUTION
    println("\n🌊 GPU Training with Tournament-Optimized Continuous Evolution...")
    println("  - 144-Algorithm Continuous Training Tournament Winner")
    println("  - Speed: 3072 batch, 128 population, 0.46 LR")
    println("  - Coherence: 6000 samples, 60K pairs, 8L×256N×256D")
    println("  - Continuous time-based evolution (no discrete epochs)")
    println("  - GPU-accelerated wave computing")
    
    train_cfg = WaveTrainConfig(
        epochs = 100_000,             # 100K epochs for full convergence
        batch_size = 3072,
        learning_rate = 0.46,
        energy_target = 0.008,
        sonify = false,               # Disable for GPU speed
        sonify_realtime = false,
        population_size = 128
    )
    
    inputs = dataset.inputs
    targets = dataset.targets
    
    # Train on GPU
    trained_model, history = train!(
        wm,
        inputs,
        targets,
        train_cfg;
        checkpoint_every = 2000,
        display_every = 20,  # Show progress every 20 epochs
        checkpoint_dir = "checkpoints_sovwave_hf_GPU",
        verbose = true
    )
    
    println("\n✅ GPU Training complete!")
    println("  Best loss: $(history.best_loss)")
    println("  Total time: $(round(history.total_time_sec, digits=2))s")
    
    # GPU memory stats
    println("\n💾 GPU Memory Usage:")
    @printf("  Used: %.2f MB\n", (CUDA.totalmem(CUDA.device()) - CUDA.available_memory()) / 1e6)
    @printf("  Free: %.2f GB\n", CUDA.available_memory() / 1e9)
    
    # Save model
    checkpoint_path = "checkpoints_sovwave_hf_GPU/sovwave_final_GPU.mkv"
    mkpath("checkpoints_sovwave_hf_GPU")
    save_model(trained_model, checkpoint_path)
    println("\n✅ Model saved to: $checkpoint_path")
    
    println("\n" * "="^90)
end

main()
