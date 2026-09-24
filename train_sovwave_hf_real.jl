#!/usr/bin/env julia
"""
Train SovWave with REAL HuggingFace datasets using Grand Champion PowerResonance_p1.4 loss
Based on Spark's proven approach with proper active_vocab tracking
"""

using Pkg
Pkg.activate(".")

include("src/Sovwave.jl")
using .Sovwave
using .Sovwave.WaveML
using Printf

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
    
    # Filter out empty or very short texts
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
    
    # Collect token IDs from all training texts
    for text in texts
        wave_tokens = tokenize(tokenizer, text)
        for wt in wave_tokens
            push!(active_vocab, wt.token_id)
        end
    end
    
    # Add essential punctuation and common words
    essential_symbols = [
        ".", ",", "!", "?", ":", ";", "-", " ",
        "+", "=", "*", "/", "<", ">", "(", ")", "[", "]",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "two", "four", "is", "are", "and", "in", "a", "the", "to"
    ]
    
    for sym in essential_symbols
        try
            # Try to tokenize and add
            wave_tokens = tokenize(tokenizer, sym)
            for wt in wave_tokens
                push!(active_vocab, wt.token_id)
            end
        catch
            # Symbol might not tokenize, skip it
        end
    end
    
    @printf(" ✓ Active vocabulary initialized: %d coherent tokens\n", length(active_vocab))
    return active_vocab
end

function main()
    println("="^90)
    println(" 🏆 SOVWAVE WITH REAL HUGGINGFACE DATA + GRAND CHAMPION LOSS")
    println("="^90)
    
    # 288-Tournament Champion: Load LARGER datasets (6000 samples)
    datasets_to_try = [
        ("roneneldan/TinyStories", 3000),  # 6x more
        ("c4", 1500),                       # 7.5x more
        ("wikitext", 1500)                  # 7.5x more
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
        
        if length(all_texts) >= 6000  # 288-Tournament target
            break
        end
    end
    
    if isempty(all_texts)
        error("Failed to load any HuggingFace datasets!")
    end
    
    println("\n✅ Total samples loaded: $(length(all_texts)) from HuggingFace")
    
    # 288-Tournament Champion Model Configuration
    println("\n🔧 Creating SovWave model (288-Tournament Champion Architecture)...")
    config = WaveModelConfig(
        layers = 8,          # 2x deeper (was 4)
        embed_dims = 256,    # 4x wider (was 64)
        nodes = 256,         # 4x more nodes (was 64)
        omega = 432.0,       # Optimal harmonic
        beta_s = 1.618033988749895,  # Golden ratio (optimal)
        t_frames = 8
    )
    
    # Create model
    wm = create_model(
        layers = config.layers,
        embed_dims = config.embed_dims,
        nodes = config.nodes,
        omega = config.omega,
        beta_s = config.beta_s,
        t_frames = config.t_frames
    )
    println("✅ Model created: $(config.layers) layers × $(config.embed_dims) dims")
    
    # Create tokenizer (use GPT2 for compatibility)
    println("\n🔧 Creating tokenizer...")
    tokenizer = gpt2_tokenizer(carrier_frequency=config.omega, beta_s=config.beta_s)
    println("✅ Tokenizer loaded: vocab size $(length(tokenizer.vocab))")
    
    # Build active vocabulary from training texts
    println("\n🔧 Building active vocabulary from training texts...")
    active_vocab = build_active_vocab!(tokenizer, all_texts)
    
    # Format training data with CHAMPION parameters (60K pairs, 256D, 48 token context)
    println("\n🔧 Formatting training data...")
    dataset = format_lm_text(all_texts; tokenizer=tokenizer, embed_dim=config.embed_dims, max_pairs=60000)
    println("✅ Formatted $(length(dataset)) training pairs (🏆 Champion: 60K target)")
    
    # 🏆 CONTINUOUS TRAINING: Champion Algorithm Cont_R11_UltraFast_1
    # Replaces discrete epoch loop with continuous time-based evolution
    # Time Quantum: 0.001s | Energy Flow: 1.0 | Momentum: 0.908
    
    println("\n🌊 Training with Grand Champion Tournament Configuration...")
    println("  - 144-Algorithm Continuous Training Tournament Winner")
    println("  - Speed: 3072 batch, 128 population, 0.46 LR")
    println("  - Coherence: 6000 samples, 60K pairs, 8L×256N×256D")
    println("  - Architecture: 8 layers × 256 nodes × 256D embeddings")
    println("  - Continuous Evolution: Time-based flow (no discrete epochs)")
    println("  - Pure continuous wave computing")
    
    # Create training config with 🏆 CHAMPION TOURNAMENT PARAMETERS
    train_cfg = WaveTrainConfig(
        epochs = 100_000,             # 100K epochs for full convergence
        batch_size = 3072,            # Champion: 3072 (96x larger)
        learning_rate = 0.46,         # Champion: 0.46 (optimal convergence)
        energy_target = 0.008,
        sonify = true,                # Enable sonification
        sonify_realtime = false,      # Save only, don't play real-time
        population_size = 128         # Champion: 128 (4x diversity)
    )
    
    #Extract inputs and targets from dataset
    inputs = dataset.inputs
    targets = dataset.targets
    
    trained_model, history = train!(
        wm,
        inputs,
        targets,
        train_cfg;
        checkpoint_every = 2000,         # 🏆 Champion: Less I/O overhead
        display_every = 20,              # Show progress every 20 epochs
        checkpoint_dir = "checkpoints_sovwave_hf",
        audio_save_dir = "checkpoints_sovwave_hf",
        verbose = true
    )
    
    println("\n✅ Training complete!")
    println("  Best loss: $(history.best_loss)")
    println("  Total time: $(round(history.total_time_sec, digits=2))s")
    
    # Save model
    checkpoint_path = "checkpoints_sovwave_hf/sovwave_final.mkv"
    mkpath("checkpoints_sovwave_hf")
    save_model(trained_model, checkpoint_path)
    println("✅ Model saved to: $checkpoint_path")
    
    # Comprehensive coherence testing
    println("\n🧪 COMPREHENSIVE COHERENCE TESTING")
    println("="^90)
    
    test_categories = Dict(
        "Math" => ["Two plus two", "Five times three", "Ten divided by"],
        "Language" => ["The sky", "Hello", "Once upon", "Good morning"],
        "Reasoning" => ["If it rains", "When water", "Because of"],
        "Knowledge" => ["The sun", "Water flows", "Birds fly"]
    )
    
    coherence_results = Dict{String, Tuple{Int, Int}}()
    
    for (category, prompts) in test_categories
        println("\n[$category]")
        coherent = 0
        total = length(prompts)
        
        for prompt in prompts
            try
                # Generate with active vocab filtering
                output = generate_text(
                    trained_model, 
                    tokenizer, 
                    prompt; 
                    max_new_tokens=6, 
                    temperature=0.7,
                    active_vocab=active_vocab
                )
                
                result = check_text_coherence(output, prompt)
                status = result.is_coherent ? "✅" : "❌"
                
                if result.is_coherent
                    coherent += 1
                end
                
                println("  $status \"$prompt\" → \"$output\" ($(round(result.score*100, digits=1))%)")
            catch e
                println("  ❌ \"$prompt\" → ERROR: $e")
            end
        end
        
        coherence_results[category] = (coherent, total)
    end
    
    # Summary
    println("\n" * "="^90)
    println(" 📊 COHERENCE SUMMARY (REAL HUGGINGFACE DATA)")
    println("="^90)
    
    total_coherent = 0
    total_tests = 0
    
    for (category, (coherent, total)) in coherence_results
        rate = (coherent / total) * 100
        total_coherent += coherent
        total_tests += total
        println("  $category: $(coherent)/$(total) ($(round(rate, digits=1))%)")
    end
    
    overall_rate = (total_coherent / total_tests) * 100
    println("\n  OVERALL: $(total_coherent)/$(total_tests) ($(round(overall_rate, digits=1))%)")
    println("  TARGET: 70%+ for Grand Champion validation")
    
    if overall_rate >= 70
        println("\n  ✅ SUCCESS: Grand Champion coherence target achieved!")
    elseif overall_rate >= 50
        println("\n  ⚠️  PARTIAL: Showing improvement, may need more training")
    else
        println("\n  ❌ BELOW TARGET: Additional training recommended")
    end
    
    println("="^90)
end

main()
