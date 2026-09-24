#!/usr/bin/env julia
"""
Test coherence of latest checkpoint before retraining
"""

using Pkg
Pkg.activate(".")

include("src/Sovwave.jl")
using .Sovwave
using .Sovwave.WaveML
using Printf

function find_latest_checkpoint(dir::String)
    if !isdir(dir)
        return nothing
    end
    
    files = readdir(dir, join=true)
    mkv_files = filter(f -> endswith(f, ".mkv") && !endswith(f, ".mp4"), files)
    
    if isempty(mkv_files)
        return nothing
    end
    
    # Sort by modification time, get newest
    return sort(mkv_files, by=mtime, rev=true)[1]
end

function main()
    println("="^90)
    println(" 🧪 TESTING LATEST CHECKPOINT COHERENCE")
    println("="^90)
    
    # Find latest checkpoint (try multiple directories)
    ckpt_path = nothing
    for dir in ["checkpoints_sovwave_hf", "checkpoints_hf_real", "checkpoints"]
        result = find_latest_checkpoint(dir)
        if result !== nothing
            ckpt_path = result
            break
        end
    end
    
    if ckpt_path === nothing
        println("❌ No checkpoint found in any checkpoint directory")
        println("   Starting fresh training instead...")
        return false
    end
    
    println("\n📦 Loading checkpoint: $ckpt_path")
    
    try
        # Load model
        model = load_model(ckpt_path)
        println("✅ Model loaded successfully")
        
        # Create tokenizer
        println("\n🔧 Creating tokenizer...")
        tokenizer = gpt2_tokenizer(carrier_frequency=432.0, beta_s=1.618033988749895)
        println("✅ Tokenizer ready")
        
        # Test prompts
        println("\n🧪 COHERENCE TESTING")
        println("-"^90)
        
        test_prompts = [
            "Two plus two",
            "The sky is",
            "Hello world",
            "Once upon a time",
            "Water flows",
            "Good morning"
        ]
        
        coherent = 0
        total = length(test_prompts)
        
        for prompt in test_prompts
            try
                output = generate_text(
                    model, 
                    tokenizer, 
                    prompt; 
                    max_new_tokens=5, 
                    temperature=0.7
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
        
        rate = (coherent / total) * 100
        
        println("\n" * "="^90)
        println(" 📊 CHECKPOINT COHERENCE: $(coherent)/$(total) ($(round(rate, digits=1))%)")
        println("="^90)
        
        if rate >= 70
            println("\n  ✅ EXCELLENT: Checkpoint shows strong coherence")
        elseif rate >= 50
            println("\n  ⚠️  MODERATE: Some coherence present, retraining may improve")
        else
            println("\n  ❌ POOR: Checkpoint needs retraining with new parameters")
        end
        
        return true
        
    catch e
        println("❌ Failed to load/test checkpoint: $e")
        println("   Will start fresh training...")
        return false
    end
end

main()
