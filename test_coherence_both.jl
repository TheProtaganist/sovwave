#!/usr/bin/env julia
"""
Test coherence of latest CPU and GPU checkpoints
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
    mkv_files = filter(f -> endswith(f, ".mkv") && !contains(f, ".mp4"), files)
    
    if isempty(mkv_files)
        return nothing
    end
    
    return sort(mkv_files, by=mtime, rev=true)[1]
end

function test_checkpoint_coherence(ckpt_path::String, label::String)
    println("\n" * "="^90)
    println(" 🧪 TESTING $label CHECKPOINT")
    println("="^90)
    println("\n📦 Loading: $ckpt_path")
    
    try
        # Load model
        model = load_model(ckpt_path)
        println("✅ Model loaded successfully")
        
        # Create tokenizer
        tokenizer = gpt2_tokenizer(carrier_frequency=432.0, beta_s=1.618033988749895)
        
        # Test prompts
        test_prompts = [
            "Two plus two",
            "The sky is",
            "Hello world",
            "Once upon a time",
            "Water flows",
            "Good morning",
            "Five times three",
            "The sun shines"
        ]
        
        println("\n🧪 COHERENCE TESTING")
        println("-"^90)
        
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
        @printf(" 📊 %s COHERENCE: %d/%d (%.1f%%)\n", label, coherent, total, rate)
        println("="^90)
        
        if rate >= 70
            println("  ✅ EXCELLENT: Strong coherence")
        elseif rate >= 50
            println("  ⚠️  MODERATE: Some coherence present")
        else
            println("  ❌ POOR: Needs more training")
        end
        
        return (coherent, total, rate)
        
    catch e
        println("❌ Failed to load/test checkpoint: $e")
        return (0, 0, 0.0)
    end
end

function main()
    println("="^90)
    println(" 🔬 DUAL CHECKPOINT COHERENCE COMPARISON")
    println("="^90)
    
    # Find CPU checkpoint
    cpu_ckpt = find_latest_checkpoint("checkpoints_sovwave_hf")
    if cpu_ckpt === nothing
        println("\n❌ No CPU checkpoint found")
        cpu_results = (0, 0, 0.0)
    else
        cpu_results = test_checkpoint_coherence(cpu_ckpt, "CPU")
    end
    
    # Find GPU checkpoint
    gpu_ckpt = find_latest_checkpoint("checkpoints_sovwave_hf_GPU")
    if gpu_ckpt === nothing
        println("\n❌ No GPU checkpoint found")
        gpu_results = (0, 0, 0.0)
    else
        gpu_results = test_checkpoint_coherence(gpu_ckpt, "GPU")
    end
    
    # Comparison summary
    println("\n" * "="^90)
    println(" 📊 COMPARISON SUMMARY")
    println("="^90)
    
    if cpu_ckpt !== nothing
        @printf("  CPU: %d/8 (%.1f%%) - %s\n", cpu_results[1], cpu_results[3], basename(cpu_ckpt))
    else
        println("  CPU: No checkpoint available")
    end
    
    if gpu_ckpt !== nothing
        @printf("  GPU: %d/8 (%.1f%%) - %s\n", gpu_results[1], gpu_results[3], basename(gpu_ckpt))
    else
        println("  GPU: No checkpoint available")
    end
    
    if cpu_ckpt !== nothing && gpu_ckpt !== nothing
        diff = cpu_results[3] - gpu_results[3]
        if abs(diff) < 5.0
            println("\n  ⚖️  TIED: Both performing similarly")
        elseif diff > 0
            @printf("\n  🏆 CPU WINS by %.1f%%\n", diff)
        else
            @printf("\n  🏆 GPU WINS by %.1f%%\n", abs(diff))
        end
    end
    
    println("="^90)
end

main()
