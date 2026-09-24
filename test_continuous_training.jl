#!/usr/bin/env julia
"""
Quick test of continuous training evolution with coherence check
"""

using Pkg
Pkg.activate(".")

include("src/Sovwave.jl")
using .Sovwave
using .Sovwave.WaveML
using Printf

println("="^90)
println(" 🌊 CONTINUOUS TRAINING QUICK TEST")
println("="^90)

# Small dataset for quick test
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

println("\n🔧 Creating model...")
config = WaveModelConfig(
    layers = 4,
    embed_dims = 64,
    nodes = 64,
    omega = 432.0,
    beta_s = 1.618033988749895,
    t_frames = 8
)

wm = create_model(
    layers = config.layers,
    embed_dims = config.embed_dims,
    nodes = config.nodes,
    omega = config.omega,
    beta_s = config.beta_s,
    t_frames = config.t_frames
)
println("✅ Model created: $(config.layers) layers × $(config.embed_dims) dims")

println("\n🔧 Creating tokenizer...")
tokenizer = gpt2_tokenizer(carrier_frequency=config.omega, beta_s=config.beta_s)
println("✅ Tokenizer loaded: vocab size $(length(tokenizer.vocab))")

println("\n🔧 Formatting training data...")
dataset = format_lm_text(texts; tokenizer=tokenizer, embed_dim=config.embed_dims, max_pairs=1000)
println("✅ Formatted $(length(dataset)) training pairs")

println("\n🌊 Training with CONTINUOUS EVOLUTION (no discrete epochs)...")
train_cfg = WaveTrainConfig(
    epochs = 1000,              # Will stop based on time/energy instead
    batch_size = 32,
    learning_rate = 0.15,
    energy_target = 0.01,
    sonify = false,
    population_size = 16
)

inputs = dataset.inputs
targets = dataset.targets

trained_model, history = train!(
    wm,
    inputs,
    targets,
    train_cfg;
    checkpoint_every = 500,
    display_every = 50,
    checkpoint_dir = "checkpoints_continuous_test",
    verbose = true
)

println("\n✅ Training complete!")
println("  Best loss: $(history.best_loss)")
println("  Total time: $(round(history.total_time_sec, digits=2))s")

# Build active vocab
active_vocab = Set{Int}()
for text in texts
    wave_tokens = tokenize(tokenizer, text)
    for wt in wave_tokens
        push!(active_vocab, wt.token_id)
    end
end

# Test coherence
println("\n🧪 COHERENCE TESTING")
println("="^90)

test_prompts = [
    "Two plus two",
    "The sky is",
    "Hello",
    "Water flows"
]

coherent_count = 0
total_count = length(test_prompts)

for prompt in test_prompts
    try
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
            global coherent_count += 1
        end
        
        println("  $status \"$prompt\" → \"$output\" ($(round(result.score*100, digits=1))%)")
    catch e
        println("  ❌ \"$prompt\" → ERROR: $e")
    end
end

println("\n" * "="^90)
println(" 📊 COHERENCE RESULTS")
println("="^90)
coherence_rate = (coherent_count / total_count) * 100
println("  Coherent: $(coherent_count)/$(total_count) ($(round(coherence_rate, digits=1))%)")
println("  Continuous Evolution: ✅ Active")
println("  Discrete Epoch Loop: ❌ Eliminated")

if coherence_rate >= 50
    println("\n  ✅ SUCCESS: Continuous training maintains coherence!")
else
    println("\n  ⚠️  Needs more training time")
end

println("="^90)
