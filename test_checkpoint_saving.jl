#!/usr/bin/env julia
"""
Test Checkpoint MKV Saving During Training
"""

using Sovwave
using Printf

println("="^80)
println(" 💾 CHECKPOINT MKV SAVING TEST")
println("="^80)

# Create model
cfg = WaveMLConfig(
    field = WaveFieldConfig(n_points=16, dimensions=2),
    model = WaveModelConfig(layers=2, embed_dims=8, nodes=8, omega=432.0),
    train = WaveTrainConfig(
        batch_size=4,
        epochs=20,
        population_size=6,
        learning_rate=0.1,
        sonify=false
    )
)

model = WaveModel(cfg)
println("\n✓ Model created")

# Generate data
n_samples = 32
inputs = [rand(8) for _ in 1:n_samples]
targets = [0.3 .* sin.(2π .* (1:8) ./ 8 .+ i*0.1) for i in 1:n_samples]
println("✓ Data generated: $n_samples samples")

# Create checkpoint directory
checkpoint_dir = "checkpoints_test"
rm(checkpoint_dir, force=true, recursive=true)
println("✓ Checkpoint directory: $checkpoint_dir")

# Train with checkpoints every 5 epochs
println("\n🎯 Training with checkpoint saving every 5 epochs...\n")

trained_model, history = train!(
    model, inputs, targets, cfg.train;
    checkpoint_dir = checkpoint_dir,
    checkpoint_every = 5,
    verbose = true
)

# Verify checkpoints
println("\n📊 Checkpoint Verification:")
if isdir(checkpoint_dir)
    ckpt_files = filter(f -> endswith(f, ".mkv"), readdir(checkpoint_dir))
    @printf("  ✓ Checkpoints saved: %d\n", length(ckpt_files))
    
    for (i, ckpt) in enumerate(ckpt_files)
        ckpt_path = joinpath(checkpoint_dir, ckpt)
        size_kb = filesize(ckpt_path) / 1024
        mp4_path = replace(ckpt_path, ".mkv" => ".mp4")
        has_mp4 = isfile(mp4_path)
        @printf("    %d. %s (%.1f KB) %s\n", 
                i, ckpt, size_kb, has_mp4 ? "✓ MP4" : "")
    end
    
    # Test loading a checkpoint
    if !isempty(ckpt_files)
        test_ckpt = joinpath(checkpoint_dir, ckpt_files[end])
        println("\n🔬 Testing checkpoint loading...")
        loaded_model = load_model(test_ckpt)
        @printf("  ✓ Loaded checkpoint: %s\n", ckpt_files[end])
        @printf("  ✓ Layers: %d\n", length(loaded_model.layers))
        @printf("  ✓ Nodes: %d\n", loaded_model.model_config.nodes)
        
        # Test inference
        test_pred = predict(loaded_model, inputs[1])
        @printf("  ✓ Inference works: [%.3f, %.3f, %.3f, ...]\n", 
                test_pred[1], test_pred[2], test_pred[3])
    end
else
    println("  ✗ Checkpoint directory not found!")
end

println("\n" * "="^80)
println(" ✅ CHECKPOINT TEST COMPLETE")
println("="^80)
println("\nTo view checkpoints:")
println("  1. MKV files: Use VLC or ffmpeg")
println("  2. MP4 files: Double-click to view in any player")
println("  3. Load in Julia: load_model(\"path/to/checkpoint.mkv\")")
println("="^80)
