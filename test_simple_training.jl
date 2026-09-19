#!/usr/bin/env julia
"""
Simple WaveML Training Demo
Trains a small wave model on synthetic data
"""

using Sovwave
using Printf

println("="^80)
println(" 🌊 SIMPLE WAVE MODEL TRAINING DEMO 🌊")
println("="^80)

# 1. Create a simple configuration
println("\n[1] Creating WaveML Configuration...")
cfg = WaveMLConfig(
    field = WaveFieldConfig(n_points=32, dimensions=2),
    model = WaveModelConfig(layers=3, embed_dims=16, nodes=16, omega=432.0),
    train = WaveTrainConfig(
        batch_size=8,
        epochs=10,
        population_size=8,
        learning_rate=0.1,
        sonify=false
    )
)
@printf("  ✓ Config: %d layers, %d embed dims, %d nodes\n", 
        cfg.model.layers, cfg.model.embed_dims, cfg.model.nodes)

# 2. Build the model
println("\n[2] Building WaveModel...")
model = WaveModel(cfg)
summary = inspect_model(model; io=devnull)  # Suppress output
@printf("  ✓ Model created with %d parameters\n", summary[:total_parameters])

# 3. Generate training data
println("\n[3] Generating Training Data...")
n_samples = 64
embed_dim = cfg.model.embed_dims
nodes = cfg.model.nodes

# Input: random wave embeddings
inputs = [rand(embed_dim) for _ in 1:n_samples]

# Target: low-energy oscillating pattern (ground state)
targets = [0.3 .* sin.(2π .* (1:nodes) ./ nodes) for _ in 1:n_samples]

@printf("  ✓ Generated %d samples (%d dims → %d output nodes)\n", 
        n_samples, embed_dim, nodes)

# 4. Train the model
println("\n[4] Training via Wave Evolution (Seeking Ground State)...")
println("    Epochs: $(cfg.train.epochs) | Population: $(cfg.train.population_size)")

trained_model, history = train!(model, inputs, targets, cfg.train; verbose=true)

# 5. Show results
println("\n[5] Training Results:")
@printf("  • Initial Loss:  %.6f\n", history.metrics[1].loss)
@printf("  • Final Loss:    %.6f\n", history.metrics[end].loss)
@printf("  • Total Time:    %.3f seconds\n", history.total_time_sec)
@printf("  • Throughput:    %.1f pts/sec\n", history.metrics[end].throughput_pts_sec)

improvement = (1.0 - history.metrics[end].loss / history.metrics[1].loss) * 100
@printf("  • Improvement:   %.1f%%\n", improvement)

# 6. Test inference
println("\n[6] Testing Inference...")
test_input = inputs[1]
prediction = predict(trained_model, test_input)
@printf("  ✓ Input shape: %d dims → Output shape: %d nodes\n", 
        length(test_input), length(prediction))
@printf("  ✓ Sample predictions: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
        prediction[1], prediction[2], prediction[3], prediction[4], prediction[5])

println("\n" * "="^80)
println(" ✅ TRAINING COMPLETED SUCCESSFULLY!")
println("="^80)
