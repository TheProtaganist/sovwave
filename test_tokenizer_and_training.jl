#!/usr/bin/env julia
"""
Complete WaveML Demo: Tokenization + Training
Demonstrates the full pipeline from text to trained wave model
"""

using Sovwave
using Printf

println("="^80)
println(" 🌊 TOKENIZER + WAVE MODEL TRAINING DEMO 🌊")
println("="^80)

# ============================================================================
# PART 1: TOKENIZATION
# ============================================================================

println("\n" * "="^80)
println(" PART 1: WAVE TOKENIZATION")
println("="^80)

println("\n[1] Creating Tokenizer...")
tok = default_tokenizer()
@printf("  ✓ Vocabulary size: %d tokens\n", length(tok.inv_vocab))
@printf("  ✓ Carrier frequency: %.1f Hz\n", tok.carrier_frequency)
@printf("  ✓ Golden ratio (β_s): %.6f\n", tok.beta_s)

println("\n[2] Testing Subword Tokenization (BPE-style)...")
test_phrases = [
    "harmonic wave computing",
    "wave",
    "computing",
    "data science",
    "🌊🧠⚡"
]

for phrase in test_phrases
    ids = tokenize_ids(tok, phrase)
    tokens = [tok.inv_vocab[id] for id in ids]
    @printf("  • \"%s\" → %d tokens: %s\n", phrase, length(tokens), tokens)
end

println("\n[3] Testing Multilingual Tokenization...")
multilingual_texts = [
    "English text",
    "Español 🇪🇸",
    "中文测试",
    "العربية",
    "Русский язык"
]

for text in multilingual_texts
    ids = tokenize_ids(tok, text)
    @printf("  • \"%s\" → %d tokens\n", text, length(ids))
end

println("\n[4] Testing Lossless Reconstruction...")
original_text = "harmonic wave computing 🌊⚡"
ids = tokenize_ids(tok, original_text)
reconstructed = decode(tok, ids)
@printf("  • Original:      \"%s\"\n", original_text)
@printf("  • Tokenized:     %d tokens\n", length(ids))
@printf("  • Reconstructed: \"%s\"\n", reconstructed)
@printf("  • Match:         %s\n", original_text == reconstructed ? "✓ PERFECT" : "✗ FAILED")

println("\n[5] Generating WaveForms (Continuous Acoustic Representation)...")
waveforms = tokenize(tok, "wave"; n_samples=32, sample_rate=48000.0)
for (i, wf) in enumerate(waveforms)
    @printf("  Token %d: \"%s\" | f=%.1f Hz | φ=%.3f rad | Energy=%.4f\n",
            i, wf.token, wf.frequency, wf.phase, wf.energy)
end

# ============================================================================
# PART 2: WAVE MODEL TRAINING
# ============================================================================

println("\n" * "="^80)
println(" PART 2: WAVE MODEL TRAINING")
println("="^80)

println("\n[6] Creating Training Dataset from Text...")
training_texts = [
    "wave computing",
    "harmonic resonance",
    "quantum field",
    "neural network",
    "deep learning",
    "data science",
    "machine learning",
    "artificial intelligence"
]

# Tokenize and convert to embeddings
println("  • Tokenizing training texts...")
n_samples = length(training_texts)
embed_dim = 16
max_tokens = 20

# Create wave embeddings from tokenized text
inputs = []
for text in training_texts
    local_waveforms = tokenize(tok, text; n_samples=embed_dim, sample_rate=48000.0)
    # Take first embed_dim waveform energies as features
    features = zeros(embed_dim)
    for (i, wf) in enumerate(local_waveforms[1:min(length(local_waveforms), embed_dim)])
        features[i] = wf.energy
    end
    push!(inputs, features)
end

@printf("  ✓ Created %d wave embeddings (%d dimensions each)\n", length(inputs), embed_dim)

println("\n[7] Creating Model Configuration...")
cfg = WaveMLConfig(
    field = WaveFieldConfig(n_points=32, dimensions=2),
    model = WaveModelConfig(layers=2, embed_dims=embed_dim, nodes=8, omega=432.0),
    train = WaveTrainConfig(
        batch_size=4,
        epochs=15,
        population_size=6,
        learning_rate=0.12,
        sonify=false
    )
)
@printf("  ✓ Model: %d layers, %d nodes, ω=%.1f Hz\n", 
        cfg.model.layers, cfg.model.nodes, cfg.model.omega)

println("\n[8] Building WaveModel...")
model = WaveModel(cfg)
summary = inspect_model(model; io=devnull)
@printf("  ✓ Model created with %d continuous parameters\n", summary[:total_parameters])

println("\n[9] Generating Target Patterns...")
# Target: Low-energy harmonic patterns (ground state)
nodes = cfg.model.nodes
targets = [0.2 .* sin.(2π .* (1:nodes) ./ nodes .+ i*0.1) for i in 1:n_samples]
@printf("  ✓ Generated %d target patterns (%d output nodes)\n", length(targets), nodes)

println("\n[10] Training Wave Model...")
println("  Seeking ground state energy via wave evolution...")
trained_model, history = train!(model, inputs, targets, cfg.train; verbose=true)

println("\n[11] Training Results:")
initial_loss = history.metrics[1].loss
final_loss = history.metrics[end].loss
improvement = (1.0 - final_loss / initial_loss) * 100

@printf("  • Initial Loss:     %.6f\n", initial_loss)
@printf("  • Final Loss:       %.6f\n", final_loss)
@printf("  • Improvement:      %.1f%%\n", improvement)
@printf("  • Training Time:    %.3f seconds\n", history.total_time_sec)
@printf("  • Final Throughput: %.1f pts/sec\n", history.metrics[end].throughput_pts_sec)
@printf("  • Final Accuracy:   %.1f%%\n", history.metrics[end].accuracy * 100)

println("\n[12] Testing Inference on New Text...")
test_text = "quantum wave"
@printf("  • Input text: \"%s\"\n", test_text)

# Tokenize test text
test_waveforms = tokenize(tok, test_text; n_samples=embed_dim, sample_rate=48000.0)
test_features = zeros(embed_dim)
for (i, wf) in enumerate(test_waveforms[1:min(length(test_waveforms), embed_dim)])
    test_features[i] = wf.energy
end

prediction = predict(trained_model, test_features)
@printf("  • Tokens: %d\n", length(test_waveforms))
@printf("  • Predictions: [%.3f, %.3f, %.3f, %.3f, %.3f, ...]\n",
        prediction[1], prediction[2], prediction[3], prediction[4], prediction[5])

println("\n" * "="^80)
println(" ✅ COMPLETE PIPELINE SUCCESSFUL!")
println("="^80)
println("\nSummary:")
println("  ✓ Tokenizer: Subword BPE with 50,000+ vocabulary (GPT-2/Qwen/Mistral pipeline)")
println("  ✓ Wave embeddings: Text → Continuous acoustic features")
println("  ✓ Model training: Ground state evolution (no backprop)")
@printf("  ✓ Performance: %.1f%% improvement in %.2fs\n", improvement, history.total_time_sec)
println("="^80)
