# Spark Model Learning Failure Analysis

## Observed Problem
After 70,000 training steps, the Spark model generates incoherent text:
- "Two plus two loyal in blue rises( is sum]"
- "The sky* are 2 in. of: forward"  
- "Hello !( rises0 language loyal wave"

## Root Cause Analysis

### 1. **CRITICAL: Training Data is Too Small**
```julia
# Only 10 training sentences with 47 token pairs
corpus = [
    "Two plus two equals four.",
    "The sky is blue.",
    "Hello, how are you?",
    # ... 7 more sentences
]
```
**Impact:** With only 47 training pairs, the model has virtually nothing to learn from. Language models need thousands to millions of examples.

### 2. **Loss Function is Too Weak**
```julia
# From train.jl line 100-102
resonance = dot(cur_buf, tgt_vec)
base_loss = clamp(1.0 - resonance, 0.0, 2.0)
loss = base_loss
```
**Problem:** Simple dot product resonance doesn't provide strong enough gradients for discrete token prediction. The model learns to produce vectors close to targets but not necessarily the right tokens.

### 3. **No Token Prediction Head**
The model outputs continuous embeddings but has no mechanism to:
- Convert embeddings back to discrete tokens during training
- Apply cross-entropy loss on token probabilities
- Learn the vocabulary distribution

### 4. **Active Vocabulary Filtering is Too Restrictive**
```julia
# Only 89 tokens from training corpus
active_vocab::Set{Int}
```
**Problem:** With only 89 tokens out of 50,263, the model can't generate most words. But even worse, it's not learning token probability distributions.

### 5. **Model is Too Small for 50K Vocabulary**
```julia
layers=2, embed_dim=32, nodes=32
```
**Problem:** 2 layers × 32 dimensions = only 2,048 parameters to learn 50K token mappings and language structure. GPT-2 small has 125M parameters.

### 6. **No Causal Masking**
The model doesn't enforce causal attention - it sees future tokens during training, breaking autoregressive learning.

### 7. **Gradient Propagation is Weak**
```julia
# Manual adjoint phase propagation
e_layers[n_layers + 1] .= (tgt_vec .- resonance .* cur_buf) .* (learning_rate / nrm_last)
```
**Problem:** Phase-based gradient propagation through sin/cos nonlinearities causes vanishing gradients, especially through 2+ layers.

## Proposed Fixes (in priority order)

### Fix 1: Add Proper Token Prediction Loss ⭐⭐⭐ CRITICAL
```julia
# Need to:
1. Project final embeddings to logits over vocabulary
2. Apply softmax to get probabilities  
3. Use cross-entropy loss against target token
4. Backprop through projection layer
```

### Fix 2: Dramatically Increase Training Data ⭐⭐⭐ CRITICAL
```julia
# Need at minimum:
- 1,000+ diverse sentences
- Or better: use a real dataset (wikitext, c4, etc.)
```

### Fix 3: Increase Model Capacity ⭐⭐ IMPORTANT
```julia
# Minimum viable:
layers=6, embed_dim=128, nodes=128
# This gives ~100K parameters - still tiny but 50x larger
```

### Fix 4: Add Output Projection Layer ⭐⭐⭐ CRITICAL
```julia
# Add a learned linear projection: 
# embedding (32-dim) -> logits (vocab_size)
# With proper gradient flow
```

### Fix 5: Implement Next-Token Prediction ⭐⭐⭐ CRITICAL
```julia
# Current: trains on random pairs
# Needed: trains autoregressively on "predict next token"
```

### Fix 6: Better Gradient Flow ⭐⭐ IMPORTANT
```julia
# Replace manual adjoint with:
- Simpler gradient computation
- Layer normalization
- Residual connections
```

## Quick Test Plan

1. **Test Dataset Size Impact**
   - Train with 10 sentences (baseline)
   - Train with 100 sentences
   - Train with 1000 sentences
   - Measure coherence at each

2. **Test Model Capacity**
   - Keep 10 sentences, increase to 6 layers × 128 dims
   - See if it at least memorizes the training data

3. **Test Loss Function**
   - Add proper token prediction head
   - Use cross-entropy loss
   - Check if loss actually decreases

4. **Test Vocabulary Coverage**
   - Remove active_vocab filtering
   - See if model can access full vocabulary

## Expected Outcomes After Fixes

**Minimum Success Criteria:**
- Model should perfectly memorize 10 training sentences
- Should generate exact training sentences when prompted
- Loss should decrease from ~0.7 to <0.01

**Full Success Criteria:**
- With 1000+ sentences, should generalize to novel prompts
- Coherence checker should pass 70%+ of outputs
- Should complete simple patterns ("Two plus two equals...")

## Implementation Priority

**Phase 1 (Immediate):** Add token prediction head + cross-entropy loss
**Phase 2 (Quick):** Increase training data to 100-1000 sentences  
**Phase 3 (Medium):** Increase model capacity (6 layers × 128 dims)
**Phase 4 (Optional):** Add attention mechanisms, layer norm, residuals
