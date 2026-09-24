# SovWave Training Configuration Guide

## Understanding Epochs vs Batches in Evolutionary Wave Training

### Current Training Paradigm
Unlike traditional deep learning where **1 epoch = 1 full pass through dataset**, SovWave uses **evolutionary wave optimization**:

- **1 "epoch" = 1 evolutionary generation = 1 batch evaluation**
- Each generation evaluates a population of 24 wave models on one batch
- The best models evolve and mutate to the next generation

### Configuration Options

#### Current Setup (Modified for Speed)
```julia
epochs = 5_000          # 5,000 evolutionary generations
batch_size = 128        # 128 samples per generation (increased from 32)
population_size = 24    # 24 competing models per generation
```

**Total Work:** 5,000 × 128 × 24 = **15.36M wave evaluations**
**Training Time:** ~60-75 minutes (vs 275-310 min with smaller batches)

#### Alternative Configurations

##### Ultra-Fast Training (Recommended for Testing)
```julia
epochs = 1_000
batch_size = 256
population_size = 24
```
- **Total:** 6.14M evaluations
- **Time:** ~15-20 minutes
- **Best for:** Quick prototyping and testing

##### Balanced Training (Good Quality/Speed)
```julia
epochs = 5_000
batch_size = 128
population_size = 24
```
- **Total:** 15.36M evaluations  
- **Time:** ~60-75 minutes
- **Best for:** Production training with good coherence

##### High-Quality Training (Maximum Coherence)
```julia
epochs = 10_000
batch_size = 64
population_size = 32
```
- **Total:** 20.48M evaluations
- **Time:** ~180-240 minutes
- **Best for:** Maximum coherence targeting 80%+

##### Full-Batch Training (Dataset-Level Optimization)
```julia
epochs = 500
batch_size = 10_000  # Entire dataset
population_size = 24
```
- **Total:** 120M evaluations
- **Time:** ~300-400 minutes
- **Best for:** When you want each iteration to see all data

### Key Parameters Explained

#### `epochs`
- Number of evolutionary generations
- Each generation processes one batch
- More epochs = more refinement opportunities

#### `batch_size`
- Samples per evolutionary generation
- Larger batches = more stable gradients, faster wall-clock time
- Smaller batches = more stochastic, more iterations

#### `population_size`
- Number of competing models per generation
- Larger population = better exploration, slower per-iteration
- Typical range: 16-32

#### `learning_rate` (mutation_rate)
- Initial mutation strength
- Automatically decays using Golden Ratio schedule
- Higher = more exploration early on

### Training Metrics Interpretation

```
[SOVWAVE EVOLUTION] Ep 171/5000 | E: 0.77431 (tgt: 0.0080) | Acc: 55.3%
```

- **Ep 171/5000**: On generation 171 of 5,000
- **E: 0.77431**: Current wave energy (loss), target is 0.008
- **Acc: 55.3%**: Accuracy on current batch
- **Gamma: 36.5 Hz**: Brainwave frequency for sonification

### Coherence Expectations

Based on the Path of Least Resistance Framework:

- **E > 0.5**: BROKEN 🚫 (0-30% coherence)
- **E: 0.1-0.5**: COMPROMISE ⚖️ (30-60% coherence)  
- **E: 0.01-0.1**: EXCELLENT 🌟 (60-80% coherence)
- **E < 0.01**: PERFECT ✨ (80%+ coherence)

### Recommendations

1. **Start with Ultra-Fast** to verify everything works
2. **Use Balanced** for most production training
3. **Use High-Quality** when targeting Grand Champion coherence (70%+)
4. **Monitor energy (E)** - it should steadily decrease
5. **Check accuracy** - should stabilize around 55-65% during training

### Active Vocabulary Impact

The **active_vocab** size is critical:
- **< 500 tokens**: Insufficient, will produce poor coherence
- **500-2000 tokens**: Minimum for basic coherence
- **2000-5000 tokens**: Good for most tasks
- **5000+ tokens**: Excellent for complex language

Current training: **4,142 tokens** ✅ (Good range!)

---

*This guide is for the continuous wave evolutionary training paradigm used in Sovwave/WaveML.*
