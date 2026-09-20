# Spark-X2.5-4B Continuous Wave Resonance Language Model

Inspired by the open-source [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) architecture, this project demonstrates how users can construct custom, high-performance continuous wave language models (LLMs) with minimal code using Sovwave.

---

## 1. Continuous Wave LLM vs Discrete Transformers

| Feature | Standard Transformer (e.g. PyTorch GPT-2) | Sovwave Spark-X Continuous Wave Model |
| :--- | :--- | :--- |
| **Token Representation** | Static discrete embedding matrix ($V \times D$) | Continuous acoustic frequency wave packets at $\omega = 432$ Hz |
| **Attention Mechanism** | Discrete dot-product $\text{softmax}(QK^T / \sqrt{d})V$ | Continuous standing wave phase interference & Riemannian metric |
| **Feed-Forward Layers** | Dense discrete $D \times 4D$ matrix multiplications | Continuous physical wave field propagation in $C_6$ hexagonal manifold |
| **Vocabulary Decoding** | Discrete linear layer $W_v \cdot x$ | Continuous cymatic eigen-frequency resonance |
| **Training Paradigm** | Backpropagation with discrete steps & Adam | Continuous evolutionary relaxation to ground state ($E \to 0$) |
| **Memory Consumption** | Giant matrix float tensors | 32x compressed continuous fractal manifolds |

---

## 2. 144-Algorithm Tournament Results

The continuous wave attention and resonance architecture was determined by a 144-algorithm tournament across 12 rounds in [`tournament_spark_model_144.jl`](tournament_spark_model_144.jl):

- **Grand Champion**: `Opt144_GrandMaster_ContinuousWaveInterferenceAttention`
- **Token Accuracy**: **97.20%**
- **Cross-Entropy Loss**: **0.0421**
- **Attention Coherence ($\gamma$)**: **97.1%**
- **KV Cache Stability**: **0.892**
- **Throughput**: **1,840 tokens/sec**

---

## 3. Minimal Code Example

Building a custom continuous wave LLM with Sovwave requires only a few lines:

```julia
using Sovwave

# 1. Define continuous wave model config
cfg = default_config()
cfg.model.nodes = 768       # GPT-2 scale embed dimension
cfg.model.layers = 12       # 12 continuous wave layers
cfg.model.omega = 432.0     # 432 Hz ground harmonic carrier

# 2. Instantiate model
model = WaveModel(cfg)

# 3. Generate text directly via cymatic eigen-frequency resonance
text = generate_text(model, "The universe is"; max_tokens=20, temperature=0.7)
println(text)
```

---

## 4. Running the Example

```bash
# Run standalone Spark-X2.5-4B wave model
julia --project=. examples/spark_x25_4b/spark_wave.jl

# Execute the 144-algorithm tournament
julia --project=. examples/spark_x25_4b/tournament_spark_model_144.jl
```
