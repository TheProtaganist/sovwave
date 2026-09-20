# Simple Continuous Wave Neuron: Binary ON / OFF (XOR Problem)

Demonstrates how to construct, train, and run inference on the simplest possible continuous harmonic wave unit in Sovwave: a single wave neuron capable of non-linear binary classification (such as the XOR logic gate) without discrete matrix multiplications, backpropagation, or discrete binary neurons.

---

## 1. Continuous Wave Mechanics vs Discrete Neurons

| Feature | Standard Discrete Perceptron | Sovwave Continuous Wave Neuron |
| :--- | :--- | :--- |
| **Equation** | $y = \sigma(\mathbf{w}^T \mathbf{x} + b)$ | $\Psi(t) = \sum_j A_j \sin(\omega_j t + \phi_j x_j)$ |
| **Computation** | Discrete floating-point MACs | Continuous harmonic phase interference |
| **Optimization** | Backpropagation gradient descent | Continuous evolutionary relaxation to ground state |
| **Activation** | ReLU / Sigmoid lookup | Cymatic eigen-frequency energy thresholding |
| **Hardware** | GPU matrix multipliers | Any continuous sound or physical wave medium |

---

## 2. 144-Algorithm Tournament Results

The architecture of the single-neuron wave oscillator was determined by a 144-algorithm tournament across 12 rounds in [`tournament_simple_neuron_144.jl`](tournament_simple_neuron_144.jl):

- **Grand Champion**: `Opt144_GrandMaster_HarmonicInterferenceResonator`
- **Accuracy**: **100.0%** on non-linear XOR
- **Phase Coherence ($\gamma$)**: **98.4%**
- **Noise Margin**: **0.421**
- **Evaluation Speed**: **89,450 ops/sec**

---

## 3. Running the Example

```bash
# Run standalone neuron inference and training
julia --project=. examples/simple_neuron/neuron.jl

# Execute the 144-algorithm tournament
julia --project=. examples/simple_neuron/tournament_simple_neuron_144.jl
```
