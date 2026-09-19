# Ginzburg-Landau Free Energy Functional

## Overview

The Ginzburg-Landau free energy functional provides the optimization framework for Aetheria.jl's wave-based computation. Instead of using gradient descent on discrete weights, the system minimizes a continuous energy functional to find optimal wave configurations. This approach replaces backpropagation with physics-based energy minimization.

## The Free Energy Functional

The Ginzburg-Landau free energy for the wave state Ψ(r) is:

```
F[Ψ] = ∫ [ (1/2)|∇Ψ|² + (α/2)|Ψ|² + (β/4)|Ψ|⁴ - ΩΨ ] d³r
```

Where:
- **F[Ψ]**: Total free energy (functional of the wave field)
- **∇Ψ**: Spatial gradient of the wave function
- **|∇Ψ|²**: Gradient energy (penalizes rapid spatial variations)
- **α**: Quadratic coefficient (controls wave amplitude)
- **β**: Quartic coefficient (controls nonlinearity strength)
- **|Ψ|²**: Wave intensity
- **|Ψ|⁴**: Fourth-order nonlinearity
- **Ω**: External driving field (encodes input data)
- **d³r**: Volume element for 3D integration

## Term-by-Term Explanation

### 1. Gradient Energy: (1/2)|∇Ψ|²

This term penalizes rapid spatial variations in the wave field, favoring smooth configurations.

**Physical Meaning**: Represents the "stiffness" or "surface tension" of the wave field. Creating sharp boundaries or rapid oscillations costs energy.

**Computational Role**: Acts as a regularizer, preventing overfitting by favoring smooth wave patterns. Similar to L2 regularization in traditional neural networks, but applied to spatial derivatives rather than weights.

**Mathematical Form**:
```
|∇Ψ|² = |∂Ψ/∂x|² + |∂Ψ/∂y|² + |∂Ψ/∂z|²
```

### 2. Quadratic Term: (α/2)|Ψ|²

The quadratic term controls the overall amplitude of the wave field.

**Physical Meaning**: 
- If α > 0: Penalizes large wave amplitudes (favors Ψ → 0)
- If α < 0: Favors non-zero wave amplitudes (spontaneous symmetry breaking)

**Computational Role**: Controls the "activation level" of the wave field. Negative α encourages the system to develop non-trivial patterns, while positive α suppresses them.

**Phase Transition**: The sign of α determines whether the system is in an "ordered" (α < 0) or "disordered" (α > 0) phase, analogous to temperature-driven phase transitions in physics.

### 3. Quartic Term: (β/4)|Ψ|⁴

The quartic term provides nonlinear self-interaction and stabilization.

**Physical Meaning**: Prevents the wave amplitude from growing unbounded. When |Ψ| becomes large, this term dominates and limits further growth.

**Computational Role**: 
- Enables universal function approximation (nonlinearity is essential)
- Stabilizes the system by preventing blow-up
- Creates multiple stable states (attractors) in the energy landscape

**Requirement**: β > 0 for stability (ensures energy is bounded from below)

### 4. External Driving: -ΩΨ

The coupling to the external field Ω represents the input data encoded as a spatial frequency pattern.

**Physical Meaning**: Ω acts as a "force" that drives the wave field toward specific configurations. The system tries to align Ψ with Ω to minimize energy.

**Computational Role**: 
- Ω encodes the input data (produced by the Harmonic Encoder)
- The optimal Ψ that minimizes F represents the "processed" or "learned" representation
- Different inputs Ω lead to different optimal wave states Ψ

**Coupling Strength**: The relative magnitude of Ω compared to other terms determines how strongly the input influences the output.

## Energy Minimization Principle

The system evolves to minimize the free energy functional:

```
δF/δΨ* = 0
```

Taking the functional derivative yields the Euler-Lagrange equation:

```
-(1/2)∇²Ψ + (α/2)Ψ + (β/2)|Ψ|²Ψ - Ω = 0
```

This is the equilibrium condition for the wave field.

**Connection to Schrödinger Equation**: The time-dependent Schrödinger equation can be viewed as gradient descent on this energy functional in imaginary time:

```
∂Ψ/∂τ = -δF/δΨ*
```

Where τ is imaginary time (τ = it).

## Physical Interpretation

### Energy Landscape

The free energy functional defines an energy landscape over all possible wave configurations:
- **Minima**: Stable wave patterns (attractors)
- **Maxima**: Unstable configurations
- **Saddle points**: Transition states between stable patterns

**Computational Interpretation**: 
- Each minimum represents a learned "concept" or "feature"
- The system naturally flows downhill to the nearest minimum
- Multiple minima enable multi-stable computation (like attractor networks)

### Spontaneous Symmetry Breaking

When α < 0, the system undergoes spontaneous symmetry breaking:
- The trivial solution Ψ = 0 becomes unstable
- The system spontaneously develops non-zero wave patterns
- Multiple equivalent minima emerge (degenerate ground states)

**Computational Interpretation**: Symmetry breaking allows the system to "choose" specific patterns from a continuum of possibilities, enabling decision-making and classification.

### Metastability and Local Minima

The energy landscape typically has multiple local minima:
- **Global minimum**: Lowest energy configuration
- **Local minima**: Stable but not optimal configurations
- **Energy barriers**: Separate different minima

**Computational Interpretation**: 
- Local minima can represent different learned solutions
- Energy barriers prevent the system from easily switching between solutions
- This is analogous to the loss landscape in traditional neural networks

## Optimization Dynamics

### Gradient Flow

The system evolves according to gradient flow (steepest descent):

```
∂Ψ/∂t = -γ δF/δΨ*
```

Where γ is a relaxation rate (learning rate analog).

**Properties**:
- Energy decreases monotonically: dF/dt ≤ 0
- System converges to a local minimum
- Convergence rate depends on γ and the energy landscape curvature

### Momentum and Adaptive Methods

For faster convergence, momentum-based methods can be used:

```
∂Ψ/∂t = -γ δF/δΨ* + μ v
∂v/∂t = -γ δF/δΨ* - λv
```

Where:
- v is the "velocity" (momentum)
- μ is the momentum coefficient
- λ is the damping coefficient

**Connection to AdamW**: The AdamWave optimizer adapts these principles to wave states, incorporating first and second moment estimates.

## Numerical Computation

### Discretization

On a spatial grid with points r_i:

```
F ≈ Σ_i [ (1/2)|∇Ψ_i|² + (α/2)|Ψ_i|² + (β/4)|Ψ_i|⁴ - Ω_i Ψ_i ] Δx Δy Δz
```

Where Δx Δy Δz is the grid cell volume.

### Gradient Computation

The functional derivative at grid point i:

```
δF/δΨ_i* ≈ -(1/2)∇²Ψ_i + (α/2)Ψ_i + (β/2)|Ψ_i|²Ψ_i - Ω_i
```

Using finite differences for ∇²Ψ_i.

### Energy Monitoring

During optimization, monitor:
1. **Total energy F**: Should decrease monotonically
2. **Energy components**: Track gradient, quadratic, quartic, and coupling terms separately
3. **Convergence criterion**: |δF/δΨ| < ε for all grid points

## Connection to Traditional Optimization

| Traditional NN | Wave-Based (Ginzburg-Landau) |
|----------------|------------------------------|
| Loss function L(w) | Free energy functional F[Ψ] |
| Weights w | Wave field Ψ(r) |
| Gradient ∂L/∂w | Functional derivative δF/δΨ |
| SGD update | Gradient flow |
| Momentum | Wave momentum |
| Regularization | Gradient energy term |

## Parameter Selection

### α (Quadratic Coefficient)

- **α > 0**: Ordered phase, small amplitude waves
- **α < 0**: Symmetry-broken phase, large amplitude patterns
- **Typical range**: α ∈ [-1, 1]

**Tuning**: Start with α ≈ -0.1 for pattern formation, adjust based on task.

### β (Quartic Coefficient)

- **β > 0**: Required for stability
- **Larger β**: Stronger amplitude saturation
- **Typical range**: β ∈ [0.1, 10]

**Tuning**: Start with β ≈ 1, increase if wave amplitudes grow too large.

### γ (Relaxation Rate)

- **Larger γ**: Faster convergence but risk of instability
- **Smaller γ**: Slower but more stable
- **Typical range**: γ ∈ [0.001, 0.1]

**Tuning**: Use adaptive methods (AdamWave) to automatically adjust γ.

## Advanced Topics

### Annealing

Gradually change α from positive to negative to guide the system through phase transition:

```
α(t) = α_initial + (α_final - α_initial) × t/T
```

This can help find better minima by avoiding early trapping in local minima.

### Multi-Component Fields

For complex tasks, use multiple coupled wave fields Ψ₁, Ψ₂, ..., Ψₙ with interaction terms:

```
F = Σ_i F_i[Ψ_i] + Σ_{i<j} F_int[Ψ_i, Ψ_j]
```

### Temperature and Noise

Add thermal fluctuations for stochastic optimization:

```
∂Ψ/∂t = -γ δF/δΨ* + √(2γT) η(t)
```

Where η(t) is white noise and T is effective temperature.

## Practical Implementation Guidelines

1. **Initialize**: Start with small random Ψ or use previous solution
2. **Compute Energy**: Calculate F[Ψ] and all components
3. **Compute Gradient**: Calculate δF/δΨ* using finite differences
4. **Update**: Apply gradient flow or momentum-based update
5. **Monitor**: Check energy decrease and convergence
6. **Iterate**: Repeat until convergence or max iterations

## Connection to Other Components

- **Schrödinger Equation**: Time evolution follows gradient flow on F
- **Sacred Geometry Potential**: Can be incorporated as additional term in F
- **Cymatic Extraction**: Minima of F correspond to standing wave patterns
- **Wave Optimizer**: Implements sophisticated minimization of F

## References

- Ginzburg, V. L., & Landau, L. D. (1950). On the theory of superconductivity.
- Cross, M. C., & Hohenberg, P. C. (1993). Pattern formation outside of equilibrium.
- See `schrodinger_equation.md` for time-dependent evolution
- See `wave_interference.md` for understanding stable patterns
