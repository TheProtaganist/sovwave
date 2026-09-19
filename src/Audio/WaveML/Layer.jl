"""
    WaveML.Layer

Wave Layer architecture for Pure Wave Computing.
Unlike traditional dense/convolutional layers based on matrix multiplication and
discrete 2^n logic, a WaveLayer represents a physical wave parameter lattice
(amplitudes, phases, frequencies, and fractal scaling) distributed over quantum nodes.

Supports flexible non-power-of-two dimensions and continuous temporal superposition.
"""

using Random

export WaveLayer
export create_layer, forward!, layer_energy, mutate!, crossover

"""
    WaveLayer

A wave-based computational layer.
- `nodes::Int`: n quantum lattice nodes
- `embed_dim::Int`: d embedding dimensions for this layer
- `amplitudes::Matrix{Float64}`: (nodes × embed_dim) learnable wave amplitudes
- `phases::Matrix{Float64}`: (nodes × embed_dim) learnable wave phases [0, 2π]
- `frequencies::Matrix{Float64}`: (nodes × embed_dim) learnable relative frequencies
- `fractal_scales::Vector{Float64}`: per-node fractal scaling parameters β_s
- `omega::Float64`: Harmonic driving frequency ω in Hz
- `layer_energy::Float64`: Accumulated physical energy of this layer
"""
mutable struct WaveLayer
    nodes::Int
    embed_dim::Int
    amplitudes::Matrix{Float64}
    phases::Matrix{Float64}
    frequencies::Matrix{Float64}
    fractal_scales::Vector{Float64}
    omega::Float64
    layer_energy::Float64

    function WaveLayer(
        nodes::Int,
        embed_dim::Int,
        amps::Matrix{Float64},
        phs::Matrix{Float64},
        freqs::Matrix{Float64},
        fractals::Vector{Float64},
        omega::Float64
    )
        new(nodes, embed_dim, amps, phs, freqs, fractals, omega, 0.0)
    end
end

"""
    create_layer(nodes::Int, embed_dim::Int; omega::Float64=432.0, beta_s::Float64=1.618033988749895)::WaveLayer

Constructs a new `WaveLayer` with initial wave parameters set to natural harmonic ground states.
"""
function create_layer(
    nodes::Int,
    embed_dim::Int;
    omega::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveLayer
    # Initialize amplitudes around ground state [0.1, 1.0]
    amps = 0.5 .+ 0.3 .* randn(nodes, embed_dim)
    clamp!(amps, 0.01, 2.0)

    # Initial phases evenly distributed to preserve initial coherence
    phs = Matrix{Float64}(undef, nodes, embed_dim)
    for i in 1:nodes, j in 1:embed_dim
        phs[i, j] = mod(2π * ((i - 1) / nodes + (j - 1) / embed_dim), 2π)
    end

    # Relative frequencies centered at 1.0 (fundamental) with harmonic overtones
    freqs = Matrix{Float64}(undef, nodes, embed_dim)
    for i in 1:nodes, j in 1:embed_dim
        freqs[i, j] = 1.0 + 0.1 * ((j % 4) + 1)
    end

    # Fractal scales initialized around Golden Ratio β_s
    fractals = [beta_s * (1.0 + 0.05 * sin(2π * i / nodes)) for i in 1:nodes]

    return WaveLayer(nodes, embed_dim, amps, phs, freqs, fractals, omega)
end

"""
    forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}

Propagates incoming wave information through the layer's quantum nodes.
Computes wave superposition across embeddings and nodes:
    \\psi_i = \\sum_{j=1}^d A_{ij} \\cdot \\sin(\\omega f_{ij} \\cdot in_j + \\phi_{ij} - t) \\cdot \\beta_{s,i}
"""
function forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = Vector{Float64}(undef, n)
    in_len = length(input_values)

    total_layer_energy = 0.0

    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]

        for j in 1:d
            in_val = j <= in_len ? input_values[j] : 0.5
            amp = layer.amplitudes[i, j]
            ph = layer.phases[i, j]
            freq = layer.frequencies[i, j]

            # Wave phase modulation: ω·f·x + φ - t
            angle = muladd(layer.omega * 0.001 * freq, in_val, ph - t)
            wave_val = amp * sin(angle)
            node_sum += wave_val
        end

        # Modulate by node fractal structure β_s
        node_wave = (node_sum / sqrt(Float64(d))) * beta
        output[i] = node_wave
        total_layer_energy += 0.5 * (node_wave * node_wave)
    end

    layer.layer_energy = total_layer_energy
    return output
end

"""
    layer_energy(layer::WaveLayer)::Float64

Returns the current potential energy of the wave layer.
"""
function layer_energy(layer::WaveLayer)::Float64
    return layer.layer_energy
end

"""
    mutate!(layer::WaveLayer, rate::Float64)::Nothing

Applies evolutionary mutations to the wave parameters of this layer:
- Amplitudes: Perturbed with rate-scaled fluctuations
- Phases: Modulated continuously on [0, 2π]
- Frequencies: Drift according to harmonic scaling
- Fractal scales: Fine-tuned around ground state
"""
function mutate!(layer::WaveLayer, rate::Float64)::Nothing
    nodes = layer.nodes
    embed_dim = layer.embed_dim

    # Correlated amplitude perturbation
    for i in 1:nodes, j in 1:embed_dim
        layer.amplitudes[i, j] += rate * 0.2 * randn()
        layer.amplitudes[i, j] = clamp(layer.amplitudes[i, j], 0.001, 3.0)

        # Phase mutation (periodic on circle)
        layer.phases[i, j] = mod(layer.phases[i, j] + rate * 0.5 * (rand() - 0.5) * 2π, 2π)

        # Frequency drift
        layer.frequencies[i, j] = clamp(layer.frequencies[i, j] + rate * 0.05 * randn(), 0.1, 10.0)
    end

    # Fractal scale tuning
    for i in 1:nodes
        layer.fractal_scales[i] += rate * 0.02 * randn()
        layer.fractal_scales[i] = clamp(layer.fractal_scales[i], 0.5, 4.0)
    end

    return nothing
end

"""
    crossover(parent_a::WaveLayer, parent_b::WaveLayer)::WaveLayer

Combines two wave layers using the tournament champion **Arithmetic Wave Blend**:
Parameters are superposed in phase space with equal energy weighting.
"""
function crossover(parent_a::WaveLayer, parent_b::WaveLayer)::WaveLayer
    nodes = parent_a.nodes
    embed_dim = parent_a.embed_dim

    # Arithmetic blend for amplitudes
    child_amps = 0.5 .* (parent_a.amplitudes .+ parent_b.amplitudes)

    # Circular phase interpolation
    child_phases = Matrix{Float64}(undef, nodes, embed_dim)
    for i in 1:nodes, j in 1:embed_dim
        th_a = parent_a.phases[i, j]
        th_b = parent_b.phases[i, j]
        # Circular mean: atan(sin(a)+sin(b), cos(a)+cos(b))
        child_phases[i, j] = mod(atan(sin(th_a) + sin(th_b), cos(th_a) + cos(th_b)), 2π)
    end

    # Geometric mean for frequencies
    child_freqs = sqrt.(parent_a.frequencies .* parent_b.frequencies)

    # Blend fractal scales
    child_fractals = 0.5 .* (parent_a.fractal_scales .+ parent_b.fractal_scales)

    return WaveLayer(nodes, embed_dim, child_amps, child_phases, child_freqs, child_fractals, parent_a.omega)
end

function Base.deepcopy(layer::WaveLayer)::WaveLayer
    return WaveLayer(
        layer.nodes,
        layer.embed_dim,
        copy(layer.amplitudes),
        copy(layer.phases),
        copy(layer.frequencies),
        copy(layer.fractal_scales),
        layer.omega
    )
end
