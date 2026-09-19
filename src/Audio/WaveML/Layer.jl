"""
    WaveML.Layer

Wave Layer architecture for Pure Wave Computing.
Unlike traditional dense/convolutional layers based on matrix multiplication and
discrete 2^n logic, a WaveLayer represents a physical wave parameter lattice
(amplitudes, phases, frequencies, fractal scaling, fractal dimension, and wave speed)
distributed over quantum nodes.

Each node tracks six learnable parameters:
- **amplitude** `A`: radiant intensity
- **phase** `φ`: wave offset ∈ [0, 2π]
- **frequency** `f`: relative frequency multiplier
- **fractal_scale** `β_s`: self-similar amplitude envelope
- **fractal_dim** `D_f ∈ [1.0, 3.0]`: Hausdorff fractal dimension (default 1.5)
- **wave_speed** `v`: propagation speed (default 1.0; -1.0 = unlimited / instantaneous)

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
- `fractal_dims::Vector{Float64}`: per-node Hausdorff fractal dimension D_f ∈ [1.0, 3.0]
- `wave_speeds::Vector{Float64}`: per-node propagation speed v (−1.0 = unlimited)
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
    fractal_dims::Vector{Float64}
    wave_speeds::Vector{Float64}
    omega::Float64
    layer_energy::Float64

    function WaveLayer(
        nodes::Int,
        embed_dim::Int,
        amps::Matrix{Float64},
        phs::Matrix{Float64},
        freqs::Matrix{Float64},
        fractals::Vector{Float64},
        omega::Float64;
        fractal_dims::Vector{Float64} = fill(1.5, nodes),
        wave_speeds::Vector{Float64} = fill(1.0, nodes)
    )
        new(nodes, embed_dim, amps, phs, freqs, fractals, fractal_dims, wave_speeds, omega, 0.0)
    end
end

"""
    create_layer(nodes, embed_dim; omega, beta_s, fractal_dim, wave_speed)::WaveLayer

Constructs a new `WaveLayer` with initial wave parameters set to natural harmonic ground states.

# Keyword Arguments
- `omega::Float64 = 432.0`: harmonic driving frequency
- `beta_s::Float64 = φ`: golden-ratio fractal scale
- `fractal_dim::Float64 = 1.5`: initial Hausdorff fractal dimension per node ∈ [1.0, 3.0]
- `wave_speed::Float64 = 1.0`: initial wave propagation speed per node; -1.0 = unlimited
"""
function create_layer(
    nodes::Int,
    embed_dim::Int;
    omega::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895,
    fractal_dim::Float64 = 1.5,
    wave_speed::Float64 = 1.0
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

    # Fractal dimensions initialized around fractal_dim with slight harmonic variation
    frac_dims = [clamp(fractal_dim + 0.02 * sin(2π * i / nodes), 1.0, 3.0) for i in 1:nodes]

    # Wave speeds: if unlimited (-1.0) use uniformly; else small per-node variation
    w_speeds = if wave_speed == -1.0
        fill(-1.0, nodes)
    else
        [clamp(wave_speed * (1.0 + 0.01 * sin(2π * i / nodes)), 0.1, 10.0) for i in 1:nodes]
    end

    return WaveLayer(nodes, embed_dim, amps, phs, freqs, fractals, omega;
                     fractal_dims=frac_dims, wave_speeds=w_speeds)
end

"""
    forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}

Propagates incoming wave information through the layer's quantum nodes.
Computes wave superposition across embeddings and nodes:

    ψᵢ = β_{s,i} · D_{f,i} · Σⱼ Aᵢⱼ · sin(ω·fᵢⱼ·(xⱼ/vᵢ) + φᵢⱼ − t)

where:
- `β_{s,i}` is the per-node fractal scale
- `D_{f,i}` is the per-node Hausdorff fractal dimension (modulates the envelope)
- `vᵢ` is the per-node wave speed (`-1.0` bypasses the divisor for unlimited speed)
"""
function forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = Vector{Float64}(undef, n)
    in_len = length(input_values)

    total_layer_energy = 0.0

    @inbounds for i in 1:n
        node_sum = 0.0
        beta       = layer.fractal_scales[i]
        d_f        = layer.fractal_dims[i]    # Hausdorff fractal dimension
        v_spd      = layer.wave_speeds[i]     # wave speed (-1.0 = unlimited)

        # Fractal dimension envelope factor: D_f modulates output amplitude
        # D_f=1 → linear, D_f=2 → planar, D_f∈(1,3) → fractal
        frac_env = d_f / 1.5  # normalized so default D_f=1.5 → envelope=1.0

        for j in 1:d
            in_val = j <= in_len ? input_values[j] : 0.5
            amp  = layer.amplitudes[i, j]
            ph   = layer.phases[i, j]
            freq = layer.frequencies[i, j]

            # Speed-aware wave phase modulation:
            # v == -1.0 (unlimited): angle = ω·f·x + φ - t   (no delay)
            # v > 0    (finite):     angle = ω·f·(x/v) + φ - t
            x_eff = v_spd == -1.0 ? in_val : in_val / max(v_spd, 1e-12)
            angle = muladd(layer.omega * 0.001 * freq, x_eff, ph - t)
            wave_val = amp * sin(angle)
            node_sum += wave_val
        end

        # Modulate by fractal geometry: β_s + fractal dimension envelope
        node_wave = (node_sum / sqrt(Float64(d))) * beta * frac_env
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
- Fractal dims: Drift within [1.0, 3.0]
- Wave speeds: Drift within [0.1, 10.0] — never drifts into the -1.0 unlimited sentinel
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

        # Fractal dimension drift ∈ [1.0, 3.0]
        layer.fractal_dims[i] += rate * 0.01 * randn()
        layer.fractal_dims[i] = clamp(layer.fractal_dims[i], 1.0, 3.0)

        # Wave speed drift ∈ [0.1, 10.0]
        # Nodes with unlimited speed (-1.0) are never mutated out of sentinel
        if layer.wave_speeds[i] != -1.0
            layer.wave_speeds[i] += rate * 0.05 * randn()
            layer.wave_speeds[i] = clamp(layer.wave_speeds[i], 0.1, 10.0)
        end
    end

    return nothing
end

"""
    crossover(parent_a::WaveLayer, parent_b::WaveLayer)::WaveLayer

Combines two wave layers using the tournament champion **Arithmetic Wave Blend**:
Parameters are superposed in phase space with equal energy weighting.
`fractal_dims` and `wave_speeds` are arithmetically blended (unlimited nodes
retain -1.0 if both parents are unlimited; otherwise blended normally).
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

    # Blend fractal dims ∈ [1.0, 3.0]
    child_fdims = clamp.(0.5 .* (parent_a.fractal_dims .+ parent_b.fractal_dims), 1.0, 3.0)

    # Blend wave speeds — if both parents are unlimited, child is unlimited
    child_wspeeds = Vector{Float64}(undef, nodes)
    for i in 1:nodes
        va = parent_a.wave_speeds[i]
        vb = parent_b.wave_speeds[i]
        if va == -1.0 && vb == -1.0
            child_wspeeds[i] = -1.0
        elseif va == -1.0
            child_wspeeds[i] = vb
        elseif vb == -1.0
            child_wspeeds[i] = va
        else
            child_wspeeds[i] = clamp(0.5 * (va + vb), 0.1, 10.0)
        end
    end

    return WaveLayer(nodes, embed_dim, child_amps, child_phases, child_freqs, child_fractals, parent_a.omega;
                     fractal_dims=child_fdims, wave_speeds=child_wspeeds)
end

function Base.deepcopy(layer::WaveLayer)::WaveLayer
    return WaveLayer(
        layer.nodes,
        layer.embed_dim,
        copy(layer.amplitudes),
        copy(layer.phases),
        copy(layer.frequencies),
        copy(layer.fractal_scales),
        layer.omega;
        fractal_dims = copy(layer.fractal_dims),
        wave_speeds  = copy(layer.wave_speeds)
    )
end
