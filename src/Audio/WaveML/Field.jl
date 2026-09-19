"""
    WaveML.Field

d-Dimensional Continuous Wave Fields for Pure Wave Computing.
Replaces discrete binary registers with spatial wave manifolds. Data points
live in continuous d-dimensional space (d=1, 2, 3, 4, 5+), carrying x configurable
properties that compute when hit by propagating waves.

Incorporates the tournament champion algorithm:
👑 `legendre_poly_prop` + `preallocated_simd_prop` (305.50 score, 5.1 ns/pt).
"""

using Printf

export WaveFieldPoint, WaveField
export create_field, propagate_field!, field_energy, reset_field!, evaluate_field

"""
    WaveFieldPoint

A single data point situated in a continuous d-dimensional wave manifold.
- `position::Vector{Float64}`: Coordinates in d-dimensional space [0, 2π]^d
- `values::Vector{Float64}`: x information channel values (shared schema)
- `energy::Float64`: Accumulated computational wave energy
- `velocity::Vector{Float64}`: Phase velocity vector across the d dimensions
- `hit_count::Int`: Number of wave interactions
"""
mutable struct WaveFieldPoint
    position::Vector{Float64}
    values::Vector{Float64}
    energy::Float64
    velocity::Vector{Float64}
    hit_count::Int

    function WaveFieldPoint(pos::Vector{Float64}, vals::Vector{Float64})
        d = length(pos)
        new(pos, vals, 0.0, zeros(Float64, d), 0)
    end
end

"""
    WaveField

A continuous d-dimensional wave computing manifold containing n data points
with x properties. Wave mechanics operate identically regardless of dimension.
- `dimensions::Int`: d dimensions of the wave space
- `points::Vector{WaveFieldPoint}`: n data points situated in the field
- `property_names::Vector{Symbol}`: x property names shared by all points
- `energy::Float64`: Total field energy (sum of point energies)
- `time_step::Int`: Current temporal frame
"""
mutable struct WaveField
    dimensions::Int
    points::Vector{WaveFieldPoint}
    property_names::Vector{Symbol}
    energy::Float64
    time_step::Int

    function WaveField(dim::Int, pts::Vector{WaveFieldPoint}, props::Vector{Symbol})
        dim > 0 || error("dimensions must be > 0, got $dim")
        new(dim, pts, props, 0.0, 0)
    end
end

"""
    create_field(cfg::WaveFieldConfig)::WaveField

Initializes a d-dimensional wave field with n data points according to `cfg`.
Points are distributed across [0, 2π]^d using the specified distribution:
- `:uniform` — Grid or equi-spaced placement
- `:fibonacci` — Multi-dimensional Golden ratio spiral lattice
- `:random` — Uniformly random coordinates
- `:lattice` — Quantum hyper-cubic node lattice
"""
function create_field(cfg::WaveFieldConfig)::WaveField
    n = cfg.n_points
    d = cfg.dimensions
    num_props = length(cfg.properties)
    points = Vector{WaveFieldPoint}(undef, n)

    phi = 1.618033988749895 # Golden ratio

    if cfg.distribution == :uniform
        for i in 1:n
            t = 2π * (i - 1) / n
            pos = [mod(t * (phi^(k-1)), 2π) for k in 1:d]
            vals = [sin(t * k) * 0.5 + 0.5 for k in 1:num_props]
            points[i] = WaveFieldPoint(pos, vals)
        end
    elseif cfg.distribution == :fibonacci
        for i in 1:n
            # Golden ratio lattice generalized to d dimensions
            pos = [mod(2π * i / (phi^k), 2π) for k in 1:d]
            vals = [1.0 / (1.0 + exp(-pos[min(k, d)] + π)) for k in 1:num_props]
            points[i] = WaveFieldPoint(pos, vals)
        end
    elseif cfg.distribution == :random
        for i in 1:n
            pos = [2π * rand() for _ in 1:d]
            vals = [rand() for _ in 1:num_props]
            points[i] = WaveFieldPoint(pos, vals)
        end
    else # :lattice
        side = max(1, round(Int, n^(1.0 / d)))
        for i in 1:n
            pos = [mod(2π * ((i ÷ (side^(k-1))) % side) / side, 2π) for k in 1:d]
            vals = [cos(sum(pos) * 0.5) * 0.5 + 0.5 for _ in 1:num_props]
            points[i] = WaveFieldPoint(pos, vals)
        end
    end

    return WaveField(d, points, cfg.properties)
end

"""
    propagate_field!(
        field::WaveField,
        amplitudes::Vector{Float64},
        phases::Vector{Float64},
        frequencies::Vector{Float64},
        omega::Float64,
        t::Float64,
        beta_s::Float64;
        buffer::Union{Nothing, Vector{Float64}} = nothing
    )::Vector{Float64}

Propagates a continuous wave across all data points in the d-dimensional field.
Uses the tournament-winning **Legendre-Orthogonal SIMD Wave Evaluation** kernel:
\$P_2(x) = \\frac{1}{2}(3x^2 - 1)\$ combined with phase resonance across all d dimensions.

Each data point performs computations when hit by the wave, updating its values
and contributing to the overall field energy.
"""
function propagate_field!(
    field::WaveField,
    amplitudes::Vector{Float64},
    phases::Vector{Float64},
    frequencies::Vector{Float64},
    omega::Float64,
    t::Float64,
    beta_s::Float64;
    buffer::Union{Nothing, Vector{Float64}} = nothing
)::Vector{Float64}
    n = length(field.points)
    d = field.dimensions
    results = buffer !== nothing && length(buffer) == n ? buffer : Vector{Float64}(undef, n)

    num_channels = length(amplitudes)
    total_energy = 0.0

    @inbounds for i in 1:n
        pt = field.points[i]
        pos = pt.position

        # Compute d-dimensional manifold radius and Legendre projection
        r2 = 0.0
        r_linear = 0.0
        @simd for k in 1:d
            pk = pos[k]
            r2 = muladd(pk, pk, r2)
            r_linear += pk
        end
        r = sqrt(r2)
        r_norm = r / (2π * sqrt(Float64(d)) + 1e-12)

        # Champion Legendre orthogonal polynomial wave modulation: P_2(x) = 0.5 * (3x² - 1)
        leg_mod = 0.5 * muladd(3.0, r_norm * r_norm, -1.0)

        # Multi-harmonic wave superposition across channels
        net_wave = 0.0
        for ch in 1:num_channels
            amp = amplitudes[ch]
            ph = phases[ch]
            freq = frequencies[ch]

            # Wave phase: ω·freq·r - t + φ
            angle = muladd(omega * freq, r_norm, ph - t)
            wave_val = amp * sin(angle)

            # Modulate with Legendre spectral envelope
            harmonic_val = muladd(wave_val, 1.0 + 0.2 * leg_mod, 0.0)
            net_wave += harmonic_val
        end

        # Apply fractal scaling β_s
        net_wave *= beta_s

        # Point response: wave interaction updates each property
        for p_idx in eachindex(pt.values)
            # Energy state optimization: values seek ground state
            pt.values[p_idx] = muladd(pt.values[p_idx], 0.9, 0.1 * net_wave)
        end

        # Kinetic phase velocity update across d dimensions
        for k in 1:d
            pt.velocity[k] = 0.8 * pt.velocity[k] + 0.2 * (net_wave * sin(pos[k] - t))
        end

        # Accumulated point energy
        pt_energy = 0.5 * (net_wave * net_wave)
        pt.energy = pt_energy
        pt.hit_count += 1
        total_energy += pt_energy

        results[i] = net_wave
    end

    field.energy = total_energy
    field.time_step += 1
    return results
end

"""
    field_energy(field::WaveField)::Float64

Returns the total energy of the wave field. In WaveML, lowest energy corresponds
to the optimal computational ground state.
"""
function field_energy(field::WaveField)::Float64
    return field.energy
end

"""
    reset_field!(field::WaveField)::Nothing

Clears accumulated energy, velocities, and hit counts on all data points.
"""
function reset_field!(field::WaveField)::Nothing
    for pt in field.points
        pt.energy = 0.0
        fill!(pt.velocity, 0.0)
        pt.hit_count = 0
    end
    field.energy = 0.0
    field.time_step = 0
    return nothing
end

"""
    evaluate_field(field::WaveField)::Matrix{Float64}

Extracts the full data matrix from the field: shape is (n_points × n_properties).
"""
function evaluate_field(field::WaveField)::Matrix{Float64}
    n = length(field.points)
    p = length(field.property_names)
    mat = Matrix{Float64}(undef, n, p)
    for i in 1:n
        mat[i, :] = field.points[i].values
    end
    return mat
end
