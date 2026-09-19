"""
    WaveDataPoint

Core type representing a data point in the wave computation field. Each data point
carries multiple properties (information channels) and performs computation when
"hit" by a wave evaluation.

# Concept

In wave-based computation, data is not stored in registers or memory cells but
in spatial positions within a wave field. When a wave sweeps through the field,
each data point responds to the wave's amplitude at its position, performing
a computation that transforms its value.

    result_i = f(wave(position_i)) ⊗ properties_i

Where:
- position_i ∈ [0, 2π] is the data point's location in the wave field
- wave(x) is the wave function evaluated at that position
- properties_i are the data point's information channels
- ⊗ is the interaction operator (varies by algorithm)

# Properties

Each data point can carry N properties, enabling multi-channel computation:
- :mass — inertial response to wave energy
- :charge — polarity of interaction (+/-)
- :energy — stored computational energy
- :spin — rotational coupling to wave phase
- :frequency — resonance frequency for selective computation
- :amplitude — sensitivity to wave amplitude
- :phase — phase offset for interference patterns
- :dimension — fractal dimension coupling

# Performance

Data points are designed for cache-friendly sequential access. The properties
Dict uses Symbol keys for O(1) lookup.

# References

- Wave-based computation: See `specs/Requirements.md` §2
- Sacred geometry data encoding: See `docs/audio/sacred_geometry.md`
"""
mutable struct WaveDataPoint
    """Position in the wave field [0, 2π] radians"""
    position::Float64
    
    """Current computed value (updated when wave hits this point)"""
    value::Float64
    
    """Extensible properties dictionary for multi-channel information"""
    properties::Dict{Symbol, Float64}
    
    """Number of times this point has been computed on"""
    computation_count::Int
    
    """Last wave amplitude that hit this point"""
    last_wave_amplitude::Float64
    
    """Accumulated energy from wave interactions"""
    accumulated_energy::Float64
    
    """
        WaveDataPoint(position; value=0.0, properties=Dict{Symbol,Float64}())
    
    Construct a data point at the given position in the wave field.
    
    # Arguments
    - `position::Float64`: Position in [0, 2π] radians
    - `value::Float64=0.0`: Initial value
    - `properties::Dict{Symbol,Float64}`: Initial properties
    
    # Examples
    ```julia
    # Simple data point
    dp = WaveDataPoint(π/4)
    
    # Data point with properties
    dp = WaveDataPoint(π/2, value=1.0, properties=Dict(
        :mass => 1.0, :charge => -0.5, :energy => 0.0, :spin => 0.5
    ))
    ```
    """
    function WaveDataPoint(
        position::Float64;
        value::Float64 = 0.0,
        properties::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
    )
        # Normalize position to [0, 2π]
        pos_normalized = mod(position, 2π)
        new(pos_normalized, value, properties, 0, 0.0, 0.0)
    end
end

export WaveDataPoint

"""
    create_data_points(n::Int; distribution::Symbol=:uniform)::Vector{WaveDataPoint}

Create N data points distributed across the wave field [0, 2π].

# Distributions
- `:uniform` — evenly spaced across [0, 2π]
- `:random` — uniformly random positions
- `:fibonacci` — Fibonacci spiral positions (golden angle spacing)
- `:gaussian` — Gaussian distribution centered at π

# Examples
```julia
points = create_data_points(64)                        # 64 uniform points
points = create_data_points(128, distribution=:fibonacci) # Golden angle spacing
```
"""
function create_data_points(n::Int; distribution::Symbol=:uniform)::Vector{WaveDataPoint}
    if n <= 0
        error("Number of data points must be positive, got: $n")
    end
    
    positions = if distribution == :uniform
        [2π * (i - 1) / n for i in 1:n]
    elseif distribution == :random
        [2π * rand() for _ in 1:n]
    elseif distribution == :fibonacci
        # Golden angle spacing: θ_i = i * 2π/φ²
        phi = (1.0 + sqrt(5.0)) / 2.0
        golden_angle = 2π / (phi * phi)
        [mod(i * golden_angle, 2π) for i in 1:n]
    elseif distribution == :gaussian
        # Gaussian centered at π, σ = π/3
        [mod(π + randn() * (π/3), 2π) for _ in 1:n]
    else
        error("Unknown distribution: $distribution. Use :uniform, :random, :fibonacci, or :gaussian")
    end
    
    # Create data points with default properties
    points = WaveDataPoint[]
    for (i, pos) in enumerate(positions)
        dp = WaveDataPoint(pos, properties=Dict{Symbol, Float64}(
            :mass => 1.0,
            :charge => (-1.0)^i * 0.5,  # Alternating charge
            :energy => 0.0,
            :spin => mod(pos / π, 1.0),  # Spin derived from position
            :frequency => 432.0,         # Default resonance frequency
            :amplitude => 1.0,
            :phase => 0.0,
            :dimension => 2.0
        ))
        push!(points, dp)
    end
    
    return points
end

export create_data_points

"""
    wave_hit!(dp::WaveDataPoint, wave_amplitude::Float64, 
              interaction::Symbol=:multiply)::Float64

Compute the result of a wave hitting this data point.

# Interaction Types
- `:multiply` — result = value × wave_amplitude × mass
- `:add` — result = value + wave_amplitude × charge
- `:resonate` — result = value × sin(wave_amplitude × frequency)
- `:interference` — result = value × cos(wave_amplitude × phase + spin × 2π)
- `:energy` — result = 0.5 × mass × wave_amplitude²
- `:quantum` — result = |wave_amplitude|² × energy

Returns the computed result and updates the data point's internal state.
"""
function wave_hit!(dp::WaveDataPoint, wave_amplitude::Float64, 
                   interaction::Symbol=:multiply)::Float64
    mass = get(dp.properties, :mass, 1.0)
    charge = get(dp.properties, :charge, 0.0)
    energy = get(dp.properties, :energy, 0.0)
    spin = get(dp.properties, :spin, 0.0)
    freq = get(dp.properties, :frequency, 432.0)
    phase = get(dp.properties, :phase, 0.0)
    
    result = if interaction == :multiply
        dp.value * wave_amplitude * mass
    elseif interaction == :add
        dp.value + wave_amplitude * charge
    elseif interaction == :resonate
        dp.value * sin(wave_amplitude * freq * 2π / 48000.0)
    elseif interaction == :interference
        dp.value * cos(wave_amplitude * phase + spin * 2π)
    elseif interaction == :energy
        0.5 * mass * wave_amplitude^2
    elseif interaction == :quantum
        abs2(wave_amplitude) * energy
    else
        dp.value * wave_amplitude  # Default: simple multiply
    end
    
    # Update data point state
    dp.value = result
    dp.computation_count += 1
    dp.last_wave_amplitude = wave_amplitude
    dp.accumulated_energy += abs(wave_amplitude)
    
    return result
end

export wave_hit!

"""
    evaluate_wave_at_points(wf::WaveFunction, points::Vector{WaveDataPoint};
                            interaction::Symbol=:multiply)::Vector{Float64}

Evaluate a wave function at all data point positions and compute interactions.

This is the core wave-computation function: the wave sweeps through the data
points and each point performs its computation when "hit".

Returns a vector of computed results, one per data point.

# Example
```julia
wf = WaveFunction(frequency=432.0, amplitude=0.8)
points = create_data_points(64)
results = evaluate_wave_at_points(wf, points)
```
"""
function evaluate_wave_at_points(wf::WaveFunction, points::Vector{WaveDataPoint};
                                  interaction::Symbol=:multiply)::Vector{Float64}
    # Get the trig function
    trig_fn = get_trig_function(wf.trig_func)
    omega = angular_frequency(wf)
    
    results = Vector{Float64}(undef, length(points))
    
    for (i, dp) in enumerate(points)
        # Evaluate wave at this data point's position
        wave_value = wf.amplitude * trig_fn(omega * dp.position / (2π * wf.frequency) + wf.current_phase[])
        
        # Hit the data point with the wave
        results[i] = wave_hit!(dp, wave_value, interaction)
    end
    
    return results
end

export evaluate_wave_at_points

"""
    evaluate_wave_at_points_timed(wf::WaveFunction, points::Vector{WaveDataPoint};
                                   interaction::Symbol=:multiply)

Same as `evaluate_wave_at_points` but returns (results, elapsed_ns) tuple
with nanosecond-precision timing.

# Example
```julia
results, elapsed_ns = evaluate_wave_at_points_timed(wf, points)
println("Computed in \$(elapsed_ns / 1e6) ms")
```
"""
function evaluate_wave_at_points_timed(wf::WaveFunction, points::Vector{WaveDataPoint};
                                        interaction::Symbol=:multiply)
    t_start = time_ns()
    results = evaluate_wave_at_points(wf, points, interaction=interaction)
    t_end = time_ns()
    elapsed_ns = t_end - t_start
    
    return results, elapsed_ns
end

export evaluate_wave_at_points_timed

"""
    reset_data_points!(points::Vector{WaveDataPoint}, value::Float64=0.0)::Nothing

Reset all data points to a given value and clear accumulated state.
"""
function reset_data_points!(points::Vector{WaveDataPoint}, value::Float64=0.0)::Nothing
    for dp in points
        dp.value = value
        dp.computation_count = 0
        dp.last_wave_amplitude = 0.0
        dp.accumulated_energy = 0.0
    end
    nothing
end

export reset_data_points!

"""
    data_point_summary(points::Vector{WaveDataPoint})::Dict{Symbol, Float64}

Get summary statistics of data point values.
"""
function data_point_summary(points::Vector{WaveDataPoint})::Dict{Symbol, Float64}
    values = [dp.value for dp in points]
    energies = [dp.accumulated_energy for dp in points]
    
    Dict{Symbol, Float64}(
        :mean_value => sum(values) / length(values),
        :max_value => maximum(values),
        :min_value => minimum(values),
        :std_value => sqrt(sum((v - sum(values)/length(values))^2 for v in values) / length(values)),
        :total_energy => sum(energies),
        :mean_computations => sum(dp.computation_count for dp in points) / length(points),
    )
end

export data_point_summary
