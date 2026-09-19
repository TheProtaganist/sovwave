export get_trig_function, generate_buffer!, generate_buffer_fractal!

using .AudioConstants

"""
    get_trig_function(sym::Symbol)

Returns the actual Julia trig function for the given symbol.
Supports all 12: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch.
"""
function get_trig_function(sym::Symbol)
    sym == :sin  && return sin
    sym == :cos  && return cos
    sym == :tan  && return tan
    sym == :cot  && return cot
    sym == :sec  && return sec
    sym == :csc  && return csc
    sym == :sinh && return sinh
    sym == :cosh && return cosh
    sym == :tanh && return tanh
    sym == :coth && return coth
    sym == :sech && return sech
    sym == :csch && return csch
    return sin  # Default fallback
end

"""
    evaluate_wave(wf::WaveFunction, t::Float64)::Float64

Evaluate a WaveFunction at time t (seconds).

    ψ(t) = A · f(ωt + φ)

Where ω = 2πf, A = amplitude, f = trig function, φ = current phase.
"""
function evaluate_wave(wf::WaveFunction, t::Float64)::Float64
    trig_fn = get_trig_function(wf.trig_func)
    ω = 2π * wf.frequency
    return wf.amplitude * trig_fn(ω * t + wf.current_phase[])
end

export evaluate_wave

"""
    generate_buffer!(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Nothing

Fills buffer with wave samples using `wf.current_phase[]` for phase continuity,
advances phase after generation.

Formula: `buffer[i] = amplitude * trig_func(current_phase + Δφ * (i-1))`
where Δφ = 2π·f / sample_rate.

Phase is advanced by `length(buffer)` samples after filling to maintain
continuity across consecutive buffer generations.
"""
function generate_buffer!(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Nothing
    trig_fn = get_trig_function(wf.trig_func)
    Δφ = Float64(BigFloat(2) * BigFloat(π) * BigFloat(wf.frequency) / BigFloat(sample_rate))
    cp = wf.current_phase[]
    amp = wf.amplitude
    
    @inbounds for i in 1:length(buffer)
        buffer[i] = amp * trig_fn(cp + Δφ * (i - 1))
    end
    
    # Advance phase for next buffer
    update_phase!(wf, length(buffer))
    return nothing
end

"""
    generate_buffer_fractal!(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Nothing

If `fractal_depth > 0`, generates sum of harmonics at golden ratio frequencies
with amplitude decay according to fractal dimension.

Formula:
    ψ(t) = Σ_{d=0}^{D} (1/φ^{d·dim}) · f(φ^d · Δφ · (i-1) + phase)

Where:
- D = fractal_depth
- φ = golden ratio (PHI ≈ 1.618)
- dim = fractal_dimension (controls amplitude decay rate)
- f = trigonometric function
"""
function generate_buffer_fractal!(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Nothing
    trig_fn = get_trig_function(wf.trig_func)
    Δφ = Float64(BigFloat(2) * BigFloat(π) * BigFloat(wf.frequency) / BigFloat(sample_rate))
    cp = wf.current_phase[]
    amp = wf.amplitude
    depth = wf.fractal_depth
    dim = wf.fractal_dimension
    
    fill!(buffer, 0.0)
    
    @inbounds for d in 0:depth
        amp_d = amp / (PHI ^ (d * dim))
        freq_mult = PHI ^ d
        
        for i in 1:length(buffer)
            buffer[i] += amp_d * trig_fn(cp + freq_mult * Δφ * (i - 1))
        end
    end
    
    # Advance phase for next buffer
    update_phase!(wf, length(buffer))
    return nothing
end
