export superpose!, normalize_preserve_phase!, interference_pattern

"""
    superpose!(output::Vector{Float64}, waves::Vector{WaveFunction}, sample_rate::Float64)::Nothing

Linear superposition of N waves into the output buffer.
Generates each wave individually then sums sample-wise.

    Ψ(t) = Σᵢ ψᵢ(t) = Σᵢ Aᵢ · fᵢ(ωᵢt + φᵢ)

Supports up to 16+ simultaneous waves.
"""
function superpose!(output::Vector{Float64}, waves::Vector{WaveFunction}, sample_rate::Float64)::Nothing
    fill!(output, 0.0)
    num_samples = length(output)
    temp_buffer = Vector{Float64}(undef, num_samples)
    
    for wave in waves
        generate_buffer!(temp_buffer, wave, sample_rate)
        @inbounds for i in 1:num_samples
            output[i] += temp_buffer[i]
        end
    end
    
    return nothing
end

"""
    normalize_preserve_phase!(buffer::Vector{Float64})::Float64

Normalize to [-1, 1] preserving zero crossings.
Returns the normalization factor used.
"""
function normalize_preserve_phase!(buffer::Vector{Float64})::Float64
    max_val = maximum(abs, buffer)
    factor = max_val > 0.0 ? 1.0 / max_val : 1.0
    
    if max_val > 0.0
        @inbounds for i in eachindex(buffer)
            buffer[i] *= factor
        end
    end
    
    return factor
end

"""
    interference_pattern(wave1::WaveFunction, wave2::WaveFunction, 
                          sample_rate::Float64, num_samples::Int)::Vector{Float64}

Returns the interference pattern between two waves by evaluating both
at each sample point and summing.
"""
function interference_pattern(wave1::WaveFunction, wave2::WaveFunction, 
                               sample_rate::Float64, num_samples::Int)::Vector{Float64}
    buf1 = zeros(Float64, num_samples)
    buf2 = zeros(Float64, num_samples)
    
    generate_buffer!(buf1, wave1, sample_rate)
    generate_buffer!(buf2, wave2, sample_rate)
    
    output = Vector{Float64}(undef, num_samples)
    @inbounds for i in 1:num_samples
        output[i] = buf1[i] + buf2[i]
    end
    return output
end
