export apply_fractal, fibonacci_fractal_wave, multiscale_pattern, fractal_dimension_spectrum

using .AudioConstants

"""
    apply_fractal(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Vector{Float64}

Applies fractal harmonics to existing buffer.
"""
function apply_fractal(buffer::Vector{Float64}, wf::WaveFunction, sample_rate::Float64)::Vector{Float64}
    temp_buffer = similar(buffer)
    generate_buffer_fractal!(temp_buffer, wf, sample_rate)
    
    for i in 1:length(buffer)
        buffer[i] += temp_buffer[i]
    end
    
    return buffer
end

"""
    fibonacci_fractal_wave(base_freq::Float64, depth::Int, sample_rate::Float64, num_samples::Int)::Vector{Float64}

Generates a wave with Fibonacci ratio harmonics.
Uses frequency ratios from the FIBONACCI sequence.
"""
function fibonacci_fractal_wave(base_freq::Float64, depth::Int, sample_rate::Float64, num_samples::Int)::Vector{Float64}
    buffer = zeros(Float64, num_samples)
    b_freq = BigFloat(base_freq)
    
    for n in 1:depth
        if n+1 <= length(FIBONACCI)
            ratio = BigFloat(FIBONACCI[n+1]) / BigFloat(FIBONACCI[n])
            freq = Float64(b_freq * ratio)
            
            if freq < sample_rate / 2 # Nyquist limit
                amp = 1.0 / n
                adv = 2.0 * pi * freq / sample_rate
                
                for i in 1:num_samples
                    buffer[i] += amp * sin(adv * (i - 1))
                end
            end
        end
    end
    return buffer
end

"""
    multiscale_pattern(base_freq::Float64, scales::Vector{Float64}, sample_rate::Float64, num_samples::Int)::Vector{Float64}

Creates self-similar pattern at multiple scales.
"""
function multiscale_pattern(base_freq::Float64, scales::Vector{Float64}, sample_rate::Float64, num_samples::Int)::Vector{Float64}
    buffer = zeros(Float64, num_samples)
    
    for scale in scales
        freq = base_freq * scale
        if freq < sample_rate / 2 # Nyquist limit
            amp = 1.0 / scale
            adv = 2.0 * pi * freq / sample_rate
            
            for i in 1:num_samples
                buffer[i] += amp * sin(adv * (i - 1))
            end
        end
    end
    
    return buffer
end

"""
    fractal_dimension_spectrum(buffer::Vector{Float64})::Float64

Estimates fractal dimension of a signal.
Uses a variation metric to approximate dimension properties.
"""
function fractal_dimension_spectrum(buffer::Vector{Float64})::Float64
    N = length(buffer)
    if N < 2
        return 1.0
    end
    
    L = 0.0
    for i in 1:N-1
        L += abs(buffer[i+1] - buffer[i])
    end
    
    if L == 0.0
        return 1.0
    end
    
    dim = 1.0 + log(L) / log(N)
    return clamp(dim, 1.0, 2.0)
end
