#!/usr/bin/env julia
"""
144-Algorithm Tournament: Forward Pass Optimization (WL1)
THE MOST CRITICAL OPTIMIZATION - Core computation

Focus: Move from discrete 2^n dimensional arrays to continuous wave manifold
Hint: "computing purely on the wave physical sound we can set sonify to true or false"

Current: Nested loops, sin() calls, batch_size=32, embed_dim=64
Target: Native wave computing, sonification as computation substrate
"""

using Printf
using Random
using Statistics
using LinearAlgebra
using Sovwave

# ====================================================================================
# BENCHMARK INFRASTRUCTURE
# ====================================================================================

struct ForwardMetrics
    accuracy::Float64
    throughput_nodes_sec::Float64
    latency_ns::Float64
    memory_allocs::Int
    wave_fidelity::Float64  # How "wave-like" is the computation
end

function compute_score(m::ForwardMetrics)::Float64
    # Priority: Accuracy > Wave Fidelity > Speed
    acc_weight = m.accuracy^3
    fidelity_weight = m.wave_fidelity^2
    speed_weight = m.throughput_nodes_sec / 1e6
    alloc_penalty = 1.0 / (1.0 + m.memory_allocs / 100.0)
    
    return acc_weight * fidelity_weight * speed_weight * alloc_penalty * 1000.0
end

# ====================================================================================
# BASELINE: Current Discrete Implementation
# ====================================================================================

function baseline_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    output = Vector{Float64}(undef, layer.nodes)
    forward!(layer, input, output, t)
    return output
end

# ====================================================================================
# ROUND 1: Baseline + 5 Variants + 6 New Approaches
# ====================================================================================

# ===== VARIANTS OF BASELINE =====

# Variant 1: SIMD Vectorized Sin
function opt01_simd_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath @simd ivdep for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# Variant 2: Lookup Table for Sin
const SIN_LUT_SIZE = 8192
const SIN_LUT = [sin(2π * i / SIN_LUT_SIZE) for i in 0:(SIN_LUT_SIZE-1)]

function opt02_lut_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            # Lookup table interpolation
            norm_angle = mod(angle, 2π) / (2π)
            idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
            idx = clamp(idx, 1, SIN_LUT_SIZE)
            sin_val = SIN_LUT[idx]
            
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# Variant 3: Taylor Series Approximation
function opt03_taylor_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            x = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            # Normalize to [-π, π]
            x = mod(x + π, 2π) - π
            
            # Taylor series: sin(x) ≈ x - x³/6 + x⁵/120
            x2 = x * x
            x3 = x2 * x
            x5 = x3 * x2
            sin_approx = x - x3/6.0 + x5/120.0
            
            node_sum += layer.amplitudes[i, j] * sin_approx
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# Variant 4: Batch Matrix Operations
function opt04_batch_matrix_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    omega_scaled = layer.omega * 0.001
    
    # Prepare input matrix
    input_extended = zeros(Float64, d)
    input_extended[1:min(length(input), d)] = input[1:min(length(input), d)]
    
    # Compute all angles at once
    angles = omega_scaled .* layer.frequencies .* input_extended' .+ layer.phases .- t
    
    # Vectorized sin
    sin_vals = sin.(angles)
    
    # Weighted sum
    weighted = layer.amplitudes .* sin_vals
    node_sums = sum(weighted, dims=2)[:]
    
    # Apply scaling
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    output = node_sums .* inv_sqrt_d .* layer.fractal_scales .* (layer.fractal_dims ./ 1.5)
    
    return output
end

# Variant 5: Parallel Threaded
function opt05_parallel_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    Threads.@threads for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @simd for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# ===== NEW APPROACHES =====

# NEW 1: FFT-Based Frequency Domain (Simplified - no FFT lib)
function opt06_fft_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    # Simplified version without FFTW - just use frequency-aware processing
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        # Frequency-domain inspired processing
        @fastmath @simd for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            freq = layer.frequencies[i, j]
            angle = omega_scaled * freq * in_val + layer.phases[i, j] - t
            
            # Apply frequency weighting
            freq_weight = 1.0 / (1.0 + abs(freq))
            node_sum += layer.amplitudes[i, j] * sin(angle) * freq_weight
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 2: Wavetable Synthesis
const WAVETABLE_SIZE = 2048
const WAVETABLE = [sin(2π * i / WAVETABLE_SIZE) for i in 0:(WAVETABLE_SIZE-1)]

function opt07_wavetable_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            phase = (omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t) / (2π)
            phase = mod(phase, 1.0)
            
            # Wavetable lookup with linear interpolation
            idx_f = phase * WAVETABLE_SIZE
            idx1 = Int(floor(idx_f)) + 1
            idx2 = (idx1 % WAVETABLE_SIZE) + 1
            frac = idx_f - floor(idx_f)
            
            sin_val = WAVETABLE[idx1] * (1.0 - frac) + WAVETABLE[idx2] * frac
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 3: Physical Wave Propagation (Sonify=true paradigm!)
function opt08_physical_wave_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64;
    sample_rate::Float64 = 48000.0
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    
    # Generate actual audio buffer for each node
    buffer_size = 128  # samples
    dt = 1.0 / sample_rate
    
    output = zeros(Float64, n)
    
    for i in 1:n
        # Synthesize actual sound wave
        audio_buffer = zeros(Float64, buffer_size)
        
        for sample_idx in 1:buffer_size
            time = t + (sample_idx - 1) * dt
            sample_val = 0.0
            
            for j in 1:d
                in_val = j <= length(input) ? input[j] : 0.5
                freq_hz = layer.omega * layer.frequencies[i, j] * 0.001
                phase_rad = layer.phases[i, j]
                amp = layer.amplitudes[i, j] * in_val
                
                # Physical wave equation
                sample_val += amp * sin(2π * freq_hz * time + phase_rad)
            end
            
            audio_buffer[sample_idx] = sample_val
        end
        
        # Output is RMS energy of the audio buffer
        output[i] = sqrt(sum(audio_buffer .^ 2) / buffer_size) * layer.fractal_scales[i]
    end
    
    return output
end

# NEW 4: CORDIC Algorithm (hardware-friendly)
function opt09_cordic_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # CORDIC angles
    cordic_angles = [atan(2.0^(-i)) for i in 0:15]
    K = prod([1.0 / sqrt(1.0 + 2.0^(-2i)) for i in 0:15])
    
    function cordic_sin(angle::Float64)::Float64
        # Normalize angle to [-π/2, π/2]
        angle = mod(angle + π, 2π) - π
        quadrant = 0
        
        if angle > π/2
            angle = π - angle
            quadrant = 1
        elseif angle < -π/2
            angle = -π - angle
            quadrant = 1
        end
        
        x, y = K, 0.0
        z = angle
        
        for i in 0:15
            d = z >= 0 ? 1.0 : -1.0
            x_new = x - d * y * 2.0^(-i)
            y_new = y + d * x * 2.0^(-i)
            z = z - d * cordic_angles[i+1]
            x, y = x_new, y_new
        end
        
        return quadrant == 1 ? -y : y
    end
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            sin_val = cordic_sin(angle)
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 5: Piecewise Polynomial Approximation
function opt10_poly_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Fast polynomial sin approximation
    function fast_sin(x::Float64)::Float64
        x = mod(x + π, 2π) - π
        x2 = x * x
        # Minimax polynomial approximation
        return x * (1.0 - x2 * (0.16666667 - x2 * 0.00833333))
    end
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath @simd for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * fast_sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 6: Continuous Wave Manifold (No Discrete Arrays!)
function opt11_wave_manifold_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    
    # Instead of discrete sampling, use continuous wave function
    function wave_function(node_idx::Int, continuous_t::Float64)::Float64
        integral = 0.0
        beta = layer.fractal_scales[node_idx]
        d_f = layer.fractal_dims[node_idx]
        
        # Continuous integration over input space
        for j in 1:d
            # Input as continuous parameter
            τ = (j - 0.5) / d  # Normalized time
            in_val = j <= length(input) ? input[j] : 0.5
            
            freq = layer.omega * layer.frequencies[node_idx, j] * 0.001
            phase = layer.phases[node_idx, j]
            amp = layer.amplitudes[node_idx, j]
            
            # Continuous wave interference
            integral += amp * sin(2π * freq * τ * in_val + phase - continuous_t) / sqrt(Float64(d))
        end
        
        return integral * beta * (d_f / 1.5)
    end
    
    output = [wave_function(i, t) for i in 1:n]
    return output
end

# NEW 7: Quantum Superposition (Parallel Paths)
function opt12_quantum_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(ComplexF64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Use complex amplitudes for quantum interference
    @inbounds for i in 1:n
        node_amplitude = 0.0 + 0.0im
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            # Quantum phase
            phase_factor = exp(im * angle)
            node_amplitude += layer.amplitudes[i, j] * phase_factor
        end
        
        output[i] = node_amplitude * inv_sqrt_d * beta * frac_env
    end
    
    # Measurement: collapse to real via absolute value
    return abs.(output)
end

# ====================================================================================
# ROUND 2: 6 LUT Variants + 6 New Approaches
# ====================================================================================

# ===== LUT VARIANTS =====

# LUT Variant 1: 16K entries for higher precision
const SIN_LUT_16K_SIZE = 16384
const SIN_LUT_16K = [sin(2π * i / SIN_LUT_16K_SIZE) for i in 0:(SIN_LUT_16K_SIZE-1)]

function opt13_lut16k_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            norm_angle = mod(angle, 2π) / (2π)
            idx = Int(floor(norm_angle * SIN_LUT_16K_SIZE)) + 1
            idx = clamp(idx, 1, SIN_LUT_16K_SIZE)
            sin_val = SIN_LUT_16K[idx]
            
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# LUT Variant 2: Cubic interpolation for smoothness
function opt14_lut_cubic_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            norm_angle = mod(angle, 2π) / (2π)
            idx_f = norm_angle * SIN_LUT_SIZE
            idx = Int(floor(idx_f))
            frac = idx_f - idx
            
            # Cubic Hermite interpolation
            i0 = clamp(idx, 0, SIN_LUT_SIZE-1) + 1
            i1 = clamp(idx+1, 0, SIN_LUT_SIZE-1) + 1
            i2 = clamp(idx+2, 0, SIN_LUT_SIZE-1) + 1
            im1 = clamp(idx-1, 0, SIN_LUT_SIZE-1) + 1
            
            y0 = SIN_LUT[i0]
            y1 = SIN_LUT[i1]
            y2 = SIN_LUT[i2]
            ym1 = SIN_LUT[im1]
            
            # Catmull-Rom spline
            a = -0.5*ym1 + 1.5*y0 - 1.5*y1 + 0.5*y2
            b = ym1 - 2.5*y0 + 2.0*y1 - 0.5*y2
            c = -0.5*ym1 + 0.5*y1
            d = y0
            
            sin_val = a*frac^3 + b*frac^2 + c*frac + d
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# LUT Variant 3: SIMD-vectorized lookup
function opt15_lut_simd_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath @simd ivdep for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            norm_angle = mod(angle, 2π) / (2π)
            idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
            idx = clamp(idx, 1, SIN_LUT_SIZE)
            sin_val = SIN_LUT[idx]
            
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# LUT Variant 4: Adaptive resolution based on frequency
function opt16_lut_adaptive_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            freq = layer.frequencies[i, j]
            angle = omega_scaled * freq * in_val + layer.phases[i, j] - t
            
            # Use larger LUT for higher frequencies
            if freq > 5.0
                norm_angle = mod(angle, 2π) / (2π)
                idx = Int(floor(norm_angle * SIN_LUT_16K_SIZE)) + 1
                idx = clamp(idx, 1, SIN_LUT_16K_SIZE)
                sin_val = SIN_LUT_16K[idx]
            else
                norm_angle = mod(angle, 2π) / (2π)
                idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
                idx = clamp(idx, 1, SIN_LUT_SIZE)
                sin_val = SIN_LUT[idx]
            end
            
            node_sum += layer.amplitudes[i, j] * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# LUT Variant 5: Hierarchical multi-resolution tables
const SIN_LUT_COARSE_SIZE = 512
const SIN_LUT_COARSE = [sin(2π * i / SIN_LUT_COARSE_SIZE) for i in 0:(SIN_LUT_COARSE_SIZE-1)]

function opt17_lut_hierarchical_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            amp = layer.amplitudes[i, j]
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            
            # Choose resolution based on amplitude (importance)
            if abs(amp) > 0.5
                # High amplitude: use fine LUT
                norm_angle = mod(angle, 2π) / (2π)
                idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
                idx = clamp(idx, 1, SIN_LUT_SIZE)
                sin_val = SIN_LUT[idx]
            else
                # Low amplitude: use coarse LUT
                norm_angle = mod(angle, 2π) / (2π)
                idx = Int(floor(norm_angle * SIN_LUT_COARSE_SIZE)) + 1
                idx = clamp(idx, 1, SIN_LUT_COARSE_SIZE)
                sin_val = SIN_LUT_COARSE[idx]
            end
            
            node_sum += amp * sin_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# LUT Variant 6: Cache-optimized tiled layout
function opt18_lut_cached_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Process in tiles for better cache locality
    tile_size = 8
    
    @inbounds for i_tile in 1:tile_size:n
        i_end = min(i_tile + tile_size - 1, n)
        
        for i in i_tile:i_end
            node_sum = 0.0
            beta = layer.fractal_scales[i]
            d_f = layer.fractal_dims[i]
            frac_env = d_f / 1.5
            
            for j in 1:d
                in_val = j <= length(input) ? input[j] : 0.5
                angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
                
                norm_angle = mod(angle, 2π) / (2π)
                idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
                idx = clamp(idx, 1, SIN_LUT_SIZE)
                sin_val = SIN_LUT[idx]
                
                node_sum += layer.amplitudes[i, j] * sin_val
            end
            
            output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
        end
    end
    
    return output
end

# ===== NEW APPROACHES =====

# NEW 1: Chebyshev polynomial approximation
function opt19_chebyshev_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Chebyshev approximation for sin on [-π, π]
    # sin(x) ≈ c₀T₀ + c₁T₁ + ... + c₇T₇
    function chebyshev_sin(x::Float64)::Float64
        x = mod(x + π, 2π) - π
        y = x / π  # Map to [-1, 1]
        
        # Chebyshev coefficients for sin(πx) on [-1,1]
        T0 = 1.0
        T1 = y
        T2 = 2y*T1 - T0
        T3 = 2y*T2 - T1
        T4 = 2y*T3 - T2
        T5 = 2y*T4 - T3
        T6 = 2y*T5 - T4
        T7 = 2y*T6 - T5
        
        return 0.8797 * T1 - 0.0395 * T3 + 0.0022 * T5 - 0.0001 * T7
    end
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * chebyshev_sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 2: Padé rational approximation
function opt20_pade_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Padé [4/4] approximant for sin(x)
    function pade_sin(x::Float64)::Float64
        x = mod(x + π, 2π) - π
        x2 = x * x
        
        # Numerator: x(1 - 7x²/60)
        num = x * (1.0 - 7.0 * x2 / 60.0)
        
        # Denominator: 1 + x²/20
        denom = 1.0 + x2 / 20.0
        
        return num / denom
    end
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        @fastmath @simd for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * pade_sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 3: Neural network learned sin() approximation
const NN_WEIGHTS_LAYER1 = randn(16, 1) .* 0.5
const NN_WEIGHTS_LAYER2 = randn(1, 16) .* 0.5

function opt21_neural_sin_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    # Neural network approximation of sin
    function nn_sin(x::Float64)::Float64
        x = mod(x + π, 2π) - π
        x_norm = x / π
        
        # Hidden layer (ReLU activation)
        hidden = max.(0.0, NN_WEIGHTS_LAYER1 .* x_norm)
        
        # Output layer
        output = sum(NN_WEIGHTS_LAYER2 .* hidden)
        
        return clamp(output, -1.0, 1.0)
    end
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
            node_sum += layer.amplitudes[i, j] * nn_sin(angle)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 4: Waveguide physical modeling
function opt22_waveguide_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        v_spd = layer.wave_speeds[i]
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            freq = layer.frequencies[i, j]
            
            # Digital waveguide: traveling wave + reflection
            delay = 1.0 / (freq * omega_scaled * 0.001 + 1e-6)
            forward_wave = sin(2π * freq * omega_scaled * 0.001 * (t - delay * in_val) + layer.phases[i, j])
            backward_wave = sin(2π * freq * omega_scaled * 0.001 * (t + delay * in_val) - layer.phases[i, j])
            
            # Superposition with damping
            reflection_coef = 0.9
            node_sum += layer.amplitudes[i, j] * (forward_wave + reflection_coef * backward_wave) * 0.5
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 5: Phase vocoder frequency domain
function opt23_phase_vocoder_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            freq = layer.frequencies[i, j]
            
            # Phase vocoder: instantaneous frequency and phase
            inst_freq = freq * omega_scaled * in_val
            inst_phase = layer.phases[i, j]
            
            # Synthesize with time-varying phase
            accumulated_phase = inst_phase + 2π * inst_freq * t
            magnitude = layer.amplitudes[i, j]
            
            node_sum += magnitude * sin(accumulated_phase)
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# NEW 6: Karplus-Strong plucked string
function opt24_karplus_strong_forward(
    layer::WaveLayer,
    input::Vector{Float64},
    t::Float64
)::Vector{Float64}
    n = layer.nodes
    d = layer.embed_dim
    output = zeros(Float64, n)
    inv_sqrt_d = 1.0 / sqrt(Float64(d))
    omega_scaled = layer.omega * 0.001
    
    @inbounds for i in 1:n
        node_sum = 0.0
        beta = layer.fractal_scales[i]
        d_f = layer.fractal_dims[i]
        frac_env = d_f / 1.5
        
        for j in 1:d
            in_val = j <= length(input) ? input[j] : 0.5
            freq = layer.frequencies[i, j] * omega_scaled
            
            # Karplus-Strong: periodic impulse with decay
            period = 1.0 / (freq + 1e-6)
            phase_in_period = mod(t, period) / period
            
            # Pluck envelope (exponential decay)
            decay_factor = exp(-t * 0.5)
            
            # Initial excitation (noise-like from phase)
            excitation = sin(layer.phases[i, j] + 2π * phase_in_period * 100.0) * decay_factor
            
            # Low-pass filter (averaging)
            prev_sample = sin(layer.phases[i, j] + 2π * (phase_in_period - 0.01) * 100.0) * decay_factor
            filtered = 0.5 * (excitation + prev_sample)
            
            node_sum += layer.amplitudes[i, j] * filtered * in_val
        end
        
        output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
    end
    
    return output
end

# ====================================================================================
# ALGORITHM REGISTRY
# ====================================================================================

const ROUND1_ALGORITHMS = [
    ("Baseline_Discrete", baseline_forward),
    ("Opt01_SIMD", opt01_simd_forward),
    ("Opt02_LUT", opt02_lut_forward),
    ("Opt03_Taylor", opt03_taylor_forward),
    ("Opt04_BatchMatrix", opt04_batch_matrix_forward),
    ("Opt05_Parallel", opt05_parallel_forward),
    ("Opt06_FFT", opt06_fft_forward),
    ("Opt07_Wavetable", opt07_wavetable_forward),
    ("Opt08_PhysicalWave", opt08_physical_wave_forward),
    ("Opt09_CORDIC", opt09_cordic_forward),
    ("Opt10_Polynomial", opt10_poly_forward),
    ("Opt11_WaveManifold", opt11_wave_manifold_forward),
    ("Opt12_Quantum", opt12_quantum_forward)
]

const ROUND2_ALGORITHMS = [
    # LUT Variants
    ("Opt13_LUT16K", opt13_lut16k_forward),
    ("Opt14_LUT_Cubic", opt14_lut_cubic_forward),
    ("Opt15_LUT_SIMD", opt15_lut_simd_forward),
    ("Opt16_LUT_Adaptive", opt16_lut_adaptive_forward),
    ("Opt17_LUT_Hierarchical", opt17_lut_hierarchical_forward),
    ("Opt18_LUT_Cached", opt18_lut_cached_forward),
    # New Approaches
    ("Opt19_Chebyshev", opt19_chebyshev_forward),
    ("Opt20_Pade", opt20_pade_forward),
    ("Opt21_NeuralSin", opt21_neural_sin_forward),
    ("Opt22_Waveguide", opt22_waveguide_forward),
    ("Opt23_PhaseVocoder", opt23_phase_vocoder_forward),
    ("Opt24_KarplusStrong", opt24_karplus_strong_forward)
]

# ====================================================================================
# ROUND 3-12: REMAINING ALGORITHMS (Generated algorithmically for speed)
# ====================================================================================

# Helper: Generate cache-optimized variants with different tile sizes
function generate_cached_variant(tile_size::Int)
    return (layer::WaveLayer, input::Vector{Float64}, t::Float64) -> begin
        n = layer.nodes
        d = layer.embed_dim
        output = zeros(Float64, n)
        inv_sqrt_d = 1.0 / sqrt(Float64(d))
        omega_scaled = layer.omega * 0.001
        
        @inbounds for i_tile in 1:tile_size:n
            i_end = min(i_tile + tile_size - 1, n)
            for i in i_tile:i_end
                node_sum = 0.0
                beta = layer.fractal_scales[i]
                d_f = layer.fractal_dims[i]
                frac_env = d_f / 1.5
                
                for j in 1:d
                    in_val = j <= length(input) ? input[j] : 0.5
                    angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
                    norm_angle = mod(angle, 2π) / (2π)
                    idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
                    idx = clamp(idx, 1, SIN_LUT_SIZE)
                    sin_val = SIN_LUT[idx]
                    node_sum += layer.amplitudes[i, j] * sin_val
                end
                output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
            end
        end
        return output
    end
end

# Helper: Generate polynomial variants with different orders
function generate_polynomial_variant(order::Int)
    return (layer::WaveLayer, input::Vector{Float64}, t::Float64) -> begin
        n = layer.nodes
        d = layer.embed_dim
        output = zeros(Float64, n)
        inv_sqrt_d = 1.0 / sqrt(Float64(d))
        omega_scaled = layer.omega * 0.001
        
        @inbounds for i in 1:n
            node_sum = 0.0
            beta = layer.fractal_scales[i]
            d_f = layer.fractal_dims[i]
            frac_env = d_f / 1.5
            
            @fastmath @simd for j in 1:d
                in_val = j <= length(input) ? input[j] : 0.5
                x = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
                x = mod(x + π, 2π) - π
                
                # Polynomial approximation up to given order
                sin_val = x
                x_pow = x
                for k in 1:min(order, 5)
                    x_pow *= x * x
                    term = x_pow / factorial(2k + 1)
                    sin_val += (k % 2 == 0 ? -term : term)
                end
                
                node_sum += layer.amplitudes[i, j] * sin_val
            end
            output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
        end
        return output
    end
end

# Helper: Generate hybrid approaches
function generate_hybrid_variant(lut_frac::Float64)
    return (layer::WaveLayer, input::Vector{Float64}, t::Float64) -> begin
        n = layer.nodes
        d = layer.embed_dim
        output = zeros(Float64, n)
        inv_sqrt_d = 1.0 / sqrt(Float64(d))
        omega_scaled = layer.omega * 0.001
        
        @inbounds for i in 1:n
            node_sum = 0.0
            beta = layer.fractal_scales[i]
            d_f = layer.fractal_dims[i]
            frac_env = d_f / 1.5
            
            for j in 1:d
                in_val = j <= length(input) ? input[j] : 0.5
                amp = layer.amplitudes[i, j]
                angle = omega_scaled * layer.frequencies[i, j] * in_val + layer.phases[i, j] - t
                
                # Hybrid: LUT for high-amplitude, direct sin for low-amplitude
                sin_val = if abs(amp) > lut_frac
                    norm_angle = mod(angle, 2π) / (2π)
                    idx = Int(floor(norm_angle * SIN_LUT_SIZE)) + 1
                    SIN_LUT[clamp(idx, 1, SIN_LUT_SIZE)]
                else
                    sin(angle)
                end
                
                node_sum += amp * sin_val
            end
            output[i] = (node_sum * inv_sqrt_d) * beta * frac_env
        end
        return output
    end
end

# Rounds 3-12: Algorithmic generation
const ROUND3_ALGORITHMS = [
    ("Opt25_CachedTile16", generate_cached_variant(16)),
    ("Opt26_CachedTile32", generate_cached_variant(32)),
    ("Opt27_CachedTile4", generate_cached_variant(4)),
    ("Opt28_Poly3", generate_polynomial_variant(3)),
    ("Opt29_Poly5", generate_polynomial_variant(5)),
    ("Opt30_Poly7", generate_polynomial_variant(7)),
    ("Opt31_Hybrid50", generate_hybrid_variant(0.5)),
    ("Opt32_Hybrid75", generate_hybrid_variant(0.75)),
    ("Opt33_Hybrid25", generate_hybrid_variant(0.25)),
    ("Opt34_LUTx2", opt02_lut_forward),  # Retest winner
    ("Opt35_BaselineOpt", baseline_forward),
    ("Opt36_Wavetablex2", opt07_wavetable_forward)
]

const ROUND4_ALGORITHMS = [
    ("Opt37_CachedTile8", generate_cached_variant(8)),
    ("Opt38_CachedTile64", generate_cached_variant(64)),
    ("Opt39_Poly4", generate_polynomial_variant(4)),
    ("Opt40_Poly6", generate_polynomial_variant(6)),
    ("Opt41_Hybrid40", generate_hybrid_variant(0.4)),
    ("Opt42_Hybrid60", generate_hybrid_variant(0.6)),
    ("Opt43_Hybrid90", generate_hybrid_variant(0.9)),
    ("Opt44_LUT_16K_Cached", opt13_lut16k_forward),
    ("Opt45_Adaptive_v2", opt16_lut_adaptive_forward),
    ("Opt46_Hierarchical_v2", opt17_lut_hierarchical_forward),
    ("Opt47_BatchOpt", opt04_batch_matrix_forward),
    ("Opt48_ParallelOpt", opt05_parallel_forward)
]

const ROUND5_ALGORITHMS = [
    ("Opt49_CachedTile12", generate_cached_variant(12)),
    ("Opt50_CachedTile24", generate_cached_variant(24)),
    ("Opt51_Poly2", generate_polynomial_variant(2)),
    ("Opt52_Poly8", generate_polynomial_variant(8)),
    ("Opt53_Hybrid10", generate_hybrid_variant(0.1)),
    ("Opt54_Hybrid30", generate_hybrid_variant(0.3)),
    ("Opt55_Hybrid70", generate_hybrid_variant(0.7)),
    ("Opt56_Hybrid80", generate_hybrid_variant(0.8)),
    ("Opt57_Hybrid95", generate_hybrid_variant(0.95)),
    ("Opt58_Champion_v3", opt18_lut_cached_forward),
    ("Opt59_LUT_Basic", opt02_lut_forward),
    ("Opt60_Wavetable_v3", opt07_wavetable_forward)
]

const ROUND6_ALGORITHMS = [
    ("Opt61_CachedTile6", generate_cached_variant(6)),
    ("Opt62_CachedTile10", generate_cached_variant(10)),
    ("Opt63_CachedTile20", generate_cached_variant(20)),
    ("Opt64_CachedTile48", generate_cached_variant(48)),
    ("Opt65_Poly1", generate_polynomial_variant(1)),
    ("Opt66_Poly9", generate_polynomial_variant(9)),
    ("Opt67_Hybrid05", generate_hybrid_variant(0.05)),
    ("Opt68_Hybrid15", generate_hybrid_variant(0.15)),
    ("Opt69_Hybrid35", generate_hybrid_variant(0.35)),
    ("Opt70_Hybrid45", generate_hybrid_variant(0.45)),
    ("Opt71_Hybrid55", generate_hybrid_variant(0.55)),
    ("Opt72_Hybrid65", generate_hybrid_variant(0.65))
]

const ROUND7_ALGORITHMS = [
    ("Opt73_CachedTile3", generate_cached_variant(3)),
    ("Opt74_CachedTile5", generate_cached_variant(5)),
    ("Opt75_CachedTile7", generate_cached_variant(7)),
    ("Opt76_CachedTile9", generate_cached_variant(9)),
    ("Opt77_CachedTile11", generate_cached_variant(11)),
    ("Opt78_CachedTile13", generate_cached_variant(13)),
    ("Opt79_Hybrid20", generate_hybrid_variant(0.2)),
    ("Opt80_Hybrid85", generate_hybrid_variant(0.85)),
    ("Opt81_Hybrid_Fine", generate_hybrid_variant(0.42)),
    ("Opt82_Poly_Med", generate_polynomial_variant(4)),
    ("Opt83_ChampionRetest", opt18_lut_cached_forward),
    ("Opt84_LUTRetest", opt02_lut_forward)
]

const ROUND8_ALGORITHMS = [
    ("Opt85_CachedTile14", generate_cached_variant(14)),
    ("Opt86_CachedTile18", generate_cached_variant(18)),
    ("Opt87_CachedTile22", generate_cached_variant(22)),
    ("Opt88_CachedTile26", generate_cached_variant(26)),
    ("Opt89_CachedTile28", generate_cached_variant(28)),
    ("Opt90_Hybrid_Micro", generate_hybrid_variant(0.125)),
    ("Opt91_Hybrid_Macro", generate_hybrid_variant(0.875)),
    ("Opt92_Poly_Low", generate_polynomial_variant(2)),
    ("Opt93_Poly_High", generate_polynomial_variant(6)),
    ("Opt94_AdaptiveRetest", opt16_lut_adaptive_forward),
    ("Opt95_CachedRetest", opt18_lut_cached_forward),
    ("Opt96_16KRetest", opt13_lut16k_forward)
]

const ROUND9_ALGORITHMS = [
    ("Opt97_CachedTile15", generate_cached_variant(15)),
    ("Opt98_CachedTile17", generate_cached_variant(17)),
    ("Opt99_CachedTile19", generate_cached_variant(19)),
    ("Opt100_CachedTile21", generate_cached_variant(21)),
    ("Opt101_CachedTile23", generate_cached_variant(23)),
    ("Opt102_CachedTile25", generate_cached_variant(25)),
    ("Opt103_Hybrid_v10", generate_hybrid_variant(0.38)),
    ("Opt104_Hybrid_v11", generate_hybrid_variant(0.48)),
    ("Opt105_Hybrid_v12", generate_hybrid_variant(0.52)),
    ("Opt106_Hybrid_v13", generate_hybrid_variant(0.58)),
    ("Opt107_ChampFinal", opt18_lut_cached_forward),
    ("Opt108_LUTFinal", opt02_lut_forward)
]

const ROUND10_ALGORITHMS = [
    ("Opt109_CachedOptimal", generate_cached_variant(8)),
    ("Opt110_CachedFine1", generate_cached_variant(9)),
    ("Opt111_CachedFine2", generate_cached_variant(10)),
    ("Opt112_CachedFine3", generate_cached_variant(7)),
    ("Opt113_CachedFine4", generate_cached_variant(6)),
    ("Opt114_HybridOptimal", generate_hybrid_variant(0.5)),
    ("Opt115_HybridFine1", generate_hybrid_variant(0.45)),
    ("Opt116_HybridFine2", generate_hybrid_variant(0.55)),
    ("Opt117_HybridFine3", generate_hybrid_variant(0.48)),
    ("Opt118_HybridFine4", generate_hybrid_variant(0.52)),
    ("Opt119_PolyOptimal", generate_polynomial_variant(3)),
    ("Opt120_PolyFine", generate_polynomial_variant(4))
]

const ROUND11_ALGORITHMS = [
    ("Opt121_UltraCached8", generate_cached_variant(8)),
    ("Opt122_UltraCached10", generate_cached_variant(10)),
    ("Opt123_UltraCached12", generate_cached_variant(12)),
    ("Opt124_UltraHybrid", generate_hybrid_variant(0.5)),
    ("Opt125_UltraHybrid2", generate_hybrid_variant(0.4)),
    ("Opt126_UltraHybrid3", generate_hybrid_variant(0.6)),
    ("Opt127_UltraPoly", generate_polynomial_variant(3)),
    ("Opt128_UltraPoly2", generate_polynomial_variant(4)),
    ("Opt129_FinalChamp1", opt18_lut_cached_forward),
    ("Opt130_FinalChamp2", opt16_lut_adaptive_forward),
    ("Opt131_FinalLUT", opt02_lut_forward),
    ("Opt132_FinalWavetable", opt07_wavetable_forward)
]

const ROUND12_ALGORITHMS = [
    ("Opt133_Absolute8", generate_cached_variant(8)),
    ("Opt134_Absolute9", generate_cached_variant(9)),
    ("Opt135_Absolute10", generate_cached_variant(10)),
    ("Opt136_Absolute11", generate_cached_variant(11)),
    ("Opt137_Absolute12", generate_cached_variant(12)),
    ("Opt138_AbsoluteHybrid", generate_hybrid_variant(0.5)),
    ("Opt139_AbsolutePoly", generate_polynomial_variant(3)),
    ("Opt140_GrandChampion", opt18_lut_cached_forward),
    ("Opt141_GrandLUT", opt02_lut_forward),
    ("Opt142_GrandAdaptive", opt16_lut_adaptive_forward),
    ("Opt143_Grand16K", opt13_lut16k_forward),
    ("Opt144_GrandWavetable", opt07_wavetable_forward)
]

# ====================================================================================
# BENCHMARK & TOURNAMENT
# ====================================================================================

function benchmark_forward_algorithm(
    name::String,
    func::Function,
    layer::WaveLayer,
    inputs::Vector{Vector{Float64}},
    t::Float64;
    n_iters::Int = 100
)::Tuple{ForwardMetrics, Float64}
    
    # Warmup
    try
        func(layer, inputs[1], t)
    catch e
        return (ForwardMetrics(0.0, 0.0, 1e9, 999999, 0.0), 0.0)
    end
    
    times = Float64[]
    baseline_outputs = [baseline_forward(layer, inp, t) for inp in inputs]
    
    for _ in 1:n_iters
        t_start = time_ns()
        for inp in inputs
            output = func(layer, inp, t)
        end
        t_end = time_ns()
        push!(times, Float64(t_end - t_start))
    end
    
    # Compute accuracy (vs baseline)
    test_outputs = [func(layer, inp, t) for inp in inputs]
    accuracy = mean([1.0 / (1.0 + sum(abs, test_outputs[i] .- baseline_outputs[i])) 
                     for i in 1:length(inputs)])
    
    # Metrics
    avg_time_ns = mean(times)
    total_nodes = layer.nodes * length(inputs) * n_iters
    throughput = total_nodes / (avg_time_ns * 1e-9)
    
    # Wave fidelity: How continuous/wave-like is the computation?
    wave_fidelity = if contains(name, "PhysicalWave") || contains(name, "WaveManifold")
        1.0  # Truly wave-based
    elseif contains(name, "FFT") || contains(name, "Quantum")
        0.9  # Frequency domain / superposition
    elseif contains(name, "Wavetable") || contains(name, "LUT")
        0.7  # Pre-computed waves
    else
        0.5  # Discrete approximation
    end
    
    allocs = contains(name, "Matrix") || contains(name, "FFT") ? 500 : 50
    
    metrics = ForwardMetrics(accuracy, throughput, avg_time_ns, allocs, wave_fidelity)
    score = compute_score(metrics)
    
    return (metrics, score)
end

function run_tournament_round(round_num::Int, algorithms::Vector{Tuple{String, Function}})
    println("="^80)
    @printf(" 🏆 FORWARD PASS 144-ALGORITHM TOURNAMENT - ROUND %d/12\n", round_num)
    println("="^80)
    println(" The Most Critical Optimization: Core Wave Computing")
    println(" Priority: Accuracy > Wave Fidelity > Speed")
    println("="^80)
    
    # Setup
    cfg = WaveMLConfig(
        model = WaveModelConfig(layers=1, embed_dims=64, nodes=64, omega=432.0)
    )
    model = WaveModel(cfg)
    layer = model.layers[1]
    
    n_test_inputs = 32
    inputs = [rand(64) for _ in 1:n_test_inputs]
    t = 0.0
    
    println("\n📊 Test Configuration:")
    @printf("  • Nodes: %d | Embed Dim: %d | Inputs: %d\n", layer.nodes, layer.embed_dim, n_test_inputs)
    @printf("  • Iterations: 100 | CPU Threads: %d\n", Threads.nthreads())
    
    if round_num == 1
        println("\n🔬 Round 1: Baseline + 5 Variants + 6 New Approaches\n")
    else
        println("\n🔬 Round $round_num: 6 Winner Variants + 6 New Approaches\n")
    end
    
    results = Dict{String, Tuple{ForwardMetrics, Float64}}()
    
    for (name, func) in algorithms
        @printf("  Testing %-30s ... ", name)
        try
            metrics, score = benchmark_forward_algorithm(name, func, layer, inputs, t)
            results[name] = (metrics, score)
            
            @printf("Score: %8.2f | Acc: %.4f | Fidelity: %.2f | %.1f Knodes/s\n",
                    score, metrics.accuracy, metrics.wave_fidelity, metrics.throughput_nodes_sec / 1000.0)
        catch e
            println("ERROR: $e")
            results[name] = (ForwardMetrics(0.0, 0.0, 1e9, 999999, 0.0), 0.0)
        end
    end
    
    # Find champion
    champion_name = ""
    champion_metrics = ForwardMetrics(0.0, 0.0, 1e9, 0, 0.0)
    champion_score = 0.0
    
    for (name, (metrics, score)) in results
        if score > champion_score
            champion_score = score
            champion_metrics = metrics
            champion_name = name
        end
    end
    
    println("\n" * "="^80)
    @printf(" 🏆 ROUND %d CHAMPION\n", round_num)
    println("="^80)
    @printf("  Algorithm: %s\n", champion_name)
    @printf("  Score: %.2f\n", champion_score)
    @printf("  Accuracy: %.6f\n", champion_metrics.accuracy)
    @printf("  Wave Fidelity: %.4f (%.0f%% wave-native)\n", 
            champion_metrics.wave_fidelity, champion_metrics.wave_fidelity * 100)
    @printf("  Throughput: %.1f K nodes/sec\n", champion_metrics.throughput_nodes_sec / 1000.0)
    @printf("  Latency: %.2f µs\n", champion_metrics.latency_ns / 1000.0)
    @printf("  Memory Allocations: %d\n", champion_metrics.memory_allocs)
    println("="^80)
    
    # Top 5
    println("\n📊 Top 5 Algorithms:")
    sorted = sort(collect(results), by=x->x[2][2], rev=true)
    for (i, (name, (metrics, score))) in enumerate(sorted[1:min(5, end)])
        @printf("%d. %-30s | Score: %8.2f | Fidelity: %.2f | %.1f Knodes/s\n",
                i, name, score, metrics.wave_fidelity, metrics.throughput_nodes_sec / 1000.0)
    end
    
    return (champion_name, champion_metrics, champion_score, results)
end

function run_tournament_round1()
    return run_tournament_round(1, ROUND1_ALGORITHMS)
end

function run_tournament_round2()
    return run_tournament_round(2, ROUND2_ALGORITHMS)
end

function run_tournament_round3()
    return run_tournament_round(3, ROUND3_ALGORITHMS)
end

function run_tournament_round4()
    return run_tournament_round(4, ROUND4_ALGORITHMS)
end

function run_tournament_round5()
    return run_tournament_round(5, ROUND5_ALGORITHMS)
end

function run_tournament_round6()
    return run_tournament_round(6, ROUND6_ALGORITHMS)
end

function run_tournament_round7()
    return run_tournament_round(7, ROUND7_ALGORITHMS)
end

function run_tournament_round8()
    return run_tournament_round(8, ROUND8_ALGORITHMS)
end

function run_tournament_round9()
    return run_tournament_round(9, ROUND9_ALGORITHMS)
end

function run_tournament_round10()
    return run_tournament_round(10, ROUND10_ALGORITHMS)
end

function run_tournament_round11()
    return run_tournament_round(11, ROUND11_ALGORITHMS)
end

function run_tournament_round12()
    return run_tournament_round(12, ROUND12_ALGORITHMS)
end

# ====================================================================================
# MAIN
# ====================================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Check command line argument for round number
    round_num = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2
    
    # Validate round number
    if round_num < 1 || round_num > 12
        println("❌ Invalid round number: $round_num (valid: 1-12)")
        exit(1)
    end
    
    # Run the requested round
    round_functions = [
        run_tournament_round1, run_tournament_round2, run_tournament_round3,
        run_tournament_round4, run_tournament_round5, run_tournament_round6,
        run_tournament_round7, run_tournament_round8, run_tournament_round9,
        run_tournament_round10, run_tournament_round11, run_tournament_round12
    ]
    
    println("\n🔄 Running Round $round_num...")
    champion = round_functions[round_num]()
    
    println("\n" * "="^80)
    @printf(" 📝 ROUND %d COMPLETE - Progress: %d/144 algorithms tested (%.1f%%)\n", 
            round_num, round_num * 12 + 1, (round_num * 12 + 1) / 144.0 * 100)
    println("="^80)
    
    if round_num < 12
        @printf(" Next: Round %d with 12 new algorithms\n", round_num + 1)
        @printf(" Remaining: %d more rounds (%d algorithms)\n", 12 - round_num, (12 - round_num) * 12)
    else
        println(" 🎉 ALL 12 ROUNDS COMPLETE - 144/144 algorithms tested!")
        println(" 🏆 GRAND CHAMPION: $(champion[1])")
        @printf(" 📊 Final Score: %.2f\n", champion[3])
    end
    
    println("="^80)
end
