"""
    SovwaveCUDAExt

Julia package extension: loaded automatically when both `Sovwave` and `CUDA`
are imported in the same session. Provides GPU-accelerated forward pass and
training loop via `CUDA.@cuda` kernels.

Activated by:
```julia
using CUDA, Sovwave
Sovwave.enable_cuda!()
```
"""
module SovwaveCUDAExt

using CUDA
using Sovwave
using Sovwave.Audio.WaveML

# Register this extension module with the CUDASupport internal state
function __init__()
    Sovwave.Audio.WaveML._CUDA_EXT[] = @__MODULE__
    @info "Sovwave: CUDA extension loaded ($(CUDA.name(CUDA.device())))"
end

# ── GPU Kernel ───────────────────────────────────────────────────────────────

"""
    _gpu_forward_kernel!(output, amps, phs, freqs, frac_scales, frac_dims, wave_speeds, omega, input_vals, t, n, d, in_len)

CUDA kernel for WaveLayer forward pass. Each GPU thread handles one node.
"""
function _gpu_forward_kernel!(
    output::CuDeviceVector{Float32},
    amps::CuDeviceMatrix{Float32},
    phs::CuDeviceMatrix{Float32},
    freqs::CuDeviceMatrix{Float32},
    frac_scales::CuDeviceVector{Float32},
    frac_dims::CuDeviceVector{Float32},
    wave_speeds::CuDeviceVector{Float32},
    omega::Float32,
    input_vals::CuDeviceVector{Float32},
    t::Float32,
    n::Int32,
    d::Int32,
    in_len::Int32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return

    beta    = frac_scales[i]
    d_f     = frac_dims[i]
    v_spd   = wave_speeds[i]
    frac_env = d_f / 1.5f0   # normalized envelope

    node_sum = 0.0f0
    for j in 1:d
        in_val = j <= in_len ? input_vals[j] : 0.5f0
        amp  = amps[i, j]
        ph   = phs[i, j]
        freq = freqs[i, j]

        x_eff = v_spd == -1.0f0 ? in_val : in_val / max(v_spd, 1f-12)
        angle = omega * 0.001f0 * freq * x_eff + ph - t
        node_sum += amp * CUDA.sin(angle)
    end

    node_wave = (node_sum / CUDA.sqrt(Float32(d))) * beta * frac_env
    output[i] = node_wave
    return
end

# ── GPU Layer Helpers ─────────────────────────────────────────────────────────

"""
    gpu_layer(layer::WaveLayer)::WaveLayer

Returns a new WaveLayer with all parameter arrays moved to GPU (CuArray).
"""
function gpu_layer(layer::WaveLayer)::WaveLayer
    return WaveLayer(
        layer.nodes,
        layer.embed_dim,
        cu(Float32.(layer.amplitudes)),
        cu(Float32.(layer.phases)),
        cu(Float32.(layer.frequencies)),
        cu(Float32.(layer.fractal_scales)),
        layer.omega;
        fractal_dims = cu(Float32.(layer.fractal_dims)),
        wave_speeds  = cu(Float32.(layer.wave_speeds))
    )
end

"""
    cpu_layer(layer::WaveLayer)::WaveLayer

Returns a new WaveLayer with all GPU arrays moved back to CPU (Array).
"""
function cpu_layer(layer::WaveLayer)::WaveLayer
    to_f64(x) = x isa CuArray ? Float64.(Array(x)) : Float64.(x)
    return WaveLayer(
        layer.nodes,
        layer.embed_dim,
        to_f64(layer.amplitudes),
        to_f64(layer.phases),
        to_f64(layer.frequencies),
        to_f64(layer.fractal_scales),
        layer.omega;
        fractal_dims = to_f64(layer.fractal_dims),
        wave_speeds  = to_f64(layer.wave_speeds)
    )
end

# ── GPU Forward Pass ──────────────────────────────────────────────────────────

"""
    gpu_forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}

GPU-accelerated forward pass. Launches one CUDA thread per node.
Falls back to CPU `forward!` if arrays are not on GPU.
"""
function gpu_forward!(layer::WaveLayer, input_values::Vector{Float64}, t::Float64)::Vector{Float64}
    n = Int32(layer.nodes)
    d = Int32(layer.embed_dim)
    in_len = Int32(length(input_values))

    cu_input = cu(Float32.(input_values))
    cu_output = CUDA.zeros(Float32, n)

    # Pull GPU arrays (or wrap CPU arrays in cu() if needed)
    cu_amps   = layer.amplitudes isa CuArray ? layer.amplitudes : cu(Float32.(layer.amplitudes))
    cu_phs    = layer.phases     isa CuArray ? layer.phases     : cu(Float32.(layer.phases))
    cu_freqs  = layer.frequencies isa CuArray ? layer.frequencies : cu(Float32.(layer.frequencies))
    cu_fscale = layer.fractal_scales isa CuArray ? layer.fractal_scales : cu(Float32.(layer.fractal_scales))
    cu_fdims  = layer.fractal_dims  isa CuArray ? layer.fractal_dims  : cu(Float32.(layer.fractal_dims))
    cu_wspeeds= layer.wave_speeds   isa CuArray ? layer.wave_speeds   : cu(Float32.(layer.wave_speeds))

    threads = min(256, n)
    blocks  = cld(n, threads)

    CUDA.@cuda threads=threads blocks=blocks _gpu_forward_kernel!(
        cu_output, cu_amps, cu_phs, cu_freqs, cu_fscale, cu_fdims, cu_wspeeds,
        Float32(layer.omega), cu_input, Float32(t), n, d, in_len
    )
    CUDA.synchronize()

    result = Float64.(Array(cu_output))
    layer.layer_energy = 0.5 * sum(x -> x^2, result)
    return result
end

end # module SovwaveCUDAExt
