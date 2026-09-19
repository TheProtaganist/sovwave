"""
    WaveML.CUDASupport

Optional CUDA GPU acceleration API for Sovwave.

This module exposes the public API (`enable_cuda!`, `disable_cuda!`, `cuda_available`,
`to_gpu`, `to_cpu`). The actual GPU kernels live in `ext/SovwaveCUDAExt.jl` and are
only loaded when CUDA.jl is installed and imported by the user.

# Usage
```julia
using CUDA, Sovwave          # CUDA.jl must be installed first
enable_cuda!()               # moves all subsequent new layers to GPU
model = WaveModel(cfg)       # arrays live on GPU
train!(model, ds...)         # GPU-accelerated training loop
disable_cuda!()              # move back to CPU
```

When CUDA.jl is NOT installed, `enable_cuda!()` prints a friendly message and returns false.
"""

export enable_cuda!, disable_cuda!, cuda_available, to_gpu, to_cpu

# ── Internal state ──────────────────────────────────────────────────────────
const _CUDA_ENABLED = Ref{Bool}(false)
const _CUDA_EXT     = Ref{Union{Nothing, Module}}(nothing)

"""
    cuda_available()::Bool

Returns `true` if CUDA.jl is loaded and a GPU device is accessible.
"""
function cuda_available()::Bool
    return _CUDA_ENABLED[]
end

"""
    enable_cuda!()::Bool

Enables GPU acceleration. Requires CUDA.jl to be installed and imported
in the user's session (`using CUDA`).

Returns `true` on success, `false` if CUDA is not available.
"""
function enable_cuda!()::Bool
    if _CUDA_EXT[] !== nothing
        _CUDA_ENABLED[] = true
        @info "Sovwave CUDA: GPU acceleration enabled ✅"
        return true
    end
    # Try to detect if CUDA extension was loaded via Julia package extension
    if isdefined(Main, :CUDA)
        _CUDA_ENABLED[] = true
        @info "Sovwave CUDA: GPU acceleration enabled ✅"
        return true
    end
    @warn """
    Sovwave CUDA: CUDA.jl not found. To enable GPU acceleration:
      1. Install CUDA.jl:  using Pkg; Pkg.add("CUDA")
      2. In your script:   using CUDA, Sovwave
      3. Then call:        enable_cuda!()
    """
    return false
end

"""
    disable_cuda!()::Nothing

Disables GPU acceleration and moves computation back to CPU.
"""
function disable_cuda!()::Nothing
    _CUDA_ENABLED[] = false
    @info "Sovwave CUDA: GPU acceleration disabled — using CPU"
    return nothing
end

"""
    to_gpu(layer::WaveLayer)::WaveLayer

Moves a WaveLayer's parameter arrays to the GPU (when CUDA is enabled).
Returns the layer unchanged if CUDA is not enabled.
"""
function to_gpu(layer::WaveLayer)::WaveLayer
    if _CUDA_ENABLED[] && _CUDA_EXT[] !== nothing
        return _CUDA_EXT[].gpu_layer(layer)
    elseif _CUDA_ENABLED[]
        @warn "Sovwave: to_gpu called but CUDA extension not loaded. Use `using CUDA, Sovwave`."
    end
    return layer
end

"""
    to_cpu(layer::WaveLayer)::WaveLayer

Moves a WaveLayer's parameter arrays back to the CPU.
Returns the layer unchanged if already on CPU.
"""
function to_cpu(layer::WaveLayer)::WaveLayer
    if _CUDA_EXT[] !== nothing
        return _CUDA_EXT[].cpu_layer(layer)
    end
    return layer
end

"""
    to_gpu(model::WaveModel)::WaveModel

Moves all layers of a WaveModel to the GPU.
"""
function to_gpu(model::WaveModel)::WaveModel
    if _CUDA_ENABLED[]
        for i in eachindex(model.layers)
            model.layers[i] = to_gpu(model.layers[i])
        end
    end
    return model
end

"""
    to_cpu(model::WaveModel)::WaveModel

Moves all layers of a WaveModel back to the CPU.
"""
function to_cpu(model::WaveModel)::WaveModel
    for i in eachindex(model.layers)
        model.layers[i] = to_cpu(model.layers[i])
    end
    return model
end
