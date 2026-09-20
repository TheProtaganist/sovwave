"""
    WaveML.Model

Full WaveModel representation: a stack of l wave layers operating over
n quantum lattice nodes and d embedding dimensions.
"""

export WaveModel
export model_energy, clone, forward!, mutate!, crossover

"""
    WaveModel

A complete wave-based neural model.
- `layers::Vector{WaveLayer}`: l layers
- `field_config::WaveFieldConfig`: Specification of input data point manifolds
- `model_config::WaveModelConfig`: Specification of model lattice parameters
- `total_energy::Float64`: Total ground-state energy across all layers
- `generation::Int`: Evolution generation of this model
"""
mutable struct WaveModel
    layers::Vector{WaveLayer}
    field_config::WaveFieldConfig
    model_config::WaveModelConfig
    total_energy::Float64
    generation::Int
    _buf_a::Vector{Float64}
    _buf_b::Vector{Float64}
    _accum::Vector{Float64}
    _frame_buf::Vector{Float64}

    function WaveModel(layers::Vector{WaveLayer}, f_cfg::WaveFieldConfig, m_cfg::WaveModelConfig)
        max_nodes = isempty(layers) ? m_cfg.nodes : maximum(l.nodes for l in layers)
        max_dim = max(max_nodes, m_cfg.embed_dims, f_cfg.dimensions)
        new(
            layers, f_cfg, m_cfg, 0.0, 0,
            zeros(Float64, max_dim),
            zeros(Float64, max_dim),
            zeros(Float64, max_dim),
            zeros(Float64, max_dim)
        )
    end
end

@inline function _ensure_buffers!(model::WaveModel, needed::Int)
    if length(model._buf_a) < needed
        resize!(model._buf_a, needed)
        resize!(model._buf_b, needed)
        resize!(model._accum, needed)
        resize!(model._frame_buf, needed)
    end
end

"""
    WaveModel(cfg::WaveMLConfig)::WaveModel

Constructs a `WaveModel` initialized according to the provided `WaveMLConfig` using list comprehensions.
"""
function WaveModel(cfg::WaveMLConfig)::WaveModel
    l = cfg.model.layers
    nodes = cfg.model.nodes
    embed_dim = cfg.model.embed_dims
    omega = cfg.model.omega
    beta_s = cfg.model.beta_s

    layers = [create_layer(nodes, embed_dim; omega=omega, beta_s=beta_s) for _ in 1:l]
    return WaveModel(layers, cfg.field, cfg.model)
end

"""
    forward!(model::WaveModel, input_data::AbstractVector{Float64}, output::AbstractVector{Float64}; t::Float64 = 0.0)::AbstractVector{Float64}

Executes an end-to-end forward wave pass across all l layers with temporal superposition
in-place into `output` with zero heap allocations.
"""
function forward!(
    model::WaveModel,
    input_data::AbstractVector{Float64},
    output::AbstractVector{Float64};
    t::Float64 = 0.0,
    mode::Symbol = :sound_native,
    sonify::Bool = false,
    sound_buf = nothing
)::AbstractVector{Float64}
    if mode == :sound_native
        s_cfg = sonify ? DEFAULT_AUDIBLE_SOUND_CFG : DEFAULT_SILENT_SOUND_CFG
        return sound_native_model_forward!(model, input_data, output; t=t, cfg=s_cfg, buf=sound_buf)
    end

    in_len = length(input_data)
    max_nodes = isempty(model.layers) ? 0 : maximum(l.nodes for l in model.layers)
    needed = max(in_len, max_nodes, model.model_config.embed_dims)
    _ensure_buffers!(model, needed)

    t_frames = model.model_config.t_frames
    omega = model.model_config.omega
    inv_sqrt_tf = 1.0 / sqrt(Float64(t_frames))
    inv_omega = 1.0 / (omega + 1e-12)

    # In-place copy input into buf_a
    @inbounds for i in 1:in_len
        model._buf_a[i] = input_data[i]
    end

    current_len = in_len
    use_a_as_input = true
    tot_e = 0.0

    @inbounds for layer in model.layers
        n = layer.nodes
        in_buf = use_a_as_input ? view(model._buf_a, 1:current_len) : view(model._buf_b, 1:current_len)
        out_buf = use_a_as_input ? view(model._buf_b, 1:n) : view(model._buf_a, 1:n)
        accum = view(model._accum, 1:n)
        frame_buf = view(model._frame_buf, 1:n)

        fill!(accum, 0.0)

        for frame in 1:t_frames
            t_offset = muladd(2π * (frame - 1), inv_omega, t)
            forward!(layer, in_buf, frame_buf, t_offset; mode=mode, sonify=sonify, sound_buf=sound_buf)
            @simd for k in 1:n
                accum[k] += frame_buf[k]
            end
        end

        @simd for k in 1:n
            out_buf[k] = accum[k] * inv_sqrt_tf
        end

        tot_e += layer_energy(layer)
        current_len = n
        use_a_as_input = !use_a_as_input
    end

    model.total_energy = tot_e

    # Copy to destination output
    final_buf = use_a_as_input ? view(model._buf_a, 1:current_len) : view(model._buf_b, 1:current_len)
    copyto!(output, 1, final_buf, 1, min(length(output), current_len))
    return output
end

"""
    forward!(model::WaveModel, input_data::AbstractVector{Float64}; t::Float64 = 0.0, kwargs...)::Vector{Float64}

Executes an end-to-end forward wave pass across all l layers with temporal superposition.
Allocates only a single output vector.
"""
function forward!(model::WaveModel, input_data::AbstractVector{Float64}; t::Float64 = 0.0, kwargs...)::Vector{Float64}
    out_dim = isempty(model.layers) ? length(input_data) : model.layers[end].nodes
    out = Vector{Float64}(undef, out_dim)
    forward!(model, input_data, out; t=t, kwargs...)
    return out
end

"""
    model_energy(model::WaveModel)::Float64

Returns the current total energy of the model.
"""
function model_energy(model::WaveModel)::Float64
    return model.total_energy
end

"""
    mutate!(model::WaveModel, rate::Float64)::Nothing

Mutates all layers within the model in-place.
"""
function mutate!(model::WaveModel, rate::Float64)::Nothing
    for layer in model.layers
        mutate!(layer, rate)
    end
    return nothing
end

"""
    crossover(model_a::WaveModel, model_b::WaveModel)::WaveModel

Recombines two models layer-by-layer using list comprehensions.
"""
function crossover(model_a::WaveModel, model_b::WaveModel)::WaveModel
    child_layers = [crossover(la, lb) for (la, lb) in zip(model_a.layers, model_b.layers)]
    child = WaveModel(child_layers, model_a.field_config, model_a.model_config)
    child.generation = max(model_a.generation, model_b.generation) + 1
    return child
end

"""
    clone(model::WaveModel)::WaveModel

Creates an exact deep copy of the wave model using list comprehensions.
"""
function clone(model::WaveModel)::WaveModel
    cloned_layers = [deepcopy(l) for l in model.layers]
    cloned = WaveModel(cloned_layers, model.field_config, model.model_config)
    cloned.total_energy = model.total_energy
    cloned.generation = model.generation
    return cloned
end
