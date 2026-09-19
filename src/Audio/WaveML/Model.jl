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

    function WaveModel(layers::Vector{WaveLayer}, f_cfg::WaveFieldConfig, m_cfg::WaveModelConfig)
        new(layers, f_cfg, m_cfg, 0.0, 0)
    end
end

"""
    WaveModel(cfg::WaveMLConfig)::WaveModel

Constructs a `WaveModel` initialized according to the provided `WaveMLConfig`.
"""
function WaveModel(cfg::WaveMLConfig)::WaveModel
    l = cfg.model.layers
    nodes = cfg.model.nodes
    embed_dim = cfg.model.embed_dims
    omega = cfg.model.omega
    beta_s = cfg.model.beta_s

    layers = Vector{WaveLayer}(undef, l)
    for i in 1:l
        layers[i] = create_layer(nodes, embed_dim; omega=omega, beta_s=beta_s)
    end

    return WaveModel(layers, cfg.field, cfg.model)
end

"""
    forward!(model::WaveModel, input_data::Vector{Float64}; t::Float64 = 0.0)::Vector{Float64}

Executes an end-to-end forward wave pass across all l layers with temporal superposition
driven by harmonic frequency ω across t_frames.
"""
function forward!(model::WaveModel, input_data::Vector{Float64}; t::Float64 = 0.0)::Vector{Float64}
    current = input_data
    t_frames = model.model_config.t_frames
    omega = model.model_config.omega

    # Propagate sequentially through all layers
    for layer in model.layers
        # Temporal superposition across t_frames
        accum = zeros(Float64, layer.nodes)
        for frame in 1:t_frames
            t_offset = t + 2π * (frame - 1) / (omega + 1e-12)
            accum .+= forward!(layer, current, t_offset)
        end
        current = accum ./ sqrt(Float64(t_frames))
    end

    # Sum total energy across all layers
    tot_e = 0.0
    for layer in model.layers
        tot_e += layer_energy(layer)
    end
    model.total_energy = tot_e

    return current
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

Mutates all layers within the model.
"""
function mutate!(model::WaveModel, rate::Float64)::Nothing
    for layer in model.layers
        mutate!(layer, rate)
    end
    return nothing
end

"""
    crossover(model_a::WaveModel, model_b::WaveModel)::WaveModel

Recombines two models layer-by-layer to produce an offspring model.
"""
function crossover(model_a::WaveModel, model_b::WaveModel)::WaveModel
    num_layers = length(model_a.layers)
    child_layers = Vector{WaveLayer}(undef, num_layers)
    for i in 1:num_layers
        child_layers[i] = crossover(model_a.layers[i], model_b.layers[i])
    end
    child = WaveModel(child_layers, model_a.field_config, model_a.model_config)
    child.generation = max(model_a.generation, model_b.generation) + 1
    return child
end

"""
    clone(model::WaveModel)::WaveModel

Creates an exact deep copy of the wave model.
"""
function clone(model::WaveModel)::WaveModel
    cloned_layers = [deepcopy(l) for l in model.layers]
    cloned = WaveModel(cloned_layers, model.field_config, model.model_config)
    cloned.total_energy = model.total_energy
    cloned.generation = model.generation
    return cloned
end
