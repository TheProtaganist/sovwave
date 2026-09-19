"""
    WaveML.Inference

Inference and Output Generation from MKV Wave Models.
Enables running predictions directly from MKV model video files or in-memory models.
"""

export infer, predict

"""
    infer(
        mkv_model_path::String,
        inputs::Vector{Vector{Float64}};
        meta_path::Union{Nothing, String} = nothing
    )::Vector{Vector{Float64}}

Runs inference on inputs by directly loading the wave model from an MKV video file.
Decodes the visual wave layers, reconstructs the quantum lattice, and propagates
the input data points through the field.
"""
function infer(
    mkv_model_path::String,
    inputs::Vector{Vector{Float64}};
    meta_path::Union{Nothing, String} = nothing
)::Vector{Vector{Float64}}
    model = load_model(mkv_model_path; meta_path=meta_path)
    return predict(model, inputs)
end

"""
    infer(
        mkv_model_path::String,
        single_input::Vector{Float64};
        meta_path::Union{Nothing, String} = nothing
    )::Vector{Float64}

Overload for a single input vector.
"""
function infer(
    mkv_model_path::String,
    single_input::Vector{Float64};
    meta_path::Union{Nothing, String} = nothing
)::Vector{Float64}
    res = infer(mkv_model_path, [single_input]; meta_path=meta_path)
    return res[1]
end

"""
    predict(model::WaveModel, inputs::Vector{Vector{Float64}})::Vector{Vector{Float64}}

Generates output predictions for a batch of input data vectors using in-memory model.
"""
function predict(model::WaveModel, inputs::Vector{Vector{Float64}})::Vector{Vector{Float64}}
    n_samples = length(inputs)
    outputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        outputs[i] = forward!(model, inputs[i])
    end

    return outputs
end

"""
    predict(model::WaveModel, single_input::Vector{Float64})::Vector{Float64}

Generates output predictions for a single input data vector.
"""
function predict(model::WaveModel, single_input::Vector{Float64})::Vector{Float64}
    return forward!(model, single_input)
end
