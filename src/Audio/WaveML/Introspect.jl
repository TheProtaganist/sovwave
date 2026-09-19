"""
    WaveML.Introspect

Model Reading and Introspection API for Sovwave models.
Provides comprehensive inspection into wave architectures, quantum node lattices,
learned physical wave values, layer parameter counts, energy spectra, and
direct video (.mkv / .mp4) inspections.
"""

using Printf
using Statistics
using LinearAlgebra

export num_layers, parameter_count, get_layer, layer_details, model_summary, inspect_model

"""
    num_layers(model::WaveModel)::Int

Returns the total number of wave layers in the model.
"""
function num_layers(model::WaveModel)::Int
    return length(model.layers)
end

"""
    parameter_count(layer::WaveLayer)::Int

Calculates the number of learned continuous wave parameters in a single `WaveLayer`.
Parameters include amplitudes, phases, frequencies, fractal scales, and harmonic frequency.
"""
function parameter_count(layer::WaveLayer)::Int
    # Amplitudes: nodes × embed_dim
    # Phases: nodes × embed_dim
    # Frequencies: nodes × embed_dim
    # Fractal scales: nodes
    # Omega: 1
    return 3 * (layer.nodes * layer.embed_dim) + layer.nodes + 1
end

"""
    parameter_count(model::WaveModel)::Int

Calculates the total number of learned continuous wave parameters across all layers.
"""
function parameter_count(model::WaveModel)::Int
    total = 0
    for layer in model.layers
        total += parameter_count(layer)
    end
    return total
end

"""
    get_layer(model::WaveModel, idx::Int)::WaveLayer

Retrieves the `WaveLayer` at 1-based index `idx`.
"""
function get_layer(model::WaveModel, idx::Int)::WaveLayer
    if idx < 1 || idx > length(model.layers)
        throw(BoundsError(model.layers, idx))
    end
    return model.layers[idx]
end

"""
    layer_details(layer::WaveLayer, idx::Int = 1)::Dict{Symbol, Any}

Extracts a detailed statistical breakdown of learned parameters and physical values for a wave layer.
"""
function layer_details(layer::WaveLayer, idx::Int = 1)::Dict{Symbol, Any}
    amp_min, amp_max = extrema(layer.amplitudes)
    amp_mean = mean(layer.amplitudes)
    amp_std = std(layer.amplitudes)

    ph_min, ph_max = extrema(layer.phases)
    ph_mean = mean(layer.phases)

    freq_min, freq_max = extrema(layer.frequencies)
    freq_mean = mean(layer.frequencies)

    beta_min, beta_max = extrema(layer.fractal_scales)
    beta_mean = mean(layer.fractal_scales)

    return Dict{Symbol, Any}(
        :layer_index => idx,
        :nodes => layer.nodes,
        :embed_dim => layer.embed_dim,
        :omega => layer.omega,
        :layer_energy => layer.layer_energy,
        :parameter_count => parameter_count(layer),
        :amplitudes => Dict(
            :min => amp_min, :max => amp_max, :mean => amp_mean, :std => amp_std,
            :raw => layer.amplitudes
        ),
        :phases => Dict(
            :min => ph_min, :max => ph_max, :mean => ph_mean,
            :raw => layer.phases
        ),
        :frequencies => Dict(
            :min => freq_min, :max => freq_max, :mean => freq_mean,
            :raw => layer.frequencies
        ),
        :fractal_scales => Dict(
            :min => beta_min, :max => beta_max, :mean => beta_mean,
            :raw => layer.fractal_scales
        )
    )
end

"""
    layer_details(model::WaveModel, idx::Int)::Dict{Symbol, Any}

Extracts detailed parameter values and statistics for layer `idx` of `model`.
"""
function layer_details(model::WaveModel, idx::Int)::Dict{Symbol, Any}
    layer = get_layer(model, idx)
    return layer_details(layer, idx)
end

"""
    model_summary(model::WaveModel; io::IO = stdout)::String

Generates and displays a structured tabular summary of the wave model architecture,
quantum node lattices, parameter distributions, and ground-state energies.
"""
function model_summary(model::WaveModel; io::IO = stdout)::String
    buf = IOBuffer()
    n_lay = num_layers(model)
    tot_params = parameter_count(model)

    println(buf, "="^78)
    println(buf, "                   SOVWAVE NEURAL MODEL SUMMARY                   ")
    println(buf, "="^78)
    @printf(buf, "Generation:           %d\n", model.generation)
    @printf(buf, "Total Layers:         %d\n", n_lay)
    @printf(buf, "Total Parameters:     %d (continuous wave values)\n", tot_params)
    @printf(buf, "Total Ground Energy:  %.6e\n", model.total_energy)
    @printf(buf, "Driving Omega (ω):    %.2f Hz\n", model.model_config.omega)
    @printf(buf, "Temporal Frames:      %d frames\n", model.model_config.t_frames)
    println(buf, "-"^78)
    @printf(buf, "%-6s | %-7s | %-9s | %-12s | %-10s | %-18s\n",
            "Layer", "Nodes", "Embed Dim", "Parameters", "Energy", "Amp Range [Min, Max]")
    println(buf, "-"^78)

    for i in 1:n_lay
        l = model.layers[i]
        p_cnt = parameter_count(l)
        amin, amax = extrema(l.amplitudes)
        @printf(buf, "#%-5d | %-7d | %-9d | %-12d | %-10.4e | [%.3f, %.3f]\n",
                i, l.nodes, l.embed_dim, p_cnt, l.layer_energy, amin, amax)
    end
    println(buf, "="^78)

    summary_str = String(take!(buf))
    print(io, summary_str)
    return summary_str
end

"""
    inspect_model(model::WaveModel; io::IO = stdout)::Dict{Symbol, Any}

Inspects an in-memory `WaveModel`, printing its summary and returning a structured
dictionary of all learned parts, layer values, and architectural configurations.
"""
function inspect_model(model::WaveModel; io::IO = stdout)::Dict{Symbol, Any}
    summary_str = model_summary(model; io=io)
    layers_data = [layer_details(model, i) for i in 1:num_layers(model)]

    return Dict{Symbol, Any}(
        :generation => model.generation,
        :num_layers => num_layers(model),
        :total_parameters => parameter_count(model),
        :total_energy => model.total_energy,
        :omega => model.model_config.omega,
        :t_frames => model.model_config.t_frames,
        :layers => layers_data,
        :summary => summary_str
    )
end

"""
    inspect_model(path::String; io::IO = stdout)::Dict{Symbol, Any}

Inspects a serialized `.mkv` or `.mp4` Sovwave model file directly from disk.
Reconstructs the model and returns all architectural parameters, layer details, and video metadata.
"""
function inspect_model(path::String; io::IO = stdout)::Dict{Symbol, Any}
    isfile(path) || error("File not found: $path")
    model = load_model(path)
    res = inspect_model(model; io=io)
    res[:file_path] = abspath(path)
    res[:file_size_bytes] = filesize(path)
    return res
end
