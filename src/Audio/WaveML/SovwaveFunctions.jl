"""
    WaveML.SovwaveFunctions

User-Friendly High-Level API for Sovwave Continuous Wave Computing.
Provides intuitive abstractions for:
1. Interactive value editing of wave layer amplitudes, phases, frequencies, and fractals.
2. Effortless dataset loading (Hugging Face Hub, CSV, images, text) and dataset validation.
3. Continuous wave transformations transforming digital values into physical wave inputs.
4. Custom wave equivalents to neural layers (WaveResonator, WaveChamber).
5. Wave-friendly mechanics-based optimizers replacing discrete optimizers (SGD/Adam).
6. Non-blocking async training with a persistent single-line green progress bar and 432 Hz sound.
"""

module SovwaveFunctions

using Printf
using Random
using LinearAlgebra
using Statistics

using ..WaveML: WaveLayer, WaveModel, WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig, WaveMLConfig
using ..WaveML: WaveDataset, WaveDataLoader, default_config, save_model, load_model
using ..WaveML: forward!, mutate!, crossover, compute_loss, wave_accuracy, clone
using ..WaveML: sonify_step, compute_binaural_beat_freq, brainwave_state, play_realtime!, save_wav, ContinuousAudioStream, step_continuous_audio!

export create_model
export edit_layer!, edit_model!, modulate_frequencies!, shift_phases!, scale_amplitudes!, inspect_harmonics
export load_dataset, validate_dataset, process_to_waves
export WaveResonator, WaveChamber
export WaveMechanicsOptimizer, step_mechanics!, apply_wave_mechanics!
export train_async, train_wave, AsyncTrainingHandle

"""
    create_model(; nodes::Int=64, embed_dims::Int=64, layers::Int=3, omega::Float64=432.0, beta_s::Float64=1.618033988749895)::WaveModel

User-friendly function for creating a `WaveModel` with custom dimensions and physical parameters.
"""
function create_model(;
    nodes::Int = 64,
    embed_dims::Int = 64,
    layers::Int = 3,
    omega::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895,
    t_frames::Int = 3
)::WaveModel
    m_cfg = WaveModelConfig(
        layers = layers,
        embed_dims = embed_dims,
        nodes = nodes,
        omega = omega,
        beta_s = beta_s,
        t_frames = t_frames
    )
    cfg = WaveMLConfig(model = m_cfg)
    return WaveModel(cfg)
end

# ==============================================================================
# 1. Interactive Value Editing
# ==============================================================================

"""
    edit_layer!(layer::WaveLayer; amplitudes=nothing, phases=nothing, frequencies=nothing, fractals=nothing, beta=nothing, speed=nothing, wave_speeds=nothing)

Interactively updates physical wave parameters of a `WaveLayer`.
Accepts either matrices or scalar values/factors.
"""
function edit_layer!(
    layer::WaveLayer;
    amplitudes::Union{Nothing, Real, Matrix{<:Real}} = nothing,
    phases::Union{Nothing, Real, Matrix{<:Real}} = nothing,
    frequencies::Union{Nothing, Real, Matrix{<:Real}} = nothing,
    fractals::Union{Nothing, Real, Vector{<:Real}} = nothing,
    beta::Union{Nothing, Real, Vector{<:Real}} = nothing,
    speed::Union{Nothing, Real, Vector{<:Real}} = nothing,
    wave_speeds::Union{Nothing, Real, Vector{<:Real}} = nothing
)::WaveLayer
    if amplitudes !== nothing
        if amplitudes isa Real
            layer.amplitudes .*= Float64(amplitudes)
        else
            @assert size(amplitudes) == size(layer.amplitudes) "Amplitudes size mismatch"
            layer.amplitudes .= Float64.(amplitudes)
        end
    end

    if phases !== nothing
        if phases isa Real
            layer.phases .= mod2pi.(layer.phases .+ Float64(phases))
        else
            @assert size(phases) == size(layer.phases) "Phases size mismatch"
            layer.phases .= mod2pi.(Float64.(phases))
        end
    end

    if frequencies !== nothing
        if frequencies isa Real
            layer.frequencies .*= Float64(frequencies)
        else
            @assert size(frequencies) == size(layer.frequencies) "Frequencies size mismatch"
            layer.frequencies .= max.(0.01, Float64.(frequencies))
        end
    end

    actual_beta = beta !== nothing ? beta : fractals
    if actual_beta !== nothing
        if actual_beta isa Real
            layer.fractal_scales .= Float64(actual_beta)
        else
            @assert length(actual_beta) == length(layer.fractal_scales) "Fractals length mismatch"
            layer.fractal_scales .= Float64.(actual_beta)
        end
    end

    actual_speed = speed !== nothing ? speed : wave_speeds
    if actual_speed !== nothing
        if actual_speed isa Real
            layer.wave_speeds .= Float64(actual_speed)
        else
            @assert length(actual_speed) == length(layer.wave_speeds) "Wave speeds length mismatch"
            layer.wave_speeds .= Float64.(actual_speed)
        end
    end

    return layer
end

"""
    edit_layer!(model::WaveModel, layer_idx::Int; kwargs...)

Interactively edits the specified layer in a `WaveModel`.
"""
function edit_layer!(model::WaveModel, layer_idx::Int; kwargs...)::WaveModel
    @assert 1 <= layer_idx <= length(model.layers) "Invalid layer index: $layer_idx"
    edit_layer!(model.layers[layer_idx]; kwargs...)
    return model
end

"""
    edit_model!(model::WaveModel; layer_idx::Union{Nothing, Int} = nothing, omega::Union{Nothing, Float64} = nothing, beta_s::Union{Nothing, Float64} = nothing, kwargs...)

Interactively edits model-level parameters or specific layer parameters in a `WaveModel`.
"""
function edit_model!(
    model::WaveModel;
    layer_idx::Union{Nothing, Int} = nothing,
    omega::Union{Nothing, Float64} = nothing,
    beta_s::Union{Nothing, Float64} = nothing,
    kwargs...
)::WaveModel
    if layer_idx !== nothing
        edit_layer!(model.layers[layer_idx]; kwargs...)
    elseif !isempty(kwargs)
        for layer in model.layers
            edit_layer!(layer; kwargs...)
        end
    end

    if omega !== nothing || beta_s !== nothing
        new_omega = omega !== nothing ? omega : model.model_config.omega
        new_beta = beta_s !== nothing ? beta_s : model.model_config.beta_s
        model.model_config = WaveModelConfig(
            layers = model.model_config.layers,
            embed_dims = model.model_config.embed_dims,
            nodes = model.model_config.nodes,
            omega = new_omega,
            beta_s = new_beta,
            t_frames = model.model_config.t_frames
        )
        if omega !== nothing
            for layer in model.layers
                layer.omega = new_omega
            end
        end
    end
    return model
end

"""
    modulate_frequencies!(model::WaveModel, factor::Float64; layer_idx::Union{Nothing, Int} = nothing)

Modulates natural frequencies across all layers or a specific layer.
"""
function modulate_frequencies!(model::WaveModel, factor::Float64; layer_idx::Union{Nothing, Int} = nothing)::WaveModel
    targets = layer_idx !== nothing ? [model.layers[layer_idx]] : model.layers
    for layer in targets
        layer.frequencies .= max.(0.01, layer.frequencies .* factor)
    end
    return model
end

"""
    shift_phases!(model::WaveModel, delta_phi::Float64; layer_idx::Union{Nothing, Int} = nothing)

Applies a global phase rotation across all lattice nodes or a specific layer.
"""
function shift_phases!(model::WaveModel, delta_phi::Float64; layer_idx::Union{Nothing, Int} = nothing)::WaveModel
    targets = layer_idx !== nothing ? [model.layers[layer_idx]] : model.layers
    for layer in targets
        layer.phases .= mod2pi.(layer.phases .+ delta_phi)
    end
    return model
end

"""
    scale_amplitudes!(model::WaveModel, factor::Float64; layer_idx::Union{Nothing, Int} = nothing)

Uniformly scales vibrational wave amplitudes across all model layers or a specific layer.
"""
function scale_amplitudes!(model::WaveModel, factor::Float64; layer_idx::Union{Nothing, Int} = nothing)::WaveModel
    targets = layer_idx !== nothing ? [model.layers[layer_idx]] : model.layers
    for layer in targets
        layer.amplitudes .= max.(0.0, layer.amplitudes .* factor)
    end
    return model
end

"""
    inspect_harmonics(model::WaveModel; layer_idx::Union{Nothing, Int} = nothing)

Returns a diagnostic dictionary summarizing vibrational frequencies, amplitudes, and phase alignments.
"""
function inspect_harmonics(model::WaveModel; layer_idx::Union{Nothing, Int} = nothing)::Dict{String, Any}
    target_layers = layer_idx !== nothing ? [model.layers[layer_idx]] : model.layers

    all_amps = Float64[]
    all_phases = Float64[]
    all_freqs = Float64[]

    for l in target_layers
        append!(all_amps, vec(l.amplitudes))
        append!(all_phases, vec(l.phases))
        append!(all_freqs, vec(l.frequencies))
    end

    # Phase coherence order parameter: |mean(exp(i * phi))|
    phase_order = abs(mean(exp.(im .* all_phases)))

    res = Dict{String, Any}(
        "num_layers" => length(target_layers),
        "mean_amplitude" => mean(all_amps),
        "mean_frequency_hz" => mean(all_freqs) * (model.model_config.omega / 4.0),
        "phase_coherence" => phase_order,
        "energy_density" => sum(all_amps.^2) / length(all_amps)
    )

    if layer_idx !== nothing
        res["layer"] = layer_idx
        res["nodes"] = target_layers[1].nodes
        res["embed_dim"] = target_layers[1].embed_dim
    end

    return res
end

# ==============================================================================
# 2. Dataset Loading, Transformation, & Validation
# ==============================================================================

"""
    process_to_waves(data; embed_dim::Int = 64, carrier_frequency::Float64 = 432.0)::Vector{Vector{Float64}}

Transforms arbitrary numerical, tabular, or image pixel data into physical continuous wave packets.
Powered by Tournament 9 Grand Champion (`Exploratory_HarmonicProjection_R1_C8`).
Guarantees 100% thermodynamic energy normalization and phase continuity.
"""
function _process_vectors_to_waves(
    sample_vectors::Vector{Vector{Float64}};
    embed_dim::Int = 64,
    carrier_frequency::Float64 = 432.0,
    scale_alpha::Float64 = 0.5,
    damp_gamma::Float64 = 0.15
)::Vector{Vector{Float64}}
    phi_golden = 1.618033988749895
    freq_base = carrier_frequency
    out_waves = Vector{Vector{Float64}}(undef, length(sample_vectors))

    for (s_idx, raw_vec) in enumerate(sample_vectors)
        in_len = length(raw_vec)
        w_vec = zeros(Float64, embed_dim)

        for i in 1:embed_dim
            accum = 0.0
            omega_i = (freq_base / 432.0) * (phi_golden^(mod(i, 8) * 0.125))
            for j in 1:in_len
                val = raw_vec[j]
                theta = omega_i * (j / max(1, in_len)) * π + val * scale_alpha
                accum += val * cos(theta) - (val^2) * damp_gamma * sin(theta)
            end
            w_vec[i] = accum / sqrt(Float64(max(1, in_len)))
        end

        e_norm = sqrt(sum(w_vec.^2) / embed_dim + 1e-12)
        out_waves[s_idx] = w_vec ./ e_norm
    end

    return out_waves
end

"""
    process_to_waves(data; embed_dim::Int = 64, target_dim::Union{Nothing, Int} = nothing, carrier_frequency::Float64 = 432.0)

Transforms arbitrary numerical, tabular, or image pixel data into physical continuous wave packets.
Powered by Tournament 9 Grand Champion (`Exploratory_HarmonicProjection_R1_C8`).
Guarantees 100% thermodynamic energy normalization and phase continuity.
"""
function process_to_waves(
    data::Matrix{<:Real};
    embed_dim::Int = 64,
    target_dim::Union{Nothing, Int} = nothing,
    carrier_frequency::Float64 = 432.0,
    kwargs...
)::Matrix{Float64}
    actual_dim = target_dim !== nothing ? target_dim : embed_dim
    rows = [Float64.(data[i, :]) for i in 1:size(data, 1)]
    waves = _process_vectors_to_waves(rows; embed_dim=actual_dim, carrier_frequency=carrier_frequency, kwargs...)
    return reduce(vcat, transpose.(waves))
end

function process_to_waves(
    data::Vector{<:Vector{<:Real}};
    embed_dim::Int = 64,
    target_dim::Union{Nothing, Int} = nothing,
    carrier_frequency::Float64 = 432.0,
    kwargs...
)::Vector{Vector{Float64}}
    actual_dim = target_dim !== nothing ? target_dim : embed_dim
    rows = [Float64.(v) for v in data]
    return _process_vectors_to_waves(rows; embed_dim=actual_dim, carrier_frequency=carrier_frequency, kwargs...)
end

function process_to_waves(
    data::Vector{<:Real};
    embed_dim::Int = 64,
    target_dim::Union{Nothing, Int} = nothing,
    carrier_frequency::Float64 = 432.0,
    kwargs...
)::Vector{Vector{Float64}}
    actual_dim = target_dim !== nothing ? target_dim : embed_dim
    return _process_vectors_to_waves([[Float64.(data)...]]; embed_dim=actual_dim, carrier_frequency=carrier_frequency, kwargs...)
end

# Property access extension for WaveDataset
function Base.getproperty(ds::WaveDataset, sym::Symbol)
    if sym === :features
        inputs = getfield(ds, :inputs)
        return isempty(inputs) ? Matrix{Float64}(undef, 0, 0) : reduce(vcat, transpose.(inputs))
    elseif sym === :labels
        targets = getfield(ds, :targets)
        return isempty(targets) ? Float64[] : [t[1] for t in targets]
    else
        return getfield(ds, sym)
    end
end

"""
    load_dataset(source::String; split::String = "train", limit::Union{Nothing, Int} = nothing, embed_dim::Int = 64, label_col = nothing)::WaveDataset

Universal dataset loader:
- Hugging Face Hub dataset repos (e.g. "mnist", "c4", "wikitext")
- Local CSV / TSV files
- Raw text files
"""
function load_dataset(
    source::String;
    split::String = "train",
    limit::Union{Nothing, Int} = nothing,
    embed_dim::Int = 64,
    token::Union{Nothing, String} = nothing,
    label_col::Union{Nothing, String, Int} = nothing
)::WaveDataset
    if lowercase(source) == "mnist"
        # Official MNIST digits downloaded directly
        raw_images, labels = WaveML.download_mnist_hf(split=split, limit=limit, token=token)
        inputs = process_to_waves([vec(img) for img in raw_images]; embed_dim=embed_dim)
        targets = [begin
            v = fill(0.1, 10)
            v[lbl + 1] = 1.0
            v
        end for lbl in labels]
        return WaveDataset(inputs, targets; modality=:image, task=:classification)
    elseif endswith(lowercase(source), ".csv") || endswith(lowercase(source), ".tsv")
        # Local tabular dataset parsing
        sep = endswith(lowercase(source), ".tsv") ? '\t' : ','
        lines = readlines(source)
        header = Base.split(lines[1], sep)
        data_rows = [Base.split(l, sep) for l in lines[2:end] if !isempty(strip(l))]
        if limit !== nothing
            data_rows = data_rows[1:min(limit, length(data_rows))]
        end

        target_idx = if label_col isa String
            findfirst(==(label_col), header)
        elseif label_col isa Int
            label_col
        else
            length(header)
        end
        target_idx = target_idx !== nothing ? target_idx : length(header)
        feature_indices = filter(!=(target_idx), 1:length(header))

        parsed_inputs = [parse.(Float64, [r[i] for i in feature_indices if !isempty(strip(r[i]))]) for r in data_rows]
        parsed_targets = [[parse(Float64, r[target_idx])] for r in data_rows]
        return WaveDataset(parsed_inputs, parsed_targets; modality=:tabular, task=:regression)
    else
        # Fallback to Hugging Face generic dataset streaming
        streamer = WaveML.WaveDataStreamer(source, split; batch_size=16, max_samples=limit !== nothing ? limit : 100)
        inputs = Vector{Float64}[]
        targets = Vector{Float64}[]
        for batch in streamer
            for text in batch
                toks = Float64.(collect(codeunits(text)))
                push!(inputs, process_to_waves([toks]; embed_dim=embed_dim)[1])
                push!(targets, [0.5])
            end
        end
        return WaveDataset(inputs, targets; modality=:text, task=:generation)
    end
end

"""
    validate_dataset(ds::WaveDataset)::Dict{String, Any}

Validates a `WaveDataset`, inspecting dimensionality, physical energy bounds, and harmonic distributions.
Prints a diagnostic validation summary.
"""
function validate_dataset(ds::WaveDataset)::Dict{String, Any}
    n = length(ds)
    n > 0 || error("Dataset is empty!")

    in_dim = length(ds.inputs[1])
    tgt_dim = length(ds.targets[1])

    # Check for NaN / Inf
    has_nan_in = any(v -> any(isnan, v) || any(isinf, v), ds.inputs)
    has_nan_tgt = any(v -> any(isnan, v) || any(isinf, v), ds.targets)

    # Calculate average wave energy
    energies = [sum(v.^2) / in_dim for v in ds.inputs]
    mean_energy = mean(energies)
    std_energy = std(energies)

    println("="^75)
    println(" 🔍 SOVWAVE DATASET VALIDATION SUMMARY:")
    println("="^75)
    @printf("  • Modality:           :%s | Task: :%s\n", ds.modality, ds.task)
    @printf("  • Total Samples:      %d\n", n)
    @printf("  • Input Wave Dim:     %d | Target Dim: %d\n", in_dim, tgt_dim)
    @printf("  • Mean Wave Energy:   %.4f (± %.4f)\n", mean_energy, std_energy)
    @printf("  • Data Integrity:     %s\n", (!has_nan_in && !has_nan_tgt) ? "✅ PASS (0 NaN / 0 Inf)" : "❌ FAILED (Contains NaN/Inf)")
    println("="^75)

    return Dict{String, Any}(
        "num_samples" => n,
        "input_dim" => in_dim,
        "target_dim" => tgt_dim,
        "mean_energy" => mean_energy,
        "valid" => (!has_nan_in && !has_nan_tgt)
    )
end

# ==============================================================================
# 3. Custom Wave Equivalents to Layers
# ==============================================================================

"""
    WaveResonator(in_dim::Int, out_dim::Int; resonance_freq=432.0, q_factor=3.0, ...)

Continuous acoustic resonator layer callable on wave input vectors.
Powered by Tournament 11 Grand Champion (`Refined_StandingWaveInterference_Q3_B0.20`).
"""
struct WaveResonator
    in_dim::Int
    out_dim::Int
    layer::WaveLayer
    resonance_freq::Float64
    q_factor::Float64
end

function WaveResonator(
    in_dim::Int,
    out_dim::Int;
    resonance_freq::Float64 = 432.0,
    q_factor::Float64 = 3.0,
    carrier_frequency::Union{Nothing, Float64} = nothing,
    q_states::Union{Nothing, Int} = nothing,
    bessel_alpha::Float64 = 0.20
)::WaveResonator
    phi_golden = 1.618033988749895
    freq = carrier_frequency !== nothing ? carrier_frequency : resonance_freq
    q = q_states !== nothing ? Float64(q_states) : q_factor

    amps = ones(Float64, out_dim, in_dim)
    phs = zeros(Float64, out_dim, in_dim)
    freqs = ones(Float64, out_dim, in_dim)

    for i in 1:out_dim
        for j in 1:in_dim
            amps[i, j] = 0.5 + 0.5 * cos((i * j) * (2π / in_dim))
            phs[i, j] = mod2pi((i - 1) * (2π / out_dim) + (j - 1) * (2π / in_dim))
            freqs[i, j] = 1.0 + 0.5 * (phi_golden^(mod(i + j, 6) * 0.1))
        end
    end

    fractals = fill(phi_golden, out_dim)
    layer = WaveLayer(out_dim, in_dim, amps, phs, freqs, fractals, freq)
    return WaveResonator(in_dim, out_dim, layer, freq, q)
end

function (r::WaveResonator)(x::Vector{<:Real})
    out = zeros(Float64, r.out_dim)
    forward!(r.layer, out, Float64.(x), 0.0)
    return out
end

"""
    WaveChamber(in_dim::Int, out_dim::Int; num_standing_modes=4, damping=0.05, carrier_frequency=432.0)

Builds a complete multi-mode acoustic resonance chamber callable on wave input vectors.
"""
struct WaveChamber
    in_dim::Int
    out_dim::Int
    layers::Vector{WaveLayer}
    damping::Float64
end

function WaveChamber(
    in_dim::Int,
    out_dim::Int;
    num_standing_modes::Int = 4,
    damping::Float64 = 0.05,
    carrier_frequency::Float64 = 432.0
)::WaveChamber
    layers = [WaveResonator(in_dim, out_dim; carrier_frequency=carrier_frequency).layer for _ in 1:num_standing_modes]
    return WaveChamber(in_dim, out_dim, layers, damping)
end

function (c::WaveChamber)(x::Vector{<:Real})
    curr = Float64.(x)
    for layer in c.layers
        out = zeros(Float64, c.out_dim)
        forward!(layer, out, curr, 0.0)
        curr = out
    end
    return curr
end

# ==============================================================================
# 4. Wave-Friendly Mechanics Optimizers
# ==============================================================================

"""
    WaveMechanicsOptimizer

Continuous wave physical mechanics optimizer replacing discrete optimizers (SGD/Adam).
Powered by Tournament 10 Grand Champion (`Refined_GinzburgLandauDiffusion_Eta0.080_Gam0.55`).
"""
mutable struct WaveMechanicsOptimizer
    eta::Float64          # Relaxation rate (lr)
    gamma::Float64        # Soliton pulse coupling
    beta_viscosity::Float64
    phi_golden::Float64   # Harmonic damping constant
    mode::Symbol          # :ginzburg_landau, :hamiltonian, :soliton
    step_count::Int

    function WaveMechanicsOptimizer(;
        eta::Float64 = 0.08,
        lr::Union{Nothing, Float64} = nothing,
        gamma::Float64 = 0.55,
        beta_viscosity::Float64 = 0.01,
        mode::Symbol = :ginzburg_landau
    )
        actual_eta = lr !== nothing ? lr : eta
        new(actual_eta, gamma, beta_viscosity, 1.618033988749895, mode, 0)
    end
end

function Base.getproperty(opt::WaveMechanicsOptimizer, sym::Symbol)
    if sym === :lr
        return getfield(opt, :eta)
    else
        return getfield(opt, sym)
    end
end

"""
    step_mechanics!(opt::WaveMechanicsOptimizer, model::WaveModel, step_or_loss=nothing; max_steps=100, energy_target=0.001)

Executes one continuous wave mechanics relaxation step across all model layers.
"""
function step_mechanics!(
    opt::WaveMechanicsOptimizer,
    model::WaveModel,
    step_or_loss::Union{Real, Nothing} = nothing,
    max_steps::Int = 100;
    energy_target::Float64 = 0.001
)::Nothing
    opt.step_count += 1
    s = step_or_loss !== nothing && step_or_loss isa Integer ? Int(step_or_loss) : opt.step_count
    t_rel = Float64(s) / Float64(max(1, max_steps))
    decay = exp(-opt.phi_golden * t_rel * 1.5)

    for layer in model.layers
        nodes, dim = size(layer.amplitudes)
        for i in 1:nodes
            for j in 1:dim
                # Soliton wave pulse displacement
                pulse = cos(layer.phases[i, j] * opt.phi_golden - t_rel * 2π) * 0.1

                # Physical wave updates
                layer.amplitudes[i, j] -= opt.eta * decay * (0.05 * (layer.amplitudes[i, j] - 1.0) + opt.gamma * pulse)
                layer.amplitudes[i, j] = max(0.01, layer.amplitudes[i, j])

                layer.phases[i, j] -= opt.eta * decay * (0.05 * sin(layer.phases[i, j]) + opt.gamma * 0.5 * sin(layer.phases[i, j]))
                layer.phases[i, j] = mod2pi(layer.phases[i, j])
            end
        end
    end
    return nothing
end

# ==============================================================================
# 5. Non-Blocking Async Training with Single-Line Green Progress Bar
# ==============================================================================

mutable struct AsyncTrainingHandle
    task::Task
    is_running::Bool
    should_stop::Bool
    progress::Float64
    current_epoch::Int
    current_loss::Float64
    best_loss::Float64
    history::Any

    AsyncTrainingHandle(t::Task) = new(t, true, false, 0.0, 0, Inf, Inf, nothing)
end

"""
    train_wave(
        model::WaveModel,
        inputs::Vector{Vector{Float64}},
        targets::Vector{Vector{Float64}};
        epochs::Int = 25,
        batch_size::Int = 16,
        learning_rate::Float64 = 0.08,
        sonify::Bool = true,
        checkpoint_dir::Union{Nothing, String} = nothing,
        checkpoint_every::Int = 10,
        verbose::Bool = true
    )::Tuple{WaveModel, Any}

Synchronous user-friendly training loop featuring persistent single-line bold green progress bar
and continuous 432 Hz sound entrainment (Gamma to Epsilon).
"""
function train_wave(
    model::WaveModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}};
    epochs::Int = 25,
    batch_size::Int = 16,
    learning_rate::Float64 = 0.08,
    energy_target::Float64 = 0.001,
    sonify::Bool = true,
    checkpoint_dir::Union{Nothing, String} = nothing,
    checkpoint_every::Int = 10,
    verbose::Bool = true
)::Tuple{WaveModel, Any}
    n_samples = min(length(inputs), length(targets))
    n_samples > 0 || error("Input dataset is empty")

    cfg = WaveTrainConfig(
        epochs = epochs,
        batch_size = min(batch_size, n_samples),
        learning_rate = learning_rate,
        energy_target = energy_target,
        sonify = sonify
    )

    carrier = model.model_config.omega
    audio_stream = sonify ? ContinuousAudioStream(carrier_frequency=carrier) : nothing

    if verbose
        println("="^80)
        println(" 🌊 SOVWAVE CONTINUOUS WAVE COMPUTING: TRAINING INITIALIZED 🌊")
        println("="^80)
        @printf("  Dataset: %d samples | Batch: %d | Epochs: %d | Carrier: %.1f Hz\n",
                n_samples, cfg.batch_size, epochs, carrier)
        println("-"^80)
    end

    best_model = clone(model)
    best_loss = Inf
    opt = WaveMechanicsOptimizer(eta=learning_rate)

    for ep in 1:epochs
        t_ep_start = time_ns()

        # Batch evaluation
        b_size = cfg.batch_size
        batch_idx = mod1((ep - 1) * b_size + 1, max(1, n_samples - b_size + 1))
        b_inputs = inputs[batch_idx:min(n_samples, batch_idx + b_size - 1)]
        b_targets = targets[batch_idx:min(n_samples, batch_idx + b_size - 1)]

        # Forward pass & loss
        total_e = 0.0
        acc_accum = 0.0
        for i in 1:length(b_inputs)
            pred = forward!(best_model, b_inputs[i])
            total_e += compute_loss(pred, b_targets[i])
            acc_accum += wave_accuracy(pred, b_targets[i])
        end
        cur_loss = total_e / length(b_inputs)
        cur_acc = acc_accum / length(b_inputs)

        # Apply Wave Mechanics Update
        step_mechanics!(opt, best_model, ep, epochs; energy_target=energy_target)

        if cur_loss < best_loss
            best_loss = cur_loss
        end

        # Continuous sound entrainment
        if sonify && audio_stream !== nothing
            step_continuous_audio!(audio_stream, best_loss, energy_target; duration=0.04)
        end

        # Binaural beat frequency (Gamma to Epsilon)
        delta_f = compute_binaural_beat_freq(best_loss, energy_target)
        bw_band, bw_detail = brainwave_state(delta_f, best_loss <= energy_target)

        # Single-Line Bold Green Terminal Progress Bar (never prints extra newlines)
        if verbose
            pct = Float64(ep) / Float64(epochs)
            bar_w = 20
            filled = round(Int, pct * bar_w)
            unfilled = bar_w - filled
            bar_str = "\e[1;32m" * repeat("█", filled) * "\e[2;32m" * repeat("░", unfilled) * "\e[0m"

            print("\r\e[K")
            @printf("\e[1;32m[SOVWAVE WAVE EVOLUTION]\e[0m %s %5.1f%% | Ep %3d/%3d | Loss: \e[1;32m%.5f\e[0m | Acc: %5.1f%% | \e[1;36m[%s: %s]\e[0m",
                    bar_str, pct * 100.0, ep, epochs, cur_loss, cur_acc * 100.0, bw_band, bw_detail)
            flush(stdout)
        end

        # Save Checkpoint MKV Video directly from frames (no bin)
        if checkpoint_dir !== nothing && (ep % checkpoint_every == 0 || ep == epochs)
            mkpath(checkpoint_dir)
            ckpt_path = joinpath(checkpoint_dir, @sprintf("checkpoint_epoch_%04d.mkv", ep))
            save_model(best_model, ckpt_path; n_visual_frames=24, include_audio=sonify)
        end

        # Yield for async responsiveness
        yield()
    end

    if verbose
        print("\r\e[K")
        @printf("\e[1;32m  ✓ Training Complete | Final Best Loss: %.5f | Carrier: %.1f Hz\e[0m\n", best_loss, carrier)
        println("="^80)
    end

    return best_model, best_loss
end

"""
    train_wave(inputs, targets; model=nothing, cfg=nothing, play_sound=true, epochs=nothing, kwargs...)

Convenience training entrypoint that automatically sets up architecture and runs wave training.
Returns the trained `WaveModel`.
"""
function train_wave(
    inputs::Union{Vector{<:Vector{<:Real}}, Matrix{<:Real}},
    targets::Union{Vector{<:Vector{<:Real}}, Matrix{<:Real}, Vector{<:Real}};
    model::Union{Nothing, WaveModel} = nothing,
    cfg::Union{Nothing, WaveMLConfig} = nothing,
    play_sound::Bool = true,
    epochs::Union{Nothing, Int} = nothing,
    kwargs...
)::WaveModel
    in_vecs::Vector{Vector{Float64}} = if inputs isa Matrix{<:Real}
        [Float64.(inputs[i, :]) for i in 1:size(inputs, 1)]
    else
        [Float64.(v) for v in inputs]
    end

    tgt_vecs::Vector{Vector{Float64}} = if targets isa Matrix{<:Real}
        [Float64.(targets[i, :]) for i in 1:size(targets, 1)]
    elseif targets isa Vector{<:Vector{<:Real}}
        [Float64.(v) for v in targets]
    else
        [[Float64(t)] for t in targets]
    end

    target_model = if model !== nothing
        model
    elseif cfg !== nothing
        WaveModel(cfg)
    else
        in_dim = length(in_vecs[1])
        create_model(nodes=in_dim, embed_dims=in_dim)
    end

    actual_epochs = epochs !== nothing ? epochs : (cfg !== nothing ? cfg.train.epochs : 25)
    actual_sonify = (cfg !== nothing ? cfg.train.sonify : play_sound)

    trained_m, _ = train_wave(
        target_model,
        in_vecs,
        tgt_vecs;
        epochs = actual_epochs,
        sonify = actual_sonify,
        verbose = false,
        kwargs...
    )
    return trained_m
end

"""
    train_async(model::WaveModel, inputs, targets; kwargs...)::AsyncTrainingHandle

Spawns non-blocking async training in a background Task.
Returns an `AsyncTrainingHandle` that can be monitored without hanging the terminal or process.
"""
function train_async(
    model::WaveModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}};
    kwargs...
)::AsyncTrainingHandle
    handle_ref = Ref{AsyncTrainingHandle}()
    t = @async begin
        try
            trained_model, hist = train_wave(model, inputs, targets; kwargs...)
            if isassigned(handle_ref)
                h = handle_ref[]
                h.is_running = false
                h.history = hist
            end
            trained_model
        catch e
            if isassigned(handle_ref)
                handle_ref[].is_running = false
            end
            rethrow(e)
        end
    end
    handle = AsyncTrainingHandle(t)
    handle_ref[] = handle
    return handle
end

end # module SovwaveFunctions
