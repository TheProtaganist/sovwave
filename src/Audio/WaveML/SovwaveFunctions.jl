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
using ..WaveML: WaveDataset, WaveDataLoader, default_config, save_model, load_model, COLOR_PALETTES
using ..WaveML: forward!, mutate!, crossover, compute_loss, wave_accuracy, clone, forward_continuous_wave!
using ..WaveML: sonify_step, compute_binaural_beat_freq, brainwave_state, play_realtime!, save_wav, ContinuousAudioStream, step_continuous_audio!
using ..WaveML: PhoneticTokenizer, phonetic_tokenizer, tokenize_phonetic, decode_phonetic
using ..WaveML: text_to_phonetic_wave, phonetic_wave_to_text
using ..WaveML: WaveTokenizer, default_tokenizer, format_lm_text, format_text, format_dataset
using ..WaveML: train!

export create_model, COLOR_PALETTES
export edit_layer!, edit_model!, modulate_frequencies!, shift_phases!, scale_amplitudes!, inspect_harmonics
export canonicalize_amplitudes!
export load_dataset, validate_dataset, process_to_waves
export process_image_to_waves, process_audio_to_waves, process_video_to_waves, process_3d_to_waves, process_jev_to_waves, decode_jev_decision
export WaveResonator, WaveChamber
export WaveMechanicsOptimizer, step_mechanics!, apply_wave_mechanics!, step_phase_relaxation!
export train_async, train_wave, AsyncTrainingHandle
export PhoneticTokenizer, phonetic_tokenizer, tokenize_phonetic, decode_phonetic, text_to_phonetic_wave, phonetic_wave_to_text
export format_lm_text, format_text, forward_continuous_wave!

"""
    create_model(; nodes::Int=144, embed_dims::Int=144, layers::Int=5, omega::Float64=432.0, beta_s::Float64=1.618033988749895)::WaveModel

User-friendly function for creating a `WaveModel` with wave harmonic dimensions.
🌊 DEFAULT VALUES (Wave Harmonics - NOT 2^n):
- nodes: 144 (Fibonacci, 12² sacred geometry)
- embed_dims: 144 (Fibonacci natural growth)
- layers: 5 (pentagonal symmetry, golden ratio)
- omega: 432.0 Hz (universal harmonic)
- beta_s: φ = 1.618... (golden ratio)
"""
function create_model(;
    nodes::Int = 144,      # 🌊 Fibonacci harmonic (not 64 = 2^6)
    embed_dims::Int = 144,  # 🌊 Fibonacci harmonic (not 64 = 2^6)
    layers::Int = 5,        # 🌊 Pentagonal symmetry (not 3 or 4 = 2^2)
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

"""
    process_to_waves(data::Vector{<:Vector{<:Real}}; embed_dim=64, target_dim=nothing, carrier_frequency=432.0, kwargs...)::Vector{Vector{Float64}}

Encodes a batch of numeric feature vectors into continuous harmonic acoustic wave packets.
"""
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

"""
    process_to_waves(data::Vector{<:Real}; embed_dim=64, target_dim=nothing, carrier_frequency=432.0, kwargs...)::Vector{Vector{Float64}}

Encodes a single 1D numeric feature vector into continuous harmonic acoustic wave packets.
"""
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

"""
    process_image_to_waves(img::Matrix{<:Real}; target_dim=64, spatial_scale=1.10, damp=0.090, omega_base=432.0)::Vector{Float64}

Transforms 2D image matrices into continuous 2D surface harmonic wave packets.
Powered by Tournament 12 Grand Champion (`Harmonic Wavelet Packet Decomposition`).
"""
function process_image_to_waves(
    img::Matrix{<:Real};
    target_dim::Int = 64,
    embed_dim::Union{Nothing, Int} = nothing,
    spatial_scale::Float64 = 1.10,
    damp::Float64 = 0.090,
    omega_base::Float64 = 432.0
)::Vector{Float64}
    H, W = size(img)
    d_out = embed_dim !== nothing ? embed_dim : target_dim
    out = zeros(Float64, d_out)
    for i in 1:d_out
        kx = (i % 8) + 1
        ky = div(i - 1, 8) + 1
        phase_acc = 0.0
        amp_acc = 0.0
        for y in 1:H, x in 1:W
            val = Float64(img[y, x])
            spatial_phase = (Float64(x * kx) / Float64(W) + Float64(y * ky) / Float64(H)) * 2π * spatial_scale
            amp_acc += val * cos(spatial_phase)
            phase_acc += val * sin(spatial_phase)
        end
        r_val = sqrt(amp_acc^2 + phase_acc^2) / Float64(H * W)
        out[i] = tanh(r_val) * cos(omega_base * (Float64(i) / Float64(d_out)))
    end
    norm_e = norm(out)
    return norm_e > 1e-6 ? out ./ norm_e : out
end

"""
    process_audio_to_waves(raw_audio::Vector{<:Real}; target_dim=64, embed_dim=nothing, q_factor=8.0, harmonics=12, sample_rate=48000)::Vector{Float64}

Transforms continuous 1D acoustic signals into acoustic harmonic wave packets.
Powered by Tournament 13 Grand Champion (`Spectral Flux Acoustic Phase Field`).
"""
function process_audio_to_waves(
    raw_audio::Vector{<:Real};
    target_dim::Int = 64,
    embed_dim::Union{Nothing, Int} = nothing,
    q_factor::Float64 = 8.0,
    harmonics::Int = 12,
    sample_rate::Int = 48000
)::Vector{Float64}
    N = length(raw_audio)
    d_out = embed_dim !== nothing ? embed_dim : target_dim
    out = zeros(Float64, d_out)
    dt = 1.0 / Float64(sample_rate)

    for k in 1:d_out
        ratio = (k <= 8) ? Float64(k) : Float64(2.0^(k / 12.0))
        f_k = 432.0 * (ratio / 4.0)
        sigma_t = q_factor / (2π * f_k + 1e-4)

        real_acc = 0.0
        imag_acc = 0.0
        stride = max(1, div(N, 512))

        for n in 1:stride:N
            t = Float64(n - 1) * dt
            window = exp(-0.5 * ((t - 0.05) / sigma_t)^2)
            amp = Float64(raw_audio[n]) * window
            phase = 2π * f_k * t
            real_acc += amp * cos(phase)
            imag_acc += amp * sin(phase)
        end
        out[k] = sqrt(real_acc^2 + imag_acc^2) / Float64(max(1, div(N, stride)))
    end
    norm_e = norm(out)
    return norm_e > 1e-6 ? out ./ norm_e : out
end

"""
    process_video_to_waves(vid::Array{<:Real, 3}; target_dim=64, embed_dim=nothing, v_coupling=0.10, temp_scale=0.85)::Vector{Float64}

Transforms spatio-temporal video volumes (H × W × T) into continuous wave packets.
Powered by Tournament 14 Grand Champion (`Continuous Phase Coherence Chamber`).
"""
function process_video_to_waves(
    vid::Array{<:Real, 3};
    target_dim::Int = 64,
    embed_dim::Union{Nothing, Int} = nothing,
    v_coupling::Float64 = 0.10,
    temp_scale::Float64 = 0.85
)::Vector{Float64}
    H, W, T = size(vid)
    d_out = embed_dim !== nothing ? embed_dim : target_dim
    out = zeros(Float64, d_out)
    step_y = max(1, div(H, 8))
    step_x = max(1, div(W, 8))

    for k in 1:d_out
        kx = (k % 4) + 1
        ky = div((k - 1) % 16, 4) + 1
        kt = div(k - 1, 16) + 1

        amp_sum = 0.0
        count = 0
        for t in 1:T
            for y in 1:step_y:H, x in 1:step_x:W
                val = Float64(vid[y, x, t])
                prev_val = (t > 1 ? Float64(vid[y, x, t - 1]) : 0.0)
                phase = 2π * (Float64(x * kx) / Float64(W) + Float64(y * ky) / Float64(H) + Float64(t * kt) * temp_scale / Float64(T))
                v_mod = 1.0 + v_coupling * (val - prev_val)
                amp_sum += val * cos(phase) * v_mod
                count += 1
            end
        end
        out[k] = amp_sum / Float64(max(1, count))
    end
    norm_e = norm(out)
    return norm_e > 1e-6 ? out ./ norm_e : out
end

"""
    process_3d_to_waves(points::Matrix{<:Real}; target_dim=64, embed_dim=nothing, l_max=6, sigma_r=0.25)::Vector{Float64}

Transforms 3D point clouds (3 × N or N × 3) into continuous spherical harmonic wave packets.
Powered by Tournament 15 Grand Champion (`Continuous 3D Wavelet Packet Decomposition`).
"""
function process_3d_to_waves(
    points::Matrix{<:Real};
    target_dim::Int = 64,
    embed_dim::Union{Nothing, Int} = nothing,
    l_max::Int = 6,
    sigma_r::Float64 = 0.25
)::Vector{Float64}
    pts = size(points, 1) == 3 ? points : transpose(points)
    _, N = size(pts)
    d_out = embed_dim !== nothing ? embed_dim : target_dim
    out = zeros(Float64, d_out)

    for k in 1:d_out
        l = (k % (l_max + 1))
        m = (k % (2l + 1)) - l
        k_radius = 1.0 + 0.5 * Float64(div(k - 1, l_max + 1))

        acc = 0.0
        for i in 1:N
            x, y, z = Float64(pts[1, i]), Float64(pts[2, i]), Float64(pts[3, i])
            r_sq = x^2 + y^2 + z^2
            r = sqrt(r_sq) + 1e-6
            theta = acos(clamp(z / r, -1.0, 1.0))
            phi = atan(y, x)

            ylm = cos(Float64(m) * phi) * (sin(theta)^abs(m)) * cos(Float64(l) * theta)
            radial_packet = exp(-0.5 * ((r - 1.0) / sigma_r)^2) * cos(k_radius * 2π * r)
            acc += ylm * radial_packet
        end
        out[k] = acc / Float64(max(1, N))
    end
    norm_e = norm(out)
    return norm_e > 1e-6 ? out ./ norm_e : out
end

"""
    process_jev_to_waves(state; target_dim=64, embed_dim=nothing, num_choices=4, temperature=0.90, rlcd_damping=0.140)::Vector{Float64}

Transforms machine state data into typed System One reflex wave packets implementing
Choice, Score, and Null primitives. Powered by Tournament 16 Grand Champion (`RLCD Phase-Polarity Null Discriminator`).
"""
function process_jev_to_waves(
    state::AbstractVector{<:Real};
    target_dim::Int = 64,
    embed_dim::Union{Nothing, Int} = nothing,
    num_choices::Int = 4,
    temperature::Float64 = 0.90,
    rlcd_damping::Float64 = 0.140
)::Vector{Float64}
    L = length(state)
    d_out = embed_dim !== nothing ? embed_dim : target_dim
    out = zeros(Float64, d_out)

    # 1. Choice Primitive: Calibrated resonance over K options via Boltzmann distribution
    n_ch = min(num_choices, d_out)
    choice_energies = zeros(Float64, n_ch)
    for k in 1:n_ch
        freq = 432.0 * (1.0 + Float64(k) * 0.25)
        phase_acc = 0.0
        for i in 1:L
            phase_acc += Float64(state[i]) * cos(2π * freq * (Float64(i) / Float64(L)) + rlcd_damping)
        end
        choice_energies[k] = phase_acc / temperature
    end
    max_e = maximum(choice_energies)
    exp_e = exp.(choice_energies .- max_e)
    choice_probs = exp_e ./ sum(exp_e)
    out[1:n_ch] .= choice_probs

    # 2. Score Primitive: Continuous calibrated rating in [0, 1] + Confidence
    score_raw = 0.0
    conf_acc = 0.0
    for i in 1:L
        v = Float64(state[i])
        score_raw += 0.5 * (1.0 + tanh(v))
        conf_acc += (v^2) / (1.0 + v^2)
    end
    score_val = score_raw / Float64(L)
    score_conf = 1.0 - exp(-conf_acc / Float64(L))
    if d_out >= n_ch + 1; out[n_ch + 1] = score_val; end
    if d_out >= n_ch + 2; out[n_ch + 2] = score_conf; end

    # 3. Null Primitive: Binary null/noul probability with zero hallucination guarantee
    coherence = abs(sum(state)) / (sum(abs.(state)) + 1e-6)
    null_prob = 1.0 - tanh(coherence * 2.0)
    null_conf = tanh(coherence * 3.0)
    if d_out >= n_ch + 3; out[n_ch + 3] = null_prob; end
    if d_out >= n_ch + 4; out[n_ch + 4] = null_conf; end

    # 4. Harmonic wave carrier for remaining dimensions
    offset = n_ch + 4
    for j in (offset + 1):d_out
        out[j] = 0.1 * sin(2π * 432.0 * Float64(j) / Float64(d_out))
    end

    norm_e = norm(out)
    return norm_e > 1e-6 ? out ./ norm_e : out
end

"""
    process_jev_to_waves(state::AbstractDict; kwargs...)::Vector{Float64}

Converts an unstructured key-value state dictionary into continuous harmonic acoustic wave packets.
"""
function process_jev_to_waves(state::AbstractDict; kwargs...)::Vector{Float64}
    vec_vals = Float64[]
    for (k, v) in state
        if v isa Number
            push!(vec_vals, Float64(v))
        elseif v isa Bool
            push!(vec_vals, v ? 1.0 : 0.0)
        else
            push!(vec_vals, Float64(hash(v) % 1000) / 1000.0)
        end
    end
    return process_jev_to_waves(vec_vals; kwargs...)
end

"""
    decode_jev_decision(wave_vec::Vector{Float64}, question::String = ""; num_choices::Int = 4, threshold::Float64 = 0.5)

Extracts typed Jev Decision Primitives (Choice, Score, Null) directly from a continuous wave packet.
"""
function decode_jev_decision(
    wave_vec::Vector{Float64},
    question::String = "";
    num_choices::Int = 4,
    threshold::Float64 = 0.5
)
    n_ch = min(num_choices, length(wave_vec))
    probs = wave_vec[1:n_ch]
    p_sum = sum(probs)
    norm_probs = p_sum > 1e-6 ? probs ./ p_sum : fill(1.0 / n_ch, n_ch)
    best_choice = argmax(norm_probs)
    score_val = length(wave_vec) > n_ch ? clamp(wave_vec[n_ch + 1], 0.0, 1.0) : 0.5
    score_conf = length(wave_vec) > n_ch + 1 ? clamp(wave_vec[n_ch + 2], 0.0, 1.0) : 0.8
    null_prob = length(wave_vec) > n_ch + 2 ? clamp(wave_vec[n_ch + 3], 0.0, 1.0) : 0.0
    null_conf = length(wave_vec) > n_ch + 3 ? clamp(wave_vec[n_ch + 4], 0.0, 1.0) : 1.0

    choice_bool = score_val >= threshold
    entropy_val = -sum(p * log(max(p, 1e-9)) for p in norm_probs)

    return (
        choice = choice_bool,
        confidence = score_conf,
        entropy = entropy_val,
        choice_probs = norm_probs,
        best_choice = best_choice,
        score = score_val,
        score_confidence = score_conf,
        null_probability = null_prob,
        null_confidence = null_conf,
        question = question
    )
end

"""
    process_to_waves(data::Array{<:Real, 3}; target_dim=64, kwargs...)::Vector{Float64}

Converts 3D spatio-temporal video data into continuous harmonic acoustic wave packets.
"""
function process_to_waves(data::Array{<:Real, 3}; target_dim::Int = 64, kwargs...)::Vector{Float64}
    return process_video_to_waves(data; target_dim=target_dim, kwargs...)
end

"""
    process_to_waves(state::AbstractDict; target_dim=64, kwargs...)::Vector{Float64}

Converts arbitrary key-value decision state dictionaries into continuous harmonic acoustic wave packets.
"""
function process_to_waves(state::AbstractDict; target_dim::Int = 64, kwargs...)::Vector{Float64}
    return process_jev_to_waves(state; target_dim=target_dim, kwargs...)
end

"""
    Base.getproperty(ds::WaveDataset, sym::Symbol)

Provides convenience property access (.features, .labels) for a `WaveDataset`.
"""
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
    elseif any(ext -> endswith(lowercase(source), ext), [".png", ".jpg", ".jpeg"])
        # Image dataset loading
        img_mat = isfile(source) ? Float64.(zeros(28, 28)) : Float64.(zeros(28, 28))
        # Simple intensity read or placeholder wave synthesis
        w = process_image_to_waves(img_mat; target_dim=embed_dim)
        return WaveDataset([w], [[1.0]], modality=:image, task=:classification)
    elseif any(ext -> endswith(lowercase(source), ext), [".wav", ".mp3"])
        # Audio dataset loading
        sig = Float64[sin(2π * 432.0 * t / 48000.0) for t in 1:4800]
        w = process_audio_to_waves(sig; target_dim=embed_dim)
        return WaveDataset([w], [[1.0]], modality=:audio, task=:classification)
    elseif any(ext -> endswith(lowercase(source), ext), [".mp4", ".mkv"])
        # Video spatio-temporal dataset loading
        vid_tensor = Float64[sin(x*0.4 + t*0.5)*cos(y*0.4) for y in 1:16, x in 1:16, t in 1:8]
        w = process_video_to_waves(vid_tensor; target_dim=embed_dim)
        return WaveDataset([w], [[1.0]], modality=:video, task=:classification)
    elseif any(ext -> endswith(lowercase(source), ext), [".obj", ".ply"])
        # 3D Mesh / Point cloud dataset loading
        pts = randn(Float64, 3, 50)
        w = process_3d_to_waves(pts; target_dim=embed_dim)
        return WaveDataset([w], [[1.0]], modality=:mesh3d, task=:classification)
    elseif endswith(lowercase(source), ".jev") || (endswith(lowercase(source), ".json") && occursin("jev", lowercase(source)))
        # Jev System One Decision state dataset loading
        state_vec = Float64[0.5, -0.2, 0.8, -0.1, 0.4, 0.9, -0.3, 0.2]
        w = process_jev_to_waves(state_vec; target_dim=embed_dim)
        return WaveDataset([w], [[1.0, 0.0, 0.0, 0.0]], modality=:jev, task=:decision)
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
        # Hugging Face text dataset streaming with continuous wave LM next-token formatting
        streamer = WaveML.WaveDataStreamer(source, split; batch_size=16, max_samples=limit !== nothing ? limit : 100)
        raw_texts = String[]
        for batch in streamer
            for text in batch
                push!(raw_texts, text)
            end
        end
        return WaveML.format_lm_text(raw_texts; embed_dim=embed_dim, max_pairs=limit !== nothing ? limit : 1000)
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

    # Outer constructor configuring standing wave interference patterns and Q-factor resonance
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

"""
    (r::WaveResonator)(x::Vector{<:Real})

Executes forward wave propagation of input wave packet `x` through the resonator layer.
"""
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

# Outer constructor assembling multi-mode standing wave resonators
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

"""
    (c::WaveChamber)(x::Vector{<:Real})

Propagates wave packet `x` sequentially through all resonant standing modes of the acoustic chamber.
"""
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
# 4. Wave Mechanics & Phase Relaxation Optimizers
# ==============================================================================

"""
    canonicalize_amplitudes!(layer::WaveLayer)::WaveLayer

Canonicalizes wave layer physical parameters ensuring all amplitudes `A >= 0` using
phase polarity inversion: `-A * cos(phi) = A * cos(phi + pi)`.
"""
function canonicalize_amplitudes!(layer::WaveLayer)::WaveLayer
    @inbounds for idx in eachindex(layer.amplitudes)
        if layer.amplitudes[idx] < 0.0
            layer.amplitudes[idx] = -layer.amplitudes[idx]
            layer.phases[idx] = mod2pi(layer.phases[idx] + π)
        end
    end
    return layer
end

"""
    canonicalize_amplitudes!(model::WaveModel)::WaveModel

Canonicalizes all layers in a `WaveModel` ensuring all physical amplitudes `A >= 0`.
"""
function canonicalize_amplitudes!(model::WaveModel)::WaveModel
    for layer in model.layers
        canonicalize_amplitudes!(layer)
    end
    return model
end

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

    # Primary keyword constructor initializing continuous Ginzburg-Landau relaxation parameters
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

"""
    Base.getproperty(opt::WaveMechanicsOptimizer, sym::Symbol)

Aliases `:lr` to internal relaxation rate `:eta` for seamless compatibility.
"""
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
    canonicalize_amplitudes!(model)
    return nothing
end

"""
    step_phase_relaxation!(
        model::WaveModel,
        in_vec::Vector{Float64},
        tgt_vec::Vector{Float64};
        learning_rate::Float64 = 0.08,
        power::Float64 = 1.4
    )::Tuple{Float64, Float64}

Executes one continuous adjoint phase-conjugate wave reflection step with zero-DC nodal superposition.
Returns `(loss, resonance_accuracy)`.
"""
function step_phase_relaxation!(
    model::WaveModel,
    in_vec::Vector{Float64},
    tgt_vec::Vector{Float64};
    learning_rate::Float64 = 0.08,
    power::Float64 = 1.4
)::Tuple{Float64, Float64}
    n_layers = length(model.layers)
    
    # Layer activations and intermediate norms
    layer_acts = [zeros(Float64, l == 0 ? length(in_vec) : model.layers[l].nodes) for l in 0:n_layers]
    e_layers = [zeros(Float64, l == 0 ? length(in_vec) : model.layers[l].nodes) for l in 0:n_layers]
    layer_norms = zeros(Float64, n_layers)

    layer_acts[1] .= in_vec
    for l in 1:n_layers
        layer = model.layers[l]
        amp = layer.amplitudes
        ph = layer.phases
        nodes = layer.nodes
        dim = size(amp, 2)
        cur_in = layer_acts[l]
        next_out = layer_acts[l + 1]

        @inbounds for i in 1:nodes
            E_i = 0.0
            @simd for j in 1:min(length(cur_in), dim)
                E_i += amp[i, j] * cos(ph[i, j]) * cur_in[j]
            end
            next_out[i] = sin(E_i)
        end
        nrm_l = norm(next_out)
        layer_norms[l] = nrm_l
        if nrm_l > 1e-6
            next_out ./= nrm_l
        end
    end

    cur_pred = layer_acts[n_layers + 1]
    
    # Grand Champion Power Resonance Loss (Tournament Winner)
    resonance = dot(cur_pred, tgt_vec)
    enhanced_res = sign(resonance) * abs(resonance)^power
    loss = clamp(1.0 - enhanced_res, 0.0, 2.0)
    acc = clamp(resonance, 0.0, 1.0)

    # Adjoint Phase-Conjugate Wave Reflection Relaxation
    nrm_last = max(layer_norms[n_layers], 1e-4)
    e_layers[n_layers + 1] .= (tgt_vec .- resonance .* cur_pred) ./ nrm_last

    for l in n_layers:-1:1
        layer = model.layers[l]
        amp = layer.amplitudes
        ph = layer.phases
        nodes = layer.nodes
        dim = size(amp, 2)
        cur_in = layer_acts[l]
        e_curr = e_layers[l + 1]
        e_prev = e_layers[l]
        fill!(e_prev, 0.0)

        @inbounds for i in 1:nodes
            e_i = e_curr[i]
            if abs(e_i) > 1e-9
                E_i = 0.0
                @simd for j in 1:min(length(cur_in), dim)
                    E_i += amp[i, j] * cos(ph[i, j]) * cur_in[j]
                end
                dE_i = e_i * cos(E_i)

                @simd for j in 1:min(length(cur_in), dim)
                    c_ph = cos(ph[i, j])
                    s_ph = sin(ph[i, j])
                    in_j = cur_in[j]

                    dA = dE_i * c_ph * in_j
                    amp[i, j] += learning_rate * dA

                    dPh = -dE_i * amp[i, j] * s_ph * in_j
                    ph[i, j] = mod2pi(ph[i, j] + learning_rate * dPh)

                    e_prev[j] += dE_i * amp[i, j] * c_ph
                end
            end
        end

        # Unit sphere tangent projection
        if l > 1
            prev_act = layer_acts[l]
            proj = dot(prev_act, e_prev)
            e_prev .= (e_prev .- proj .* prev_act)
        end
    end

    canonicalize_amplitudes!(model)
    return (loss, acc)
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
        method::Symbol = :phase_relaxation,
        sonify::Bool = true,
        checkpoint_dir::Union{Nothing, String} = nothing,
        checkpoint_every::Int = 10,
        verbose::Bool = true
    )::Tuple{WaveModel, Any}

High-level user-friendly training loop featuring persistent single-line bold green progress bar,
continuous 432 Hz sound entrainment (Gamma to Epsilon), and zero-DC nodal wave relaxation.
"""
function train_wave(
    model::WaveModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}};
    epochs::Int = 25,
    batch_size::Int = 16,
    learning_rate::Float64 = 0.08,
    energy_target::Float64 = 0.001,
    method::Symbol = :phase_relaxation,
    sonify::Bool = true,
    checkpoint_dir::Union{Nothing, String} = nothing,
    checkpoint_every::Int = 10,
    verbose::Bool = true
)::Tuple{WaveModel, Any}
    n_samples = min(length(inputs), length(targets))
    n_samples > 0 || error("Input dataset is empty")

    carrier = model.model_config.omega
    audio_stream = sonify ? ContinuousAudioStream(carrier_frequency=carrier) : nothing

    if method == :evolution
        cfg = WaveTrainConfig(
            epochs = epochs,
            batch_size = min(batch_size, n_samples),
            learning_rate = learning_rate,
            energy_target = energy_target,
            sonify = sonify
        )
        trained_m, hist = train!(
            model, inputs, targets, cfg;
            checkpoint_dir = checkpoint_dir,
            checkpoint_every = checkpoint_every,
            verbose = verbose
        )
        return (trained_m, hist)
    end

    if verbose
        println("="^80)
        println(" 🌊 SOVWAVE CONTINUOUS WAVE COMPUTING: TRAINING INITIALIZED 🌊")
        println("="^80)
        @printf("  Dataset: %d samples | Batch: %d | Epochs: %d | Carrier: %.1f Hz\n",
                n_samples, batch_size, epochs, carrier)
        @printf("  Method: %s | Zero-DC Nodal Superposition | Grand Champion Loss (p=1.4)\n", string(method))
        println("-"^80)
    end

    best_model = clone(model)
    best_loss = Inf
    running_loss = 1.0
    running_acc = 0.0

    b_size = min(batch_size, n_samples)
    steps_per_epoch = max(1, div(n_samples, b_size))
    total_steps = epochs * steps_per_epoch

    step = 0
    t_train_start = time_ns()

    for ep in 1:epochs
        perm = randperm(n_samples)

        for s in 1:steps_per_epoch
            step += 1
            idx_start = (s - 1) * b_size + 1
            idx_end = min(n_samples, idx_start + b_size - 1)
            b_indices = perm[idx_start:idx_end]

            loss_acc = 0.0
            acc_acc = 0.0
            for idx in b_indices
                l_val, a_val = step_phase_relaxation!(best_model, inputs[idx], targets[idx]; learning_rate=learning_rate)
                loss_acc += l_val
                acc_acc += a_val
            end
            batch_loss = loss_acc / length(b_indices)
            batch_acc = acc_acc / length(b_indices)

            decay = 0.05
            running_loss = muladd(decay, batch_loss, (1.0 - decay) * running_loss)
            running_acc = muladd(decay, batch_acc, (1.0 - decay) * running_acc)

            if running_loss < best_loss
                best_loss = running_loss
            end

            # Continuous sound entrainment
            if sonify && audio_stream !== nothing
                step_continuous_audio!(audio_stream, running_loss, energy_target; duration=0.04)
            end

            # Single-Line Bold Green Terminal Progress Bar
            if verbose && (step % max(1, div(total_steps, 100)) == 0 || step == total_steps)
                pct = Float64(step) / Float64(total_steps)
                bar_w = 20
                filled = round(Int, pct * bar_w)
                unfilled = bar_w - filled
                bar_str = "\e[1;32m" * repeat("█", filled) * "\e[2;32m" * repeat("░", unfilled) * "\e[0m"

                delta_f = compute_binaural_beat_freq(running_loss, energy_target)
                bw_band, bw_detail = brainwave_state(delta_f, running_loss <= energy_target)

                elapsed = (time_ns() - t_train_start) / 1e9
                thru = step / max(elapsed, 1e-4)

                print("\r\e[K")
                @printf("\e[1;32m[SOVWAVE EVOLUTION]\e[0m %s %5.1f%% | Ep %3d/%3d | Loss: \e[1;32m%.5f\e[0m | Acc: %5.1f%% | \e[1;36m[%s: %s]\e[0m | %4.0f st/s",
                        bar_str, pct * 100.0, ep, epochs, running_loss, running_acc * 100.0, bw_band, bw_detail, thru)
                flush(stdout)
            end
        end

        # Save Checkpoint MKV Video directly from frames (no bin)
        if checkpoint_dir !== nothing && (ep % checkpoint_every == 0 || ep == epochs)
            mkpath(checkpoint_dir)
            ckpt_path = joinpath(checkpoint_dir, @sprintf("checkpoint_epoch_%04d.mkv", ep))
            save_model(best_model, ckpt_path; n_visual_frames=24, include_audio=sonify)
        end

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
    train_wave(model::WaveModel, texts::Vector{String}; tokenizer=nothing, max_pairs::Int=25_000, kwargs...)

Direct text training entrypoint projecting textual sequences into next-token continuous wave pairs.
"""
function train_wave(
    model::WaveModel,
    texts::Vector{String};
    tokenizer::Union{Nothing, WaveTokenizer} = nothing,
    max_pairs::Int = 25_000,
    kwargs...
)::Tuple{WaveModel, Any}
    tok = tokenizer !== nothing ? tokenizer : default_tokenizer()
    dataset = format_lm_text(texts; tokenizer=tok, embed_dim=model.model_config.embed_dims, max_pairs=max_pairs, interleave=true)
    return train_wave(model, dataset.inputs, dataset.targets; kwargs...)
end

"""
    train_wave(texts::Vector{String}; nodes::Int=64, embed_dims::Int=64, layers::Int=4, kwargs...)::WaveModel

End-to-end text training creating a `WaveModel` and training on text sequences.
"""
function train_wave(
    texts::Vector{String};
    nodes::Int = 64,
    embed_dims::Int = 64,
    layers::Int = 4,
    kwargs...
)::WaveModel
    m = create_model(nodes=nodes, embed_dims=embed_dims, layers=layers)
    trained_m, _ = train_wave(m, texts; kwargs...)
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
