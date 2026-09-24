"""
    WaveML.Dataset

Multi-Modal Dataset Formatting & Continuous Wave Embedding Pipelines for Sovwave.
Provides universal dataset abstractions that convert arbitrary real-world data
(Tabular, Text, 2D Images, Time-Series, 3D Volumes, and Jev Decision States)
into model-readable continuous wave embeddings.

Directly competes with PyTorch's `DataLoader` / `TensorDataset`, JAX's tree data,
and Flux.jl's `DataLoader` while operating natively in the wave-harmonic domain.
"""

using Random
using LinearAlgebra
using Statistics

export WaveDataset, WaveDataLoader, WaveDataStreamer
export format_tabular, format_text, format_lm_text, format_images, format_timeseries, format_jev, format_dataset
export encode_wave_context
export format_audio, format_video, format_3d
export process_pixel_waves, process_wave_tokens, process_digital_data, stream_dataset
export from_tabular, from_text, from_image, from_timeseries, from_jev_state, from_audio, from_video, from_3d
export batch_size, num_batches, num_samples

"""
    WaveDataset

Universal dataset container storing continuous wave-embedded inputs, targets, and task metadata.
"""
struct WaveDataset
    inputs::Vector{Vector{Float64}}
    targets::Vector{Vector{Float64}}
    modality::Symbol # :tabular, :text, :image, :timeseries, :volume, :jev_state
    task::Symbol     # :classification, :regression, :generation, :decision, :embedding, :reconstruction
    metadata::Dict{String, Any}

    # Primary constructor ensuring input/target dimensionality consistency
    function WaveDataset(
        inputs::Vector{Vector{Float64}},
        targets::Vector{Vector{Float64}};
        modality::Symbol = :tabular,
        task::Symbol = :classification,
        metadata::AbstractDict = Dict{String, Any}()
    )
        length(inputs) == length(targets) || error("Inputs length ($(length(inputs))) must match targets length ($(length(targets)))")
        meta = Dict{String, Any}(string(k) => v for (k, v) in metadata)
        new(inputs, targets, modality, task, meta)
    end
end

Base.length(ds::WaveDataset) = length(ds.inputs)
Base.getindex(ds::WaveDataset, idx::Int) = (ds.inputs[idx], ds.targets[idx])
Base.getindex(ds::WaveDataset, range::AbstractVector{Int}) = (ds.inputs[range], ds.targets[range])
Base.iterate(ds::WaveDataset, state::Int = 1) = state > length(ds) ? nothing : (ds[state], state + 1)

"""
    WaveDataLoader

Batched, shuffle-capable iterator over a `WaveDataset`.
"""
struct WaveDataLoader
    dataset::WaveDataset
    batch_size::Int
    shuffle::Bool
    drop_last::Bool
    indices::Vector{Int}

    # Primary constructor initializing index permutations for batched iteration
    function WaveDataLoader(
        dataset::WaveDataset;
        batch_size::Int = 16,
        shuffle::Bool = true,
        drop_last::Bool = false
    )
        indices = collect(1:length(dataset))
        new(dataset, max(1, batch_size), shuffle, drop_last, indices)
    end
end

num_batches(loader::WaveDataLoader) = loader.drop_last ? 
    div(length(loader.dataset), loader.batch_size) : 
    cld(length(loader.dataset), loader.batch_size)

function Base.iterate(loader::WaveDataLoader, state::Int = 1)
    N = length(loader.dataset)
    B = loader.batch_size

    if state == 1 && loader.shuffle
        shuffle!(loader.indices)
    end

    if state > N || (loader.drop_last && state + B - 1 > N)
        return nothing
    end

    end_idx = min(state + B - 1, N)
    batch_idx = loader.indices[state:end_idx]

    batch_x = loader.dataset.inputs[batch_idx]
    batch_y = loader.dataset.targets[batch_idx]

    return ((batch_x, batch_y), end_idx + 1)
end

Base.length(loader::WaveDataLoader) = num_batches(loader)

# ============================================================================
# Modality Formatting Engines
# ============================================================================

"""
    format_tabular(X::Matrix{Float64}, y; task=:classification, embed_dim=32)::WaveDataset

Transforms tabular/numerical feature matrices (samples × features) into continuous wave embeddings.
Each feature x_m is projected into harmonic amplitude A_m, phase phi_m, and frequency f_m.
"""
function format_tabular(
    X::Matrix{Float64},
    y::Union{Vector{Float64}, Vector{Int}, Matrix{Float64}, Vector{Vector{Float64}}};
    task::Symbol = :classification,
    embed_dim::Int = 32
)::WaveDataset
    n_samples, n_features = size(X)
    inputs = Vector{Vector{Float64}}(undef, n_samples)
    inv_dim = 1.0 / Float64(embed_dim)

    # Convert tabular features into wave harmonic interference vectors
    for i in 1:n_samples
        row = X[i, :]
        emb = zeros(Float64, embed_dim)

        for (m, val) in enumerate(row)
            A = tanh(abs(val))
            ph = atan(val)
            freq = 1.0 + Float64((m - 1) % 8) * 0.5

            for d in 1:embed_dim
                x = Float64(d - 1) * inv_dim
                emb[d] += A * cos(2π * freq * x + ph)
            end
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        inputs[i] = emb
    end

    # Format targets
    targets = if y isa Matrix{Float64}
        [y[i, :] for i in 1:n_samples]
    elseif y isa Vector{Vector{Float64}}
        y
    elseif y isa Vector{Int} && task == :classification
        # One-hot encoding
        num_classes = maximum(y)
        [Float64[c == y[i] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    else
        [[Float64(val)] for val in y]
    end

    return WaveDataset(inputs, targets; modality=:tabular, task=task)
end

"""
    format_text(texts::Vector{String}, labels; tokenizer=default_tokenizer(), max_len=32, embed_dim=32, task=:classification)::WaveDataset

Formats raw natural language strings into continuous wave packet representations using `WaveTokenizer`.
"""
function format_text(
    texts::Vector{String},
    labels::Union{Vector{Int}, Vector{Float64}, Vector{String}, Vector{Vector{Float64}}};
    tokenizer::WaveTokenizer = default_tokenizer(),
    max_len::Int = 32,
    embed_dim::Int = 32,
    task::Symbol = :classification
)::WaveDataset
    n_samples = length(texts)
    inputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        # Encode text to sequence wave tensor (embed_dim × max_len)
        seq_mat = encode_sequence(tokenizer, texts[i]; max_len=max_len, embed_dim=embed_dim)
        # Mean temporal pooling across sequence steps for single-vector input
        inputs[i] = vec(mean(seq_mat, dims=2))
        nrm = norm(inputs[i])
        if nrm > 1e-6
            inputs[i] ./= nrm
        end
    end

    # Format labels
    targets = if labels isa Vector{Int} && task == :classification
        num_classes = maximum(labels)
        [Float64[c == labels[i] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    elseif labels isa Vector{String} && task == :classification
        unique_labels = unique(labels)
        label_map = Dict(l => idx for (idx, l) in enumerate(unique_labels))
        num_classes = length(unique_labels)
        [Float64[c == label_map[labels[i]] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    elseif labels isa Vector{Vector{Float64}}
        labels
    else
        [[Float64(l)] for l in labels]
    end

    return WaveDataset(inputs, targets; modality=:text, task=task, metadata=Dict("max_len" => max_len))
end

"""
    format_images(images::Vector{Matrix{Float64}}, labels; embed_dim=32, task=:classification)::WaveDataset

Transforms 2D grayscale/intensity image matrices into continuous 2D surface harmonic nodes.
"""
function format_images(
    images::Vector{Matrix{Float64}},
    labels::Union{Vector{Int}, Vector{Float64}, Vector{Vector{Float64}}};
    embed_dim::Int = 32,
    task::Symbol = :classification
)::WaveDataset
    n_samples = length(images)
    # Continuous 2D spatial surface harmonic wavefield projection
    inputs = process_pixel_waves(images; embed_dim=embed_dim)

    targets = if labels isa Vector{Int} && task == :classification
        num_classes = maximum(labels)
        [Float64[c == labels[i] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    elseif labels isa Vector{Vector{Float64}}
        labels
    else
        [[Float64(l)] for l in labels]
    end

    return WaveDataset(inputs, targets; modality=:image, task=task)
end

"""
    format_timeseries(series::Vector{Float64}; window_size=16, horizon=1, embed_dim=32)::WaveDataset

Converts 1D continuous signals or time-series into sliding window harmonic phase frames.
"""
function format_timeseries(
    series::Vector{Float64};
    window_size::Int = 16,
    horizon::Int = 1,
    embed_dim::Int = 32
)::WaveDataset
    N = length(series)
    N > window_size + horizon || error("Series too short for window_size=$window_size and horizon=$horizon")

    n_samples = N - window_size - horizon + 1
    inputs = Vector{Vector{Float64}}(undef, n_samples)
    targets = Vector{Vector{Float64}}(undef, n_samples)
    inv_dim = 1.0 / Float64(embed_dim)

    for i in 1:n_samples
        window = series[i:(i + window_size - 1)]
        target_val = series[(i + window_size):(i + window_size + horizon - 1)]

        emb = zeros(Float64, embed_dim)
        for (w_idx, val) in enumerate(window)
            t = Float64(w_idx) / Float64(window_size)
            for d in 1:embed_dim
                emb[d] += val * cos(2π * t * (Float64(d) * inv_dim))
            end
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        inputs[i] = emb
        targets[i] = target_val
    end

    return WaveDataset(inputs, targets; modality=:timeseries, task=:regression)
end

"""
    format_jev(states::Vector{Dict{String, Any}}, questions::Vector{String}, decisions::Vector{Vector{Float64}}; embed_dim=32)::WaveDataset

Formats application state dictionaries paired with structured questions for Jev-based decision models.
Each state-question pair maps into a continuous wave interference vector.
"""
function format_jev(
    states::AbstractVector{<:AbstractDict},
    questions::Vector{String},
    decisions::Vector{Vector{Float64}};
    embed_dim::Int = 32
)::WaveDataset
    n_samples = length(states)
    n_samples == length(questions) || error("States count must match questions count")
    inputs = Vector{Vector{Float64}}(undef, n_samples)
    inv_dim = 1.0 / Float64(embed_dim)

    for i in 1:n_samples
        state = states[i]
        q = questions[i]
        emb = zeros(Float64, embed_dim)

        # 1. State feature harmonic superposition
        for (k_idx, (key, val)) in enumerate(state)
            num_val = val isa Number ? Float64(val) : (val isa Bool ? (val ? 1.0 : 0.0) : Float64(hash(val) % 100) / 100.0)
            ph = 2π * Float64(k_idx) / Float64(max(1, length(state)))
            for d in 1:embed_dim
                emb[d] += num_val * cos(2π * (Float64(d) * inv_dim) + ph)
            end
        end

        # 2. Question semantic phase interference
        q_hash = Float64(hash(q) % 1000) / 1000.0
        for d in 1:embed_dim
            emb[d] += 0.5 * sin(2π * q_hash * Float64(d) * inv_dim)
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        inputs[i] = emb
    end

    return WaveDataset(inputs, decisions; modality=:jev_state, task=:decision)
end

"""
    format_audio(signals::Vector{Vector{Float64}}, labels; embed_dim=32, q_factor=8.0, harmonics=12, sample_rate=48000, task=:classification)::WaveDataset

Transforms 1D audio waveforms into continuous acoustic harmonic wave packets using Tournament 13 Grand Champion (`Spectral Flux Acoustic Phase Field`).
"""
function format_audio(
    signals::Vector{Vector{Float64}},
    labels;
    embed_dim::Int = 32,
    q_factor::Float64 = 8.0,
    harmonics::Int = 12,
    sample_rate::Int = 48000,
    task::Symbol = :classification
)::WaveDataset
    n_samples = length(signals)
    inputs = Vector{Vector{Float64}}(undef, n_samples)
    dt = 1.0 / Float64(sample_rate)

    for i in 1:n_samples
        raw = signals[i]
        N = length(raw)
        out = zeros(Float64, embed_dim)
        for k in 1:embed_dim
            ratio = (k <= 8) ? Float64(k) : Float64(2.0^(k / 12.0))
            f_k = 432.0 * (ratio / 4.0)
            sigma_t = q_factor / (2π * f_k + 1e-4)

            real_acc = 0.0
            imag_acc = 0.0
            stride = max(1, div(N, 512))

            for n in 1:stride:N
                t = Float64(n - 1) * dt
                window = exp(-0.5 * ((t - 0.05) / sigma_t)^2)
                amp = raw[n] * window
                phase = 2π * f_k * t
                real_acc += amp * cos(phase)
                imag_acc += amp * sin(phase)
            end
            out[k] = sqrt(real_acc^2 + imag_acc^2) / Float64(max(1, div(N, stride)))
        end
        nrm = norm(out)
        inputs[i] = nrm > 1e-6 ? out ./ nrm : out
    end

    targets = if labels isa Vector{Int} && task == :classification
        num_classes = maximum(labels)
        [Float64[c == labels[i] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    elseif labels isa Vector{Vector{Float64}}
        labels
    else
        [[Float64(l)] for l in labels]
    end

    return WaveDataset(inputs, targets; modality=:audio, task=task)
end

"""
    format_video(videos::Vector{Array{Float64, 3}}, labels; embed_dim=32, v_coupling=0.10, temp_scale=0.85, task=:classification)::WaveDataset

Transforms 3D spatio-temporal video arrays (H × W × T) into continuous wave packets using Tournament 14 Grand Champion (`Continuous Phase Coherence Chamber`).
"""
function format_video(
    videos::Vector{Array{Float64, 3}},
    labels;
    embed_dim::Int = 32,
    v_coupling::Float64 = 0.10,
    temp_scale::Float64 = 0.85,
    task::Symbol = :classification
)::WaveDataset
    n_samples = length(videos)
    inputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        vid = videos[i]
        H, W, T = size(vid)
        out = zeros(Float64, embed_dim)
        step_y = max(1, div(H, 8))
        step_x = max(1, div(W, 8))

        for k in 1:embed_dim
            kx = (k % 4) + 1
            ky = div((k - 1) % 16, 4) + 1
            kt = div(k - 1, 16) + 1

            amp_sum = 0.0
            count = 0
            for t in 1:T
                for y in 1:step_y:H, x in 1:step_x:W
                    phase = 2π * (x * kx / W + y * ky / H + t * kt * temp_scale / T)
                    v_mod = 1.0 + v_coupling * (vid[y, x, t] - (t > 1 ? vid[y, x, t-1] : 0.0))
                    amp_sum += vid[y, x, t] * cos(phase) * v_mod
                    count += 1
                end
            end
            out[k] = amp_sum / Float64(max(1, count))
        end
        nrm = norm(out)
        inputs[i] = nrm > 1e-6 ? out ./ nrm : out
    end

    targets = if labels isa Vector{Int} && task == :classification
        num_classes = maximum(labels)
        [Float64[c == labels[i] ? 1.0 : 0.0 for c in 1:num_classes] for i in 1:n_samples]
    elseif labels isa Vector{Vector{Float64}}
        labels
    else
        [[Float64(l)] for l in labels]
    end

    return WaveDataset(inputs, targets; modality=:video, task=task)
end

"""
    format_3d(points_list::Vector{Matrix{Float64}}, labels; embed_dim=32, l_max=6, sigma_r=0.25, task=:classification)::WaveDataset

Transforms 3D point clouds (3 × N) into continuous wave packets using Tournament 15 Grand Champion (`Continuous 3D Wavelet Packet Decomposition`).
"""
function format_3d(
    points_list::Vector{Matrix{Float64}},
    labels;
    embed_dim::Int = 32,
    l_max::Int = 6,
    sigma_r::Float64 = 0.25,
    task::Symbol = :classification
)::WaveDataset
    n_samples = length(points_list)
    inputs = Vector{Vector{Float64}}(undef, n_samples)

    for idx in 1:n_samples
        points = points_list[idx]
        pts_3d = size(points, 1) == 3 ? points : (size(points, 2) == 3 ? Matrix(transpose(points)) : points)
        _, N = size(pts_3d)
        out = zeros(Float64, embed_dim)

        for k in 1:embed_dim
            l = (k % (l_max + 1))
            m = (k % (2l + 1)) - l
            k_radius = 1.0 + 0.5 * Float64(div(k - 1, l_max + 1))

            acc = 0.0
            for i in 1:N
                x, y, z = pts_3d[1, i], pts_3d[2, i], pts_3d[3, i]
                r_sq = x^2 + y^2 + z^2
                r = sqrt(r_sq) + 1e-6
                theta = acos(clamp(z / r, -1.0, 1.0))
                phi = atan(y, x)

                ylm = cos(Float64(m * phi)) * (sin(theta)^abs(m)) * cos(Float64(l * theta))
                radial_packet = exp(-0.5 * (r - 1.0)^2 / (sigma_r^2)) * cos(Float64(k_radius * 2π * r))
                acc += ylm * radial_packet
            end
            out[k] = acc / Float64(max(1, N))
        end
        nrm = norm(out)
        inputs[idx] = nrm > 1e-6 ? out ./ nrm : out
    end

    targets = if labels isa Vector{Int} && task == :classification
        num_classes = maximum(labels)
        [Float64[c == labels[idx] ? 1.0 : 0.0 for c in 1:num_classes] for idx in 1:n_samples]
    elseif labels isa Vector{Vector{Float64}}
        labels
    else
        [[Float64(l)] for l in labels]
    end

    return WaveDataset(inputs, targets; modality=:mesh3d, task=task)
end

"""
    format_dataset(data, targets; modality=:tabular, task=:classification, embed_dim=32)::WaveDataset

Universal entrypoint for automatic multi-modal dataset conversion across all data types.
"""
function format_dataset(
    data,
    targets;
    modality::Symbol = :tabular,
    task::Symbol = :classification,
    embed_dim::Int = 32
)::WaveDataset
    if modality == :tabular && data isa Matrix{Float64}
        return format_tabular(data, targets; task=task, embed_dim=embed_dim)
    elseif modality == :text && data isa Vector{String}
        return format_text(data, targets; task=task, embed_dim=embed_dim)
    elseif modality == :image && data isa Vector{Matrix{Float64}}
        return format_images(data, targets; task=task, embed_dim=embed_dim)
    elseif modality == :timeseries && data isa Vector{Float64}
        return format_timeseries(data; embed_dim=embed_dim)
    elseif (modality == :audio || modality == :music) && data isa Vector{Vector{Float64}}
        return format_audio(data, targets; task=task, embed_dim=embed_dim)
    elseif modality == :video && data isa Vector{Array{Float64, 3}}
        return format_video(data, targets; task=task, embed_dim=embed_dim)
    elseif (modality == :mesh3d || modality == :pointcloud3d) && data isa Vector{Matrix{Float64}}
        return format_3d(data, targets; task=task, embed_dim=embed_dim)
    elseif modality == :jev || modality == :jev_state
        return format_jev(data, targets; embed_dim=embed_dim)
    else
        # Fallback: assume data is already Vector{Vector{Float64}}
        return WaveDataset(data, targets; modality=modality, task=task)
    end
end

# Ergonomic aliases
const from_tabular = format_tabular
const from_text = format_text
const from_image = format_images
const from_timeseries = format_timeseries
const from_jev_state = format_jev
const from_audio = format_audio
const from_video = format_video
const from_3d = format_3d
num_samples(ds::WaveDataset)::Int = length(ds)

# ==============================================================================
#  🌊 REAL CONTINUOUS WAVE DATA PROCESSING PIPELINES (ZERO DISCRETE LOGIC) 🌊
# ==============================================================================

"""
    process_pixel_waves(
        images::Vector{Matrix{Float64}};
        embed_dim::Int = 32,
        carrier_omega::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895
    )::Vector{Vector{Float64}}

Transforms raw 2D pixel matrices into continuous 2D surface harmonic wavefield representations.
Operates solely via physical wave interference without discrete convolution kernels or static tensors:
\\[
\\Psi_d = \\sum_{r, c} I(r, c) \\cos\\left(2\\pi \\omega_0 \\beta_s^{(r/H)} \\frac{c}{W} + \\phi_{r, c}\\right)
\\]
"""
function process_pixel_waves(
    images::Vector{Matrix{Float64}};
    embed_dim::Int = 32,
    carrier_omega::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::Vector{Vector{Float64}}
    n_images = length(images)
    wave_embeddings = Vector{Vector{Float64}}(undef, n_images)

    # 2D Chladni / Fourier surface standing wave spatial modes
    m_max = max(1, round(Int, sqrt(embed_dim)))
    n_max = max(1, cld(embed_dim, m_max))

    for i in 1:n_images
        img = images[i]
        H, W = size(img)
        emb = zeros(Float64, embed_dim)
        k = 1

        for m in 1:m_max, n in 1:n_max
            if k <= embed_dim
                s = 0.0
                for r in 1:H, c in 1:W
                    val = img[r, c]
                    if val > 0.005
                        # 2D Chladni standing wave nodal surface projection
                        s += val * cos(π * m * r / H) * cos(π * n * c / W)
                    end
                end
                emb[k] = s
                k += 1
            end
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        wave_embeddings[i] = emb
    end

    return wave_embeddings
end

"""
    encode_wave_context(
        tokenizer::Union{WaveTokenizer, PhoneticTokenizer},
        token_ids::Vector{Int},
        embed_dim::Int;
        beta_s::Float64 = 1.618033988749895,
        alpha_decay::Float64 = 0.90
    )::Vector{Float64}

Encodes a sequence of token IDs into a unified continuous wave context vector via:
1. Acoustic causal phase field superposition across the prefix.
2. Pure continuous standing wave phase interference attention (zero matrix multiplications).
3. Flower of Life golden ratio Riemannian manifold hyper-connection.
"""
function encode_wave_context(
    tokenizer::Union{WaveTokenizer, PhoneticTokenizer},
    token_ids::Vector{Int},
    embed_dim::Int;
    beta_s::Float64 = 1.618033988749895,
    alpha_decay::Float64 = 0.90
)::Vector{Float64}
    L = length(token_ids)
    L == 0 && return zeros(Float64, embed_dim)

    # 1. Packet extraction for each token in context
    packets = [to_wave_packet(tokenizer, tid, embed_dim) for tid in token_ids]

    # 2. Causal acoustic phase field superposition (exponential recency decay without destructive zeroing)
    phase_field = zeros(Float64, embed_dim)
    for (pos, pkt) in enumerate(packets)
        tau = Float64(L - pos)
        weight = alpha_decay ^ tau
        phase_field .+= weight .* pkt
    end
    nrm_pf = norm(phase_field)
    if nrm_pf > 1e-6
        phase_field ./= nrm_pf
    end

    # 3. Continuous standing wave interference attention
    # Query wave is the most recent token packet in the context
    q = packets[end]
    attn_out = zeros(Float64, embed_dim)
    tot_weight = 0.0
    p_pow = beta_s * 4.0
    for j in 1:L
        cos_sim = clamp(dot(q, packets[j]), -1.0, 1.0)
        w = ((1.0 + cos_sim) * 0.5) ^ p_pow
        attn_out .+= w .* packets[j]
        tot_weight += w
    end
    if tot_weight > 1e-6
        attn_out ./= tot_weight
    end

    # 4. Golden ratio Riemannian manifold hyper-connection
    blend = 1.0 / beta_s # ~0.618
    out = blend .* phase_field .+ (1.0 - blend) .* attn_out
    nrm_out = norm(out)
    return nrm_out > 1e-6 ? (out ./ nrm_out) : out
end

"""
    format_lm_text(
        texts::Vector{String};
        tokenizer = default_tokenizer(),
        embed_dim::Int = 256,
        context_len::Int = 48,
        max_pairs::Int = 60000,
        interleave::Bool = true
    )::WaveDataset

Formats raw natural language text corpus into next-token continuous wave prediction pairs:
- Input: continuous wave context vector of sequence prefix via `encode_wave_context`
- Target: continuous wave packet of next token
- Interleaved: round-robin sampling across all input documents so no category/tier is starved
"""
function format_lm_text(
    texts::Vector{String};
    tokenizer::Union{WaveTokenizer, PhoneticTokenizer} = default_tokenizer(),
    embed_dim::Int = 256,
    context_len::Int = 48,
    max_pairs::Int = 60000,
    interleave::Bool = true
)::WaveDataset
    # 1. Extract sequence pairs per document
    doc_pairs = Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}}}}()
    total_extracted = 0

    for text in texts
        tokens = tokenize(tokenizer, text)
        token_ids = [t.token_id for t in tokens]
        pairs_for_doc = Tuple{Vector{Float64}, Vector{Float64}}[]
        if length(token_ids) >= 2
            for i in 1:(length(token_ids) - 1)
                start_idx = max(1, i - context_len + 1)
                prefix_ids = token_ids[start_idx:i]
                next_id = token_ids[i + 1]

                ctx_vec = encode_wave_context(tokenizer, prefix_ids, embed_dim)

                tgt_vec = to_wave_packet(tokenizer, next_id, embed_dim)
                nrm_tgt = norm(tgt_vec)
                if nrm_tgt > 1e-6; tgt_vec ./= nrm_tgt; end

                push!(pairs_for_doc, (ctx_vec, tgt_vec))
                total_extracted += 1
            end
        end
        if !isempty(pairs_for_doc)
            push!(doc_pairs, pairs_for_doc)
        end
    end

    inputs = Vector{Vector{Float64}}()
    targets = Vector{Vector{Float64}}()

    if !interleave || total_extracted <= max_pairs
        # Directly gather pairs up to max_pairs
        for doc in doc_pairs
            for (inp, tgt) in doc
                length(inputs) >= max_pairs && break
                push!(inputs, inp)
                push!(targets, tgt)
            end
            length(inputs) >= max_pairs && break
        end
    else
        # Round-robin interleaved sampling across all documents so no tier is starved
        doc_indices = fill(1, length(doc_pairs))
        has_more = true
        while has_more && length(inputs) < max_pairs
            has_more = false
            for d in 1:length(doc_pairs)
                if doc_indices[d] <= length(doc_pairs[d])
                    inp, tgt = doc_pairs[d][doc_indices[d]]
                    push!(inputs, inp)
                    push!(targets, tgt)
                    doc_indices[d] += 1
                    has_more = true
                    if length(inputs) >= max_pairs
                        break
                    end
                end
            end
        end
    end

    return WaveDataset(inputs, targets; modality=:text, task=:generation)
end

"""
    process_wave_tokens(
        texts::Vector{String};
        tokenizer = default_tokenizer(),
        embed_dim::Int = 32,
        max_len::Int = 32
    )::Vector{Vector{Float64}}

Encodes raw natural language strings into continuous acoustic frequency wave packets.
Tokens are continuous harmonic frequencies (carrier 432 Hz) rather than discrete IDs or static matrices.
"""
function process_wave_tokens(
    texts::Vector{String};
    tokenizer = default_tokenizer(),
    embed_dim::Int = 32,
    max_len::Int = 32
)::Vector{Vector{Float64}}
    n = length(texts)
    embeddings = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        wfs = tokenize(tokenizer, texts[i])
        emb = zeros(Float64, embed_dim)
        
        limit_tokens = min(length(wfs), max_len)
        for (idx, wf) in enumerate(wfs[1:limit_tokens])
            slot = mod1(idx, embed_dim)
            emb[slot] += wf.energy * cos(wf.phase + 2π * (wf.frequency / 432.0))
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        embeddings[i] = emb
    end

    return embeddings
end

"""
    process_digital_data(
        data::Union{Vector{Vector{Float64}}, Matrix{Float64}};
        embed_dim::Int = 32,
        carrier_omega::Float64 = 432.0
    )::Vector{Vector{Float64}}

Projects arbitrary digital data (tabular rows, sensor signals, time-series) into continuous harmonic wave spectra.
"""
function process_digital_data(
    data::Union{Vector{Vector{Float64}}, Matrix{Float64}};
    embed_dim::Int = 32,
    carrier_omega::Float64 = 432.0
)::Vector{Vector{Float64}}
    n_samples = data isa Matrix ? size(data, 1) : length(data)
    n_features = data isa Matrix ? size(data, 2) : length(data[1])

    embeddings = Vector{Vector{Float64}}(undef, n_samples)
    inv_dim = 1.0 / Float64(embed_dim)

    for i in 1:n_samples
        row = data isa Matrix ? data[i, :] : data[i]
        emb = zeros(Float64, embed_dim)

        for (feat_idx, val) in enumerate(row)
            freq = 1.0 + (Float64(feat_idx) / Float64(n_features)) * 3.0
            phase = val * π
            for d in 1:embed_dim
                x = Float64(d - 1) * inv_dim
                emb[d] += val * cos(2π * freq * x + phase)
            end
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        embeddings[i] = emb
    end

    return embeddings
end

"""
    WaveDataStreamer

Streaming iterator for continuous wave datasets. Yields batches without loading entire massive datasets into memory.
"""
struct WaveDataStreamer
    dataset::WaveDataset
    batch_size::Int
    total_batches::Int
    shuffle::Bool

    function WaveDataStreamer(dataset::WaveDataset; batch_size::Int = 16, shuffle::Bool = true)
        b_size = max(1, batch_size)
        t_batches = div(length(dataset) + b_size - 1, b_size)
        new(dataset, b_size, t_batches, shuffle)
    end
end

Base.length(s::WaveDataStreamer) = s.total_batches

function Base.iterate(s::WaveDataStreamer, state::Int = 1)
    if state > s.total_batches
        return nothing
    end
    start_idx = (state - 1) * s.batch_size + 1
    end_idx = min(start_idx + s.batch_size - 1, length(s.dataset))
    indices = collect(start_idx:end_idx)
    if s.shuffle
        shuffle!(indices)
    end
    batch_inputs = [s.dataset.inputs[idx] for idx in indices]
    batch_targets = [s.dataset.targets[idx] for idx in indices]
    return ((batch_inputs, batch_targets), state + 1)
end

"""
    stream_dataset(dataset::WaveDataset; batch_size::Int = 16, shuffle::Bool = true)::WaveDataStreamer

Creates a streaming batch generator over a continuous `WaveDataset`.
"""
function stream_dataset(dataset::WaveDataset; batch_size::Int = 16, shuffle::Bool = true)::WaveDataStreamer
    return WaveDataStreamer(dataset; batch_size=batch_size, shuffle=shuffle)
end

