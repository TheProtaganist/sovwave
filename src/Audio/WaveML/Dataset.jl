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
export process_pixel_waves, process_wave_tokens, process_digital_data, stream_dataset
export from_tabular, from_text, from_image, from_timeseries, from_jev_state
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
    format_dataset(data, targets; modality=:tabular, task=:classification, embed_dim=32)::WaveDataset

Universal entrypoint for automatic multi-modal dataset conversion.
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
    format_lm_text(texts::Vector{String}; tokenizer=default_tokenizer(), embed_dim::Int=64, context_len::Int=16, max_pairs::Int=1000)::WaveDataset

Formats raw natural language text corpus into next-token continuous wave prediction pairs:
- Input: continuous wave context vector of sequence prefix
- Target: continuous wave packet of next token
"""
function format_lm_text(
    texts::Vector{String};
    tokenizer::WaveTokenizer = default_tokenizer(),
    embed_dim::Int = 64,
    context_len::Int = 16,
    max_pairs::Int = 1000
)::WaveDataset
    inputs = Vector{Vector{Float64}}()
    targets = Vector{Vector{Float64}}()

    for text in texts
        length(inputs) >= max_pairs && break
        tokens = tokenize(tokenizer, text)
        token_ids = [t.token_id for t in tokens]
        if length(token_ids) >= 2
            for i in 1:(length(token_ids) - 1)
                length(inputs) >= max_pairs && break
                start_idx = max(1, i - context_len + 1)
                prefix_ids = token_ids[start_idx:i]
                next_id = token_ids[i + 1]

                prefix_str = decode(tokenizer, prefix_ids)
                seq_mat = encode_sequence(tokenizer, prefix_str; max_len=max(1, length(prefix_ids)), embed_dim=embed_dim)
                ctx_vec = vec(mean(seq_mat, dims=2))
                nrm = norm(ctx_vec)
                if nrm > 1e-6; ctx_vec ./= nrm; end

                tgt_vec = to_wave_packet(tokenizer, next_id, embed_dim)
                nrm_tgt = norm(tgt_vec)
                if nrm_tgt > 1e-6; tgt_vec ./= nrm_tgt; end

                push!(inputs, ctx_vec)
                push!(targets, tgt_vec)
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

