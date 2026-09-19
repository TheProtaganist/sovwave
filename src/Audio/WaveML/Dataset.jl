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

export WaveDataset, WaveDataLoader
export format_tabular, format_text, format_images, format_timeseries, format_jev, format_dataset
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
    inputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        img = images[i]
        h, w = size(img)
        emb = zeros(Float64, embed_dim)

        # 2D surface spatial harmonic projection
        for r in 1:min(h, 16), c in 1:min(w, 16)
            pixel_val = img[r, c]
            ph = 2π * Float64(r) / Float64(h)
            freq = 1.0 + Float64(c) / Float64(w)

            for d in 1:embed_dim
                emb[d] += pixel_val * cos(2π * freq * (Float64(d) / embed_dim) + ph)
            end
        end

        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end
        inputs[i] = emb
    end

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
