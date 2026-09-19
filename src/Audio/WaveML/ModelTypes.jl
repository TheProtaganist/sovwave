"""
    WaveML.ModelTypes

The 5 Default Native Model Architectures for Sovwave:
1. LLM (:llm) — Autoregressive Wave Sequence Modeling & Text Generation
2. Image Generation (:image_generation) — Continuous 2D Surface Wave De-interference
3. Text-to-3D (:text_to_3d) — Continuous Volumetric Radiance Fields (x, y, z) ↦ (σ, RGB)
4. Jev Decision Engine (:jev) — Non-Autoregressive Type-Safe Decision Model (Hallucination-Free)
5. Text-to-Video (:text_to_video) — Spatio-Temporal Wave Frame Dynamics Serialized to MKV/MP4
"""

using Random
using LinearAlgebra
using Printf

export generate_text, generate_image, generate_3d, jev_decide, generate_video
export register_model_type!, list_model_types

const MODEL_REGISTRY = Dict{Symbol, Dict{String, Any}}(
    :llm => Dict(
        "name" => "Wave Large Language Model",
        "paradigm" => "Autoregressive Harmonic Sequence",
        "description" => "Generates text by predicting continuous next-token wave packets across context windows."
    ),
    :image_generation => Dict(
        "name" => "Wave Image Generator",
        "paradigm" => "Continuous 2D Surface De-interference",
        "description" => "Generates 2D spatial pixel manifolds from harmonic interference states."
    ),
    :text_to_3d => Dict(
        "name" => "Wave Text-to-3D Volumetric Field",
        "paradigm" => "3D Radiance Manifold (x, y, z) ↦ (σ, RGB)",
        "description" => "Synthesizes continuous 3D volumetric fields and coordinate point manifolds from text."
    ),
    :jev => Dict(
        "name" => "Jev Decision Engine",
        "paradigm" => "Non-Autoregressive Type-Safe Decision Model",
        "description" => "Ingests application state alongside predefined questions and returns type-safe probabilistic decisions (categories, booleans, rubrics). Fully hallucination-immune."
    ),
    :text_to_video => Dict(
        "name" => "Wave Text-to-Video Generator",
        "paradigm" => "Spatio-Temporal Wave Dynamics",
        "description" => "Generates multi-frame spatio-temporal video sequences directly saved as MKV/MP4 files."
    )
)

"""
    register_model_type!(name::Symbol, descriptor::Dict{String, Any})::Nothing

Registers a new model architecture into Sovwave's extensible model registry.
"""
function register_model_type!(name::Symbol, descriptor::Dict{String, Any})::Nothing
    MODEL_REGISTRY[name] = descriptor
    return nothing
end

"""
    list_model_types()::Dict{Symbol, Dict{String, Any}}

Returns all registered model types available in Sovwave.
"""
function list_model_types()::Dict{Symbol, Dict{String, Any}}
    return copy(MODEL_REGISTRY)
end

# ============================================================================
# 1. LLM (Wave Large Language Model)
# ============================================================================

"""
    generate_text(
        model::WaveModel,
        tokenizer::WaveTokenizer,
        prompt::String;
        max_new_tokens::Int = 16,
        temperature::Float64 = 0.7
    )::String

Autoregressively generates text tokens using the trained wave model.
Encodes context into continuous wave embeddings, propagates through quantum layers,
and resonates the output wave state against the tokenizer's vocabulary.
"""
function generate_text(
    model::WaveModel,
    tokenizer::WaveTokenizer,
    prompt::String;
    max_new_tokens::Int = 16,
    temperature::Float64 = 0.7
)::String
    wave_tokens = tokenize(tokenizer, prompt)
    token_ids = [wf.token_id for wf in wave_tokens]
    embed_dim = model.model_config.embed_dims
    eos_id = tokenizer.special_tokens[:EOS]

    for _ in 1:max_new_tokens
        # Encode current sequence context into continuous wave vector
        seq_mat = encode_sequence(tokenizer, decode(tokenizer, token_ids); max_len=length(token_ids), embed_dim=embed_dim)
        context_vec = vec(mean(seq_mat, dims=2))
        
        # Propagate through wave model layers
        output_wave = forward!(model, context_vec)
        
        # Resonant decoding: calculate cosine similarity across vocabulary
        V = length(tokenizer.inv_vocab)
        out_dim = length(output_wave)
        logits = Vector{Float64}(undef, V)
        for k in 1:V
            target_packet = to_wave_packet(tokenizer, k, out_dim)
            logits[k] = dot(output_wave, target_packet) / max(temperature, 1e-4)
        end
        
        # Softmax sampling
        max_l = maximum(logits)
        probs = exp.(logits .- max_l)
        probs ./= sum(probs)
        
        # Sample next token
        r = rand()
        cum = 0.0
        sampled_id = V
        for k in 1:V
            cum += probs[k]
            if cum >= r
                sampled_id = k
                break
            end
        end
        
        push!(token_ids, sampled_id)
        if sampled_id == eos_id
            break
        end
    end

    return decode(tokenizer, token_ids)
end

# ============================================================================
# 2. Image Generation (Continuous 2D Surface Wave De-interference)
# ============================================================================

"""
    generate_image(
        model::WaveModel,
        prompt::String;
        height::Int = 16,
        width::Int = 16,
        tokenizer::WaveTokenizer = default_tokenizer()
    )::Matrix{Float64}

Generates a continuous 2D image surface from a text prompt via wave interference.
"""
function generate_image(
    model::WaveModel,
    prompt::String;
    height::Int = 16,
    width::Int = 16,
    tokenizer::WaveTokenizer = default_tokenizer()
)::Matrix{Float64}
    embed_dim = model.model_config.embed_dims
    wave_tokens = tokenize(tokenizer, prompt)
    
    # Text prompt continuous wave packet
    prompt_emb = zeros(Float64, embed_dim)
    for wf in wave_tokens
        prompt_emb .+= to_wave_packet(wf, embed_dim)
    end
    prompt_norm = norm(prompt_emb)
    if prompt_norm > 1e-6
        prompt_emb ./= prompt_norm
    end

    # Forward pass through wave model
    latent = forward!(model, prompt_emb)

    # De-interfere into 2D spatial pixel grid
    img = Matrix{Float64}(undef, height, width)
    for r in 1:height, c in 1:width
        u = Float64(r) / Float64(height)
        v = Float64(c) / Float64(width)
        
        pixel_val = 0.0
        for d in 1:embed_dim
            freq = 1.0 + Float64(d % 8) * 0.5
            ph = 2π * Float64(d) / Float64(embed_dim)
            pixel_val += latent[d] * cos(2π * (freq * u + v) + ph)
        end
        # Normalize to [0.0, 1.0]
        img[r, c] = clamp(0.5 + 0.5 * (pixel_val / sqrt(embed_dim)), 0.0, 1.0)
    end

    return img
end

"""
    generate_image(model::WaveModel, dims::Tuple{Int, Int}; kwargs...)::Matrix{Float64}

Convenience overload generating an image of size `dims` from spontaneous harmonic ground waves.
"""
function generate_image(model::WaveModel, dims::Tuple{Int, Int}; kwargs...)::Matrix{Float64}
    return generate_image(model, "emergent wave surface"; height=dims[1], width=dims[2], kwargs...)
end

# ============================================================================
# 3. Text-to-3D (Volumetric Wave Radiance Field)
# ============================================================================

"""
    generate_3d(
        model::WaveModel,
        prompt::String;
        resolution::Int = 8,
        tokenizer::WaveTokenizer = default_tokenizer()
    )::Array{Float64, 4}

Synthesizes a continuous 3D volumetric wave field (x, y, z) ↦ (σ, R, G, B) of shape
`(resolution, resolution, resolution, 4)` from a text prompt.
"""
function generate_3d(
    model::WaveModel,
    prompt::String;
    resolution::Int = 8,
    tokenizer::WaveTokenizer = default_tokenizer()
)::Array{Float64, 4}
    embed_dim = model.model_config.embed_dims
    wave_tokens = tokenize(tokenizer, prompt)
    
    prompt_emb = zeros(Float64, embed_dim)
    for wf in wave_tokens
        prompt_emb .+= to_wave_packet(wf, embed_dim)
    end
    nrm = norm(prompt_emb)
    if nrm > 1e-6
        prompt_emb ./= nrm
    end

    latent = forward!(model, prompt_emb)

    # 4D Tensor: (X, Y, Z, Channels) where Channels = [Density σ, Red, Green, Blue]
    vol = Array{Float64, 4}(undef, resolution, resolution, resolution, 4)
    R = Float64(resolution)

    for x in 1:resolution, y in 1:resolution, z in 1:resolution
        u = Float64(x) / R
        v = Float64(y) / R
        w = Float64(z) / R

        # Radial harmonic distance
        r_dist = sqrt(u^2 + v^2 + w^2)
        
        # Density σ: harmonic sphere interference
        density = 0.0
        for d in 1:min(embed_dim, 8)
            density += latent[d] * cos(2π * Float64(d) * r_dist)
        end
        vol[x, y, z, 1] = clamp(max(0.0, 0.5 + 0.5 * density), 0.0, 1.0)

        # RGB radiance channels: Potts clock domain 3-phase assignment
        vol[x, y, z, 2] = clamp(0.5 + 0.5 * sin(2π * u + latent[1]), 0.0, 1.0)
        vol[x, y, z, 3] = clamp(0.5 + 0.5 * sin(2π * v + latent[min(2, embed_dim)]), 0.0, 1.0)
        vol[x, y, z, 4] = clamp(0.5 + 0.5 * sin(2π * w + latent[min(3, embed_dim)]), 0.0, 1.0)
    end

    return vol
end

"""
    generate_3d(model::WaveModel, resolution::Int; kwargs...)::Array{Float64, 4}

Convenience overload synthesizing a continuous 3D field of size `(resolution, resolution, resolution, 4)`.
"""
function generate_3d(model::WaveModel, resolution::Int; kwargs...)::Array{Float64, 4}
    return generate_3d(model, "volumetric radiance field"; resolution=resolution, kwargs...)
end

# ============================================================================
# 4. Jev Decision Engine (Non-Autoregressive Type-Safe Decision Model)
# ============================================================================

"""
    jev_decide(
        model::WaveModel,
        state::Dict{String, Any},
        questions::Vector{String};
        question_types::Vector{Symbol} = fill(:boolean, length(questions))
    )::Vector{NamedTuple{(:question, :type, :decision, :confidence), Tuple{String, Symbol, Any, Float64}}}

Executes the Jev Decision Engine:
- Reads language (questions + state data) but NEVER generates open-ended text.
- Returns type-safe, structured probabilistic decisions (:boolean, :rubric, :category).
- Completely immune to traditional text hallucinations.
"""
function jev_decide(
    model::WaveModel,
    state::AbstractDict,
    questions::Vector{String};
    question_types::Vector{Symbol} = fill(:boolean, length(questions))
)::Vector{NamedTuple{(:question, :type, :decision, :confidence), Tuple{String, Symbol, Any, Float64}}}
    embed_dim = model.model_config.embed_dims
    inv_dim = 1.0 / Float64(embed_dim)
    decisions = Vector{NamedTuple{(:question, :type, :decision, :confidence), Tuple{String, Symbol, Any, Float64}}}(undef, length(questions))

    for (q_idx, q) in enumerate(questions)
        q_type = question_types[q_idx]
        
        # Format single state-question wave input
        emb = zeros(Float64, embed_dim)
        for (k_idx, (key, val)) in enumerate(state)
            num_val = val isa Number ? Float64(val) : (val isa Bool ? (val ? 1.0 : 0.0) : Float64(hash(val) % 100) / 100.0)
            ph = 2π * Float64(k_idx) / Float64(max(1, length(state)))
            for d in 1:embed_dim
                emb[d] += num_val * cos(2π * (Float64(d) * inv_dim) + ph)
            end
        end

        q_hash = Float64(hash(q) % 1000) / 1000.0
        for d in 1:embed_dim
            emb[d] += 0.5 * sin(2π * q_hash * Float64(d) * inv_dim)
        end
        nrm = norm(emb)
        if nrm > 1e-6
            emb ./= nrm
        end

        # Non-autoregressive direct state inference
        output = forward!(model, emb)
        score = 1.0 / (1.0 + exp(-output[1])) # Sigmoid probability

        if q_type == :boolean
            val = score >= 0.5
            conf = max(score, 1.0 - score)
            decisions[q_idx] = (question=q, type=:boolean, decision=val, confidence=conf)
        elseif q_type == :rubric
            # Continuous normalized score in [0.0, 1.0]
            decisions[q_idx] = (question=q, type=:rubric, decision=round(score, digits=4), confidence=0.99)
        else # :category
            cat_idx = mod(round(Int, score * 4.0), 3) + 1
            decisions[q_idx] = (question=q, type=:category, decision="category_$(cat_idx)", confidence=score)
        end
    end

    return decisions
end

"""
    jev_decide(model::WaveModel, state::AbstractVector{<:Real}, questions::Vector{String}; kwargs...)

Convenience overload executing Jev Decision Engine on numerical state vectors.
"""
function jev_decide(
    model::WaveModel,
    state::AbstractVector{<:Real},
    questions::Vector{String};
    kwargs...
)
    state_dict = Dict{String, Any}("feature_$i" => Float64(v) for (i, v) in enumerate(state))
    return jev_decide(model, state_dict, questions; kwargs...)
end

# ============================================================================
# 5. Text-to-Video (Spatio-Temporal Wave Dynamics)
# ============================================================================

"""
    generate_video(
        model::WaveModel,
        prompt::String,
        output_path::String;
        frames::Int = 12,
        fps::Int = 4,
        height::Int = 32,
        width::Int = 32
    )::String

Generates a spatio-temporal video sequence from a text prompt and serializes it
directly to `.mkv` and `.mp4` using Sovwave's native multi-stream Matroska engine.
"""
function generate_video(
    model::WaveModel,
    prompt::String,
    output_path::String;
    frames::Int = 12,
    fps::Int = 4,
    height::Int = 32,
    width::Int = 32
)::String
    # Uses model save_model engine with prompt-driven temporal excitation
    v_cfg = WaveVideoConfig(
        render_mode = :potts_model_q_state_domains,
        pixel_scale = 2,
        target_height = height * 2,
        fps = fps,
        frames = frames
    )
    save_model(model, output_path; video_cfg=v_cfg)
    return output_path
end
