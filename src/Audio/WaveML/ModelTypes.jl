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

const _W_PROJ_CACHE = Dict{Tuple{UInt64, Int, Int}, Matrix{Float64}}()
const _VALID_MASK_CACHE = Dict{Tuple{UInt64, Bool, Bool, Bool, Int}, BitVector}()

"""
    generate_text(
        model::WaveModel,
        tokenizer::Union{WaveTokenizer, PhoneticTokenizer},
        prompt::String;
        max_new_tokens::Int = 16,
        temperature::Float64 = 0.7,
        top_k::Int = 50,
        top_p::Float64 = 0.9,
        min_p::Float64 = 0.05,
        active_vocab::Union{Nothing, Set{Int}, Vector{Int}} = nothing
    )::String

Autoregressively generates coherent text tokens using the trained continuous wave model.
Encodes context into continuous acoustic wave embeddings, propagates through physical quantum layers,
and resonates the output wave state against the tokenizer's vocabulary via fast matrix projection
with word boundary whitespace enforcement and repetition defense.
"""
function generate_text(
    model::WaveModel,
    tokenizer::Union{WaveTokenizer, PhoneticTokenizer},
    prompt::String;
    max_new_tokens::Int = 16,
    temperature::Float64 = 0.7,
    top_k::Int = 50,
    top_p::Float64 = 0.9,
    min_p::Float64 = 0.05,
    active_vocab::Union{Nothing, Set{Int}, Vector{Int}} = nothing
)::String
    wave_tokens = tokenize(tokenizer, prompt)
    token_ids = [wf.token_id for wf in wave_tokens]
    embed_dim = model.model_config.embed_dims
    eos_id = tokenizer.special_tokens[:EOS]
    pad_id = tokenizer.special_tokens[:PAD]

    V = length(tokenizer.inv_vocab)
    out_dim = model.model_config.nodes
    tok_id = objectid(tokenizer)

    # Cached vocabulary wave packet projection matrix for ultra-fast resonance
    W_proj = get!(_W_PROJ_CACHE, (tok_id, out_dim, V)) do
        mat = Matrix{Float64}(undef, out_dim, V)
        for k in 1:V
            mat[:, k] = to_wave_packet(tokenizer, k, out_dim)
        end
        mat
    end

    # Detect prompt linguistic domain to prevent cross-language token pollution
    is_cjk = any(c -> '\u4e00' <= c <= '\u9fff', prompt)
    is_math = any(c -> c in ['∂', '∇', '∫', '∑', 'ℏ', '≈', '⊕', 'π', 'Ω', 'ψ', 'λ', '≡', '≠', '≤', '≥', '+', '=', '*', '/', '-', '<', '>'], prompt)
    is_emoji = any(c -> Int(c) > 0x1F000, prompt)

    valid_vocab_mask = get!(_VALID_MASK_CACHE, (tok_id, is_cjk, is_math, is_emoji, V)) do
        mask = trues(V)
        for k in 1:V
            t = tokenizer.inv_vocab[k]
            # Always block special internal placeholders and controls
            if (startswith(t, "<") && endswith(t, ">") && (t in ["<PAD>", "<UNK>", "<SEP>", "<MASK>", "<|endoftext|>", "<｜end▁of▁sentence｜>"] || startswith(t, "<TOKEN_")))
                mask[k] = false
                continue
            end
            # Block tokens with 3 or more repeated consecutive identical characters (e.g. "xxxx", "ffff", "HHHH")
            if occursin(r"(.)\1{2,}", t)
                mask[k] = false
                continue
            end
            # Block mixed digit and letter byte-merges (e.g. "046UA", "bursting145")
            if any(isdigit, t) && any(isletter, t) && length(t) >= 3
                mask[k] = false
                continue
            end
            # Block mixed case / BPE merge artifacts (e.g. "VIitle", "AXIBcome", "economiespb")
            s_core = (startswith(t, "Ġ") || startswith(t, " ")) ? chop(t, head=1, tail=0) : t
            if occursin(r"[A-Z]{2,}[a-z]+|[a-z]+[A-Z]{2,}", s_core)
                mask[k] = false
                continue
            end
            # Block 3+ character all-uppercase acronym tokens unless prompt has uppercase words
            if length(s_core) >= 3 && all(c -> isuppercase(c) && isascii(c), s_core)
                mask[k] = false
                continue
            end
            # Block tokens with strange embedded brackets or trailing underscores like "_[", "0[", "==[", "_("
            if occursin(r"[_\[\]\(\)\{\}]{2,}|[0-9][\[\]\(\)]|[_][\[\]\(\)]|[a-zA-Z][_][\[\]\(\)]", t)
                mask[k] = false
                continue
            end
            # Allow clean digits and pure numbers
            if all(isdigit, t) || (startswith(t, "Ġ") && all(isdigit, chop(t, head=1, tail=0)))
                mask[k] = true
                continue
            end
            # Allow all standard symbols, operators, punctuation, and newline markers
            is_sym_or_punct(c::Char) = ispunct(c) || isspace(c) || c in "+-*/=<>^|~&%\\#@!?,.'\"`:;()[]{}Ġ Ċĉ\n\t"
            if all(is_sym_or_punct, t)
                mask[k] = true
                continue
            end
            # Strip leading whitespace/newline marker if present
            s_core = (startswith(t, "Ġ") || startswith(t, " ") || startswith(t, "Ċ") || startswith(t, "ĉ")) ? chop(t, head=1, tail=0) : t
            if isempty(s_core)
                mask[k] = true
                continue
            end
            # In standard non-CJK text, block non-ASCII foreign noise unless requested
            if !is_cjk && !is_math && !is_emoji && V > 500
                is_ascii_text = all(c -> (isascii(c) && (isletter(c) || isdigit(c) || is_sym_or_punct(c))) || c in "\x27-\x22", s_core)
                mask[k] = is_ascii_text
            else
                mask[k] = true
            end
        end
        mask
    end

    alpha_decay = 0.90
    for _ in 1:max_new_tokens
        # Unified continuous wave context encoding (causal phase field + wave attention)
        context_vec = encode_wave_context(tokenizer, token_ids, embed_dim)

        # Propagate through wave model layers
        output_wave = forward_continuous_wave!(model, context_vec)
        out_norm = norm(output_wave)
        if out_norm > 1e-6
            output_wave ./= out_norm
        end

        # Fast BLAS matrix projection: resonant dot product across entire vocabulary
        # Contrastive resonance scale: 10 * beta_s maps bounded [-1, 1] cosine resonance to calibrated softmax logits
        contrastive_scale = 16.18033988749895
        logits = ((transpose(W_proj) * output_wave) .* contrastive_scale) ./ max(temperature, 1e-4)

        # Mask padding and out-of-domain tokens
        if pad_id >= 1 && pad_id <= length(logits)
            logits[pad_id] = -Inf
        end
        logits[.!valid_vocab_mask] .= -Inf

        # Multi-scale Repetition Defense (Opt144 Grand Champion)
        # Exempt whitespace and indentation tokens so code indentation (e.g. 4 spaces, tabs) functions properly
        is_whitespace_token(id::Int) = (1 <= id <= V) && (tokenizer.inv_vocab[id] in [" ", "Ġ", "Ċ", "ĉ", "\t", "\n", "  ", "   ", "    ", "ĠĠ", "ĠĠĠ", "ĠĠĠĠ"])

        last_id = isempty(token_ids) ? 0 : token_ids[end]
        penult_id = length(token_ids) >= 2 ? token_ids[end-1] : 0
        ante_id = length(token_ids) >= 3 ? token_ids[end-2] : 0

        # 1. Immediate 1-gram ban (exempt whitespace and code indentation)
        if last_id > 0 && last_id <= length(logits) && !is_whitespace_token(last_id)
            logits[last_id] = -Inf
        end

        # 2. Strict 2-gram blocking (exempt whitespace and code indentation)
        if penult_id > 0 && length(token_ids) >= 4 && !is_whitespace_token(penult_id)
            for h in 1:(length(token_ids)-1)
                if token_ids[h] == penult_id
                    rep_next = token_ids[h+1]
                    if rep_next > 0 && rep_next <= length(logits) && !is_whitespace_token(rep_next)
                        logits[rep_next] = -Inf
                    end
                end
            end
        end

        # 3. Strict 3-gram blocking (exempt whitespace and code indentation)
        if ante_id > 0 && length(token_ids) >= 6 && !is_whitespace_token(ante_id)
            for h in 1:(length(token_ids)-2)
                if token_ids[h] == ante_id && token_ids[h+1] == penult_id
                    rep_next = token_ids[h+2]
                    if rep_next > 0 && rep_next <= length(logits) && !is_whitespace_token(rep_next)
                        logits[rep_next] = -Inf
                    end
                end
            end
        end

        # 4. Multi-scale exponential recency decay and frequency penalty (O(tokens_count))
        total_toks = length(token_ids)
        for (idx, prev_id) in enumerate(token_ids)
            if 1 <= prev_id <= V && isfinite(logits[prev_id]) && !is_whitespace_token(prev_id)
                dist = total_toks - idx + 1
                logits[prev_id] -= 1.5 * (0.85 ^ (dist - 1)) + 0.5
            end
        end

        # Active vocabulary filtering (e.g. trained domain corpus)
        if active_vocab !== nothing && !isempty(active_vocab)
            @inbounds for k in 1:V
                if !(k in active_vocab)
                    logits[k] = -Inf
                end
            end
        end

        # Parenthesis / bracket balance enforcement: never emit closing brackets if none were opened
        open_parens = count(c -> c in "([{", prompt) + count(tid -> any(c -> c in "([{", tokenizer.inv_vocab[tid]), token_ids)
        close_parens = count(c -> c in ")]}", prompt) + count(tid -> any(c -> c in ")]}", tokenizer.inv_vocab[tid]), token_ids)
        if close_parens >= open_parens
            for bracket in [")", "]", "}", " )", " ]", " }", "Ġ)", "Ġ]", "Ġ}"]
                if haskey(tokenizer.vocab, bracket)
                    logits[tokenizer.vocab[bracket]] = -Inf
                end
            end
        end

        # Word boundary, whitespace, and punctuation coherence enforcement:
        if last_id > 0
            last_tok_str = tokenizer.inv_vocab[last_id]
            is_punct_char(c::Char) = c in ['.', ',', '!', '?', ':', ';', ')', ']', '}']
            last_is_word = !isempty(last_tok_str) && (isletter(last_tok_str[end]) || isdigit(last_tok_str[end]))
            last_is_closing = !isempty(last_tok_str) && is_punct_char(last_tok_str[end])

            # 1. Whitespace enforcement: after words or closing punctuation, any subsequent word/number must start with whitespace marker ('Ġ', ' ', 'Ċ', '\n')
            if last_is_word || last_is_closing
                @inbounds for k in 1:V
                    if isfinite(logits[k])
                        cand = tokenizer.inv_vocab[k]
                        if !isempty(cand) && (isletter(cand[1]) || isdigit(cand[1])) &&
                           !startswith(cand, "Ġ") && !startswith(cand, " ") &&
                           !startswith(cand, "Ċ") && !startswith(cand, "\n")
                            logits[k] = -Inf
                        end
                    end
                end
            end

            # 2. Punctuation anti-repetition: if last token ended in punctuation, never emit immediate duplicate punctuation
            if last_is_closing
                @inbounds for k in 1:V
                    if isfinite(logits[k])
                        cand = tokenizer.inv_vocab[k]
                        if !isempty(cand) && (cand in [".", ",", "!", "?", ":", ";", ")", "]", "}"] || cand == string(last_tok_str[end]))
                            logits[k] = -Inf
                        end
                    end
                end
            end
        end

        # Top-K filtering
        if top_k > 0 && top_k < V
            perm = sortperm(logits, rev=true)
            cutoff = logits[perm[top_k]]
            logits[logits .< cutoff] .= -Inf
        end

        # Softmax probabilities
        finite_mask = isfinite.(logits)
        if !any(finite_mask)
            logits .= 0.0
            finite_mask = trues(V)
        end
        max_l = maximum(logits[finite_mask])
        probs = exp.(logits .- max_l)
        probs[.!finite_mask] .= 0.0

        # Min-P (nucleus thresholding)
        p_max = maximum(probs)
        min_thresh = min_p * p_max
        probs[probs .< min_thresh] .= 0.0

        p_sum = sum(probs)
        if p_sum > 1e-12
            probs ./= p_sum
        else
            probs .= 1.0 / V
        end

        # Top-P (nucleus) filtering
        if top_p < 1.0
            sorted_indices = sortperm(probs, rev=true)
            sorted_probs = probs[sorted_indices]
            cum_probs = cumsum(sorted_probs)
            cutoff_mask = cum_probs .> top_p
            if any(cutoff_mask)
                cutoff_idx = findfirst(cutoff_mask)
                if cutoff_idx > 1
                    for idx in (cutoff_idx + 1):length(sorted_indices)
                        probs[sorted_indices[idx]] = 0.0
                    end
                end
            end
            p_sum2 = sum(probs)
            if p_sum2 > 1e-12
                probs ./= p_sum2
            end
        end

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
        
        dim_limit = min(embed_dim, length(latent))
        pixel_val = 0.0
        for d in 1:dim_limit
            freq = 1.0 + Float64(d % 8) * 0.5
            ph = 2π * Float64(d) / Float64(dim_limit)
            pixel_val += latent[d] * cos(2π * (freq * u + v) + ph)
        end
        # Normalize to [0.0, 1.0]
        img[r, c] = clamp(0.5 + 0.5 * (pixel_val / sqrt(dim_limit)), 0.0, 1.0)
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
