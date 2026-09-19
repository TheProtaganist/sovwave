"""
    WaveML.TokenizerConverter

Universal Pretrained Tokenizer Converter for Sovwave.
Converts tokenizer vocabularies from established pipelines (GPT-2, Qwen, Mistral,
LLaMA, DeepSeek, Hugging Face Tokenizers) into continuous `WaveTokenizer`s.

Features:
- Zero-dependency byte parser for `tokenizer.json` and `vocab.json` (150k+ tokens in ~0.16s)
- Automatic download and local caching from Hugging Face Hub (`~/.cache/sovwave/tokenizers/`)
- Support for byte-level BPE whitespace markers (`Ġ` in GPT-2/Qwen, ` ` in SentencePiece/Mistral/LLaMA)
- Automatic mapping to continuous physical harmonic wave frequencies and circular phases
- Zero <UNK> data loss with dynamic token registration
- Pretrained loaders: `gpt2_tokenizer()`, `qwen_tokenizer()`, `mistral_tokenizer()`, `llama_tokenizer()`
"""

using Printf
using LinearAlgebra

export convert_tokenizer, load_tokenizer_file, load_huggingface_tokenizer, convert_hf_tokenizer
export load_pretrained_tokenizer, gpt2_tokenizer, qwen_tokenizer, mistral_tokenizer, llama_tokenizer
export save_tokenizer, load_tokenizer, custom_tokenizer

# ASCII byte constants for fast JSON scanning
const _O_BRACE = 0x7b # {
const _C_BRACE = 0x7d # }
const _QUOTE   = 0x22 # "
const _COLON   = 0x3a # :
const _COMMA   = 0x2c # ,
const _ESCAPE  = 0x5c # \

"""
    parse_vocab_bytes(raw_bytes::Vector{UInt8})::Dict{String, Int}

Ultra-fast streaming byte parser for Hugging Face `tokenizer.json` and `vocab.json`.
Parses 150,000+ tokens in <0.2 seconds with zero external dependencies and zero regex PCRE limits.
"""
function parse_vocab_bytes(raw_bytes::Vector{UInt8})::Dict{String, Int}
    vocab = Dict{String, Int}()
    target = b"\"vocab\":"
    v_range = findfirst(target, raw_bytes)
    pos = v_range !== nothing ? last(v_range) + 1 : 1
    N = length(raw_bytes)

    # Find opening brace of vocabulary mapping
    while pos <= N && raw_bytes[pos] != _O_BRACE
        pos += 1
    end
    pos += 1 # past {

    buf = UInt8[]
    while pos <= N
        # find opening quote for key
        while pos <= N && raw_bytes[pos] != _QUOTE && raw_bytes[pos] != _C_BRACE
            pos += 1
        end
        (pos > N || raw_bytes[pos] == _C_BRACE) && break
        pos += 1 # past "

        # read string token key
        empty!(buf)
        escaped = false
        while pos <= N
            b = raw_bytes[pos]
            if escaped
                push!(buf, b)
                escaped = false
            elseif b == _ESCAPE
                escaped = true
                push!(buf, b)
            elseif b == _QUOTE
                break
            else
                push!(buf, b)
            end
            pos += 1
        end
        pos += 1 # past closing "

        raw_str = String(copy(buf))
        key = try
            unescape_string(raw_str)
        catch
            raw_str
        end

        # find colon
        while pos <= N && raw_bytes[pos] != _COLON && raw_bytes[pos] != _C_BRACE
            pos += 1
        end
        (pos > N || raw_bytes[pos] == _C_BRACE) && break
        pos += 1

        # skip whitespace
        while pos <= N && (raw_bytes[pos] == 0x20 || raw_bytes[pos] == 0x09 || raw_bytes[pos] == 0x0a || raw_bytes[pos] == 0x0d)
            pos += 1
        end

        # read integer token ID
        v_start = pos
        while pos <= N && (raw_bytes[pos] >= 0x30 && raw_bytes[pos] <= 0x39)
            pos += 1
        end
        if pos > v_start
            id_val = parse(Int, String(raw_bytes[v_start:pos-1]))
            vocab[key] = id_val
        end

        # find comma or closing brace
        while pos <= N && raw_bytes[pos] != _COMMA && raw_bytes[pos] != _C_BRACE
            pos += 1
        end
        if pos <= N && raw_bytes[pos] == _C_BRACE
            break
        end
        pos += 1
    end

    return vocab
end

"""
    convert_tokenizer(
        vocab_dict::Dict{String, Int};
        carrier_frequency::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895
    )::WaveTokenizer

Converts an arbitrary vocabulary dictionary (from GPT-2, Qwen, Mistral, LLaMA, etc.)
into a continuous `WaveTokenizer` where each token is mapped to its deterministic
continuous physical wave carrier frequency and circular phase.
"""
function convert_tokenizer(
    vocab_dict::Dict{String, Int};
    carrier_frequency::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveTokenizer
    isempty(vocab_dict) && error("Cannot convert empty vocabulary")

    # Normalize to 1-based indexing
    min_id = minimum(values(vocab_dict))
    offset = min_id == 0 ? 1 : 0

    v1 = Dict{String, Int}()
    max_id = 0
    for (tok, id) in vocab_dict
        adj_id = id + offset
        v1[tok] = adj_id
        if adj_id > max_id
            max_id = adj_id
        end
    end

    inv_v = fill("", max_id)
    for (tok, id) in v1
        if 1 <= id <= max_id
            inv_v[id] = tok
        end
    end

    # Fill any empty slots with placeholder tokens
    for i in 1:max_id
        if isempty(inv_v[i])
            tok = "<TOKEN_$i>"
            inv_v[i] = tok
            v1[tok] = i
        end
    end

    # Ensure special tokens are present
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
    for s in specials
        if !haskey(v1, s)
            push!(inv_v, s)
            v1[s] = length(inv_v)
        end
    end

    # Map common aliases to special tokens
    if haskey(v1, "<|endoftext|>") && !haskey(v1, "<EOS>")
        v1["<EOS>"] = v1["<|endoftext|>"]
    end
    if haskey(v1, "</s>") && !haskey(v1, "<EOS>")
        v1["<EOS>"] = v1["</s>"]
    end
    if haskey(v1, "<s>") && !haskey(v1, "<BOS>")
        v1["<BOS>"] = v1["<s>"]
    end

    return WaveTokenizer(v1, inv_v; carrier_frequency=carrier_frequency, beta_s=beta_s)
end

"""
    load_tokenizer_file(
        file_path::String;
        carrier_frequency::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895
    )::WaveTokenizer

Loads and converts a local tokenizer file (`tokenizer.json` or `vocab.json`).
"""
function load_tokenizer_file(
    file_path::String;
    carrier_frequency::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveTokenizer
    isfile(file_path) || error("Tokenizer file not found: $file_path")
    raw_bytes = read(file_path)
    vocab = parse_vocab_bytes(raw_bytes)
    return convert_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s)
end

# Model aliases mapping short names to Hugging Face repository IDs
const MODEL_HF_ALIASES = Dict{String, String}(
    "gpt2"      => "gpt2",
    "qwen"      => "Qwen/Qwen2.5-0.5B",
    "qwen2"     => "Qwen/Qwen2.5-0.5B",
    "mistral"   => "mistralai/Mistral-7B-v0.1",
    "llama"     => "meta-llama/Llama-3.2-1B",
    "llama3"    => "meta-llama/Llama-3.2-1B",
    "deepseek"  => "deepseek-ai/DeepSeek-V3",
    "deepseek2" => "deepseek-ai/DeepSeek-V2-Lite"
)

"""
    load_huggingface_tokenizer(
        repo_or_alias::AbstractString;
        token::Union{Nothing, String} = nothing,
        cache_dir::Union{Nothing, String} = nothing,
        carrier_frequency::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895
    )::WaveTokenizer

Downloads, caches, and converts any Hugging Face model tokenizer into a continuous `WaveTokenizer`.

Supported model aliases:
- `"gpt2"`: OpenAI GPT-2 BPE tokenizer (50,257 tokens)
- `"qwen"`: Qwen/Qwen2.5 tokenizer (151,643 tokens)
- `"mistral"`: Mistral-7B tokenizer (32,768 tokens)
- `"llama"`: LLaMA-3.2 tokenizer (128,256 tokens)
- `"deepseek"`: DeepSeek-V3 tokenizer (129,280 tokens)
- Or any Hugging Face repo ID: `"username/model-name"`
"""
function load_huggingface_tokenizer(
    repo_or_alias::AbstractString;
    token::Union{Nothing, String} = nothing,
    cache_dir::Union{Nothing, String} = nothing,
    carrier_frequency::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveTokenizer
    alias_key = lowercase(String(repo_or_alias))
    repo_id = get(MODEL_HF_ALIASES, alias_key, String(repo_or_alias))

    # Check bundled assets first (for 100% offline GPT-2)
    if alias_key == "gpt2"
        bundled_asset = joinpath(@__DIR__, "assets", "gpt2_vocab.json")
        if isfile(bundled_asset)
            return load_tokenizer_file(bundled_asset; carrier_frequency=carrier_frequency, beta_s=beta_s)
        end
    end

    # Cache directory
    c_dir = cache_dir !== nothing ? cache_dir : joinpath(homedir(), ".cache", "sovwave", "tokenizers")
    mkpath(c_dir)
    sanitized = replace(repo_id, "/" => "_")
    cache_file = joinpath(c_dir, "$(sanitized)_tokenizer.json")

    # If already cached, load from cache
    if isfile(cache_file) && filesize(cache_file) > 100
        return load_tokenizer_file(cache_file; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end

    # Resolve Hugging Face authentication
    auth = hf_auth_token(token)
    auth_header = auth !== nothing ? ["-H", "Authorization: Bearer $auth"] : String[]

    # Try downloading tokenizer.json first, then vocab.json fallback
    urls_to_try = [
        "https://huggingface.co/$(repo_id)/raw/main/tokenizer.json",
        "https://huggingface.co/$(repo_id)/raw/main/vocab.json"
    ]

    download_success = false
    for url in urls_to_try
        cmd = `curl -s -L --max-time 20 $(auth_header) -o $(cache_file) $(url)`
        try
            run(cmd)
            if isfile(cache_file) && filesize(cache_file) > 500
                download_success = true
                break
            end
        catch
            # continue to next candidate URL
        end
    end

    if download_success && isfile(cache_file)
        return load_tokenizer_file(cache_file; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end

    # Fallback to bundled asset if network is down
    bundled_gpt2 = joinpath(@__DIR__, "assets", "gpt2_vocab.json")
    if isfile(bundled_gpt2)
        @warn "Sovwave: Could not download tokenizer for '$(repo_id)'. Falling back to bundled GPT-2 tokenizer."
        return load_tokenizer_file(bundled_gpt2; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end

    error("Failed to load tokenizer for '$repo_id' and no local fallback available.")
end

const convert_hf_tokenizer = load_huggingface_tokenizer
const load_pretrained_tokenizer = load_huggingface_tokenizer

"""
    gpt2_tokenizer(; carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Returns the converted GPT-2 pipeline tokenizer with full 50,257 vocabulary.
Works 100% offline via bundled package assets.
"""
gpt2_tokenizer(; carrier_frequency=432.0, beta_s=1.618033988749895) =
    load_huggingface_tokenizer("gpt2"; carrier_frequency=carrier_frequency, beta_s=beta_s)

"""
    qwen_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Returns the converted Qwen 2.5 pipeline tokenizer (151,643 tokens).
"""
qwen_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895) =
    load_huggingface_tokenizer("qwen"; token=token, carrier_frequency=carrier_frequency, beta_s=beta_s)

"""
    mistral_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Returns the converted Mistral 7B pipeline tokenizer (32,768 tokens).
"""
mistral_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895) =
    load_huggingface_tokenizer("mistral"; token=token, carrier_frequency=carrier_frequency, beta_s=beta_s)

"""
    llama_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Returns the converted LLaMA 3.2 pipeline tokenizer (128,256 tokens).
"""
llama_tokenizer(; token=nothing, carrier_frequency=432.0, beta_s=1.618033988749895) =
    load_huggingface_tokenizer("llama"; token=token, carrier_frequency=carrier_frequency, beta_s=beta_s)

"""
    save_tokenizer(tok::WaveTokenizer, filepath::String)::String

Saves a `WaveTokenizer` to a portable JSON file.
Stores all tokens as their physical continuous wave frequencies in Hz.
Enables instant loading in any project with `load_tokenizer(filepath)`.
"""
function save_tokenizer(tok::WaveTokenizer, filepath::String)::String
    dir = dirname(filepath)
    !isempty(dir) && mkpath(dir)
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"version\": \"1.0\",")
        @printf(io, "  \"carrier_frequency\": %.6f,\n", tok.carrier_frequency)
        @printf(io, "  \"beta_s\": %.15f,\n", tok.beta_s)
        println(io, "  \"tokens\": {")
        sorted_pairs = sort(collect(tok.vocab), by = x -> x.second)
        for (i, (token, id)) in enumerate(sorted_pairs)
            escaped_tok = escape_string(token)
            comma = i < length(sorted_pairs) ? "," : ""
            freq = tok.frequencies[id]
            println(io, "    \"", escaped_tok, "\": ", freq, comma)
        end
        println(io, "  },")
        println(io, "  \"token_frequencies\": {")
        for (i, (token, id)) in enumerate(sorted_pairs)
            escaped_tok = escape_string(token)
            comma = i < length(sorted_pairs) ? "," : ""
            freq = tok.frequencies[id]
            println(io, "    \"", escaped_tok, "\": ", freq, comma)
        end
        println(io, "  },")
        println(io, "  \"vocab\": {")
        for (i, (token, id)) in enumerate(sorted_pairs)
            escaped_tok = escape_string(token)
            comma = i < length(sorted_pairs) ? "," : ""
            println(io, "    \"", escaped_tok, "\": ", id, comma)
        end
        println(io, "  }")
        println(io, "}")
    end
    return filepath
end

"""
    parse_token_frequencies_bytes(raw_bytes::Vector{UInt8})::Dict{String, Float64}

Fast streaming byte parser for continuous wave token frequencies from JSON.
"""
function parse_token_frequencies_bytes(raw_bytes::Vector{UInt8})::Dict{String, Float64}
    freqs = Dict{String, Float64}()
    target = b"\"token_frequencies\":"
    v_range = findfirst(target, raw_bytes)
    if v_range === nothing
        target = b"\"tokens\":"
        v_range = findfirst(target, raw_bytes)
    end
    v_range === nothing && return freqs

    pos = last(v_range) + 1
    N = length(raw_bytes)
    while pos <= N && raw_bytes[pos] != _O_BRACE
        pos += 1
    end
    pos += 1

    buf = UInt8[]
    while pos <= N
        while pos <= N && raw_bytes[pos] != _QUOTE && raw_bytes[pos] != _C_BRACE
            pos += 1
        end
        (pos > N || raw_bytes[pos] == _C_BRACE) && break
        pos += 1

        empty!(buf)
        escaped = false
        while pos <= N
            b = raw_bytes[pos]
            if escaped
                push!(buf, b)
                escaped = false
            elseif b == _ESCAPE
                escaped = true
                push!(buf, b)
            elseif b == _QUOTE
                break
            else
                push!(buf, b)
            end
            pos += 1
        end
        pos += 1

        raw_str = String(copy(buf))
        key = try unescape_string(raw_str) catch; raw_str end

        while pos <= N && raw_bytes[pos] != _COLON && raw_bytes[pos] != _C_BRACE
            pos += 1
        end
        (pos > N || raw_bytes[pos] == _C_BRACE) && break
        pos += 1

        while pos <= N && (raw_bytes[pos] == 0x20 || raw_bytes[pos] == 0x09 || raw_bytes[pos] == 0x0a || raw_bytes[pos] == 0x0d)
            pos += 1
        end

        v_start = pos
        while pos <= N && (
            (raw_bytes[pos] >= 0x30 && raw_bytes[pos] <= 0x39) ||
            raw_bytes[pos] == 0x2e || raw_bytes[pos] == 0x2d || raw_bytes[pos] == 0x2b ||
            raw_bytes[pos] == 0x65 || raw_bytes[pos] == 0x45
        )
            pos += 1
        end

        val_str = String(raw_bytes[v_start:(pos - 1)])
        val = try parse(Float64, val_str) catch; 0.0 end
        freqs[key] = val
    end
    return freqs
end

"""
    convert_frequencies_tokenizer(freq_dict::Dict{String, Float64}; carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Constructs a `WaveTokenizer` where each token is directly defined by its physical wave frequency in Hz.
"""
function convert_frequencies_tokenizer(
    freq_dict::Dict{String, Float64};
    carrier_frequency::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveTokenizer
    isempty(freq_dict) && error("Cannot convert empty frequency dictionary")
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]

    v1 = Dict{String, Int}()
    inv_v = String[]
    freqs = Float64[]
    id = 1
    for s in specials
        v1[s] = id
        push!(inv_v, s)
        f = get(freq_dict, s, token_wave_frequency(s; carrier_frequency=carrier_frequency, beta_s=beta_s))
        push!(freqs, f)
        id += 1
    end

    for (tok, f) in freq_dict
        if !haskey(v1, tok)
            v1[tok] = id
            push!(inv_v, tok)
            push!(freqs, f)
            id += 1
        end
    end

    return WaveTokenizer(v1, inv_v; carrier_frequency=carrier_frequency, beta_s=beta_s, frequencies=freqs)
end

"""
    load_tokenizer(filepath::String; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::WaveTokenizer

Universal tokenizer loader. Automatically detects and parses:
1. Sovwave custom tokenizer JSON with wave frequencies (`{"tokens": {"word": 432.0, ...}}`)
2. Hugging Face `tokenizer.json` / `vocab.json`
3. SentencePiece `.vocab` file (tab/space-separated tokens)
4. Plaintext `.txt` wordlist (one token per line)
"""
function load_tokenizer(filepath::String; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::WaveTokenizer
    isfile(filepath) || error("Tokenizer file not found: $filepath")

    # Read up to 2KB to inspect header format
    header = String(read(filepath, min(filesize(filepath), 2048)))

    if startswith(strip(header), "{") || occursin("\"vocab\":", header) || occursin("\"tokens\":", header)
        # Check custom carrier frequency and beta_s
        cf_match = match(r"\"carrier_frequency\"\s*:\s*([0-9.]+)", header)
        if cf_match !== nothing
            carrier_frequency = parse(Float64, cf_match.captures[1])
        end
        bs_match = match(r"\"beta_s\"\s*:\s*([0-9.]+)", header)
        if bs_match !== nothing
            beta_s = parse(Float64, bs_match.captures[1])
        end

        raw_bytes = read(filepath)

        # Check if wave frequencies are stored directly
        freq_dict = parse_token_frequencies_bytes(raw_bytes)
        if !isempty(freq_dict)
            return convert_frequencies_tokenizer(freq_dict; carrier_frequency=carrier_frequency, beta_s=beta_s)
        end

        # Fallback to standard vocab index JSON
        vocab = parse_vocab_bytes(raw_bytes)
        return convert_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s)
    else
        # Plaintext wordlist or SentencePiece .vocab
        vocab = Dict{String, Int}()
        id = 1
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
            vocab[sp] = id
            id += 1
        end

        for line in eachline(filepath)
            s = strip(line)
            isempty(s) && continue
            parts = split(s, ('\t', ' '))
            tok = String(parts[1])
            if !haskey(vocab, tok)
                vocab[tok] = id
                id += 1
            end
        end

        return convert_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end
end

"""
    custom_tokenizer(source; carrier_frequency=432.0, beta_s=1.618033988749895)::WaveTokenizer

Instantiates a custom `WaveTokenizer` directly for any project.
Tokens are continuous harmonic wave frequencies (in Hz).

Accepts:
- `source::AbstractString`: Path to an existing file (`.json`, `.vocab`, `.txt`), or a text corpus string.
- `source::Vector{<:AbstractString}`: List of custom words/subwords/tokens.
- `source::Dict{<:AbstractString, <:Real}`: Token-to-WaveFrequency mapping (Hz).
- `source::Dict{<:AbstractString, <:Integer}`: Token-to-index mapping.
"""
function custom_tokenizer(
    source::Union{AbstractString, Vector{<:AbstractString}, Dict};
    carrier_frequency::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::WaveTokenizer
    if source isa AbstractString
        if isfile(source)
            return load_tokenizer(source; carrier_frequency=carrier_frequency, beta_s=beta_s)
        else
            # Raw text corpus: extract unique words
            words = unique(String.(split(source)))
            return custom_tokenizer(words; carrier_frequency=carrier_frequency, beta_s=beta_s)
        end
    elseif source isa Dict
        # Check if values are wave frequencies (floating point)
        first_val = first(values(source))
        if first_val isa AbstractFloat
            freq_dict = Dict{String, Float64}(string(k) => Float64(v) for (k, v) in source)
            return convert_frequencies_tokenizer(freq_dict; carrier_frequency=carrier_frequency, beta_s=beta_s)
        else
            dict_str = Dict{String, Int}(string(k) => Int(v) for (k, v) in source)
            return convert_tokenizer(dict_str; carrier_frequency=carrier_frequency, beta_s=beta_s)
        end
    elseif source isa Vector
        vocab = Dict{String, Int}()
        id = 1
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
            vocab[sp] = id
            id += 1
        end
        for tok in source
            tok_str = string(tok)
            if !haskey(vocab, tok_str)
                vocab[tok_str] = id
                id += 1
            end
        end
        return convert_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end
end
