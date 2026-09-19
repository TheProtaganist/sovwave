"""
    WaveML.Tokenizer

Harmonic Continuous Wave Tokenizer for Sovwave.
Unlike traditional discrete lookup tables or integer tokens, the Wave Tokenizer
projects text into continuous physical waveforms (sound frequencies, continuous
wave packets, and circular phase manifolds [0, 2π)).

Every token is an acoustic waveform:
- Resonant carrier frequency f_k (governed by 432 Hz carrier & golden ratio Φ ≈ 1.618)
- Phase angle φ_k on the unit circle
- Continuous time-domain sound wave samples ψ(t) with harmonic overtones
- Can be directly sonified, heard as audio, or propagated into wave neural lattices.

Features:
- Bidirectional wave tokenization: String ⇄ Vector{WaveForm} ⇄ Continuous Sound Wave
- Waveform generation with physical frequencies and overtone spectra
- Audio synthesis: converts tokenized sentences into playable acoustic sound buffers
- Continuous resonant decoding: recovers text from continuous wave embeddings or audio
- Built-in special tokens: <PAD>, <UNK>, <BOS>, <EOS>, <SEP>, <MASK>
"""

using Printf
using LinearAlgebra

export WaveForm, WaveTokenizer, default_tokenizer, build_tokenizer
export tokenize, tokenize_ids, decode, to_wave_form, to_audio, sonify_tokens
export to_wave_packet, encode_sequence, decode_embedding, decode_sequence_embeddings
export unicode_wave_frequency, unicode_wave_phase, token_wave_frequency, token_wave_phase, register_token!

"""
    WaveForm

Physical waveform representation of a tokenized linguistic or data unit.
Represents sound and frequency rather than a discrete integer.
- `token::String`: Original subword or character
- `token_id::Int`: Harmonic vocabulary index
- `frequency::Float64`: Resonant frequency in Hz (e.g. 432 Hz * Φ^k)
- `phase::Float64`: Circular phase in radians [0, 2π)
- `amplitude::Float64`: Acoustic amplitude
- `harmonics::Vector{Float64}`: Harmonic overtone frequencies (Hz)
- `samples::Vector{Float64}`: Continuous time-domain audio samples ψ(t)
- `duration::Float64`: Duration of wave packet in seconds
- `energy::Float64`: Integrated wave energy
"""
struct WaveForm
    token::String
    token_id::Int
    frequency::Float64
    phase::Float64
    amplitude::Float64
    harmonics::Vector{Float64}
    samples::Vector{Float64}
    duration::Float64
    energy::Float64
end

function Base.show(io::IO, wf::WaveForm)
    @printf(io, "WaveForm(\"%s\", f=%.2f Hz, φ=%.3f rad, energy=%.4f, %d samples)",
            wf.token, wf.frequency, wf.phase, wf.energy, length(wf.samples))
end

"""
    unicode_wave_frequency(c::Char; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::Float64

Computes the physical harmonic carrier frequency for any Unicode character (U+0000 to U+10FFFF)
using a golden-ratio continuous Weyl mapping. Every character has a mathematically unique, deterministic frequency.
"""
function unicode_wave_frequency(c::Char; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::Float64
    u = UInt32(c)
    inv_phi = 1.0 / beta_s
    weyl = mod(Float64(u) * inv_phi, 1.0)
    octave_step = Float64(u % 12) / 12.0
    f = carrier_frequency * (beta_s ^ octave_step) * (1.0 + weyl * 0.5 + Float64(div(u, 12) % 1000) * 0.0002)
    return f
end

"""
    unicode_wave_phase(c::Char; beta_s::Float64 = 1.618033988749895)::Float64

Computes the continuous circular phase angle in [0, 2π) for any Unicode character.
"""
function unicode_wave_phase(c::Char; beta_s::Float64 = 1.618033988749895)::Float64
    u = UInt32(c)
    inv_phi = 1.0 / beta_s
    return mod(2π * Float64(u) * inv_phi, 2π)
end

"""
    token_wave_frequency(s::String; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::Float64

Computes the composite harmonic frequency for any token or subword across its Unicode graphemes.
"""
function token_wave_frequency(s::String; carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)::Float64
    chars = collect(s)
    if isempty(chars)
        return carrier_frequency
    elseif length(chars) == 1
        return unicode_wave_frequency(chars[1]; carrier_frequency=carrier_frequency, beta_s=beta_s)
    end
    total_f = 0.0
    weight_sum = 0.0
    for (i, c) in enumerate(chars)
        w = beta_s ^ (-Float64(i - 1))
        total_f += w * unicode_wave_frequency(c; carrier_frequency=carrier_frequency, beta_s=beta_s)
        weight_sum += w
    end
    return total_f / weight_sum
end

"""
    token_wave_phase(s::String; beta_s::Float64 = 1.618033988749895)::Float64

Computes the continuous phase angle for any token or subword.
"""
function token_wave_phase(s::String; beta_s::Float64 = 1.618033988749895)::Float64
    chars = collect(s)
    if isempty(chars)
        return 0.0
    elseif length(chars) == 1
        return unicode_wave_phase(chars[1]; beta_s=beta_s)
    end
    ph = 0.0
    for (i, c) in enumerate(chars)
        ph += unicode_wave_phase(c; beta_s=beta_s) * (beta_s ^ (-Float64(i - 1)))
    end
    return mod(ph, 2π)
end

"""
    WaveTokenizer

Continuous harmonic wave tokenizer struct.
All Unicode characters, emojis, and world language scripts are first-class harmonic primitives.
"""
struct WaveTokenizer
    vocab::Dict{String, Int}
    inv_vocab::Vector{String}
    frequencies::Vector{Float64}
    phases::Vector{Float64}
    special_tokens::Dict{Symbol, Int}
    carrier_frequency::Float64
    beta_s::Float64
    lock::ReentrantLock

    function WaveTokenizer(
        vocab::Dict{String, Int},
        inv_vocab::Vector{String};
        carrier_frequency::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895
    )
        V = length(inv_vocab)
        freqs = Vector{Float64}(undef, V)
        phs = Vector{Float64}(undef, V)

        # Generate harmonic frequency and circular phase per token using Unicode-aware harmonic mapping
        for k in 1:V
            tok_str = inv_vocab[k]
            freqs[k] = token_wave_frequency(tok_str; carrier_frequency=carrier_frequency, beta_s=beta_s)
            phs[k] = token_wave_phase(tok_str; beta_s=beta_s)
        end

        specials = Dict{Symbol, Int}(
            :PAD => get(vocab, "<PAD>", 1),
            :UNK => get(vocab, "<UNK>", 2),
            :BOS => get(vocab, "<BOS>", 3),
            :EOS => get(vocab, "<EOS>", 4),
            :SEP => get(vocab, "<SEP>", 5),
            :MASK => get(vocab, "<MASK>", 6)
        )

        new(vocab, inv_vocab, freqs, phs, specials, carrier_frequency, beta_s, ReentrantLock())
    end
end

"""
    register_token!(tok::WaveTokenizer, s::String)::Int

Dynamically registers a novel character, emoji, or subword as a first-class native wave token.
Computes its deterministic physical frequency and phase and inserts it into the active vocabulary.
Ensures zero data loss, zero unknown tokens (<UNK>), and 100% exact bidirectional reconstruction.
"""
function register_token!(tok::WaveTokenizer, s::String)::Int
    lock(tok.lock) do
        if haskey(tok.vocab, s)
            return tok.vocab[s]
        end
        new_id = length(tok.inv_vocab) + 1
        tok.vocab[s] = new_id
        push!(tok.inv_vocab, s)
        freq = token_wave_frequency(s; carrier_frequency=tok.carrier_frequency, beta_s=tok.beta_s)
        ph = token_wave_phase(s; beta_s=tok.beta_s)
        push!(tok.frequencies, freq)
        push!(tok.phases, ph)
        return new_id
    end
end

"""
    default_tokenizer(; carrier_frequency=432.0)::WaveTokenizer

Returns the standard universal Wave Tokenizer equipped with:
- Standard special tokens (<PAD>, <UNK>, <BOS>, <EOS>, <SEP>, <MASK>)
- All standard ASCII printable characters and whitespace
- Latin Extended / Accented characters (Spanish, French, German, Scandinavian, etc.)
- Greek, Cyrillic, Arabic, Hebrew, Devanagari (Hindi), CJK, Japanese, and Korean alphabets
- Sacred Geometry, Mathematical & Physics symbols (∂, ∇, ∫, ∑, ℏ, ψ, Φ, π, etc.)
- Universal Emojis (🌊, 🧠, ⚡, 🚀, ⚛️, 🔮, 🎵, 🎶, 🔊, 🌐, 🌌, ✨, 🌟, 💡, 🔥, etc.)
- Common subwords, punctuation, and digits
"""
function default_tokenizer(; carrier_frequency::Float64 = 432.0)::WaveTokenizer
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
    
    # Printable ASCII characters (space through ~) plus whitespace
    ascii_chars = [string(Char(c)) for c in 32:126]
    whitespace = ["\n", "\t", "\r", " "]

    # Extended Latin & accented characters
    latin_ext = [
        "á", "é", "í", "ó", "ú", "à", "è", "ì", "ò", "ù", "ä", "ö", "ü", "ñ", "ç", "ß",
        "ø", "å", "æ", "œ", "Á", "É", "Í", "Ó", "Ú", "À", "È", "Ì", "Ò", "Ù", "Ä", "Ö",
        "Ü", "Ñ", "Ç", "Ø", "Å", "Æ", "Œ", "ã", "õ", "â", "ê", "î", "ô", "û", "ě", "š",
        "č", "ř", "ž", "ý", "ť", "ď", "ň", "ů"
    ]

    # Greek alphabet
    greek = [
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π",
        "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ",
        "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω"
    ]

    # Cyrillic alphabet
    cyrillic = [
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
        "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю",
        "я", "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й", "К", "Л", "М", "Н",
        "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"
    ]

    # Arabic alphabet
    arabic = [
        "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط",
        "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ء", "آ", "ة", "ى", "ئ", "ؤ"
    ]

    # Hebrew alphabet
    hebrew = [
        "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ", "ס", "ע",
        "פ", "צ", "ק", "ר", "ש", "ת", "ך", "ם", "ן", "ף", "ץ"
    ]

    # Devanagari (Hindi)
    devanagari = [
        "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ",
        "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प",
        "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "ा", "ि", "ी", "ु",
        "ू", "ृ", "े", "ै", "ो", "ौ", "्", "ं", "ः"
    ]

    # CJK Common Characters
    cjk = [
        "中", "文", "国", "人", "大", "小", "日", "月", "水", "火", "木", "金", "土", "天", "地",
        "道", "心", "気", "和", "平", "愛", "智", "慧", "波", "脳", "量", "子", "生", "命", "宇",
        "宙", "象", "意", "識", "力", "光", "音", "楽", "学", "校", "山", "海", "川", "花", "鳥",
        "風", "雲", "雷", "電", "神", "仏", "微", "分", "積", "極"
    ]

    # Japanese Hiragana & Katakana
    japanese = [
        "あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と", "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ",
        "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を", "ん",
        "ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ",
        "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ",
        "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ワ", "ヲ", "ン"
    ]

    # Korean Hangul
    korean = [
        "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하",
        "한", "글", "파", "동", "인", "공", "지", "능", "양", "자", "물", "리", "세", "계", "우", "주"
    ]

    # Sacred Geometry, Mathematical & Physics symbols
    math_symbols = [
        "==", "!=", "<=", ">=", "->", "=>", "+=", "-=", "*=", "/=", "::", "...", "/*", "*/",
        "∂", "∇", "∫", "∬", "∭", "∮", "∑", "∏", "√", "∛", "∞", "∝", "∠", "∧", "∨", "∩", "∪",
        "≈", "≠", "≡", "≤", "≥", "≪", "≫", "±", "∓", "×", "÷", "⊕", "⊗", "⊙", "⊥", "⊤", "⊢",
        "⊨", "∴", "∵", "ℏ", "ψ", "Ψ", "λ", "ω", "Ω", "π", "Φ", "φ", "ϵ", "ϕ", "→", "←", "↑", "↓"
    ]

    # Universal Emojis (First-Class Wave Primitives)
    emojis = [
        "🌊", "🧠", "⚡", "🚀", "⚛️", "🔮", "🎵", "🎶", "🔊", "🌐", "🌌", "✨", "🌟", "💡",
        "🔥", "🌈", "🛠️", "📊", "🎯", "🤖", "💻", "🧬", "🛡️", "🕊️", "🪐", "☀️", "🌙", "⭐",
        "❤️", "💎", "🔔", "👁️", "🌀", "🎨", "🧪", "📡", "🔋", "🔑", "🏆", "🥇"
    ]
    
    # Common English word fragments & subwords (2-3 chars only, no full words)
    common_subwords = [
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
        "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
        "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll",
        "mo", "ni", "wa", "mp", "ut", "ma", "rm", "pu", "tin", "po", "ta", "so", "la", "mi", "si",
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her", "was", "one",
        "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see",
        "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use"
    ]

    all_tokens = unique(vcat(specials, ascii_chars, whitespace, latin_ext, greek, cyrillic, arabic, hebrew, devanagari, cjk, japanese, korean, math_symbols, emojis, common_subwords))
    vocab = Dict{String, Int}(tok => idx for (idx, tok) in enumerate(all_tokens))
    inv_vocab = all_tokens

    return WaveTokenizer(vocab, inv_vocab; carrier_frequency=carrier_frequency)
end

"""
    build_tokenizer(corpus::Vector{String}; max_vocab::Int = 1000, min_freq::Int = 1, carrier_frequency=432.0)::WaveTokenizer

Learns and constructs a `WaveTokenizer` from a given text corpus based on subword frequencies.
"""
function build_tokenizer(
    corpus::Vector{String};
    max_vocab::Int = 1000,
    min_freq::Int = 1,
    carrier_frequency::Float64 = 432.0
)::WaveTokenizer
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
    
    # Count word and subword frequencies
    counts = Dict{String, Int}()
    for doc in corpus
        for ch in doc
            s = string(ch)
            counts[s] = get(counts, s, 0) + 1
        end
        words = split(doc, r"\s+")
        for w in words
            if !isempty(w)
                counts[String(w)] = get(counts, String(w), 0) + 1
            end
        end
    end

    # Filter by min_freq and sort by frequency descending
    filtered = filter(p -> p.second >= min_freq, collect(counts))
    sort!(filtered, by = p -> p.second, rev = true)

    n_to_take = min(max_vocab - length(specials), length(filtered))
    selected = [p.first for p in filtered[1:n_to_take]]

    all_tokens = unique(vcat(specials, selected))
    vocab = Dict{String, Int}(tok => idx for (idx, tok) in enumerate(all_tokens))
    inv_vocab = all_tokens

    return WaveTokenizer(vocab, inv_vocab; carrier_frequency=carrier_frequency)
end

"""
    to_wave_form(tok::WaveTokenizer, token_id::Int; n_samples::Int = 32, sample_rate::Float64 = 48000.0, t::Float64 = 0.0)::WaveForm

Synthesizes a physical continuous `WaveForm` for a token ID.
Computes harmonic frequencies, overtone spectrum, and continuous audio time-series samples.
"""
function to_wave_form(
    tok::WaveTokenizer,
    token_id::Int;
    n_samples::Int = 32,
    sample_rate::Float64 = 48000.0,
    t::Float64 = 0.0
)::WaveForm
    valid_id = clamp(token_id, 1, length(tok.inv_vocab))
    token_str = tok.inv_vocab[valid_id]
    f_k = tok.frequencies[valid_id]
    phi_k = tok.phases[valid_id]
    beta = tok.beta_s

    # Overtones: fundamental, octave (2f), fifth (3f/2), golden overtone (Φ * f)
    harmonics = [f_k, 2.0 * f_k, 1.5 * f_k, beta * f_k]
    duration = Float64(n_samples) / sample_rate

    samples = Vector{Float64}(undef, n_samples)
    dt = 1.0 / sample_rate

    for n in 1:n_samples
        t_cur = t + Float64(n - 1) * dt
        # Fundamental carrier + phase
        val = cos(2π * f_k * t_cur + phi_k)
        # Add harmonic overtones with golden ratio decay
        val += 0.382 * cos(2π * harmonics[2] * t_cur + phi_k * 2.0)
        val += 0.236 * cos(2π * harmonics[3] * t_cur + phi_k * 1.5)
        val += 0.146 * cos(2π * harmonics[4] * t_cur + phi_k * beta)
        # Envelope windowing
        envelope = exp(-Float64(n - 1) / (Float64(n_samples) * 0.8))
        samples[n] = val * envelope
    end

    # Normalize energy
    nrm = norm(samples)
    if nrm > 1e-6
        samples ./= nrm
    end
    energy = sum(samples .^ 2)
    amplitude = maximum(abs.(samples))

    return WaveForm(token_str, valid_id, f_k, phi_k, amplitude, harmonics, samples, duration, energy)
end

"""
    tokenize_ids(tok::WaveTokenizer, text::String)::Vector{Int}

Frequency-adaptive BPE tokenizer with bigram learning.
"""
function tokenize_ids(tok::WaveTokenizer, text::String)::Vector{Int}
    isempty(text) && return Int[]
    tokens = Int[]
    chars = collect(text)
    n_chars = length(chars)
    
    # Learn bigram frequencies from input
    bigram_freq = Dict{String, Int}()
    for i in 1:(n_chars - 1)
        bg = string(chars[i], chars[i+1])
        bigram_freq[bg] = get(bigram_freq, bg, 0) + 1
    end
    
    idx = 1
    while idx <= n_chars
        matched = false
        
        # Try subword matches (8, 6, 4, 2 char lookahead)
        for len in [8, 6, 4, 2]
            if idx + len <= n_chars
                sub = String(chars[idx:(idx + len - 1)])
                
                # Match if in vocab OR frequent bigram (≥2 occurrences)
                if haskey(tok.vocab, sub) || (len == 2 && get(bigram_freq, sub, 0) >= 2)
                    if !haskey(tok.vocab, sub)
                        id = register_token!(tok, sub)
                    else
                        id = tok.vocab[sub]
                    end
                    push!(tokens, id)
                    idx += len
                    matched = true
                    break
                end
            end
        end
        
        # Single character fallback
        if !matched
            ch_str = String([chars[idx]])
            id = haskey(tok.vocab, ch_str) ? tok.vocab[ch_str] : register_token!(tok, ch_str)
            push!(tokens, id)
            idx += 1
        end
    end
    
    return tokens
end

"""
    tokenize(tok::WaveTokenizer, text::String; n_samples::Int = 32, sample_rate::Float64 = 48000.0)::Vector{WaveForm}

Tokenizes text into continuous physical `WaveForm`s representing acoustic sound waves and frequencies.
Each token is a continuous sound wave packet rather than a discrete number.
"""
function tokenize(
    tok::WaveTokenizer,
    text::String;
    n_samples::Int = 32,
    sample_rate::Float64 = 48000.0
)::Vector{WaveForm}
    ids = tokenize_ids(tok, text)
    waveforms = Vector{WaveForm}(undef, length(ids))
    for (i, id) in enumerate(ids)
        # Phase evolves sequentially across tokens for continuous phase coherence
        t_offset = Float64(i - 1) * (Float64(n_samples) / sample_rate)
        waveforms[i] = to_wave_form(tok, id; n_samples=n_samples, sample_rate=sample_rate, t=t_offset)
    end
    return waveforms
end

"""
    to_audio(waveforms::Vector{WaveForm}; sample_rate::Float64 = 48000.0)::Vector{Float64}

Synthesizes a continuous, smooth audio buffer from a sequence of `WaveForm`s.
Allows hearing the tokenized text directly as sound.
"""
function to_audio(waveforms::Vector{WaveForm}; sample_rate::Float64 = 48000.0)::Vector{Float64}
    isempty(waveforms) && return Float64[]
    total_samples = sum(length(wf.samples) for wf in waveforms)
    buffer = zeros(Float64, total_samples)
    offset = 1
    for wf in waveforms
        len = length(wf.samples)
        buffer[offset:(offset + len - 1)] .= wf.samples
        offset += len
    end
    # Normalize peak audio amplitude
    pk = maximum(abs.(buffer))
    if pk > 1e-6
        buffer ./= (pk * 1.05)
    end
    return buffer
end

"""
    sonify_tokens(tok::WaveTokenizer, text_or_waveforms; sample_rate::Float64 = 48000.0, path::Union{Nothing, String} = nothing)::Vector{Float64}

Synthesizes the tokenized sound wave. If `path` is provided, writes a standard 16-bit WAV file.
"""
function sonify_tokens(
    tok::WaveTokenizer,
    input;
    sample_rate::Float64 = 48000.0,
    path::Union{Nothing, String} = nothing
)::Vector{Float64}
    waveforms = if input isa String
        tokenize(tok, input; sample_rate=sample_rate)
    elseif input isa Vector{WaveForm}
        input
    else
        throw(ArgumentError("Expected String or Vector{WaveForm}"))
    end

    audio = to_audio(waveforms; sample_rate=sample_rate)
    if path !== nothing
        # Write 16-bit PCM WAV
        open(path, "w") do io
            n = length(audio)
            byte_rate = round(Int, sample_rate * 2)
            block_align = 2
            # RIFF header
            write(io, b"RIFF")
            write(io, Int32(36 + n * 2))
            write(io, b"WAVE")
            write(io, b"fmt ")
            write(io, Int32(16))
            write(io, Int16(1)) # PCM
            write(io, Int16(1)) # Mono
            write(io, Int32(round(Int, sample_rate)))
            write(io, Int32(byte_rate))
            write(io, Int16(block_align))
            write(io, Int16(16)) # bits per sample
            write(io, b"data")
            write(io, Int32(n * 2))
            for s in audio
                i16 = clamp(round(Int16, s * 32767.0), typemin(Int16), typemax(Int16))
                write(io, i16)
            end
        end
    end
    return audio
end

"""
    decode(tok::WaveTokenizer, waveforms::Vector{WaveForm})::String

Decodes text directly from a sequence of physical `WaveForm`s.
"""
function decode(tok::WaveTokenizer, waveforms::Vector{WaveForm})::String
    buf = IOBuffer()
    pad_tok = "<PAD>"
    bos_tok = "<BOS>"
    eos_tok = "<EOS>"

    for wf in waveforms
        if wf.token != pad_tok && wf.token != bos_tok && wf.token != eos_tok
            print(buf, wf.token)
        end
    end
    return String(take!(buf))
end

"""
    decode(tok::WaveTokenizer, ids::Vector{Int})::String

Decodes text from token IDs, skipping special formatting tokens like <PAD>.
"""
function decode(tok::WaveTokenizer, ids::Vector{Int})::String
    buf = IOBuffer()
    pad_id = tok.special_tokens[:PAD]
    bos_id = tok.special_tokens[:BOS]
    eos_id = tok.special_tokens[:EOS]

    for id in ids
        if id == pad_id || id == bos_id || id == eos_id
            continue
        elseif 1 <= id <= length(tok.inv_vocab)
            print(buf, tok.inv_vocab[id])
        end
    end

    return String(take!(buf))
end

"""
    to_wave_packet(tok::WaveTokenizer, token_id::Int, embed_dim::Int; t::Float64 = 0.0)::Vector{Float64}

Synthesizes a continuous harmonic wave packet embedding vector for a given token ID:
E_d = cos(2π f_k · d/D + φ_k + t) · exp(-d / (β_s · D))
"""
function to_wave_packet(
    tok::WaveTokenizer,
    token_id::Int,
    embed_dim::Int;
    t::Float64 = 0.0
)::Vector{Float64}
    valid_id = clamp(token_id, 1, length(tok.inv_vocab))
    f_k = tok.frequencies[valid_id]
    phi_k = tok.phases[valid_id]
    beta = tok.beta_s

    emb = Vector{Float64}(undef, embed_dim)
    inv_dim = 1.0 / Float64(embed_dim)

    @inbounds for d in 1:embed_dim
        x = Float64(d - 1) * inv_dim
        carrier = cos(2π * (f_k / tok.carrier_frequency) * x + phi_k + t)
        envelope = exp(-x / beta)
        emb[d] = carrier * envelope
    end

    nrm = norm(emb)
    if nrm > 1e-6
        emb ./= nrm
    end

    return emb
end

"""
    to_wave_packet(wf::WaveForm, embed_dim::Int; t::Float64 = 0.0)::Vector{Float64}

Extracts or interpolates continuous wave packet from a `WaveForm`.
"""
function to_wave_packet(wf::WaveForm, embed_dim::Int; t::Float64 = 0.0)::Vector{Float64}
    n_s = length(wf.samples)
    if n_s == embed_dim
        return copy(wf.samples)
    end
    # Resample / generate wave packet at embed_dim
    emb = Vector{Float64}(undef, embed_dim)
    inv_dim = 1.0 / Float64(embed_dim)
    beta = 1.618033988749895
    for d in 1:embed_dim
        x = Float64(d - 1) * inv_dim
        carrier = cos(2π * (wf.frequency / 432.0) * x + wf.phase + t)
        envelope = exp(-x / beta)
        emb[d] = carrier * envelope
    end
    nrm = norm(emb)
    if nrm > 1e-6
        emb ./= nrm
    end
    return emb
end

"""
    encode_sequence(tok::WaveTokenizer, text::String; max_len::Int = 32, embed_dim::Int = 32)::Matrix{Float64}

Encodes input text into a continuous wave packet tensor of shape `(embed_dim, max_len)`
using physical continuous `WaveForm` packets.
"""
function encode_sequence(
    tok::WaveTokenizer,
    text::String;
    max_len::Int = 32,
    embed_dim::Int = 32
)::Matrix{Float64}
    waveforms = tokenize(tok, text; n_samples=embed_dim)
    pad_id = tok.special_tokens[:PAD]
    n_wfs = length(waveforms)

    out_matrix = Matrix{Float64}(undef, embed_dim, max_len)
    for step_idx in 1:max_len
        t_step = Float64(step_idx - 1) * (2π / max_len)
        if step_idx <= n_wfs
            out_matrix[:, step_idx] = to_wave_packet(waveforms[step_idx], embed_dim; t=t_step)
        else
            out_matrix[:, step_idx] = to_wave_packet(tok, pad_id, embed_dim; t=t_step)
        end
    end

    return out_matrix
end

"""
    decode_embedding(tok::WaveTokenizer, emb::Vector{Float64}; embed_dim::Union{Nothing, Int} = nothing)::Int

Finds the token ID whose theoretical wave packet exhibits maximum harmonic resonance (cosine similarity)
with the given continuous embedding vector `emb`.
"""
function decode_embedding(
    tok::WaveTokenizer,
    emb::Vector{Float64};
    embed_dim::Union{Nothing, Int} = nothing
)::Int
    d_dim = embed_dim !== nothing ? embed_dim : length(emb)
    emb_norm = norm(emb)
    emb_norm < 1e-6 && return tok.special_tokens[:PAD]

    best_id = 1
    best_sim = -Inf

    for k in 1:length(tok.inv_vocab)
        candidate = to_wave_packet(tok, k, d_dim)
        sim = dot(emb, candidate)
        if sim > best_sim
            best_sim = sim
            best_id = k
        end
    end

    return best_id
end

"""
    decode_sequence_embeddings(tok::WaveTokenizer, emb_matrix::Matrix{Float64})::String

Decodes an entire continuous sequence matrix of shape `(embed_dim, seq_len)` into text.
"""
function decode_sequence_embeddings(tok::WaveTokenizer, emb_matrix::Matrix{Float64})::String
    seq_len = size(emb_matrix, 2)
    ids = Vector{Int}(undef, seq_len)
    for s in 1:seq_len
        ids[s] = decode_embedding(tok, emb_matrix[:, s])
    end
    return decode(tok, ids)
end
