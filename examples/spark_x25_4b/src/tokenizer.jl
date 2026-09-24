"""
    examples/spark_x25_4b/src/tokenizer.jl

Continuous Acoustic Frequency Tokenizer for Spark-X.
Encodes text directly into continuous physical wave packets at 432 Hz carrier frequency.
Eliminates static discrete embedding tables: every token is a physical harmonic frequency.
"""

module SparkTokenizer

using Sovwave
using Sovwave.WaveML
using LinearAlgebra

export SparkWaveTokenizer, create_spark_tokenizer, encode_prompt_to_waves

struct SparkWaveTokenizer
    tok::Union{WaveTokenizer, PhoneticTokenizer}
    carrier_frequency::Float64
    beta_s::Float64
end

"""
    create_spark_tokenizer(; vocab=nothing, carrier_frequency=963.0, beta_s=1.618033988749895, use_phonetic=false)

Creates Spark tokenizer supporting both standard continuous cymatic wave and phonetic wave tokenization.
"""
function create_spark_tokenizer(;
    vocab::Union{Nothing, Vector{String}} = nothing,
    carrier_frequency::Float64 = 963.0,
    beta_s::Float64 = 1.618033988749895,
    use_phonetic::Bool = true
)
    wt = if vocab !== nothing
        use_phonetic ? phonetic_tokenizer(vocab; carrier_frequency=carrier_frequency) : custom_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s)
    else
        default_tokenizer(carrier_frequency=carrier_frequency, use_phonetic=use_phonetic)
    end
    return SparkWaveTokenizer(wt, carrier_frequency, beta_s)
end

"""
    encode_prompt_to_waves(st::SparkWaveTokenizer, text::String; embed_dim::Int = 64)::Vector{Float64}

Encodes natural language text into a continuous acoustic wave packet vector via causal phase field superposition.
Matches the physical training projection format identically.
"""
function encode_prompt_to_waves(st::SparkWaveTokenizer, text::String; embed_dim::Int = 64)::Vector{Float64}
    wfs = tokenize(st.tok, text)
    L = length(wfs)
    inv_L = 1.0 / max(1, L)
    emb = zeros(Float64, embed_dim)
    alpha_decay = 0.90

    for (pos, wf) in enumerate(wfs)
        pkt = to_wave_packet(st.tok, wf.token_id, embed_dim)
        tau = Float64(L - pos)
        weight = alpha_decay ^ tau
        phase_rot = cos(2π * tau * inv_L)
        emb .+= (weight * phase_rot) .* pkt
    end

    nrm = norm(emb)
    if nrm > 1e-6
        emb ./= nrm
    end
    return emb
end

end # module SparkTokenizer
