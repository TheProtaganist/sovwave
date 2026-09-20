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
    tok::WaveTokenizer
    carrier_frequency::Float64
    beta_s::Float64
end

function create_spark_tokenizer(; vocab::Union{Nothing, Vector{String}} = nothing, carrier_frequency::Float64 = 432.0, beta_s::Float64 = 1.618033988749895)
    wt = vocab !== nothing ? custom_tokenizer(vocab; carrier_frequency=carrier_frequency, beta_s=beta_s) : default_tokenizer()
    return SparkWaveTokenizer(wt, carrier_frequency, beta_s)
end

"""
    encode_prompt_to_waves(st::SparkWaveTokenizer, text::String; embed_dim::Int = 64)::Vector{Float64}

Encodes natural language text into a continuous acoustic wave packet vector.
"""
function encode_prompt_to_waves(st::SparkWaveTokenizer, text::String; embed_dim::Int = 64)::Vector{Float64}
    wfs = tokenize(st.tok, text)
    emb = zeros(Float64, embed_dim)
    inv_dim = 1.0 / Float64(embed_dim)

    for (idx, wf) in enumerate(wfs)
        slot = mod1(idx, embed_dim)
        freq_norm = wf.frequency / st.carrier_frequency
        emb[slot] += wf.energy * cos(wf.phase + 2π * freq_norm * (Float64(slot) * inv_dim))
    end

    nrm = norm(emb)
    if nrm > 1e-6
        emb ./= nrm
    end
    return emb
end

end # module SparkTokenizer
