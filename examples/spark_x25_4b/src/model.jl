"""
    examples/spark_x25_4b/src/model.jl

Spark-X2.5-4B Sized Continuous Wave Architecture.
Inspired by https://huggingface.co/XHToken/Spark-X2.5-4B
Integrates continuous acoustic frequency tokenization, wave-attention interference,
and Flower of Life Riemannian manifold hyper-connections.
"""

module SparkModel

using Sovwave
using Sovwave.WaveML
using LinearAlgebra
using Printf

include("tokenizer.jl")
include("attention.jl")
using .SparkTokenizer
using .SparkAttention

export SparkXConfig, SparkXModel, create_spark_model, forward_spark!

struct SparkXConfig
    layers::Int
    embed_dim::Int
    heads::Int
    carrier_omega::Float64
    beta_s::Float64
    vocab_size::Int

    function SparkXConfig(;
        layers::Int = 4,
        embed_dim::Int = 64,
        heads::Int = 4,
        carrier_omega::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895,
        vocab_size::Int = 50263
    )
        new(layers, embed_dim, heads, carrier_omega, beta_s, vocab_size)
    end
end

struct SparkXModel
    config::SparkXConfig
    tokenizer::SparkWaveTokenizer
    backbone::WaveModel
end

"""
    create_spark_model(cfg::SparkXConfig = SparkXConfig())::SparkXModel

Constructs a full continuous wave language model based on the Spark-X2.5-4B architecture.
Zero static discrete embedding tables | Zero GPU matrix multipliers.
"""
function create_spark_model(cfg::SparkXConfig = SparkXConfig(); vocab::Union{Nothing, Vector{String}} = nothing)::SparkXModel
    st = create_spark_tokenizer(vocab=vocab, carrier_frequency=cfg.carrier_omega, beta_s=cfg.beta_s)

    m_cfg = WaveModelConfig(
        nodes = cfg.embed_dim,
        embed_dims = cfg.embed_dim,
        layers = cfg.layers,
        omega = cfg.carrier_omega,
        beta_s = cfg.beta_s
    )
    ml_cfg = WaveMLConfig(model=m_cfg)
    wm = WaveModel(ml_cfg)

    return SparkXModel(cfg, st, wm)
end

"""
    forward_spark!(model::SparkXModel, wave_input::Vector{Float64})::Vector{Float64}

Executes continuous physical wave propagation across all Spark-X layers.
"""
function forward_spark!(model::SparkXModel, wave_input::Vector{Float64})::Vector{Float64}
    return forward!(model.backbone, wave_input)
end

end # module SparkModel
