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

export SparkXConfig, SparkXModel, create_spark_model, forward_spark!, load_spark_model

struct SparkXConfig
    layers::Int
    embed_dim::Int
    heads::Int
    carrier_omega::Float64
    beta_s::Float64
    vocab_size::Int

    # Primary keyword constructor setting architecture hyper-parameters
    function SparkXConfig(;
        layers::Int = 4,
        embed_dim::Int = 64,
        heads::Int = 4,
        carrier_omega::Float64 = 963.0,
        beta_s::Float64 = 1.618033988749895,
        vocab_size::Int = 50263
    )
        new(layers, embed_dim, heads, carrier_omega, beta_s, vocab_size)
    end
end

struct SparkXModel
    config::SparkXConfig
    tokenizer::Any
    backbone::WaveModel
    active_vocab::Set{Int}

    # Primary constructor assembling configured backbone, tokenizer, and active vocabulary
    function SparkXModel(
        config::SparkXConfig,
        tokenizer::Any,
        backbone::WaveModel,
        active_vocab::Set{Int} = Set{Int}()
    )
        new(config, tokenizer, backbone, active_vocab)
    end
end

"""
    create_spark_model(cfg::SparkXConfig = SparkXConfig(); vocab::Union{Nothing, Vector{String}} = nothing, use_phonetic::Bool = true)::SparkXModel

Constructs a full continuous wave language model based on the Spark-X2.5-4B architecture.
Zero static discrete embedding tables | Zero GPU matrix multipliers.
"""
function create_spark_model(
    cfg::SparkXConfig = SparkXConfig();
    vocab::Union{Nothing, Vector{String}} = nothing,
    use_phonetic::Bool = true
)::SparkXModel
    st = create_spark_tokenizer(vocab=vocab, carrier_frequency=cfg.carrier_omega, beta_s=cfg.beta_s, use_phonetic=use_phonetic)

    # Verify tokenizer vocabulary size matches expected configuration
    actual_vocab_size = length(st.tok.inv_vocab)
    if actual_vocab_size != cfg.vocab_size
        @warn "Tokenizer vocabulary size mismatch: expected $(cfg.vocab_size), got $actual_vocab_size"
        println("  ℹ️  Spark-X2.5-4B reference model uses 131,072 tokens")
        println("  ℹ️  Current tokenizer has $actual_vocab_size tokens")
        println("  ℹ️  Using actual vocabulary size for model creation")
    else
        println("  ✓ Tokenizer vocabulary verified: $actual_vocab_size tokens")
    end

    m_cfg = WaveModelConfig(
        nodes = cfg.embed_dim,
        embed_dims = cfg.embed_dim,
        layers = cfg.layers,
        omega = cfg.carrier_omega,
        beta_s = cfg.beta_s
    )
    ml_cfg = WaveMLConfig(model=m_cfg)
    wm = WaveModel(ml_cfg)

    # Initialize acoustic transmission amplitudes with variance 1/D and physical A >= 0; sign carried in phase angle phi in {0, pi}
    for layer in wm.layers
        W = randn(cfg.embed_dim, cfg.embed_dim) ./ sqrt(Float64(cfg.embed_dim))
        layer.amplitudes .= abs.(W)
        layer.phases .= [w < 0 ? π : 0.0 for w in W]
    end

    return SparkXModel(cfg, st, wm)
end

"""
    forward_spark!(model::SparkXModel, wave_input::Vector{Float64})::Vector{Float64}

Executes continuous physical wave propagation across all Spark-X layers.
"""
function forward_spark!(model::SparkXModel, wave_input::Vector{Float64})::Vector{Float64}
    return forward_continuous_wave!(model.backbone, wave_input)
end

"""
    load_spark_model(mkv_path::String; cfg::SparkXConfig = SparkXConfig(), vocab::Union{Nothing, Vector{String}} = nothing, use_phonetic::Bool = true, active_vocab::Set{Int} = Set{Int}())::SparkXModel

Restores the Spark-X continuous wave model directly from an MKV/MP4 video file.
Zero .bin files needed!
"""
function load_spark_model(
    mkv_path::String;
    cfg::SparkXConfig = SparkXConfig(),
    vocab::Union{Nothing, Vector{String}} = nothing,
    use_phonetic::Bool = true,
    active_vocab::Set{Int} = Set{Int}()
)::SparkXModel
    st = create_spark_tokenizer(vocab=vocab, carrier_frequency=cfg.carrier_omega, beta_s=cfg.beta_s, use_phonetic=use_phonetic)
    wm = load_model(mkv_path)
    return SparkXModel(cfg, st, wm, active_vocab)
end

end # module SparkModel
