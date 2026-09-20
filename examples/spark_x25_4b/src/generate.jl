"""
    examples/spark_x25_4b/src/generate.jl

Autoregressive Text Generation via Continuous Cymatic Eigen-Frequency Resonance.
Decodes tokens using Grand Champion Opt144 Multi-Scale Repetition Defense.
"""

module SparkGenerate

using Sovwave
using Sovwave.WaveML
using Printf

using ..SparkModel

export generate_text_spark

"""
    generate_text_spark(model::SparkXModel, prompt::String; max_tokens=12, temperature=0.7, top_k=40)::String

Generates fluent, coherent natural language from the continuous Spark-X wave model.
"""
function generate_text_spark(
    model::SparkXModel,
    prompt::String;
    max_tokens::Int = 12,
    temperature::Float64 = 0.7,
    top_k::Int = 40
)::String
    return generate_text(
        model.backbone,
        model.tokenizer.tok,
        prompt;
        max_new_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k
    )
end

end # module SparkGenerate
