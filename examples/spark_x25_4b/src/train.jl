"""
    examples/spark_x25_4b/src/train.jl

Continuous Evolutionary Training Loop for Spark-X Wave Model.
Evolves continuous wave parameters to ground state using the green terminal progress bar,
432 Hz Gamma-to-Epsilon binaural audio, and fluid cymatic heatmap MKV video checkpointing.
"""

module SparkTrain

using Sovwave
using Sovwave.WaveML
using Printf

using ..SparkModel

export train_spark_model

"""
    train_spark_model(model::SparkXModel, texts::Vector{String}; epochs=20, batch_size=8, sonify=false, checkpoint_dir="checkpoints/spark")

Trains the Spark-X continuous wave language model via continuous evolutionary relaxation.
"""
function train_spark_model(
    model::SparkXModel,
    texts::Vector{String};
    epochs::Int = 20,
    batch_size::Int = 8,
    sonify::Bool = true,
    checkpoint_dir::Union{Nothing, String} = "checkpoints/spark"
)::Tuple{SparkXModel, TrainingHistory}
    println("="^80)
    println(" 🌊 TRAINING SPARK-X CONTINUOUS WAVE LANGUAGE MODEL 🌊")
    println("="^80)

    # 1. Project texts into next-token continuous acoustic wave prediction pairs
    println(" Formatting $(length(texts)) text sequences into next-token continuous wave prediction pairs...")
    dataset = format_lm_text(texts; tokenizer=model.tokenizer.tok, embed_dim=model.config.embed_dim, max_pairs=1000)
    @printf(" ✓ Converted %d text samples into %d continuous %d-dim wave training pairs\n", length(texts), length(dataset), model.config.embed_dim)

    # 2. Evolutionary Training Configuration
    cfg = WaveTrainConfig(
        epochs = epochs,
        population_size = 12,
        learning_rate = 0.05,
        energy_target = 0.005,
        batch_size = batch_size,
        sonify = sonify
    )

    # 3. Train with green progress bar and 432 Hz Gamma-to-Epsilon binaural tracking
    println(" Initiating continuous wave evolution...")
    trained_wm, history = train!(
        model.backbone,
        dataset.inputs,
        dataset.targets,
        cfg;
        checkpoint_dir = checkpoint_dir,
        checkpoint_every = 10,
        verbose = true
    )

    if checkpoint_dir !== nothing
        mkpath(checkpoint_dir)
        final_ckpt = joinpath(checkpoint_dir, "spark_model.mkv")
        save_model(trained_wm, final_ckpt; include_audio=sonify, export_mp4=true)
    end

    return (SparkXModel(model.config, model.tokenizer, trained_wm), history)
end

end # module SparkTrain
