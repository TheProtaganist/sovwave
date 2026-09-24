"""
    examples/mnist/src/train.jl

Continuous Evolutionary Training for MNIST Wave Model.
Evolves physical wave parameters directly to ground state with bold green terminal progress bar,
432 Hz Gamma-to-Epsilon binaural beat sonification, and fluid cymatic heatmap MKV serialization.
"""

module MNISTTrain

using Sovwave
using Sovwave.WaveML
using LinearAlgebra
using Printf

include("dataset.jl")
include("model.jl")
using .MNISTDataset
using .MNISTModel

export train_mnist_model

"""
    train_mnist_model(; epochs=20, limit=100, batch_size=16, checkpoint_dir="checkpoints/mnist", sonify=false)

Trains the continuous wave model on real Hugging Face MNIST images.
"""
function train_mnist_model(;
    epochs::Int = 20,
    limit::Int = 100,
    batch_size::Int = 16,
    checkpoint_dir::Union{Nothing, String} = "checkpoints/mnist",
    sonify::Bool = true,
    token::Union{Nothing, String} = nothing
)::Tuple{MNISTWaveNet, TrainingHistory}
    println("="^80)
    println(" 🌊 TRAINING CONTINUOUS WAVE MNIST CLASSIFIER (HUGGING FACE) 🌊")
    println("="^80)

    # 1. Download/Load real MNIST from Hugging Face Hub (token optional)
    println(" Loading $(limit) MNIST images from Hugging Face Hub...")
    dataset = load_mnist_dataset(; split="train", limit=limit, embed_dim=64, token=token)
    @printf(" ✓ Successfully projected %d images into continuous 2D surface harmonic waves\n", length(dataset))

    # 2. Create Continuous Wave Net
    net = create_mnist_wave_model(nodes=10, embed_dim=64, layers=1, carrier_omega=432.0)

    # 3. Seed continuous wave model with empirical harmonic prototypes from dataset
    prototypes = [zeros(64) for _ in 1:10]
    counts = zeros(Int, 10)
    for i in 1:length(dataset)
        for d in 1:10
            if dataset.targets[i][d] == 1.0
                prototypes[d] .+= dataset.inputs[i]
                counts[d] += 1
            end
        end
    end
    for d in 1:10
        if counts[d] > 0
            prototypes[d] ./= counts[d]
            if norm(prototypes[d]) > 1e-6
                prototypes[d] ./= norm(prototypes[d])
            end
        end
    end

    # Continuous contrastive wave interference refinement
    for ref_ep in 1:15
        for i in 1:length(dataset)
            target_d = argmax(dataset.targets[i])
            sims = [dot(dataset.inputs[i], prototypes[d]) for d in 1:10]
            pred_d = argmax(sims)
            if pred_d != target_d
                prototypes[target_d] .+= 0.05 .* dataset.inputs[i]
                prototypes[pred_d] .-= 0.025 .* dataset.inputs[i]
                prototypes[target_d] ./= max(1e-6, norm(prototypes[target_d]))
                prototypes[pred_d] ./= max(1e-6, norm(prototypes[pred_d]))
            end
        end
    end

    for d in 1:10
        for c in 1:64
            val = prototypes[d][c]
            net.model.layers[1].amplitudes[d, c] = abs(val)
            net.model.layers[1].phases[d, c] = val >= 0.0 ? 0.0 : Float64(π)
            net.model.layers[1].frequencies[d, c] = 1.0
        end
    end

    # 4. Evolutionary Training Configuration
    cfg = WaveTrainConfig(
        epochs = epochs,
        population_size = 12,
        learning_rate = 0.02,
        energy_target = 0.005,
        batch_size = batch_size,
        sonify = sonify
    )

    # 5. Train with green progress bar and Gamma-to-Epsilon binaural tracking
    println(" Initiating continuous wave evolution...")
    trained_model, history = train!(
        net.model,
        dataset.inputs,
        dataset.targets,
        cfg;
        checkpoint_dir = checkpoint_dir,
        checkpoint_every = 10,
        verbose = true
    )

    return (MNISTWaveNet(trained_model, net.carrier_omega, net.eigen_frequencies), history)
end

end # module MNISTTrain
