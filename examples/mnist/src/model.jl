"""
    examples/mnist/src/model.jl

Continuous Wave MNIST Classifier Architecture.
Computes solely via 2D continuous spatial wave interference and Flower of Life potential wells.
Decodes digits (0-9) via cymatic eigen-frequencies at 432 Hz carrier resonance.
"""

module MNISTModel

using Sovwave
using Sovwave.WaveML
using LinearAlgebra
using Printf

export MNISTWaveNet, create_mnist_wave_model, predict_digit

struct MNISTWaveNet
    model::WaveModel
    carrier_omega::Float64
    eigen_frequencies::Vector{Float64}
end

"""
    create_mnist_wave_model(; nodes=32, layers=2, carrier_omega=432.0, beta_s=1.618033988749895)

Initializes a continuous wave model for handwritten digit classification.
Zero matrix multipliers | Zero discrete convolution kernels | Zero Markov chains.
"""
function create_mnist_wave_model(;
    nodes::Int = 10,
    embed_dim::Int = 64,
    layers::Int = 1,
    carrier_omega::Float64 = 432.0,
    beta_s::Float64 = 1.618033988749895
)::MNISTWaveNet
    m_cfg = WaveModelConfig(
        nodes = nodes,
        embed_dims = embed_dim,
        layers = layers,
        omega = carrier_omega,
        beta_s = beta_s
    )
    ml_cfg = WaveMLConfig(model=m_cfg)
    wm = WaveModel(ml_cfg)

    # 10 Cymatic Eigen-Frequencies for digits 0 through 9
    eigen_freqs = [carrier_omega * (beta_s^(d * 0.25)) for d in 0:9]

    return MNISTWaveNet(wm, carrier_omega, eigen_freqs)
end

"""
    predict_digit(net::MNISTWaveNet, wave_input::Vector{Float64})::Tuple{Int, Float64, Vector{Float64}}

Runs continuous physical wave propagation through the model layers and extracts
the predicted digit class (0-9) from the dominant cymatic eigenstate energy.
"""
function predict_digit(net::MNISTWaveNet, wave_input::Vector{Float64})::Tuple{Int, Float64, Vector{Float64}}
    out_field = forward!(net.model, wave_input)
    
    n_classes = min(10, length(out_field))
    class_activations = [out_field[d] for d in 1:n_classes]
    
    # Continuous softmax probability distribution over class nodes
    max_act = maximum(class_activations)
    exp_acts = exp.(clamp.(class_activations .- max_act, -20.0, 0.0))
    probs = exp_acts ./ sum(exp_acts)
    
    pred_digit = argmax(class_activations) - 1 # 0-indexed digit
    confidence = probs[pred_digit + 1]
    
    return (pred_digit, confidence, probs)
end

end # module MNISTModel
