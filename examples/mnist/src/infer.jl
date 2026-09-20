"""
    examples/mnist/src/infer.jl

Inference and Evaluation for MNIST Wave Model.
Supports evaluating in-memory models or directly loading trained weights from fluid heatmap MKV video files!
"""

module MNISTInfer

using Sovwave
using Sovwave.WaveML
using Printf

include("dataset.jl")
include("model.jl")
using .MNISTDataset
using .MNISTModel

export evaluate_mnist_model

"""
    evaluate_mnist_model(net::Union{Nothing, MNISTWaveNet} = nothing; mkv_path::Union{Nothing, String} = nothing, limit::Int = 50)

Runs inference on test MNIST digits from Hugging Face.
If `mkv_path` is specified, restores model weights directly from the MKV video file!
"""
function evaluate_mnist_model(
    net::Union{Nothing, MNISTWaveNet} = nothing;
    mkv_path::Union{Nothing, String} = nothing,
    limit::Int = 50,
    token::Union{Nothing, String} = nothing
)
    # Restore model from MKV if path provided
    active_net = if mkv_path !== nothing && isfile(mkv_path)
        println(" Loading continuous wave model weights directly from MKV video: $mkv_path")
        loaded_wm = load_model(mkv_path)
        carrier = loaded_wm.model_config.omega
        eigen_freqs = [carrier * (1.618033988749895^(d * 0.25)) for d in 0:9]
        MNISTWaveNet(loaded_wm, carrier, eigen_freqs)
    elseif net !== nothing
        net
    else
        error("Must provide either a trained MNISTWaveNet or a valid mkv_path")
    end

    # Download test split from Hugging Face
    println(" Loading $(limit) test MNIST samples from Hugging Face Hub (split=test)...")
    images, labels = download_mnist_hf(split="test", limit=limit, token=token)
    wave_inputs = process_pixel_waves(images; embed_dim=active_net.model.model_config.embed_dims)

    println("\n" * "="^85)
    println(" 🎯 HUGGING FACE MNIST TEST INFERENCE & CYMATIC FREQUENCY DECODING:")
    println("="^85)
    println(" Sample # | True Digit | Predicted | Match? | Confidence | Dominant Cymatic Frequency")
    println("-"^85)

    correct = 0
    total = length(labels)

    for i in 1:total
        pred_d, conf, probs = predict_digit(active_net, wave_inputs[i])
        true_d = labels[i]
        match = (pred_d == true_d)
        if match; correct += 1; end

        cymatic_freq = active_net.eigen_frequencies[pred_d + 1]
        @printf("   %3d    |     %d      |     %d     | %s |   %5.1f%%   |         %7.2f Hz\n",
                i, true_d, pred_d, match ? "✅ PASS" : "❌ FAIL", conf * 100.0, cymatic_freq)
    end

    acc = Float64(correct) / Float64(total)
    println("="^85)
    @printf(" Test Accuracy: %.2f%% (%d/%d correct) on real Hugging Face MNIST digits\n",
            acc * 100.0, correct, total)
    println("="^85)
    return acc
end

end # module MNISTInfer
