"""
    examples/simple_neuron/neuron.jl

Simple Continuous Wave Neuron: Binary ON / OFF (XOR Logic Gate)
Operates solely on wave mechanics and continuous harmonic interference.
Uses Sovwave as a library.
"""

# Include Sovwave library
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "../../")))
using Sovwave
using Sovwave.WaveML
using LinearAlgebra
using Printf

"""
    main()

Executes single continuous wave oscillator demonstration solving non-linear binary XOR.
"""
function main()
    println("="^80)
    println(" 🌊 SOVWAVE EXAMPLE: SIMPLE CONTINUOUS WAVE NEURON (ON/OFF) 🌊")
    println("="^80)
    println(" Solves non-linear binary XOR classification using a single continuous wave oscillator.")
    println(" Zero discrete matrix multiplication | Zero discrete Markov chains | Zero backprop")
    println("-"^80)

    # Truth table for non-linear XOR problem:
    # (0, 0) -> OFF (0)
    # (0, 1) -> ON  (1)
    # (1, 0) -> ON  (1)
    # (1, 1) -> OFF (0)
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    targets = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    # Create a continuous wave model configured with harmonic phase resonance
    m_cfg = WaveModelConfig(nodes=4, embed_dims=2, layers=2, omega=6283.185)
    cfg = WaveMLConfig(model=m_cfg)
    model = WaveModel(cfg)

    # Configure continuous evolutionary training
    train_cfg = WaveTrainConfig(
        epochs = 120,
        population_size = 32,
        learning_rate = 0.12,
        energy_target = 0.005,
        sonify = true, # Set to true to hear the 432 Hz Gamma-to-Epsilon binaural beat
        batch_size = 4
    )

    println(" Training single wave neuron with continuous evolution...")
    trained_model, history = train!(model, inputs, targets, train_cfg; verbose=false)

    println("\n" * "="^80)
    println(" 🎯 INFERENCE & EVALUATION RESULTS:")
    println("="^80)
    println(" Input [X1, X2]  | Predicted Continuous Wave Output | State Output | Expected | Match?")
    println("-"^80)

    all_correct = true
    for i in 1:4
        pred = forward!(trained_model, inputs[i])
        out_val = pred[1]
        # Grand Champion Cymatic Threshold
        predicted_state = out_val >= 0.5 ? "ON  [1]" : "OFF [0]"
        expected_state  = targets[i][1] == 1.0 ? "ON  [1]" : "OFF [0]"
        match = (predicted_state == expected_state)
        if !match; all_correct = false; end

        @printf("   [%1.0f, %1.0f]       |              %7.4f             |   %s   |  %s  |   %s\n",
                inputs[i][1], inputs[i][2], out_val, predicted_state, expected_state, match ? "✅ PASS" : "❌ FAIL")
    end
    println("="^80)
    @printf(" Final Training Energy: %.6f | Best Epoch: %d | Accuracy: %s\n",
            history.best_loss, history.best_epoch, all_correct ? "100.0% (Perfect)" : "Partial")
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
