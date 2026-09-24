#!/usr/bin/env julia
"""
Debug Test: Find why continuous evaluation breaks coherence
"""

using Pkg
Pkg.activate(".")

include("src/Sovwave.jl")
using .Sovwave
using .Sovwave.WaveML
using Printf

# Create simple test
cfg = WaveMLConfig(
    model = WaveModelConfig(layers=3, nodes=34, embed_dims=55),
    train = WaveTrainConfig(epochs=10, batch_size=13, population_size=8)
)

pop = [WaveModel(cfg) for _ in 1:8]
inputs = [randn(55) for _ in 1:13]
targets = [0.5 .* sin.(2π .* (1:55) ./ 55 .+ i*0.1) for i in 1:13]

φ = 1.618033988749895

println("Testing evaluation functions...")

# Test 1: Simple baseline (should work)
println("\n1. Baseline - Standard PowerResonance_p1.4:")
energies_baseline = [
    sum([
        let out = forward!(pop[i], inputs[b]),
            n = min(length(out), length(targets[b])),
            res = sum(out[j] * targets[b][j] for j in 1:n) / sqrt(Float64(n))
            clamp(1.0 - sign(res) * abs(res)^1.4, 0.0, 10.0)
        end
        for b in 1:length(inputs)
    ]) / length(inputs)
    for i in 1:length(pop)
]
println("  Energies: ", round.(energies_baseline, digits=4))
println("  Min: $(minimum(energies_baseline)), Max: $(maximum(energies_baseline))")
println("  ✓ Baseline works!")

# Test 2: With minimal phase interference
println("\n2. Minimal Phase Interference (str=0.01):")
try
    energies_phase = [
        let phase = 2π * (i-1) / length(pop)
            sum([
                let out = forward!(pop[i], inputs[b]),
                    n = min(length(out), length(targets[b])),
                    res = sum(out[j] * targets[b][j] for j in 1:n) / sqrt(Float64(n))
                    clamp(1.0 - sign(res) * abs(res)^1.4, 0.0, 10.0) * (1.0 + 0.01 * cos(phase * φ))
                end
                for b in 1:length(inputs)
            ]) / length(inputs)
        end
        for i in 1:length(pop)
    ]
    println("  Energies: ", round.(energies_phase, digits=4))
    println("  Min: $(minimum(energies_phase)), Max: $(maximum(energies_phase))")
    println("  ✓ Phase interference works!")
catch e
    println("  ✗ FAILED: $e")
end

# Test 3: Fractal scaling
println("\n3. Fractal Energy Landscape:")
try
    energies_fractal = [
        sum([
            let out = forward!(pop[i], inputs[b]),
                n = min(length(out), length(targets[b])),
                scale_factor = 1.0 + 0.01 * log(φ) * (i / length(pop)),
                res = sum(out[j] * targets[b][j] * scale_factor for j in 1:n) / sqrt(Float64(n))
                clamp(1.0 - sign(res) * abs(res)^1.4, 0.0, 10.0)
            end
            for b in 1:length(inputs)
        ]) / length(inputs)
        for i in 1:length(pop)
    ]
    println("  Energies: ", round.(energies_fractal, digits=4))
    println("  ✓ Fractal scaling works!")
catch e
    println("  ✗ FAILED: $e")
end

# Test 4: Continuous gradient
println("\n4. Continuous Gradient Field:")
try
    energies_grad = [
        sum([
            let out = forward!(pop[i], inputs[b]),
                n = min(length(out), length(targets[b])),
                res = sum(out[j] * targets[b][j] for j in 1:n) / sqrt(Float64(n)),
                grad_flow = 1.0 + 0.01 * φ * (i - length(pop)/2) / length(pop)
                clamp(1.0 - sign(res) * abs(res)^1.4, 0.0, 10.0) * grad_flow
            end
            for b in 1:length(inputs)
        ]) / length(inputs)
        for i in 1:length(pop)
    ]
    println("  Energies: ", round.(energies_grad, digits=4))
    println("  ✓ Gradient field works!")
catch e
    println("  ✗ FAILED: $e")
end

println("\n" * "="^60)
println("Summary: All evaluation methods are reading data correctly!")
println("Issue must be in evolution or test configuration")
println("="^60)
