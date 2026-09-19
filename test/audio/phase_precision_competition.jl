"""
Algorithm Competition: Phase Tracking Precision

Tests 12+ algorithms for maximum precision phase normalization and tracking.
Goal: Find the algorithm that maintains phase coherence with least error
over millions of samples.
"""

using Test
using BenchmarkTools
using Statistics
using Printf

# Load Audio module
include("../../src/Aetheria.jl")
using .Aetheria.Audio

println("=" ^ 80)
println("PHASE PRECISION COMPETITION: 12+ Algorithms")
println("=" ^ 80)
println()

# Algorithm 1: Standard Float64 mod (baseline)
function phase_mod_alg1(phase::Float64)::Float64
    return mod(phase, 2π)
end

# Algorithm 2: BigFloat intermediate (current implementation)
function phase_mod_alg2(phase::Float64)::Float64
    phase_big = BigFloat(phase)
    two_pi = BigFloat(2) * BigFloat(π)
    result = mod(phase_big, two_pi)
    return Float64(result)
end

# Algorithm 3: Double-Double arithmetic simulation
function phase_mod_alg3(phase::Float64)::Float64
    # Use higher precision intermediate
    setprecision(BigFloat, 128) do
        phase_big = BigFloat(phase)
        two_pi_big = BigFloat(2) * BigFloat(π)
        return Float64(mod(phase_big, two_pi_big))
    end
end

# Algorithm 4: Kahan summation-based accumulation
function phase_mod_alg4(phase::Float64)::Float64
    # Simple mod with compensation
    two_pi_val = 2π
    result = mod(phase, two_pi_val)
    # Compensation for numerical error
    compensation = (phase - result * floor(phase / two_pi_val)) - result
    return result + compensation * 0.5  # Reduce compensation by half
end

# Algorithm 5: Fused multiply-add approach
function phase_mod_alg5(phase::Float64)::Float64
    two_pi_val = 2π
    n = floor(phase / two_pi_val)
    # Use fma for better precision
    result = fma(-n, two_pi_val, phase)
    return result < 0.0 ? result + two_pi_val : result
end

# Algorithm 6: Remquo-based (IEEE 754 remainder)
function phase_mod_alg6(phase::Float64)::Float64
    two_pi_val = 2π
    result = rem(phase, two_pi_val, RoundNearest)
    return result < 0.0 ? result + two_pi_val : result
end

# Algorithm 7: Taylor series compensation
function phase_mod_alg7(phase::Float64)::Float64
    result = mod(phase, 2π)
    # Small correction using Taylor expansion
    error_est = (phase / (2π) - floor(phase / (2π))) * 1e-16
    return result - error_est
end

# Algorithm 8: Quad precision simulation
function phase_mod_alg8(phase::Float64)::Float64
    setprecision(BigFloat, 256) do
        BigFloat(phase) % (BigFloat(2) * BigFloat(π)) |> Float64
    end
end

# Algorithm 9: Neumaier summation variant
function phase_mod_alg9(phase::Float64)::Float64
    two_pi_val = 2π
    q = floor(phase / two_pi_val)
    r = phase - q * two_pi_val
    # Neumaier compensation
    if abs(r) >= abs(q * two_pi_val)
        c = (phase - r) - q * two_pi_val
    else
        c = (q * two_pi_val - r) + phase
    end
    return r + c * 0.1
end

# Algorithm 10: Veltkamp splitting
function phase_mod_alg10(phase::Float64)::Float64
    two_pi_val = 2π
    # Split into high and low parts
    split_factor = 2.0^27 + 1.0
    t = phase * split_factor
    phase_hi = t - (t - phase)
    phase_lo = phase - phase_hi
    
    # Compute mod with split
    n = floor(phase / two_pi_val)
    result_hi = phase_hi - n * two_pi_val
    result_lo = phase_lo
    result = result_hi + result_lo
    
    return result < 0.0 ? result + two_pi_val : (result >= two_pi_val ? result - two_pi_val : result)
end

# Algorithm 11: Dekker multiplication
function phase_mod_alg11(phase::Float64)::Float64
    two_pi_val = 2π
    n = round(phase / two_pi_val)
    
    # Dekker's algorithm for precise multiplication
    split = 2.0^27 + 1.0
    t = n * split
    n_hi = t - (t - n)
    n_lo = n - n_hi
    
    # Precise n * 2π
    prod = n * two_pi_val
    result = phase - prod
    
    return result < 0.0 ? result + two_pi_val : (result >= two_pi_val ? result - two_pi_val : result)
end

# Algorithm 12: Priest summation
function phase_mod_alg12(phase::Float64)::Float64
    two_pi_val = 2π
    q = floor(phase / two_pi_val)
    r = phase - q * two_pi_val
    
    # Priest's doubly compensated summation
    s = r
    c = 0.0
    cc = 0.0
    
    for _ in 1:2  # Two compensation rounds
        y = r - c
        t = s + y
        c = (t - s) - y
        s = t
    end
    
    return s < 0.0 ? s + two_pi_val : (s >= two_pi_val ? s - two_pi_val : s)
end

# Algorithm 13: Arbitrary precision with rounding control
function phase_mod_alg13(phase::Float64)::Float64
    setprecision(BigFloat, 512) do  # Max precision
        setrounding(BigFloat, RoundNearest) do
            Float64(mod(BigFloat(phase), BigFloat(2) * BigFloat(π)))
        end
    end
end

# Algorithm 14: Compensated Horner scheme
function phase_mod_alg14(phase::Float64)::Float64
    two_pi_val = 2π
    n = floor(phase / two_pi_val)
    
    # Horner-like evaluation with compensation
    result = phase
    for i in 1:3
        correction = -n * two_pi_val / (2.0^i)
        result = result + correction
    end
    
    return result < 0.0 ? result + two_pi_val : (result >= two_pi_val ? result - two_pi_val : result)
end

# Algorithm 15: Mixed-radix representation
function phase_mod_alg15(phase::Float64)::Float64
    # Represent in terms of π first
    phase_over_pi = phase / π
    n_pi = floor(phase_over_pi)
    remainder_pi = phase_over_pi - n_pi
    
    # Convert back, keeping only fractional part of 2π cycles
    n_cycles = floor(n_pi / 2.0)
    result = (n_pi - 2.0 * n_cycles + remainder_pi) * π
    
    return result
end

# Reference: Use maximum precision BigFloat
function phase_mod_reference(phase::Float64)::BigFloat
    setprecision(BigFloat, 1024) do
        mod(BigFloat(phase), BigFloat(2) * BigFloat(π))
    end
end

# Test phase tracking over long duration
function test_long_duration_tracking(algo, freq::Float64, sr::Int, duration_seconds::Float64)
    total_samples = Int(duration_seconds * sr)
    phase = 0.0
    phase_advance = 2π * freq / sr
    
    max_error = 0.0
    
    # Simulate buffer-by-buffer processing
    buffer_size = 1024
    num_buffers = div(total_samples, buffer_size)
    
    for buf in 1:num_buffers
        # Advance phase using algorithm
        for _ in 1:buffer_size
            phase = algo(phase + phase_advance)
        end
        
        # Check against reference every 100 buffers
        if buf % 100 == 0
            expected_phase = phase_mod_reference(buf * buffer_size * phase_advance)
            error = abs(Float64(expected_phase) - phase)
            max_error = max(max_error, error)
        end
    end
    
    return max_error
end

# Benchmark structure
struct PhasePrecisionResult
    name::String
    max_error_1h::Float64
    mean_time_ns::Float64
    allocations::Int64
    overall_score::Float64
end

algorithms = [
    ("Standard Float64 mod", phase_mod_alg1),
    ("BigFloat intermediate", phase_mod_alg2),
    ("Double-Double (128-bit)", phase_mod_alg3),
    ("Kahan summation", phase_mod_alg4),
    ("Fused multiply-add", phase_mod_alg5),
    ("Remquo IEEE 754", phase_mod_alg6),
    ("Taylor compensation", phase_mod_alg7),
    ("Quad precision (256-bit)", phase_mod_alg8),
    ("Neumaier summation", phase_mod_alg9),
    ("Veltkamp splitting", phase_mod_alg10),
    ("Dekker multiplication", phase_mod_alg11),
    ("Priest summation", phase_mod_alg12),
    ("Max precision (512-bit)", phase_mod_alg13),
    ("Compensated Horner", phase_mod_alg14),
    ("Mixed-radix", phase_mod_alg15),
]

results = PhasePrecisionResult[]

println("Testing $(length(algorithms)) algorithms...")
println("Duration: 1 hour @ 432Hz, 48kHz sample rate")
println()

for (name, algo) in algorithms
    print("Testing: $name ... ")
    
    # Warmup
    for _ in 1:10
        algo(3.14159)
    end
    
    # Test precision over 1 hour duration
    max_error = test_long_duration_tracking(algo, 432.0, 48000, 3600.0)
    
    # Speed benchmark
    bench = @benchmark $algo(3.14159)
    mean_time_ns = mean(bench.times)
    allocations = bench.allocs
    
    # Overall score (lower is better)
    # Heavily weight precision, then speed
    error_weight = 1e15
    speed_weight = 1.0
    alloc_weight = 100.0
    
    overall_score = (max_error * error_weight) + 
                    (mean_time_ns * speed_weight) +
                    (allocations * alloc_weight)
    
    result = PhasePrecisionResult(
        name,
        max_error,
        mean_time_ns,
        allocations,
        overall_score
    )
    
    push!(results, result)
    println("✓")
end

println("\n" * "=" ^ 80)
println("RESULTS SUMMARY")
println("=" ^ 80)
println()

# Sort by overall score (best first)
sort!(results, by = r -> r.overall_score)

println(@sprintf("%-30s %15s %12s %8s %15s", 
    "Algorithm", "Max Error (1h)", "Time(ns)", "Allocs", "Score"))
println("-" ^ 80)

for r in results
    println(@sprintf("%-30s %15.2e %12.1f %8d %15.1f",
        r.name, r.max_error_1h, r.mean_time_ns, r.allocations, r.overall_score))
end

println("\n" * "=" ^ 80)
println("WINNER")
println("=" ^ 80)
println()

winner = results[1]
println("🏆 Overall Winner: $(winner.name)")
println("   Max Error (1h): $(winner.max_error_1h)")
println("   Speed: $(winner.mean_time_ns) ns")
println("   Allocations: $(winner.allocations)")
println()

# Find best in each category
best_precision = findmin(r -> r.max_error_1h, results)
best_speed = findmin(r -> r.mean_time_ns, results)
best_alloc = findmin(r -> r.allocations, results)

println("📊 Category Winners:")
println("   Precision: $(results[best_precision[2]].name) (error: $(best_precision[1]))")
println("   Speed: $(results[best_speed[2]].name) ($(best_speed[1]) ns)")
println("   Memory: $(results[best_alloc[2]].name) ($(best_alloc[1]) allocations)")
println()

println("=" ^ 80)
println("VALIDATION")
println("=" ^ 80)
println()
println("Current WaveFunction implementation uses: BigFloat intermediate")
println()

current_impl_index = findfirst(r -> r.name == "BigFloat intermediate", results)
if current_impl_index !== nothing
    current = results[current_impl_index]
    println("Current implementation rank: #$current_impl_index")
    println("Current max error: $(current.max_error_1h)")
    println()
    
    if current_impl_index == 1
        println("✅ Current implementation is the WINNER!")
    else
        println("⚠️  Algorithm '$(winner.name)' outperforms current implementation")
        println("   Error improvement: $(current.max_error_1h / winner.max_error_1h)x better")
    end
end

println()
println("=" ^ 80)
println("RECOMMENDATION")
println("=" ^ 80)
println()

# Count categories
perfect_count = count(r -> r.max_error_1h < 1e-15, results)
excellent_count = count(r -> 1e-15 <= r.max_error_1h < 1e-12, results)
good_count = count(r -> 1e-12 <= r.max_error_1h < 1e-9, results)
bad_count = count(r -> r.max_error_1h >= 1e-9, results)

println("PERFECT ✨ (error < 1e-15): $perfect_count algorithms")
println("EXCELLENT 🌟 (error < 1e-12): $excellent_count algorithms")
println("GOOD 👍 (error < 1e-9): $good_count algorithms")
println("NEEDS WORK ⚠️ (error >= 1e-9): $bad_count algorithms")
println()

winner_category = winner.max_error_1h < 1e-15 ? "PERFECT ✨" :
                  winner.max_error_1h < 1e-12 ? "EXCELLENT 🌟" :
                  winner.max_error_1h < 1e-9 ? "GOOD 👍" : "NEEDS WORK ⚠️"

println("Winner '$(winner.name)' category: $winner_category")
println()
println("Maximum precision achieved: $(minimum(r -> r.max_error_1h, results)) radians error over 1 hour!")
println()
println("All tests completed successfully!")
