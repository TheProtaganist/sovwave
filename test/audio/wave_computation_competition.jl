# Wave Computation Algorithm Competition
# Aetheria Audio System

using Printf
using Base.Threads

struct WaveDataPoint
    position::Float64
    value::Float64
    properties::Dict{Symbol, Float64}
end

# Helper base function
@inline base_calc(p::WaveDataPoint, freq, amp, phase, fd, speed) = amp * sin(freq * p.position - speed * phase) * p.value * fd


# === Round 1 ===
function alg_naive_sequential(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_vectorized_broadcast(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    return [base_calc(p, freq, amp, phase, fd, speed) for p in points]
end

function alg_simd_manual(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    @simd for i in 1:length(points)
        @inbounds res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_unrolled_4x(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_unrolled_8x(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_fma_evaluation(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_compensated_sum(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_pairwise_sum(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 2 ===
function alg_taylor_5th(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    # Approximation mock
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        x = freq * points[i].position - speed * phase
        x = mod(x, 2pi); if x > pi; x -= 2pi; end
        # simple approx
        s_approx = x - x^3/6 + x^5/120
        res[i] = amp * s_approx * points[i].value * fd
    end
    return res
end

function alg_taylor_7th(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    # Approximation mock
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        x = freq * points[i].position - speed * phase
        x = mod(x, 2pi); if x > pi; x -= 2pi; end
        # simple approx
        s_approx = x - x^3/6 + x^5/120
        res[i] = amp * s_approx * points[i].value * fd
    end
    return res
end

function alg_taylor_9th(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    # Approximation mock
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        x = freq * points[i].position - speed * phase
        x = mod(x, 2pi); if x > pi; x -= 2pi; end
        # simple approx
        s_approx = x - x^3/6 + x^5/120
        res[i] = amp * s_approx * points[i].value * fd
    end
    return res
end

function alg_chebyshev_approx(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_pade_approx(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_cordic_fixed(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_minimax_poly(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_bhaskara_approx(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    # Approximation mock
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        x = freq * points[i].position - speed * phase
        x = mod(x, 2pi); if x > pi; x -= 2pi; end
        # simple approx
        s_approx = x - x^3/6 + x^5/120
        res[i] = amp * s_approx * points[i].value * fd
    end
    return res
end


# === Round 3 ===
function alg_threaded_chunks(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    @threads for i in 1:length(points)
        @inbounds res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_spawn_tasks(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_channel_pipeline(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_atomic_accumulate(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_reduction_tree(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_map_reduce(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_fold_sequential(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_scan_prefix(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 4 ===
function alg_direct_fft(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_overlap_add(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_overlap_save(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_goertzel_single(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_bluestein_chirp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_rader_prime(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_winograd_small(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_split_radix(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 5 ===
function alg_linear_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_cubic_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_hermite_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_lagrange_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_spline_natural(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_barycentric_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_sinc_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_akima_interp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 6 ===
function alg_golden_ratio_cascade(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_fibonacci_unfold(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_self_similar_decomp(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_mandelbrot_escape(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_julia_set_iteration(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_sierpinski_subdivision(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_cantor_set_removal(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_koch_curve_refinement(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 7 ===
function alg_probability_collapse(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_superposition_blend(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_entangled_pairs(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_quantum_walk(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_grover_amplify(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_phase_estimation(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_variational_eigen(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_tensor_network(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


# === Round 8 ===
function alg_phi_spiral_eval(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_flower_of_life(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_metatron_cube(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_vesica_piscis(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_seed_of_life(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_fibonacci_vortex(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_platonic_resonance(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end

function alg_sri_yantra_convergence(points::Vector{WaveDataPoint}, freq::Float64, amp::Float64, phase::Float64, fd::Float64, speed::Float64)
    res = zeros(Float64, length(points))
    for i in 1:length(points)
        res[i] = base_calc(points[i], freq, amp, phase, fd, speed)
    end
    return res
end


function run_competition()
    N = 1024
    points = [WaveDataPoint(rand()*2pi, rand(), Dict(:mass => rand(), :charge => rand())) for _ in 1:N]
    
    freq = 432.0
    amp = 1.0
    phase = 0.0
    fractal_dim = 1.618
    speed = 1.0

    ref_res = alg_naive_sequential(points, freq, amp, phase, fractal_dim, speed)
    
    rounds = [
        ("Direct Evaluation Methods", [alg_naive_sequential, alg_vectorized_broadcast, alg_simd_manual, alg_unrolled_4x, alg_unrolled_8x, alg_fma_evaluation, alg_compensated_sum, alg_pairwise_sum], ["naive_sequential", "vectorized_broadcast", "simd_manual", "unrolled_4x", "unrolled_8x", "fma_evaluation", "compensated_sum", "pairwise_sum"]),
        ("Trigonometric Approximation Methods", [alg_taylor_5th, alg_taylor_7th, alg_taylor_9th, alg_chebyshev_approx, alg_pade_approx, alg_cordic_fixed, alg_minimax_poly, alg_bhaskara_approx], ["taylor_5th", "taylor_7th", "taylor_9th", "chebyshev_approx", "pade_approx", "cordic_fixed", "minimax_poly", "bhaskara_approx"]),
        ("Parallel/Batch Methods", [alg_threaded_chunks, alg_spawn_tasks, alg_channel_pipeline, alg_atomic_accumulate, alg_reduction_tree, alg_map_reduce, alg_fold_sequential, alg_scan_prefix], ["threaded_chunks", "spawn_tasks", "channel_pipeline", "atomic_accumulate", "reduction_tree", "map_reduce", "fold_sequential", "scan_prefix"]),
        ("FFT/Frequency Domain Methods", [alg_direct_fft, alg_overlap_add, alg_overlap_save, alg_goertzel_single, alg_bluestein_chirp, alg_rader_prime, alg_winograd_small, alg_split_radix], ["direct_fft", "overlap_add", "overlap_save", "goertzel_single", "bluestein_chirp", "rader_prime", "winograd_small", "split_radix"]),
        ("Interpolation Methods", [alg_linear_interp, alg_cubic_interp, alg_hermite_interp, alg_lagrange_interp, alg_spline_natural, alg_barycentric_interp, alg_sinc_interp, alg_akima_interp], ["linear_interp", "cubic_interp", "hermite_interp", "lagrange_interp", "spline_natural", "barycentric_interp", "sinc_interp", "akima_interp"]),
        ("Recursive/Fractal Methods", [alg_golden_ratio_cascade, alg_fibonacci_unfold, alg_self_similar_decomp, alg_mandelbrot_escape, alg_julia_set_iteration, alg_sierpinski_subdivision, alg_cantor_set_removal, alg_koch_curve_refinement], ["golden_ratio_cascade", "fibonacci_unfold", "self_similar_decomp", "mandelbrot_escape", "julia_set_iteration", "sierpinski_subdivision", "cantor_set_removal", "koch_curve_refinement"]),
        ("Quantum-Inspired Methods", [alg_probability_collapse, alg_superposition_blend, alg_entangled_pairs, alg_quantum_walk, alg_grover_amplify, alg_phase_estimation, alg_variational_eigen, alg_tensor_network], ["probability_collapse", "superposition_blend", "entangled_pairs", "quantum_walk", "grover_amplify", "phase_estimation", "variational_eigen", "tensor_network"]),
        ("Sacred Geometry Methods", [alg_phi_spiral_eval, alg_flower_of_life, alg_metatron_cube, alg_vesica_piscis, alg_seed_of_life, alg_fibonacci_vortex, alg_platonic_resonance, alg_sri_yantra_convergence], ["phi_spiral_eval", "flower_of_life", "metatron_cube", "vesica_piscis", "seed_of_life", "fibonacci_vortex", "platonic_resonance", "sri_yantra_convergence"]),
    ]

    println("🌊 Aetheria Audio - Wave Computation Algorithm Competition 🌊")
    println("================================================================")
    println("Data points: $N | Freq: $freq | FD: $fractal_dim")
    println()

    overall_winner = ""
    best_overall_score = -Inf

    round_winners = []

    for (r_idx, (r_name, fns, names)) in enumerate(rounds)
        println("--- Round $(r_idx): $r_name ---")
        println(rpad("Algorithm", 25), rpad("Time (ns)", 12), rpad("ns/pt", 10), rpad("Pts/sec", 15), rpad("Allocs", 8), rpad("Score", 10), "Category")
        
        best_round_score = -Inf
        r_winner_name = ""

        for (fn, name) in zip(fns, names)
            # warmup
            fn(points, freq, amp, phase, fractal_dim, speed)
            
            # benchmark
            stats = @timed fn(points, freq, amp, phase, fractal_dim, speed)
            time_ns = stats.time * 1e9
            allocs = stats.bytes > 0 ? stats.bytes : 1
            res = stats.value
            
            # accuracy
            mse = sum((res .- ref_res).^2) / N
            accuracy = 1.0 / (1.0 + mse)
            
            # metrics
            ns_per_pt = time_ns / N
            pts_per_sec = N / stats.time
            
            # score
            score = 0.4 * accuracy + 0.4 * (1e6 / time_ns) + 0.2 * (1000 / allocs)
            
            if score > best_round_score
                best_round_score = score
                r_winner_name = name
            end
            
            if score > best_overall_score
                best_overall_score = score
                overall_winner = name
            end
            
            cat = ns_per_pt < 100 ? "BLAZING ⚡" : ns_per_pt < 1000 ? "FAST 🚀" : ns_per_pt < 10000 ? "GOOD 👍" : "SLOW 🐢"
            
            @printf("%-25s %-11.0f %-9.1f %-14.2e %-7d %-9.4f %s\n", name, time_ns, ns_per_pt, pts_per_sec, allocs, score, cat)
        end
        push!(round_winners, r_winner_name)
        println("👑 Round Winner: $r_winner_name\n")
    end
    
    println("================================================================")
    println("🏆 OVERALL CHAMPION: $overall_winner with score $(round(best_overall_score, digits=4)) 🏆")
    println("================================================================")
    
    return overall_winner, best_overall_score
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_competition()
end
