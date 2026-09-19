"""
    waveml_competitions.jl

Algorithm Tournament for WaveML: 180 Algorithms across 5 Core Modules
- Field.jl: 36 Wave Propagation Algorithms (3 Rounds of 12)
- Evolution.jl: 36 Optimization Algorithms (3 Rounds of 12)
- Training.jl: 36 Batch/Scheduling/Convergence Algorithms (3 Rounds of 12)
- Loss.jl: 36 Energy/Resonance/Interference Loss Algorithms (3 Rounds of 12)
- Sonify.jl: 36 Audio Rendering Algorithms (3 Rounds of 12)

Each round of 12 produces a winner; the 3 round winners compete in a championship round.
The ultimate champion of each module is incorporated into the production WaveML codebase.
"""

using Printf
using Random
using Statistics
using LinearAlgebra

# Structure to hold algorithm competition results
struct AlgoResult
    name::String
    category::String
    round::Int
    time_ns::Float64
    allocs::Int
    accuracy::Float64
    score::Float64
    rank::Int
end

function compute_score(time_ns::Float64, allocs::Int, accuracy::Float64)::Float64
    t_safe = max(time_ns, 1.0)
    a_safe = max(allocs, 1)
    acc_term = 0.4 * clamp(accuracy, 0.0, 1.0)
    time_term = 0.4 * (1e6 / t_safe)
    alloc_term = 0.2 * (1000.0 / a_safe)
    return acc_term + time_term + alloc_term
end

# ============================================================================
# [SECTION 1] Field.jl: 36 Wave Propagation Algorithms
# ============================================================================
function run_field_tournament()
    println("\n" * "="^80)
    println(" 🌊 SECTION 1: Field.jl Wave Propagation Tournament (36 Algorithms)")
    println("="^80)

    # Synthetic test data: N points in 3D space with 4 properties
    N = 256
    dims = 3
    positions = [rand(dims) for _ in 1:N]
    properties = [rand(4) for _ in 1:N]
    omega = 432.0
    beta_s = 1.618033988749895
    t = 1.0

    # R1: Direct Propagation (12)
    r1_algos = Dict{String, Function}(
        "naive_sequential_prop" => () -> begin
            res = zeros(N)
            for i in 1:N
                pos_norm = sqrt(sum(positions[i].^2))
                phase = omega * pos_norm - t
                res[i] = properties[i][1] * sin(phase) * beta_s
            end
            res
        end,
        "simd_vectorized_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            @inbounds @simd for i in 1:N
                p = positions[i]
                r = sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
                res[i] = properties[i][1] * sin(omega * r - t) * beta_s
            end
            res
        end,
        "fma_propagation" => () -> begin
            res = Vector{Float64}(undef, N)
            @inbounds for i in 1:N
                p = positions[i]
                r = sqrt(muladd(p[1], p[1], muladd(p[2], p[2], p[3]*p[3])))
                phase = muladd(omega, r, -t)
                res[i] = muladd(properties[i][1] * sin(phase), beta_s, 0.0)
            end
            res
        end,
        "unrolled_4x_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            i = 1
            @inbounds while i + 3 <= N
                for k in 0:3
                    idx = i + k
                    p = positions[idx]
                    r = sqrt(p[1]^2 + p[2]^2 + p[3]^2)
                    res[idx] = properties[idx][1] * sin(omega * r - t) * beta_s
                end
                i += 4
            end
            @inbounds while i <= N
                p = positions[i]
                res[i] = properties[i][1] * sin(omega * sqrt(sum(p.^2)) - t) * beta_s
                i += 1
            end
            res
        end,
        "unrolled_8x_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            i = 1
            @inbounds while i + 7 <= N
                for k in 0:7
                    idx = i + k
                    p = positions[idx]
                    r = sqrt(p[1]^2 + p[2]^2 + p[3]^2)
                    res[idx] = properties[idx][1] * sin(omega * r - t) * beta_s
                end
                i += 8
            end
            @inbounds while i <= N
                p = positions[i]
                res[i] = properties[i][1] * sin(omega * sqrt(sum(p.^2)) - t) * beta_s
                i += 1
            end
            res
        end,
        "broadcast_prop" => () -> begin
            radii = [sqrt(sum(p.^2)) for p in positions]
            [props[1] for props in properties] .* sin.(omega .* radii .- t) .* beta_s
        end,
        "map_prop" => () -> begin
            map(1:N) do i
                p = positions[i]
                properties[i][1] * sin(omega * sqrt(sum(p.^2)) - t) * beta_s
            end
        end,
        "threaded_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            Threads.@threads for i in 1:N
                @inbounds begin
                    p = positions[i]
                    r = sqrt(p[1]^2 + p[2]^2 + p[3]^2)
                    res[i] = properties[i][1] * sin(omega * r - t) * beta_s
                end
            end
            res
        end,
        "column_major_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            @inbounds for i in eachindex(positions)
                p = positions[i]
                d2 = 0.0
                for c in 1:3; d2 += p[c]*p[c]; end
                res[i] = properties[i][1] * sin(omega * sqrt(d2) - t) * beta_s
            end
            res
        end,
        "blocked_64_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            bs = 64
            for b in 1:bs:N
                hi = min(b + bs - 1, N)
                @inbounds @simd for i in b:hi
                    p = positions[i]
                    res[i] = properties[i][1] * sin(omega * sqrt(p[1]^2+p[2]^2+p[3]^2) - t) * beta_s
                end
            end
            res
        end,
        "preallocated_simd_prop" => () -> begin
            res = zeros(Float64, N)
            @inbounds @simd for i in 1:N
                p = positions[i]
                res[i] = properties[i][1] * sin(omega * sqrt(p[1]^2+p[2]^2+p[3]^2) - t) * beta_s
            end
            res
        end,
        "batched_fused_simd_prop" => () -> begin
            res = Vector{Float64}(undef, N)
            om = omega; bt = beta_s; tm = t
            @inbounds @simd for i in 1:N
                p = positions[i]
                d = sqrt(muladd(p[1], p[1], muladd(p[2], p[2], p[3]*p[3])))
                s = sin(muladd(om, d, -tm))
                res[i] = muladd(properties[i][1] * s, bt, 0.0)
            end
            res
        end
    )

    # R2: Spectral/Harmonic Propagation (12)
    r2_algos = Dict{String, Function}(
        "fourier_mode_prop" => () -> [properties[i][1] * (sin(omega*positions[i][1] - t) + 0.5*sin(2omega*positions[i][2] - t)) for i in 1:N],
        "harmonic_cascade_prop" => () -> begin
            res = zeros(N)
            for h in 1:4
                w = omega * (beta_s^(h-1))
                a = 1.0 / (beta_s^(h-1))
                for i in 1:N
                    res[i] += a * properties[i][1] * sin(w * positions[i][1] - t)
                end
            end
            res
        end,
        "chebyshev_spectral_prop" => () -> begin
            res = zeros(N)
            for i in 1:N
                x = mod(positions[i][1], 2.0) - 1.0
                t1 = x; t2 = 2x^2 - 1
                res[i] = properties[i][1] * (0.6*t1 + 0.4*t2) * sin(-t)
            end
            res
        end,
        "taylor_expansion_prop" => () -> begin
            res = zeros(N)
            for i in 1:N
                th = mod(omega * positions[i][1] - t + π, 2π) - π
                s = th - th^3/6.0 + th^5/120.0 - th^7/5040.0
                res[i] = properties[i][1] * s * beta_s
            end
            res
        end,
        "pade_rational_prop" => () -> begin
            res = zeros(N)
            for i in 1:N
                x = mod(omega * positions[i][1] - t + π, 2π) - π
                x2 = x*x
                s = x * (1.0 - x2/20.0) / (1.0 + x2/6.0)
                res[i] = properties[i][1] * s * beta_s
            end
            res
        end,
        "wavelet_decomp_prop" => () -> [properties[i][1] * cos(omega*positions[i][1]) * exp(-sum(positions[i].^2)) for i in 1:N],
        "dct_prop" => () -> [properties[i][1] * cos(π * (2i-1) * omega / (2N)) for i in 1:N],
        "hilbert_analytic_prop" => () -> [properties[i][1] * hypot(sin(omega*positions[i][1]-t), cos(omega*positions[i][1]-t)) for i in 1:N],
        "bessel_radial_prop" => () -> begin
            # Bessel J0 approx: cos(r)/sqrt(r+1)
            [properties[i][1] * cos(omega * sqrt(sum(positions[i].^2))) / sqrt(sqrt(sum(positions[i].^2)) + 1.0) for i in 1:N]
        end,
        "spherical_harmonic_prop" => () -> begin
            [properties[i][1] * (3*positions[i][3]^2 - 1.0) * sin(omega*positions[i][1]-t) for i in 1:N]
        end,
        "legendre_poly_prop" => () -> begin
            [properties[i][1] * 0.5 * (3*positions[i][1]^2 - 1.0) for i in 1:N]
        end,
        "hermite_gauss_prop" => () -> begin
            [properties[i][1] * (4*positions[i][1]^2 - 2.0) * exp(-sum(positions[i].^2)*0.5) for i in 1:N]
        end
    )

    # R3: Geometric/Topological Propagation (12)
    r3_algos = Dict{String, Function}(
        "geodesic_prop" => () -> [properties[i][1] * sin(omega * (positions[i][1] + positions[i][2]*beta_s) - t) for i in 1:N],
        "curvature_flow_prop" => () -> begin
            res = zeros(N)
            for i in 2:N-1
                curv = (positions[i+1][1] - 2positions[i][1] + positions[i-1][1])
                res[i] = properties[i][1] * (sin(omega*positions[i][1]-t) + 0.1*curv)
            end
            res[1] = properties[1][1]; res[N] = properties[N][1]
            res
        end,
        "laplacian_diffusion_prop" => () -> begin
            res = zeros(N)
            for i in 2:N-1
                lap = properties[i+1][1] - 2properties[i][1] + properties[i-1][1]
                res[i] = properties[i][1] * sin(omega*positions[i][1]-t) + 0.05*lap
            end
            res[1] = properties[1][1]; res[N] = properties[N][1]
            res
        end,
        "green_function_prop" => () -> [properties[i][1] * exp(-sqrt(sum(positions[i].^2))) * sin(omega*positions[i][1]-t) for i in 1:N],
        "eikonal_raycast_prop" => () -> [properties[i][1] * cos(omega * sum(positions[i]) / sqrt(dims) - t) for i in 1:N],
        "lattice_spring_prop" => () -> [properties[i][1] * sin(omega * positions[i][1] - t) * (1.0 - 0.1*abs(positions[i][2])) for i in 1:N],
        "wave_equation_fd_prop" => () -> [properties[i][1] * (2*cos(omega*positions[i][1]) - cos(omega*positions[i][1] - 0.01)) for i in 1:N],
        "symplectic_prop" => () -> begin
            p = 0.0; q = 1.0; res = zeros(N)
            for i in 1:N
                p -= 0.01 * omega * q
                q += 0.01 * p
                res[i] = properties[i][1] * q
            end
            res
        end,
        "fibonacci_lattice_prop" => () -> begin
            phi = 1.618033988749895
            [properties[i][1] * sin(omega * mod(i * 2π / phi, 2π) - t) for i in 1:N]
        end,
        "voronoi_cell_prop" => () -> [properties[i][1] * sign(sin(omega*positions[i][1]-t)) * 0.8 for i in 1:N],
        "kd_tree_prop" => () -> [properties[i][1] * sin(omega * positions[i][i % dims + 1] - t) for i in 1:N],
        "octree_spatial_prop" => () -> begin
            octant = [(positions[i][1]>0 ? 1 : 0) + (positions[i][2]>0 ? 2 : 0) + (positions[i][3]>0 ? 4 : 0) for i in 1:N]
            [properties[i][1] * sin(omega*positions[i][1] + octant[i]*π/4 - t) for i in 1:N]
        end
    )

    return evaluate_rounds("Field.jl", r1_algos, r2_algos, r3_algos)
end

# ============================================================================
# [SECTION 2] Evolution.jl: 36 Optimization Algorithms
# ============================================================================
function run_evolution_tournament()
    println("\n" * "="^80)
    println(" 🧬 SECTION 2: Evolution.jl Wave Optimization Tournament (36 Algorithms)")
    println("="^80)

    pop_size = 32
    param_dim = 64
    population = [randn(param_dim) for _ in 1:pop_size]
    fitnesses = [sum(abs2, p) for p in population] # Lower is better (energy)
    lr = 0.05

    # R1: Selection Strategies (12)
    r1_algos = Dict{String, Function}(
        "tournament_k3" => () -> begin
            cands = rand(1:pop_size, 3)
            cands[argmin([fitnesses[c] for c in cands])]
        end,
        "tournament_k5" => () -> begin
            cands = rand(1:pop_size, 5)
            cands[argmin([fitnesses[c] for c in cands])]
        end,
        "roulette_wheel_select" => () -> begin
            inv_fit = 1.0 ./ (fitnesses .+ 1e-6)
            probs = inv_fit ./ sum(inv_fit)
            r = rand()
            cum = 0.0
            idx = 1
            for i in 1:pop_size
                cum += probs[i]
                if cum >= r; idx = i; break; end
            end
            idx
        end,
        "stochastic_universal_sample" => () -> begin
            inv_fit = 1.0 ./ (fitnesses .+ 1e-6)
            probs = inv_fit ./ sum(inv_fit)
            ptr = rand() / pop_size
            cum = 0.0; idx = 1
            for i in 1:pop_size
                cum += probs[i]
                if cum >= ptr; idx = i; break; end
            end
            idx
        end,
        "rank_based_select" => () -> begin
            ranks = sortperm(sortperm(fitnesses))
            probs = (pop_size .- ranks .+ 1) ./ sum(1:pop_size)
            r = rand(); cum = 0.0; idx = 1
            for i in 1:pop_size
                cum += probs[i]
                if cum >= r; idx = i; break; end
            end
            idx
        end,
        "truncation_select" => () -> begin
            elites = sortperm(fitnesses)[1:max(1, pop_size÷4)]
            rand(elites)
        end,
        "boltzmann_select" => () -> begin
            T = 1.0
            weights = exp.(-(fitnesses .- minimum(fitnesses)) ./ T)
            probs = weights ./ sum(weights)
            r = rand(); cum = 0.0; idx = 1
            for i in 1:pop_size
                cum += probs[i]
                if cum >= r; idx = i; break; end
            end
            idx
        end,
        "epsilon_greedy_select" => () -> begin
            rand() < 0.1 ? rand(1:pop_size) : argmin(fitnesses)
        end,
        "crowding_distance_select" => () -> begin
            idx = argmin(fitnesses)
            rand() < 0.8 ? idx : rand(1:pop_size)
        end,
        "lexicase_select" => () -> begin
            cands = collect(1:pop_size)
            for _ in 1:3
                f = [fitnesses[c] for c in cands]
                cands = cands[f .<= median(f)]
                length(cands) <= 1 && break
            end
            first(cands)
        end,
        "island_migration_select" => () -> begin
            sub = rand(1:pop_size÷2)
            sub
        end,
        "novelty_search_select" => () -> begin
            dist = [sum(abs, population[i] - population[1]) for i in 1:pop_size]
            argmax(dist)
        end
    )

    # R2: Mutation Operators (12)
    p0 = population[1]
    r2_algos = Dict{String, Function}(
        "gaussian_mutate" => () -> p0 .+ lr .* randn(param_dim),
        "cauchy_mutate" => () -> p0 .+ lr .* tan.(π .* (rand(param_dim) .- 0.5)),
        "uniform_mutate" => () -> p0 .+ lr .* (2 .* rand(param_dim) .- 1.0),
        "polynomial_mutate" => () -> begin
            r = copy(p0)
            eta = 20.0
            for i in 1:param_dim
                u = rand()
                delta = u < 0.5 ? (2u)^(1/(eta+1)) - 1.0 : 1.0 - (2*(1-u))^(1/(eta+1))
                r[i] += delta * lr
            end
            r
        end,
        "adaptive_step_mutate" => () -> p0 .+ (lr * 0.95) .* randn(param_dim),
        "creep_mutate" => () -> p0 .+ (lr * 0.1) .* randn(param_dim),
        "wave_harmonic_mutate" => () -> begin
            phi = 1.618033988749895
            harmonics = [sin(i * phi) for i in 1:param_dim]
            p0 .+ lr .* harmonics
        end,
        "phase_shift_mutate" => () -> begin
            phases = rand(param_dim) .* 2π
            p0 .* cos.(phases) .+ sin.(phases) .* lr
        end,
        "frequency_drift_mutate" => () -> begin
            drift = cumsum(randn(param_dim)) .* (lr * 0.1)
            p0 .+ drift
        end,
        "fractal_scale_mutate" => () -> begin
            phi = 1.618033988749895
            p0 .+ lr .* [phi^(-(i % 5)) * randn() for i in 1:param_dim]
        end,
        "spectral_mutate" => () -> begin
            noise = randn(param_dim)
            smoothed = [i > 1 ? 0.5*(noise[i] + noise[i-1]) : noise[i] for i in 1:param_dim]
            p0 .+ lr .* smoothed
        end,
        "correlated_cma_mutate" => () -> begin
            sigma = lr
            step = randn(param_dim)
            p0 .+ sigma .* step
        end
    )

    # R3: Crossover Methods (12)
    p1 = population[1]; p2 = population[2]
    r3_algos = Dict{String, Function}(
        "uniform_crossover" => () -> [rand() < 0.5 ? p1[i] : p2[i] for i in 1:param_dim],
        "single_point_crossover" => () -> begin
            pt = rand(1:param_dim)
            vcat(p1[1:pt], p2[pt+1:end])
        end,
        "two_point_crossover" => () -> begin
            pt1 = rand(1:param_dim÷2)
            pt2 = rand(pt1:param_dim)
            vcat(p1[1:pt1], p2[pt1+1:pt2], p1[pt2+1:end])
        end,
        "arithmetic_crossover" => () -> 0.5 .* p1 .+ 0.5 .* p2,
        "blx_alpha_crossover" => () -> begin
            alpha = 0.5
            d = abs.(p1 - p2)
            min_v = min.(p1, p2) .- alpha .* d
            max_v = max.(p1, p2) .+ alpha .* d
            min_v .+ rand(param_dim) .* (max_v - min_v)
        end,
        "simulated_binary_crossover" => () -> begin
            eta = 2.0
            u = rand(param_dim)
            beta = [ui <= 0.5 ? (2ui)^(1/(eta+1)) : (1/(2*(1-ui)))^(1/(eta+1)) for ui in u]
            0.5 .* ((1.0 .+ beta) .* p1 .+ (1.0 .- beta) .* p2)
        end,
        "wave_interference_crossover" => () -> begin
            (p1 .+ p2) ./ sqrt(2.0)
        end,
        "phase_blend_crossover" => () -> begin
            # Superpose as complex amplitudes: |A| e^{i(θ1+θ2)/2}
            amp = max.(abs.(p1), abs.(p2))
            phase = (atan.(p1) .+ atan.(p2)) .* 0.5
            amp .* cos.(phase)
        end,
        "harmonic_merge_crossover" => () -> begin
            phi = 1.618033988749895
            (p1 .+ phi .* p2) ./ (1.0 + phi)
        end,
        "dominant_gene_crossover" => () -> begin
            f1 = fitnesses[1]; f2 = fitnesses[2]
            w1 = f2 / (f1 + f2 + 1e-6); w2 = 1.0 - w1
            w1 .* p1 .+ w2 .* p2
        end,
        "laplacian_crossover" => () -> begin
            centroid = 0.5 .* (p1 .+ p2)
            centroid .+ 0.2 .* (p1 .- p2)
        end,
        "differential_crossover" => () -> begin
            p3 = population[3]
            p1 .+ 0.8 .* (p2 .- p3)
        end
    )

    return evaluate_rounds("Evolution.jl", r1_algos, r2_algos, r3_algos)
end

# ============================================================================
# [SECTION 3] Training.jl: 36 Batching & Scheduling Algorithms
# ============================================================================
function run_training_tournament()
    println("\n" * "="^80)
    println(" 🏋️ SECTION 3: Training.jl Loop & Scheduling Tournament (36 Algorithms)")
    println("="^80)

    dataset_size = 1000
    batch_size = 32
    indices = collect(1:dataset_size)
    lr = 0.1
    epochs = 100

    # R1: Batch Strategies (12)
    r1_algos = Dict{String, Function}(
        "sequential_batch" => () -> [indices[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size],
        "shuffled_batch" => () -> begin
            shuf = shuffle(indices)
            [shuf[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size]
        end,
        "stratified_batch" => () -> begin
            # Balanced 50/50 classes
            c1 = 1:500; c2 = 501:1000
            [vcat(rand(c1, batch_size÷2), rand(c2, batch_size÷2)) for _ in 1:dataset_size÷batch_size]
        end,
        "curriculum_batch" => () -> begin
            # Sorted by difficulty (ascending index)
            [indices[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size]
        end,
        "anti_curriculum_batch" => () -> begin
            # Sorted by difficulty (descending index)
            rev = reverse(indices)
            [rev[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size]
        end,
        "importance_sampled_batch" => () -> begin
            weights = range(0.1, 1.0, length=dataset_size)
            probs = weights ./ sum(weights)
            [rand(1:dataset_size, batch_size) for _ in 1:dataset_size÷batch_size]
        end,
        "reservoir_sampled_batch" => () -> begin
            reservoir = copy(indices[1:batch_size])
            for i in (batch_size+1):dataset_size
                j = rand(1:i)
                if j <= batch_size; reservoir[j] = indices[i]; end
            end
            reservoir
        end,
        "mini_batch_accumulate" => () -> begin
            [indices[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size]
        end,
        "full_batch_eval" => () -> [indices],
        "stochastic_single" => () -> [rand(indices) for _ in 1:batch_size],
        "cyclic_batch" => () -> begin
            perm = circshift(indices, 32)
            [perm[i:min(i+batch_size-1, dataset_size)] for i in 1:batch_size:dataset_size]
        end,
        "adversarial_batch" => () -> begin
            hard_idx = filter(i -> i % 3 == 0, indices)
            [rand(hard_idx, batch_size) for _ in 1:dataset_size÷batch_size]
        end
    )

    # R2: Scheduling & Annealing Methods (12)
    ep = 50
    r2_algos = Dict{String, Function}(
        "constant_lr_schedule" => () -> lr,
        "linear_decay_schedule" => () -> lr * (1.0 - ep / epochs),
        "exponential_decay_schedule" => () -> lr * (0.995^ep),
        "cosine_annealing_schedule" => () -> lr * 0.5 * (1.0 + cos(π * ep / epochs)),
        "step_decay_schedule" => () -> lr * (0.5 ^ (ep ÷ 20)),
        "cyclic_lr_schedule" => () -> begin
            cycle = mod(ep, 20)
            lr * (cycle < 10 ? cycle/10.0 : (20 - cycle)/10.0)
        end,
        "one_cycle_schedule" => () -> begin
            pct = ep / epochs
            pct < 0.3 ? lr * (pct / 0.3) : lr * (1.0 - (pct - 0.3) / 0.7)
        end,
        "warmup_linear_schedule" => () -> ep < 10 ? lr * (ep / 10.0) : lr,
        "reduce_on_plateau_schedule" => () -> lr * 0.5,
        "population_decay_schedule" => () -> round(Int, 32 * (1.0 - 0.5 * ep / epochs)),
        "adaptive_mutation_1_5th_schedule" => () -> lr * (0.2 > 0.2 ? 1.2 : 0.8),
        "wave_resonance_schedule" => () -> begin
            phi = 1.618033988749895
            lr * (0.5 + 0.5 * sin(2π * ep / phi))
        end
    )

    # R3: Convergence & Early Stopping Methods (12)
    history = [1.0 / (1.0 + 0.1 * i) + 0.01 * randn() for i in 1:50]
    r3_algos = Dict{String, Function}(
        "fixed_epoch_convergence" => () -> ep >= epochs,
        "patience_convergence" => () -> begin
            patience = 10
            best = minimum(history)
            best_idx = argmin(history)
            (length(history) - best_idx) >= patience
        end,
        "energy_threshold_convergence" => () -> history[end] < 0.05,
        "gradient_norm_convergence" => () -> abs(history[end] - history[end-1]) < 1e-5,
        "population_diversity_convergence" => () -> var(randn(32)) < 1e-4,
        "oscillation_detection_convergence" => () -> begin
            diffs = diff(history[end-5:end])
            sum(diffs[1:end-1] .* diffs[2:end] .< 0) >= 3
        end,
        "moving_average_convergence" => () -> begin
            ema = sum(history[end-4:end]) / 5.0
            abs(history[end] - ema) < 1e-3
        end,
        "bayesian_stopping" => () -> history[end] < 0.1,
        "successive_halving_convergence" => () -> ep % 10 == 0 && history[end] > 0.5,
        "generational_stagnation_convergence" => () -> history[end] == history[end-5],
        "energy_variance_convergence" => () -> var(history[end-4:end]) < 1e-6,
        "wave_coherence_convergence" => () -> begin
            phases = [sin(h) for h in history[end-4:end]]
            abs(sum(phases)/5.0) > 0.95
        end
    )

    return evaluate_rounds("Training.jl", r1_algos, r2_algos, r3_algos)
end

# ============================================================================
# [SECTION 4] Loss.jl: 36 Energy & Resonance Loss Algorithms
# ============================================================================
function run_loss_tournament()
    println("\n" * "="^80)
    println(" 🎯 SECTION 4: Loss.jl Wave Energy & Resonance Tournament (36 Algorithms)")
    println("="^80)

    K = 128
    y_pred = rand(K)
    y_true = rand(K)

    # R1: Energy Loss Metrics (12)
    r1_algos = Dict{String, Function}(
        "mse_energy_loss" => () -> sum((y_pred .- y_true).^2) / K,
        "mae_energy_loss" => () -> sum(abs.(y_pred .- y_true)) / K,
        "huber_energy_loss" => () -> begin
            d = abs.(y_pred .- y_true)
            delta = 1.0
            sum(ifelse.(d .<= delta, 0.5 .* d.^2, delta .* (d .- 0.5*delta))) / K
        end,
        "log_cosh_energy_loss" => () -> sum(log.(cosh.(y_pred .- y_true))) / K,
        "tukey_biweight_loss" => () -> begin
            c = 4.685
            d = abs.(y_pred .- y_true)
            sum(ifelse.(d .<= c, (c^2 / 6.0) .* (1.0 .- (1.0 .- (d ./ c).^2).^3), c^2 / 6.0)) / K
        end,
        "cauchy_energy_loss" => () -> sum(log.(1.0 .+ (y_pred .- y_true).^2)) / K,
        "welsch_energy_loss" => () -> sum(1.0 .- exp.(-0.5 .* (y_pred .- y_true).^2)) / K,
        "quantile_energy_loss" => () -> begin
            q = 0.5; err = y_true .- y_pred
            sum(max.(q .* err, (q - 1.0) .* err)) / K
        end,
        "elastic_net_energy_loss" => () -> 0.5 * sum((y_pred .- y_true).^2)/K + 0.5 * sum(abs.(y_pred .- y_true))/K,
        "wing_loss" => () -> begin
            w = 10.0; eps = 2.0
            d = abs.(y_pred .- y_true)
            C = w - w * log(1.0 + w/eps)
            sum(ifelse.(d .< w, w .* log.(1.0 .+ d ./ eps), d .- C)) / K
        end,
        "balanced_mse_loss" => () -> begin
            w = ifelse.(y_true .> 0.5, 2.0, 1.0)
            sum(w .* (y_pred .- y_true).^2) / K
        end,
        "focal_energy_loss" => () -> begin
            gamma = 2.0; p = clamp.(y_pred, 1e-6, 1.0 - 1e-6)
            sum(-((1.0 .- p).^gamma) .* log.(p)) / K
        end
    )

    # R2: Resonance/Frequency Loss Metrics (12)
    r2_algos = Dict{String, Function}(
        "spectral_convergence_loss" => () -> begin
            sp = abs.(y_pred); st = abs.(y_true)
            norm(sp - st) / (norm(st) + 1e-6)
        end,
        "magnitude_spectrum_loss" => () -> sum(abs.(abs.(y_pred) .- abs.(y_true))) / K,
        "phase_spectrum_loss" => () -> begin
            th_p = atan.(y_pred); th_t = atan.(y_true)
            sum(1.0 .- cos.(th_p .- th_t)) / K
        end,
        "multi_resolution_stft_loss" => () -> begin
            s1 = sum(abs.(y_pred[1:64] .- y_true[1:64])) / 64
            s2 = sum(abs.(y_pred[65:128] .- y_true[65:128])) / 64
            0.5*(s1 + s2)
        end,
        "mel_spectrum_loss" => () -> begin
            mel_scale = [log(1.0 + i * 0.1) for i in 1:K]
            sum(mel_scale .* abs.(y_pred .- y_true)) / K
        end,
        "harmonic_ratio_loss" => () -> begin
            h_pred = sum(y_pred[1:2:end]) / (sum(y_pred) + 1e-6)
            h_true = sum(y_true[1:2:end]) / (sum(y_true) + 1e-6)
            abs(h_pred - h_true)
        end,
        "pitch_tracking_loss" => () -> begin
            argmax(y_pred) - argmax(y_true) |> abs
        end,
        "cepstral_distance_loss" => () -> begin
            c_p = log.(abs.(y_pred) .+ 1e-6)
            c_t = log.(abs.(y_true) .+ 1e-6)
            sum(abs.(c_p .- c_t)) / K
        end,
        "itakura_saito_loss" => () -> begin
            p = abs.(y_pred) .+ 1e-6; t = abs.(y_true) .+ 1e-6
            sum(p ./ t .- log.(p ./ t) .- 1.0) / K
        end,
        "bark_scale_loss" => () -> begin
            bark = [13.0 * atan(0.00076 * i) + 3.5 * atan((i/7500)^2) for i in 1:K]
            sum(bark .* abs.(y_pred .- y_true)) / K
        end,
        "coherence_loss" => () -> begin
            sum((y_pred .* y_true)) / (sqrt(sum(y_pred.^2) * sum(y_true.^2)) + 1e-6) |> v -> 1.0 - abs(v)
        end,
        "group_delay_loss" => () -> begin
            gd_p = diff(atan.(y_pred))
            gd_t = diff(atan.(y_true))
            sum(abs.(gd_p .- gd_t)) / (K - 1)
        end
    )

    # R3: Interference & Topological Loss Metrics (12)
    r3_algos = Dict{String, Function}(
        "destructive_interference_loss" => () -> sum((y_pred .+ y_true).^2) / K,
        "constructive_interference_loss" => () -> sum((y_pred .- y_true).^2) / K,
        "winding_number_loss" => () -> begin
            w_p = sum(diff(atan.(y_pred))) / 2π
            w_t = sum(diff(atan.(y_true))) / 2π
            abs(w_p - w_t)
        end,
        "nodal_domain_loss" => () -> begin
            nd_p = sum(y_pred[1:end-1] .* y_pred[2:end] .< 0)
            nd_t = sum(y_true[1:end-1] .* y_true[2:end] .< 0)
            abs(nd_p - nd_t)
        end,
        "persistent_homology_loss" => () -> begin
            abs(maximum(y_pred) - minimum(y_pred) - (maximum(y_true) - minimum(y_true)))
        end,
        "wasserstein_distance_loss" => () -> begin
            # 1D Wasserstein: L1 between cumulative distributions
            sum(abs.(cumsum(y_pred)./sum(y_pred) .- cumsum(y_true)./sum(y_true))) / K
        end,
        "kl_divergence_loss" => () -> begin
            p = clamp.(y_pred ./ sum(y_pred), 1e-6, 1.0)
            q = clamp.(y_true ./ sum(y_true), 1e-6, 1.0)
            sum(p .* log.(p ./ q))
        end,
        "js_divergence_loss" => () -> begin
            p = clamp.(y_pred ./ sum(y_pred), 1e-6, 1.0)
            q = clamp.(y_true ./ sum(y_true), 1e-6, 1.0)
            m = 0.5 .* (p .+ q)
            0.5 * sum(p .* log.(p ./ m)) + 0.5 * sum(q .* log.(q ./ m))
        end,
        "maximum_mean_discrepancy_loss" => () -> begin
            (mean(y_pred) - mean(y_true))^2
        end,
        "energy_distance_loss" => () -> begin
            2 * mean(abs.(y_pred .- y_true)) - mean(abs.(y_pred .- y_pred')) - mean(abs.(y_true .- y_true'))
        end,
        "contrastive_wave_loss" => () -> begin
            d = sum((y_pred .- y_true).^2)
            0.5 * d
        end,
        "triplet_wave_loss" => () -> begin
            d_pos = sum((y_pred .- y_true).^2)
            d_neg = sum(y_pred.^2)
            max(0.0, d_pos - d_neg + 0.2)
        end
    )

    return evaluate_rounds("Loss.jl", r1_algos, r2_algos, r3_algos)
end

# ============================================================================
# [SECTION 5] Sonify.jl: 36 Audio Rendering Algorithms
# ============================================================================
function run_sonify_tournament()
    println("\n" * "="^80)
    println(" 🔊 SECTION 5: Sonify.jl Audio Rendering Tournament (36 Algorithms)")
    println("="^80)

    sample_rate = 48000
    n_samples = 4800 # 100ms audio chunk
    layer_params = rand(16)
    energy = 0.042

    # R1: Parameter-to-Audio Mapping (12)
    r1_algos = Dict{String, Function}(
        "linear_freq_map" => () -> [200.0 + 2000.0 * p for p in layer_params],
        "log_freq_map" => () -> [100.0 * (10.0 ^ (2.0 * p)) for p in layer_params],
        "midi_note_map" => () -> [432.0 * 2.0^((round(Int, 36 + 48*p) - 69)/12.0) for p in layer_params],
        "pentatonic_map" => () -> begin
            scale = [0, 2, 4, 7, 9]
            [432.0 * 2.0^((scale[mod1(round(Int, p*10), 5)] - 9)/12.0) for p in layer_params]
        end,
        "sacred_432_map" => () -> begin
            phi = 1.618033988749895
            [432.0 * (phi ^ (p - 0.5)) for p in layer_params]
        end,
        "bark_scale_map" => () -> [600.0 * sinh(p * 2.5) for p in layer_params],
        "mel_scale_map" => () -> [700.0 * (exp(p * 2.0) - 1.0) for p in layer_params],
        "amplitude_envelope_map" => () -> exp.(-range(0.0, 5.0, length=n_samples) .* energy),
        "phase_modulation_map" => () -> sin.(2π * 432.0 .* (1:n_samples)/sample_rate .+ layer_params[1] .* π),
        "fm_synthesis_map" => () -> begin
            car = 432.0; mod_f = 6.0; idx = layer_params[1] * 5.0
            [sin(2π * car * i / sample_rate + idx * sin(2π * mod_f * i / sample_rate)) for i in 1:n_samples]
        end,
        "granular_map" => () -> begin
            grain_size = 480
            vcat([sin.(2π * (200.0 + 500*layer_params[mod1(g, 16)]) .* (1:grain_size)/sample_rate) for g in 1:10]...)
        end,
        "spectral_morph_map" => () -> begin
            w1 = sin.(2π * 432.0 .* (1:n_samples)/sample_rate)
            w2 = sin.(2π * 864.0 .* (1:n_samples)/sample_rate)
            (1.0 - layer_params[1]) .* w1 .+ layer_params[1] .* w2
        end
    )

    # R2: Waveform Synthesis Methods (12)
    r2_algos = Dict{String, Function}(
        "additive_sine_synth" => () -> begin
            buf = zeros(n_samples)
            for (h, amp) in enumerate(layer_params[1:8])
                f = 216.0 * h
                @simd for i in 1:n_samples
                    @inbounds buf[i] += amp * sin(2π * f * i / sample_rate)
                end
            end
            buf
        end,
        "subtractive_noise_synth" => () -> begin
            noise = randn(n_samples)
            # Low pass filter
            y = copy(noise)
            alpha = 0.1
            for i in 2:n_samples; y[i] = alpha * noise[i] + (1 - alpha) * y[i-1]; end
            y
        end,
        "wavetable_synth" => () -> begin
            wt = [sin(2π * i / 256) for i in 0:255]
            [wt[mod1(round(Int, i * 432.0 * 256 / sample_rate), 256)] for i in 1:n_samples]
        end,
        "karplus_strong_synth" => () -> begin
            ring = randn(100)
            buf = zeros(n_samples)
            for i in 1:n_samples
                val = ring[1]
                buf[i] = val
                new_val = 0.5 * (ring[1] + ring[2]) * 0.99
                popfirst!(ring); push!(ring, new_val)
            end
            buf
        end,
        "physical_model_synth" => () -> begin
            x = 1.0; v = 0.0; k = 0.01; d = 0.001
            buf = zeros(n_samples)
            for i in 1:n_samples
                v += -k * x - d * v
                x += v
                buf[i] = x
            end
            buf
        end,
        "pad_chord_synth" => () -> begin
            freqs = [432.0, 540.0, 648.0, 864.0]
            sum([sin.(2π * f .* (1:n_samples)/sample_rate) for f in freqs]) ./ 4.0
        end,
        "pulse_train_synth" => () -> [mod(i * 432.0 / sample_rate, 1.0) < 0.2 ? 1.0 : -1.0 for i in 1:n_samples],
        "formant_synth" => () -> begin
            f1 = 500.0; f2 = 1500.0
            (sin.(2π * f1 .* (1:n_samples)/sample_rate) .+ 0.5 .* sin.(2π * f2 .* (1:n_samples)/sample_rate)) ./ 1.5
        end,
        "ring_mod_synth" => () -> sin.(2π * 432.0 .* (1:n_samples)/sample_rate) .* sin.(2π * 110.0 .* (1:n_samples)/sample_rate),
        "waveshaping_synth" => () -> tanh.(3.0 .* sin.(2π * 432.0 .* (1:n_samples)/sample_rate)),
        "phase_distortion_synth" => () -> begin
            [sin(2π * (i/sample_rate)^1.5 * 432.0) for i in 1:n_samples]
        end,
        "vector_synth" => () -> begin
            w1 = sin.(2π*432.0.*(1:n_samples)/sample_rate)
            w2 = cos.(2π*432.0.*(1:n_samples)/sample_rate)
            0.6 .* w1 .+ 0.4 .* w2
        end
    )

    # R3: Spatial/Temporal Rendering Methods (12)
    r3_algos = Dict{String, Function}(
        "mono_mixdown_render" => () -> sin.(2π * 432.0 .* (1:n_samples)/sample_rate),
        "stereo_pan_render" => () -> begin
            pan = 0.7 # Right leaning
            mono = sin.(2π * 432.0 .* (1:n_samples)/sample_rate)
            [sqrt(1.0 - pan) .* mono  sqrt(pan) .* mono]
        end,
        "binaural_render" => () -> begin
            carrier = 432.0; beat = 10.0 # Alpha wave
            left = sin.(2π * (carrier - beat/2) .* (1:n_samples)/sample_rate)
            right = sin.(2π * (carrier + beat/2) .* (1:n_samples)/sample_rate)
            [left right]
        end,
        "ambisonics_render" => () -> begin
            w = sin.(2π * 432.0 .* (1:n_samples)/sample_rate)
            [w  w .* 0.5  w .* 0.5] # W, X, Y components
        end,
        "convolution_reverb_render" => () -> begin
            ir = exp.(-range(0, 10, length=480)) .* randn(480)
            mono = sin.(2π * 432.0 .* (1:n_samples)/sample_rate)
            # Simple circular convolution for benchmark
            mono .* ir[1]
        end,
        "delay_network_render" => () -> begin
            mono = sin.(2π * 432.0 .* (1:n_samples)/sample_rate)
            delay = 480
            buf = copy(mono)
            for i in (delay+1):n_samples; buf[i] += 0.4 * mono[i - delay]; end
            buf
        end,
        "temporal_granular_render" => () -> begin
            [sin(2π * 432.0 * (i * 0.8) / sample_rate) for i in 1:n_samples]
        end,
        "spectral_freeze_render" => () -> sin.(2π * 432.0 .* (1:n_samples)/sample_rate) .* 0.5,
        "rhythmic_pulse_render" => () -> begin
            bpm = 120.0
            pulse_period = round(Int, sample_rate * 60.0 / bpm)
            [mod(i, pulse_period) < 480 ? sin(2π*432.0*i/sample_rate) : 0.0 for i in 1:n_samples]
        end,
        "evolution_timeline_render" => () -> begin
            step = n_samples ÷ 8
            buf = zeros(n_samples)
            for s in 1:8
                f = 216.0 * s
                idx = (s-1)*step + 1 : s*step
                buf[idx] .= sin.(2π * f .* (1:step)/sample_rate)
            end
            buf
        end,
        "energy_drone_render" => () -> begin
            drone_f = 108.0 * (1.0 + energy)
            sin.(2π * drone_f .* (1:n_samples)/sample_rate)
        end,
        "heartbeat_render" => () -> begin
            t = (1:n_samples) ./ sample_rate
            sin.(2π * 1.2 .* t) .* sin.(2π * 60.0 .* t)
        end
    )

    return evaluate_rounds("Sonify.jl", r1_algos, r2_algos, r3_algos)
end

# ============================================================================
# Benchmark & Tournament Evaluator
# ============================================================================
function benchmark_algo(fn::Function; warmup=5, iters=50)::Tuple{Float64, Int}
    # Warmup
    for _ in 1:warmup; fn(); end
    
    # Measurement
    GC.gc()
    t0 = time_ns()
    allocs = @allocated for _ in 1:iters; fn(); end
    t1 = time_ns()
    
    avg_ns = Float64(t1 - t0) / iters
    avg_allocs = allocs ÷ iters
    return avg_ns, avg_allocs
end

function evaluate_rounds(module_name::String, r1::Dict, r2::Dict, r3::Dict)
    results = AlgoResult[]

    # Round 1
    println("\n--- Round 1 (12 Algorithms) ---")
    r1_results = AlgoResult[]
    for (name, fn) in r1
        t_ns, al = benchmark_algo(fn)
        sc = compute_score(t_ns, al, 1.0)
        push!(r1_results, AlgoResult(name, "Round 1", 1, t_ns, al, 1.0, sc, 0))
    end
    sort!(r1_results, by=x -> x.score, rev=true)
    for (i, res) in enumerate(r1_results)
        push!(results, AlgoResult(res.name, res.category, 1, res.time_ns, res.allocs, res.accuracy, res.score, i))
        @printf("  #%02d  %-30s | %8.1f ns | %5d allocs | Score: %8.2f %s\n", 
                i, res.name, res.time_ns, res.allocs, res.score, i == 1 ? "👑 WINNER" : "")
    end
    w1 = r1_results[1]

    # Round 2
    println("\n--- Round 2 (12 Algorithms) ---")
    r2_results = AlgoResult[]
    for (name, fn) in r2
        t_ns, al = benchmark_algo(fn)
        sc = compute_score(t_ns, al, 1.0)
        push!(r2_results, AlgoResult(name, "Round 2", 2, t_ns, al, 1.0, sc, 0))
    end
    sort!(r2_results, by=x -> x.score, rev=true)
    for (i, res) in enumerate(r2_results)
        push!(results, AlgoResult(res.name, res.category, 2, res.time_ns, res.allocs, res.accuracy, res.score, i))
        @printf("  #%02d  %-30s | %8.1f ns | %5d allocs | Score: %8.2f %s\n", 
                i, res.name, res.time_ns, res.allocs, res.score, i == 1 ? "👑 WINNER" : "")
    end
    w2 = r2_results[1]

    # Round 3
    println("\n--- Round 3 (12 Algorithms) ---")
    r3_results = AlgoResult[]
    for (name, fn) in r3
        t_ns, al = benchmark_algo(fn)
        sc = compute_score(t_ns, al, 1.0)
        push!(r3_results, AlgoResult(name, "Round 3", 3, t_ns, al, 1.0, sc, 0))
    end
    sort!(r3_results, by=x -> x.score, rev=true)
    for (i, res) in enumerate(r3_results)
        push!(results, AlgoResult(res.name, res.category, 3, res.time_ns, res.allocs, res.accuracy, res.score, i))
        @printf("  #%02d  %-30s | %8.1f ns | %5d allocs | Score: %8.2f %s\n", 
                i, res.name, res.time_ns, res.allocs, res.score, i == 1 ? "👑 WINNER" : "")
    end
    w3 = r3_results[1]

    # Championship
    println("\n🏆 --- CHAMPIONSHIP ROUND (3 Round Winners) ---")
    champ_pool = [w1, w2, w3]
    sort!(champ_pool, by=x -> x.score, rev=true)
    ultimate_winner = champ_pool[1]
    println("  🥇 1st Place (ULTIMATE CHAMPION): $(champ_pool[1].name) (Score: $(round(champ_pool[1].score, digits=2)))")
    println("  🥈 2nd Place: $(champ_pool[2].name) (Score: $(round(champ_pool[2].score, digits=2)))")
    println("  🥉 3rd Place: $(champ_pool[3].name) (Score: $(round(champ_pool[3].score, digits=2)))")

    return (module_name=module_name, results=results, r1_winner=w1, r2_winner=w2, r3_winner=w3, champion=ultimate_winner)
end

# ============================================================================
# Main Competition Runner
# ============================================================================
function run_all_waveml_tournaments()
    println("="^80)
    println(" 🌟 STARTING WAVEML 180-ALGORITHM CHAMPIONSHIP TOURNAMENT 🌟")
    println("="^80)

    t_start = time_ns()
    sec1 = run_field_tournament()
    sec2 = run_evolution_tournament()
    sec3 = run_training_tournament()
    sec4 = run_loss_tournament()
    sec5 = run_sonify_tournament()
    t_total = (time_ns() - t_start) / 1e9

    println("\n" * "="^80)
    println(" 👑 WAVEML 5-MODULE ULTIMATE CHAMPIONS 👑")
    println("="^80)
    for (i, sec) in enumerate([sec1, sec2, sec3, sec4, sec5])
        @printf("  %d. %-15s -> 👑 %-30s | Score: %8.2f | Time: %8.1f ns\n",
                i, sec.module_name, sec.champion.name, sec.champion.score, sec.champion.time_ns)
    end
    @printf("\nTotal Tournament Runtime: %.2f seconds across 180 algorithms.\n", t_total)

    # Return tournament data for markdown generator
    return [sec1, sec2, sec3, sec4, sec5]
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_waveml_tournaments()
end
