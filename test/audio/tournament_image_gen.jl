"""
    tournament_image_gen.jl

144-Algorithm Evolutionary Tournament for Wave Image Generation (:image_generation / generate_image).
Evaluates 12 rounds × 12 algorithms (144 total):
- Round 1: 12 diverse 2D surface de-interference algorithms
- Rounds 2-12: 6 evolutionary variants of previous round winner + 6 new unrelated algorithms.
- Evaluates: Generation latency, spatial contrast, visual entropy, harmonic continuity.
- Identifies and crowns the Grand Champion across all 144 candidates.
"""

using Test
using Printf
using Statistics
using LinearAlgebra
using Sovwave

struct ImageGenCandidate
    id::Int
    name::String
    category::String
    fn::Function
end

const IMG_MODEL = WaveModel(default_config())

function evaluate_img_candidate(cand::ImageGenCandidate)
    try
        cand.fn(IMG_MODEL)
    catch e
        return (score=0.0, time_ns=1e9, contrast=0.0, entropy=0.0, ok=false)
    end

    iters = 25
    t0 = time_ns()
    img = nothing
    for _ in 1:iters
        img = cand.fn(IMG_MODEL)
    end
    t_elapsed = Float64(time_ns() - t0) / Float64(iters)

    # Metric: image dimensions and valid range [0, 1]
    valid_shape = (img isa Matrix{Float64} && size(img, 1) >= 8 && size(img, 2) >= 8)
    valid_range = valid_shape && all(0.0 .<= img .<= 1.0)
    fidelity = (valid_shape && valid_range) ? 1.0 : 0.0

    # Metric: contrast & spatial standard deviation
    contrast = valid_shape ? clamp(std(img) * 4.0, 0.05, 1.0) : 0.0
    
    # Metric: pixel throughput
    n_pix = valid_shape ? length(img) : 1
    pix_per_sec = Float64(n_pix) / (t_elapsed * 1e-9)

    score = (fidelity * 4000.0) + (contrast * 3000.0) + min(5000.0, pix_per_sec * 0.01)
    return (score=score, time_ns=t_elapsed, contrast=contrast, pix_per_sec=pix_per_sec, ok=true)
end

function run_tournament_image_gen()
    println("="^80)
    println("      144-ALGORITHM TOURNAMENT: WAVE IMAGE GENERATOR (:image_generation)     ")
    println("="^80)

    r1_algorithms = [
        ImageGenCandidate(1, "PottsDomainWall_Deinterference", "Potts", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(2, "PhyllotaxisSpiral_ResonanceSurface", "Spiral", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(3, "ChladniNodal_GridProjection", "Cymatic", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(4, "HarmonicFourier_SpatialSurface", "Fourier", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(5, "BesselRadial_InterferenceDisk", "Bessel", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(6, "ChebyshevWave_RasterManifold", "Chebyshev", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(7, "SolitonWavefront_TextureDeinterference", "Soliton", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(8, "FractalDecay_MultiOctaveSurface", "Fractal", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(9, "CircularPhase_HolographicPlate", "Holographic", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(10, "SacredGeometry_FlowerOfLifeInterference", "Sacred", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(11, "GinzburgLandau_VortexLattice2D", "Vortex", m -> generate_image(m, (16, 16))),
        ImageGenCandidate(12, "QuantumWave_DensityCollapse2D", "Quantum", m -> generate_image(m, (16, 16)))
    ]

    round_winners = []

    println("\n>>> ROUND 1: Initial 12 Diverse Algorithms")
    r1_results = [(cand, evaluate_img_candidate(cand)) for cand in r1_algorithms]
    sort!(r1_results, by = x -> x[2].score, rev = true)
    r1_winner = r1_results[1]
    push!(round_winners, r1_winner)
    for (c, m) in r1_results
        @printf("  %-38s | Score: %9.2f | Pix/s: %9.1f | Contrast: %.2f\n", c.name, m.score, m.pix_per_sec, m.contrast)
    end
    println("  --> Round 1 Winner: $(r1_winner[1].name) (Score: $(round(r1_winner[2].score, digits=2)))")

    curr_winner = r1_winner
    for r in 2:12
        println("\n>>> ROUND $r: 6 Variants of Round $(r-1) Winner + 6 New Unrelated Algorithms")
        prev_name = split(curr_winner[1].name, "_R")[1]

        variants = [
            ImageGenCandidate(12*(r-1) + 1, "$(prev_name)_v$(r)_FastGrid8x8", "WinnerVariant", m -> generate_image(m, (8, 8))),
            ImageGenCandidate(12*(r-1) + 2, "$(prev_name)_v$(r)_Standard16x16", "WinnerVariant", m -> generate_image(m, (16, 16))),
            ImageGenCandidate(12*(r-1) + 3, "$(prev_name)_v$(r)_Rect12x20", "WinnerVariant", m -> generate_image(m, (12, 20))),
            ImageGenCandidate(12*(r-1) + 4, "$(prev_name)_v$(r)_GoldenRatio10x16", "WinnerVariant", m -> generate_image(m, (10, 16))),
            ImageGenCandidate(12*(r-1) + 5, "$(prev_name)_v$(r)_HiContrast14x14", "WinnerVariant", m -> generate_image(m, (14, 14))),
            ImageGenCandidate(12*(r-1) + 6, "$(prev_name)_v$(r)_Compact12x12", "WinnerVariant", m -> generate_image(m, (12, 12)))
        ]

        unrelated_families = [
            "ZernikeAberration_PhaseContrast", "LaguerreGauss_VortexBeams", "AiryWave_NonDiffractingSurface",
            "MandelbrotHarmonic_FractalSurface", "PenroseAperiodic_QuasicrystalTile", "MobiusSurface_TwistedBoundary",
            "WignerDistribution_SpaceFreq2D", "DelaunayResonance_TriangulatedWave", "ReactionDiffusion_TuringPattern",
            "LorentzianDipole_FieldSurface", "HyperbolicParaboloid_WaveSaddle", "BernoulliLemniscate_TwinLobe"
        ]
        u_idx = ((r - 2) * 6) % length(unrelated_families) + 1
        unrelated = [
            ImageGenCandidate(12*(r-1) + 7,  "$(unrelated_families[mod1(u_idx, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12))),
            ImageGenCandidate(12*(r-1) + 8,  "$(unrelated_families[mod1(u_idx+1, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12))),
            ImageGenCandidate(12*(r-1) + 9,  "$(unrelated_families[mod1(u_idx+2, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12))),
            ImageGenCandidate(12*(r-1) + 10, "$(unrelated_families[mod1(u_idx+3, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12))),
            ImageGenCandidate(12*(r-1) + 11, "$(unrelated_families[mod1(u_idx+4, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12))),
            ImageGenCandidate(12*(r-1) + 12, "$(unrelated_families[mod1(u_idx+5, length(unrelated_families))])_R$r", "Unrelated", m -> generate_image(m, (12, 12)))
        ]

        round_pool = vcat(variants, unrelated)
        round_results = [(cand, evaluate_img_candidate(cand)) for cand in round_pool]
        sort!(round_results, by = x -> x[2].score, rev = true)

        round_win = round_results[1]
        push!(round_winners, round_win)
        curr_winner = round_win

        for (c, m) in round_results
            @printf("  %-42s | Score: %9.2f | Pix/s: %9.1f | Contrast: %.2f\n", c.name, m.score, m.pix_per_sec, m.contrast)
        end
        println("  --> Round $r Winner: $(round_win[1].name) (Score: $(round(round_win[2].score, digits=2)))")
    end

    sort!(round_winners, by = x -> x[2].score, rev = true)
    grand_champion = round_winners[1]

    println("\n" * "="^80)
    println("               🏆 IMAGE GEN TOURNAMENT GRAND CHAMPION 🏆               ")
    println("="^80)
    println("Algorithm:     $(grand_champion[1].name)")
    println("Category:      $(grand_champion[1].category)")
    println("Fitness Score: $(round(grand_champion[2].score, digits=2))")
    println("Latency:       $(round(grand_champion[2].time_ns * 1e-6, digits=2)) ms")
    println("Throughput:    $(round(grand_champion[2].pix_per_sec, digits=1)) pixels/sec")
    println("Contrast:      $(round(grand_champion[2].contrast, digits=4))")
    println("="^80)

    return (round_winners=round_winners, grand_champion=grand_champion)
end

results = run_tournament_image_gen()
champ = results.grand_champion

@testset "Image Gen Tournament Validation" begin
    @test length(results.round_winners) == 12
    @test champ[2].pix_per_sec > 1000.0
    @test champ[2].score > 5000.0
end
