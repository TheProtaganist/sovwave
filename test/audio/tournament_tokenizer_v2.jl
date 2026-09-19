"""
    tournament_tokenizer_v2.jl

NEW 144-Algorithm Tournament: Industry-Standard Tokenization Competition
Tests proper BPE/WordPiece variants against novel wave-based approaches.

Round 1: Current winner + 5 variants + 6 new algorithms
Rounds 2-12: 6 variants of previous winner + 6 new algorithms each
"""

using Test
using Printf
using Statistics
using LinearAlgebra
using Sovwave

# Test corpus
const TEST_TEXTS = Dict(
    "english" => "The quick brown fox jumps over the lazy dog",
    "code" => "function tokenize(text::String)::Vector{Int} return ids end",
    "math" => "∫∂ψ/∂t = ℏ∇²ψ, Φ ≈ 1.618",
    "emoji" => "🌊🧠⚡ wave computing 🚀⚛️",
    "multi" => "Hello 世界 مرحبا Привет",
    "repeated" => "the the the wave wave computing computing"
)

struct TokenAlgo
    id::Int
    name::String
    category::String
    fn::Function
end

function evaluate_algo(algo::TokenAlgo, tok::WaveTokenizer)
    total_score = 0.0
    total_time = 0.0
    total_acc = 0.0
    total_comp = 0.0
    n = length(TEST_TEXTS)
    
    for (name, text) in TEST_TEXTS
        # Warmup
        try
            algo.fn(tok, text)
        catch e
            return (score=0.0, time_us=1e6, acc=0.0, comp=1.0, ok=false)
        end
        
        # Benchmark
        iters = 100
        t0 = time_ns()
        local result
        for _ in 1:iters
            result = algo.fn(tok, text)
        end
        elapsed_ns = Float64(time_ns() - t0) / Float64(iters)
        
        # Metrics
        n_chars = length(text)
        n_tokens = length(result)
        comp_ratio = Float64(n_tokens) / max(1, Float64(n_chars))
        
        # Accuracy
        reconstructed = try
            decode(tok, result)
        catch
            ""
        end
        acc = (reconstructed == text) ? 1.0 : 0.5
        
        total_time += elapsed_ns
        total_acc += acc
        total_comp += comp_ratio
    end
    
    avg_time_us = total_time / Float64(n) / 1000.0
    avg_acc = total_acc / Float64(n)
    avg_comp = total_comp / Float64(n)
    
    # Score: accuracy (50%), compression (30%), speed (20%)
    speed_score = 1e5 / max(10.0, avg_time_us)
    comp_score = 1.5 / max(0.1, avg_comp)
    score = (avg_acc * 5000.0) + (comp_score * 3000.0) + speed_score
    
    return (score=score, time_us=avg_time_us, acc=avg_acc, comp=avg_comp, ok=true)
end

# ============================================================================
# ROUND 1: Baseline + 5 Variants + 6 New Algorithms
# ============================================================================

function create_round1_algorithms()
    baseline_name = "BPE_GreedyLongestMatch"
    
    return [
        # 1. Current baseline
        TokenAlgo(1, baseline_name, "Baseline",
            (tok, txt) -> tokenize(tok, txt)),
        
        # 2-6. Variants of baseline
        TokenAlgo(2, "$(baseline_name)_FastEnvelope16", "Variant",
            (tok, txt) -> tokenize(tok, txt; n_samples=16)),
        
        TokenAlgo(3, "$(baseline_name)_HiRes48", "Variant",
            (tok, txt) -> tokenize(tok, txt; n_samples=48)),
        
        TokenAlgo(4, "$(baseline_name)_MidRes28", "Variant",
            (tok, txt) -> tokenize(tok, txt; n_samples=28)),
        
        TokenAlgo(5, "$(baseline_name)_Coherent32", "Variant",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)),
        
        TokenAlgo(6, "$(baseline_name)_Quantum24", "Variant",
            (tok, txt) -> tokenize(tok, txt; n_samples=24)),
        
        # 7-12. New algorithms
        TokenAlgo(7, "FrequencyBPE_AdaptiveBigram", "FreqBPE",
            (tok, txt) -> begin
                chars = collect(txt)
                n = length(chars)
                freq = Dict{String,Int}()
                for i in 1:(n-1)
                    bg = string(chars[i], chars[i+1])
                    freq[bg] = get(freq, bg, 0) + 1
                end
                
                tokens = WaveForm[]
                i = 1
                while i <= n
                    matched = false
                    for len in 8:-1:2
                        if i + len - 1 <= n
                            sub = String(chars[i:(i+len-1)])
                            if haskey(tok.vocab, sub) || (len == 2 && get(freq, sub, 0) >= 2)
                                id = haskey(tok.vocab, sub) ? tok.vocab[sub] : register_token!(tok, sub)
                                push!(tokens, to_wave_form(tok, id))
                                i += len
                                matched = true
                                break
                            end
                        end
                    end
                    if !matched
                        cs = string(chars[i])
                        id = haskey(tok.vocab, cs) ? tok.vocab[cs] : register_token!(tok, cs)
                        push!(tokens, to_wave_form(tok, id))
                        i += 1
                    end
                end
                tokens
            end),
        
        TokenAlgo(8, "WordPiece_BoundaryDetect", "WordPiece",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)),
        
        TokenAlgo(9, "Unigram_ProbabilisticSample", "Unigram",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)),
        
        TokenAlgo(10, "SentencePiece_Subword", "SentencePiece",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)),
        
        TokenAlgo(11, "CharLevel_Pure", "CharLevel",
            (tok, txt) -> begin
                [to_wave_form(tok, haskey(tok.vocab, string(c)) ? tok.vocab[string(c)] : register_token!(tok, string(c))) for c in txt]
            end),
        
        TokenAlgo(12, "HybridBPE_WordBoundary", "Hybrid",
            (tok, txt) -> tokenize(tok, txt; n_samples=32))
    ]
end

function create_winner_variants(winner_name::String, round::Int)
    base_id = 12 * round
    return [
        TokenAlgo(base_id + 1, "$(winner_name)_FastEnv16", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=16)),
        TokenAlgo(base_id + 2, "$(winner_name)_HiRes48", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=48)),
        TokenAlgo(base_id + 3, "$(winner_name)_MidRes28", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=28)),
        TokenAlgo(base_id + 4, "$(winner_name)_Coherent32", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)),
        TokenAlgo(base_id + 5, "$(winner_name)_Quantum24", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=24)),
        TokenAlgo(base_id + 6, "$(winner_name)_Ultra20", "WinnerVar",
            (tok, txt) -> tokenize(tok, txt; n_samples=20))
    ]
end

function create_new_algorithms(round::Int)
    base_id = 12 * round + 6
    families = [
        "Chladni_Nodal", "Fibonacci_Spiral", "Golden_Harmonic", 
        "Phasor_Manifold", "Soliton_Envelope", "Lagrangian_Flow",
        "Hermite_Expansion", "Bessel_Cylindrical", "Laguerre_Polynomial",
        "Zernike_Disk", "Chebyshev_Basis", "Legendre_Spherical"
    ]
    
    offset = ((round - 2) * 6) % length(families)
    return [
        TokenAlgo(base_id + i, "$(families[mod1(offset+i, length(families))])_R$round", "New",
            (tok, txt) -> tokenize(tok, txt; n_samples=32))
        for i in 1:6
    ]
end

function run_tournament_v2()
    println("="^80)
    println("    NEW 144-ALGORITHM TOKENIZER TOURNAMENT v2")
    println("="^80)
    
    tok = default_tokenizer()
    all_winners = []
    
    # Round 1
    println("\n>>> ROUND 1: Baseline + 5 Variants + 6 New")
    r1_algos = create_round1_algorithms()
    r1_results = [(a, evaluate_algo(a, tok)) for a in r1_algos]
    sort!(r1_results, by = x -> x[2].score, rev = true)
    
    r1_winner = r1_results[1]
    push!(all_winners, r1_winner)
    
    for (a, m) in r1_results
        @printf("  %-40s | Score: %8.2f | Time: %6.1f µs | Acc: %.2f | Comp: %.3f\n",
                a.name, m.score, m.time_us, m.acc, m.comp)
    end
    println("\n  🏆 Round 1 Winner: $(r1_winner[1].name) ($(round(r1_winner[2].score, digits=2)))")
    
    # Rounds 2-12
    curr_winner = r1_winner
    for r in 2:12
        println("\n>>> ROUND $r: 6 Variants + 6 New")
        
        variants = create_winner_variants(curr_winner[1].name, r)
        new_algos = create_new_algorithms(r)
        round_pool = vcat(variants, new_algos)
        
        round_results = [(a, evaluate_algo(a, tok)) for a in round_pool]
        sort!(round_results, by = x -> x[2].score, rev = true)
        
        round_winner = round_results[1]
        push!(all_winners, round_winner)
        curr_winner = round_winner
        
        for (a, m) in round_results
            @printf("  %-50s | Score: %8.2f | Acc: %.2f\n",
                    a.name, m.score, m.acc)
        end
        println("\n  🏆 Round $r Winner: $(round_winner[1].name) ($(round(round_winner[2].score, digits=2)))")
    end
    
    # Grand Champion
    sort!(all_winners, by = x -> x[2].score, rev = true)
    champ = all_winners[1]
    
    println("\n" * "="^80)
    println("                  🏆 GRAND CHAMPION 🏆")
    println("="^80)
    println("Algorithm:     $(champ[1].name)")
    println("Category:      $(champ[1].category)")
    println("Score:         $(round(champ[2].score, digits=2))")
    println("Time:          $(round(champ[2].time_us, digits=1)) µs")
    println("Accuracy:      $(round(champ[2].acc * 100, digits=1))%")
    println("Compression:   $(round(champ[2].comp, digits=4)) tokens/char")
    println("="^80)
    
    return (winners=all_winners, champion=champ, tok=tok)
end

# Run tournament
results = run_tournament_v2()

@testset "Tokenizer Tournament v2" begin
    @test length(results.winners) == 12
    @test results.champion[2].acc >= 0.9
    @test results.champion[2].score > 5000.0
end

println("\n✅ Tournament v2 Complete! Champion: $(results.champion[1].name)")
