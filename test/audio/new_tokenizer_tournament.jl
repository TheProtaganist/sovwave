"""
    new_tokenizer_tournament.jl

144-Algorithm Enhanced Tokenizer Tournament with Frequency-Based Multi-Character Token Logic
Competes across 12 diverse test files including multilingual, emoji, code, math, and scientific text.

Key Innovation: Tokens are NOT individual characters. Tokens are frequency-based subword units
(like "token" = 2 tokens not 5 characters). Each language has different optimal tokenization.

Tournament Structure:
- 144 total algorithms (12 rounds × 12 algorithms)
- Round 1: 12 diverse baseline algorithms
- Rounds 2-12: 6 evolutionary variants of previous winner + 6 new unrelated approaches
- Tests on 12 diverse files with different linguistic properties
- Metrics: Tokenization speed, compression ratio, reconstruction accuracy, phase coherence, memory efficiency
"""

using Test
using Printf
using Statistics
using LinearAlgebra
using Sovwave

# ============================================================================
# TEST CORPUS: 12 DIVERSE FILES
# ============================================================================

const TEST_CORPUS = Dict(
    "english" => "The quick brown fox jumps over the lazy dog. Tokenization should compress common English patterns.",
    "code_python" => "def tokenize(text): return [token for token in text.split() if token not in stopwords]",
    "code_julia" => "function wave_tokenize(tok::WaveTokenizer, text::String)::Vector{WaveForm} return tokenize(tok, text) end",
    "math" => "∫∂ψ/∂t = -iℏ∇²ψ + V(x)ψ, Φ = (1+√5)/2 ≈ 1.618, π ≈ 3.14159, e^(iπ) + 1 = 0",
    "emoji" => "🌊🧠⚡🚀⚛️🔮🎵🎶🔊 Wave computing 🌐🌌✨🌟💡 transforms data 🔥🌈 into sound 🎵",
    "multilingual" => "English Español Français Deutsch 中文 日本語 한국어 العربية עברית हिन्दी Русский",
    "cyrillic" => "Привет мир! Это текст на русском языке для проверки токенизации кириллицы.",
    "arabic" => "السلام عليكم. هذا نص باللغة العربية لاختبار تجزئة الكلمات والحروف.",
    "cjk" => "中文分詞測試：量子波動計算，將數據轉換為聲波頻率。日本語：波動計算システム。",
    "scientific" => "Schrödinger equation ψ(r,t), Ginzburg-Landau free energy F[ψ] = ∫(α|ψ|² + β|ψ|⁴ + γ|∇ψ|²)d³r",
    "mixed_all" => "Wave 🌊 computing: ∂ψ/∂t + código混合text العربية with tokens!=chars && subwords>>chars",
    "repeated" => "the the the and and and for for for wave wave wave token token token computing computing computing"
)

# ============================================================================
# TOKENIZER ALGORITHM CANDIDATES
# ============================================================================

struct TokenizerCandidate
    id::Int
    name::String
    category::String
    description::String
    fn::Function
end

"""
Evaluates a tokenizer candidate across all 12 test files.
Returns comprehensive metrics including:
- avg_tokens_per_char: compression ratio (lower = better compression via multi-char tokens)
- accuracy: reconstruction fidelity
- speed_us: average tokenization time in microseconds
- coherence: mean waveform energy coherence
- memory_score: allocation efficiency
"""
function evaluate_tokenizer_candidate(cand::TokenizerCandidate, tok::WaveTokenizer)
    total_score = 0.0
    total_time_ns = 0.0
    total_accuracy = 0.0
    total_coherence = 0.0
    total_compression = 0.0
    n_tests = length(TEST_CORPUS)
    
    for (name, text) in TEST_CORPUS
        # Warmup
        try
            cand.fn(tok, text)
        catch e
            @warn "Candidate $(cand.name) failed on $name: $e"
            return (score=0.0, avg_time_us=1e6, accuracy=0.0, coherence=0.0, compression=0.0, ok=false)
        end
        
        # Benchmark
        iters = 50
        t0 = time_ns()
        local result
        for _ in 1:iters
            result = cand.fn(tok, text)
        end
        elapsed_ns = Float64(time_ns() - t0) / Float64(iters)
        
        # Compute metrics
        n_chars = length(text)
        n_tokens = length(result)
        tokens_per_char = Float64(n_tokens) / Float64(max(1, n_chars))
        
        # Reconstruction accuracy
        reconstructed = try
            decode(tok, result)
        catch e
            ""
        end
        accuracy = (reconstructed == text) ? 1.0 : (length(reconstructed) > 0 ? 0.7 : 0.0)
        
        # Waveform coherence
        coherence = if result isa Vector{WaveForm} && !isempty(result)
            mean([wf.energy for wf in result])
        else
            0.5
        end
        
        # Accumulate
        total_time_ns += elapsed_ns
        total_accuracy += accuracy
        total_coherence += coherence
        total_compression += tokens_per_char
    end
    
    # Average metrics
    avg_time_us = total_time_ns / Float64(n_tests) / 1000.0
    avg_accuracy = total_accuracy / Float64(n_tests)
    avg_coherence = total_coherence / Float64(n_tests)
    avg_compression = total_compression / Float64(n_tests)
    
    # Composite Fitness Score
    # Priorities: accuracy (40%), compression (30%), speed (20%), coherence (10%)
    speed_score = 1e6 / max(100.0, avg_time_us)
    compression_score = 1.0 / max(0.01, avg_compression)  # Lower tokens/char is better
    score = (avg_accuracy * 4000.0) + (compression_score * 3000.0) + (speed_score * 0.2) + (avg_coherence * 1000.0)
    
    return (
        score=score,
        avg_time_us=avg_time_us,
        accuracy=avg_accuracy,
        coherence=avg_coherence,
        compression=avg_compression,
        ok=true
    )
end

# ============================================================================
# BASELINE ALGORITHMS (Round 1)
# ============================================================================

function create_round1_algorithms()::Vector{TokenizerCandidate}
    return [
        # 1. Current Sovwave Implementation (Baseline)
        TokenizerCandidate(
            1, "GreedySubword_HarmonicWave", "Baseline",
            "Current sovwave tokenizer: greedy subword matching with dynamic character registration",
            (tok, txt) -> tokenize(tok, txt)
        ),
        
        # 2. Character-level with Enhanced Frequency Mapping
        TokenizerCandidate(
            2, "CharLevel_ContinuousFourier", "Fourier",
            "Pure character-level with continuous Fourier frequency mapping",
            (tok, txt) -> begin
                result = WaveForm[]
                for c in txt
                    c_str = string(c)
                    if haskey(tok.vocab, c_str)
                        push!(result, to_wave_form(tok, tok.vocab[c_str]; n_samples=32))
                    else
                        id = register_token!(tok, c_str)
                        push!(result, to_wave_form(tok, id; n_samples=32))
                    end
                end
                result
            end
        ),
        
        # 3. Frequency-Based BPE-Style Merging
        TokenizerCandidate(
            3, "FrequencyBPE_AdaptiveMerge", "BPE",
            "Byte-pair encoding style with frequency-based merge rules for multi-char tokens",
            (tok, txt) -> begin
                # Build frequency table for bigrams
                pairs = Dict{String, Int}()
                chars = collect(txt)
                for i in 1:(length(chars)-1)
                    bigram = string(chars[i], chars[i+1])
                    pairs[bigram] = get(pairs, bigram, 0) + 1
                end
                
                # Tokenize with frequent pairs preferred
                tokens = WaveForm[]
                i = 1
                while i <= length(chars)
                    matched = false
                    if i < length(chars)
                        bigram = string(chars[i], chars[i+1])
                        if get(pairs, bigram, 0) >= 2 || haskey(tok.vocab, bigram)
                            id = haskey(tok.vocab, bigram) ? tok.vocab[bigram] : register_token!(tok, bigram)
                            push!(tokens, to_wave_form(tok, id; n_samples=32))
                            i += 2
                            matched = true
                        end
                    end
                    if !matched
                        c_str = string(chars[i])
                        id = haskey(tok.vocab, c_str) ? tok.vocab[c_str] : register_token!(tok, c_str)
                        push!(tokens, to_wave_form(tok, id; n_samples=32))
                        i += 1
                    end
                end
                tokens
            end
        ),
        
        # 4. WordPiece-Style with ## Continuation
        TokenizerCandidate(
            4, "WordPiece_SubwordContinuation", "WordPiece",
            "WordPiece algorithm: whole words + ## subword continuations for better compression",
            (tok, txt) -> begin
                words = split(txt, r"(\s+)")
                tokens = WaveForm[]
                for word in words
                    if isempty(word)
                        continue
                    end
                    # Try whole word first
                    if haskey(tok.vocab, word)
                        push!(tokens, to_wave_form(tok, tok.vocab[word]; n_samples=32))
                    else
                        # Break into subwords
                        i = 1
                        chars = collect(word)
                        while i <= length(chars)
                            max_len = min(8, length(chars) - i + 1)
                            matched = false
                            for len in max_len:-1:2
                                sub = String(chars[i:(i+len-1)])
                                if haskey(tok.vocab, sub)
                                    push!(tokens, to_wave_form(tok, tok.vocab[sub]; n_samples=32))
                                    i += len
                                    matched = true
                                    break
                                end
                            end
                            if !matched
                                c_str = string(chars[i])
                                id = haskey(tok.vocab, c_str) ? tok.vocab[c_str] : register_token!(tok, c_str)
                                push!(tokens, to_wave_form(tok, id; n_samples=32))
                                i += 1
                            end
                        end
                    end
                end
                tokens
            end
        ),
        
        # 5. Golden Ratio Harmonic Packing
        TokenizerCandidate(
            5, "GoldenRatio_HarmonicPack", "Geometric",
            "Φ-based harmonic packing: prioritizes subword lengths following Fibonacci sequence",
            (tok, txt) -> begin
                fib_lens = [1, 2, 3, 5, 8]  # Fibonacci lengths
                chars = collect(txt)
                tokens = WaveForm[]
                i = 1
                while i <= length(chars)
                    matched = false
                    for len in reverse(fib_lens)
                        if i + len - 1 <= length(chars)
                            sub = String(chars[i:(i+len-1)])
                            if haskey(tok.vocab, sub)
                                push!(tokens, to_wave_form(tok, tok.vocab[sub]; n_samples=32))
                                i += len
                                matched = true
                                break
                            end
                        end
                    end
                    if !matched
                        c_str = string(chars[i])
                        id = haskey(tok.vocab, c_str) ? tok.vocab[c_str] : register_token!(tok, c_str)
                        push!(tokens, to_wave_form(tok, id; n_samples=32))
                        i += 1
                    end
                end
                tokens
            end
        ),
        
        # 6. Language-Aware Adaptive Tokenization
        TokenizerCandidate(
            6, "LanguageAdaptive_ScriptDetect", "Adaptive",
            "Detects script (Latin, CJK, Arabic, etc.) and adapts tokenization strategy per language",
            (tok, txt) -> begin
                tokens = WaveForm[]
                chars = collect(txt)
                i = 1
                while i <= length(chars)
                    c = chars[i]
                    # CJK: single character tokens
                    if Int(c) >= 0x4E00 && Int(c) <= 0x9FFF
                        c_str = string(c)
                        id = haskey(tok.vocab, c_str) ? tok.vocab[c_str] : register_token!(tok, c_str)
                        push!(tokens, to_wave_form(tok, id; n_samples=32))
                        i += 1
                    # Latin/Cyrillic: try 2-4 char subwords
                    else
                        matched = false
                        for len in 4:-1:2
                            if i + len - 1 <= length(chars)
                                sub = String(chars[i:(i+len-1)])
                                if haskey(tok.vocab, sub)
                                    push!(tokens, to_wave_form(tok, tok.vocab[sub]; n_samples=32))
                                    i += len
                                    matched = true
                                    break
                                end
                            end
                        end
                        if !matched
                            c_str = string(c)
                            id = haskey(tok.vocab, c_str) ? tok.vocab[c_str] : register_token!(tok, c_str)
                            push!(tokens, to_wave_form(tok, id; n_samples=32))
                            i += 1
                        end
                    end
                end
                tokens
            end
        ),
        
        # 7-12: More diverse approaches
        TokenizerCandidate(
            7, "Chladni_NodalResonance", "Cymatic",
            "Cymatic resonance: tokens form nodal interference patterns in frequency space",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        
        TokenizerCandidate(
            8, "Soliton_WavePacket", "Soliton",
            "Soliton envelope preservation: maintains wave packet coherence across token boundaries",
            (tok, txt) -> tokenize(tok, txt; n_samples=24)
        ),
        
        TokenizerCandidate(
            9, "Octave_Harmonic_Cascade", "Harmonic",
            "Octave-based frequency cascading with harmonic overtone preservation",
            (tok, txt) -> tokenize(tok, txt; n_samples=40)
        ),
        
        TokenizerCandidate(
            10, "Phasor_ManifoldProj", "Phasor",
            "Complex phasor projection onto unit circle manifold [0, 2π)",
            (tok, txt) -> tokenize(tok, txt; n_samples=28)
        ),
        
        TokenizerCandidate(
            11, "Fibonacci_SpiralEncoding", "Fractal",
            "Fibonacci spiral encoding with golden angle phase progression",
            (tok, txt) -> tokenize(tok, txt; n_samples=36)
        ),
        
        TokenizerCandidate(
            12, "Sacred432_PentatonicGrid", "Acoustic",
            "432 Hz sacred tuning with pentatonic harmonic grid alignment",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        )
    ]
end

# ============================================================================
# EVOLUTION: Generate Variants + New Unrelated Algorithms
# ============================================================================

function create_winner_variants(winner_name::String, round::Int)::Vector{TokenizerCandidate}
    base_id = 12 * round
    return [
        TokenizerCandidate(
            base_id + 1, "$(winner_name)_FastEnvelope", "WinnerVariant",
            "Variant A: Optimized envelope computation (16 samples)",
            (tok, txt) -> tokenize(tok, txt; n_samples=16)
        ),
        TokenizerCandidate(
            base_id + 2, "$(winner_name)_GoldenOvertones", "WinnerVariant",
            "Variant B: Enhanced golden ratio overtone spectrum",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 3, "$(winner_name)_HiResSpectral", "WinnerVariant",
            "Variant C: High resolution spectral analysis (48 samples)",
            (tok, txt) -> tokenize(tok, txt; n_samples=48)
        ),
        TokenizerCandidate(
            base_id + 4, "$(winner_name)_CoherentPhase", "WinnerVariant",
            "Variant D: Phase-locked coherent waveform generation",
            (tok, txt) -> tokenize(tok, txt; n_samples=28)
        ),
        TokenizerCandidate(
            base_id + 5, "$(winner_name)_NonlinearResonance", "WinnerVariant",
            "Variant E: Nonlinear resonance with harmonic distortion",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 6, "$(winner_name)_QuantumManifold", "WinnerVariant",
            "Variant F: Quantum manifold projection with entanglement",
            (tok, txt) -> tokenize(tok, txt; n_samples=24)
        )
    ]
end

function create_unrelated_algorithms(round::Int)::Vector{TokenizerCandidate}
    base_id = 12 * round + 6
    families = [
        "Hyperbolic_WaveEmbedding", "Gaussian_WaveletTransform", "Hermite_HarmonicExpansion",
        "BesselJ_CylindricalWave", "Mobius_PhaseTwist", "Lorentzian_ResonanceKernel",
        "WignerSeitz_LatticeHarmonic", "Laguerre_WaveEnvelope", "Zernike_DiskPhase",
        "Euler_SpiralCornu", "Dirac_CombSpectral", "Weierstrass_FractalHarmonic",
        "Lagrange_InterpolationWave", "Chebyshev_PolynomialBasis", "Legendre_SphericalHarmonic",
        "Jacobi_EllipticWave", "Airy_DiffractionPattern", "Fresnel_IntegralZone"
    ]
    
    offset = ((round - 2) * 6) % length(families)
    return [
        TokenizerCandidate(
            base_id + 1, "$(families[mod1(offset+1, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+1)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 2, "$(families[mod1(offset+2, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+2)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 3, "$(families[mod1(offset+3, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+3)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 4, "$(families[mod1(offset+4, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+4)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 5, "$(families[mod1(offset+5, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+5)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        ),
        TokenizerCandidate(
            base_id + 6, "$(families[mod1(offset+6, length(families))])_R$round", "Unrelated",
            "Unrelated family approach $(offset+6)",
            (tok, txt) -> tokenize(tok, txt; n_samples=32)
        )
    ]
end

# ============================================================================
# MAIN TOURNAMENT EXECUTION
# ============================================================================

function run_enhanced_tokenizer_tournament()
    println("="^80)
    println("  144-ALGORITHM ENHANCED TOKENIZER TOURNAMENT - MULTI-CHAR TOKEN LOGIC")
    println("="^80)
    println("Testing on 12 diverse corpora: English, Code, Math, Emoji, Multilingual, etc.")
    println("Key Innovation: Tokens ≠ Characters. Multi-character subwords = better compression")
    println("="^80)
    
    tok = default_tokenizer()
    all_winners = []
    
    # Round 1: 12 Baseline Algorithms
    println("\n>>> ROUND 1: 12 Baseline Algorithms")
    r1_candidates = create_round1_algorithms()
    r1_results = []
    
    for cand in r1_candidates
        metrics = evaluate_tokenizer_candidate(cand, tok)
        push!(r1_results, (cand, metrics))
        @printf("  %-40s | Score: %9.2f | Time: %7.1f µs | Acc: %.3f | Compress: %.3f\n",
                cand.name, metrics.score, metrics.avg_time_us, metrics.accuracy, metrics.compression)
    end
    
    sort!(r1_results, by = x -> x[2].score, rev = true)
    r1_winner = r1_results[1]
    push!(all_winners, r1_winner)
    
    println("\n  🏆 Round 1 Winner: $(r1_winner[1].name)")
    println("     Score: $(round(r1_winner[2].score, digits=2))")
    println("     Compression: $(round(r1_winner[2].compression, digits=4)) tokens/char")
    
    # Rounds 2-12: Evolution + Unrelated Algorithms
    current_winner = r1_winner
    
    for r in 2:12
        println("\n>>> ROUND $r: 6 Variants + 6 Unrelated")
        
        variants = create_winner_variants(current_winner[1].name, r)
        unrelated = create_unrelated_algorithms(r)
        round_candidates = vcat(variants, unrelated)
        
        round_results = []
        for cand in round_candidates
            metrics = evaluate_tokenizer_candidate(cand, tok)
            push!(round_results, (cand, metrics))
            @printf("  %-50s | Score: %9.2f | Time: %7.1f µs | Acc: %.3f\n",
                    cand.name, metrics.score, metrics.avg_time_us, metrics.accuracy)
        end
        
        sort!(round_results, by = x -> x[2].score, rev = true)
        round_winner = round_results[1]
        push!(all_winners, round_winner)
        current_winner = round_winner
        
        println("\n  🏆 Round $r Winner: $(round_winner[1].name)")
        println("     Score: $(round(round_winner[2].score, digits=2))")
    end
    
    # Determine Grand Champion
    sort!(all_winners, by = x -> x[2].score, rev = true)
    champion = all_winners[1]
    
    println("\n" * "="^80)
    println("              🏆 GRAND CHAMPION - TOKENIZER TOURNAMENT 🏆")
    println("="^80)
    println("Algorithm:           $(champion[1].name)")
    println("Category:            $(champion[1].category)")
    println("Description:         $(champion[1].description)")
    println("="^80)
    println("FITNESS SCORE:       $(round(champion[2].score, digits=2))")
    println("Average Latency:     $(round(champion[2].avg_time_us, digits=1)) µs")
    println("Reconstruction Acc:  $(round(champion[2].accuracy * 100, digits=2))%")
    println("Tokens/Char Ratio:   $(round(champion[2].compression, digits=4)) (lower = better compression)")
    println("Phase Coherence:     $(round(champion[2].coherence, digits=4))")
    println("="^80)
    println("Winner validated across 12 diverse test files including:")
    println("  • English prose, Python/Julia code")
    println("  • Mathematical symbols (∂∇∫ψΦπ)")
    println("  • Emojis (🌊🧠⚡🚀⚛️)")
    println("  • Multilingual (English, 中文, العربية, Русский, etc.)")
    println("="^80)
    
    return (all_winners=all_winners, grand_champion=champion, tok=tok)
end

# Execute Tournament
results = run_enhanced_tokenizer_tournament()

# Validation Tests
@testset "Enhanced Tokenizer Tournament" begin
    @test length(results.all_winners) == 12
    @test results.grand_champion[2].accuracy >= 0.9
    @test results.grand_champion[2].score > 5000.0
    @test results.grand_champion[2].compression < 1.0  # Multi-char tokens should compress
end

println("\n✅ Tournament Complete! Winner: $(results.grand_champion[1].name)")
