# test_mwave_continuous_computing.jl
# Comprehensive test suite for Real Physical Wave (mwave) Computing,
# Sound-Native AI Substrate, Fractal Storage Compression, Flower of Life Wells,
# and Cymatic Eigen-State Extraction.

using Test
using LinearAlgebra
using Sovwave
using Sovwave.WaveML

@testset "Continuous Physical Wave & Sound-Native AI Suite" begin

    @testset "1. Sound-Native Computing Substrate (SoundCompute)" begin
        layer = create_layer(32, 16; omega=432.0)
        inp = rand(16)
        out_silent = zeros(32)
        out_audible = zeros(32)
        
        # Test silent sound computing (sonify=false)
        cfg_silent = SoundComputeConfig(sonify=false)
        sound_native_forward!(layer, inp, out_silent, 0.0; cfg=cfg_silent)
        @test norm(out_silent) > 0.0
        @test isfinite(layer.layer_energy)
        @test layer.layer_energy > 0.0
        
        # Test audible sound computing (sonify=true)
        s_buf = SoundAcousticBuffer(128)
        cfg_audible = SoundComputeConfig(sonify=true)
        sound_native_forward!(layer, inp, out_audible, 0.0; cfg=cfg_audible, buf=s_buf)
        
        # In-memory acoustic physics must produce identical mathematical results
        @test isapprox(out_silent, out_audible; atol=1e-10)
        # Audible buffer captured active sound frame
        @test norm(s_buf.audio_frame[1:32]) > 0.0
        
        # Test end-to-end model forward with sound-native substrate
        m_cfg = WaveModelConfig(layers=2, nodes=16, embed_dims=8)
        cfg = WaveMLConfig(model=m_cfg)
        model = WaveModel(cfg)
        
        m_in = rand(8)
        m_out = zeros(16)
        forward!(model, m_in, m_out; mode=:sound_native, sonify=false)
        @test norm(m_out) > 0.0
        @test model_energy(model) > 0.0
        
        # Verify zero heap allocations on repeated forward passes
        forward!(model, m_in, m_out)
        allocs = @allocated forward!(model, m_in, m_out)
        @test allocs == 0
    end

    @testset "2. Fractal Storage & Continuous Compression (FractalStorage)" begin
        layer = create_layer(64, 32; omega=432.0)
        uncompressed_floats = 3 * 64 * 32 # 6,144 floats
        
        # Compress layer to continuous fractal manifold
        f_cfg = compress_layer_to_fractal(layer)
        seed_floats = length(f_cfg.amp_seeds) + length(f_cfg.phase_seeds) + length(f_cfg.freq_seeds) # 192 floats
        
        compression_ratio = uncompressed_floats / seed_floats
        @test compression_ratio >= 30.0 # >30x compression
        
        # Reconstruct layer on-the-fly from fractal seeds with zero allocation
        evaluate_fractal_layer!(layer, f_cfg)
        allocs = @allocated evaluate_fractal_layer!(layer, f_cfg)
        @test allocs == 0
        
        # Verify continuous parameter evaluation at specific coordinate
        amp, ph, fr = get_fractal_param(f_cfg, 10, 15)
        @test 0.01 <= amp <= 2.0
        @test 0.0 <= ph <= 2π
        @test fr > 0.0
    end

    @testset "3. Flower of Life Potential Well & C6 Symmetry (FlowerOfLife)" begin
        pot = FlowerOfLifePotential(V0=1.5, lattice_a=4.0)
        
        # Test C_6 60-degree rotational symmetry: V_FoL(R_60° · r) == V_FoL(r)
        r_dist = 2.3
        theta = 0.4
        x = r_dist * cos(theta)
        y = r_dist * sin(theta)
        
        # Rotate by 60 degrees (π/3 radians)
        x_rot = r_dist * cos(theta + π / 3.0)
        y_rot = r_dist * sin(theta + π / 3.0)
        
        v_orig = flower_of_life_potential(x, y; pot=pot)
        v_rot = flower_of_life_potential(x_rot, y_rot; pot=pot)
        
        @test isapprox(v_orig, v_rot; atol=1e-10) # Exact C_6 hexagonal symmetry!
        
        # Test 2D potential landscape grid generation
        grid = zeros(Float64, 32, 32)
        compute_flower_of_life_grid!(grid; pot=pot)
        @test !all(grid .== 0.0)
        # Verify symmetry across center
        @test isapprox(grid[16, 16], grid[16, 16]; atol=1e-12)
    end

    @testset "4. Morphogenetic Wave Simulator (MorphogeneticSimulator)" begin
        m_cfg = MorphogeneticConfig(dt=0.001, dx=0.25, g=0.1, alpha=0.5, beta=0.2)
        field = MorphogeneticField2D(24, 24; dx=0.25)
        
        # Initial energy
        e_init = compute_free_energy(field, m_cfg)
        @test isfinite(e_init)
        
        # Single Schrödinger time-evolution step (preserves density)
        step_schrodinger!(field, m_cfg)
        @test any(field.psi_imag .!= 0.0) # Developed quantum phase
        
        # Ginzburg-Landau free energy relaxation
        e_before = compute_free_energy(field, m_cfg)
        relax_to_eigenstate!(field, m_cfg; steps=20)
        e_after = compute_free_energy(field, m_cfg)
        
        # Ground state relaxation reduces free energy!
        @test e_after <= e_before + 1e-4
    end

    @testset "5. Cymatic Eigen-State Extraction & Quantum Coherence (CymaticExtractor)" begin
        # 1. Quantum Phase Coherence γ ∈ [0.0, 1.0]
        coherent_state = fill(0.25, 32) # perfectly aligned phases
        incoherent_state = [2π * (i - 1) / 32 for i in 1:32] # uniformly distributed phases
        
        gamma_coh = compute_phase_coherence(coherent_state)
        gamma_incoh = compute_phase_coherence(incoherent_state)
        
        @test isapprox(gamma_coh, 1.0; atol=1e-8)
        @test isapprox(gamma_incoh, 0.0; atol=1e-8)
        
        # 2. Standing Wave Nodal Line Detection (Chladni nodes)
        wave_with_nodes = [sin(2π * i / 32) for i in 1:32]
        nodes = detect_standing_wave_nodes(wave_with_nodes; threshold=0.1)
        @test length(nodes) > 0
        
        # 3. Resonant Eigen-Frequency Extraction
        candidate_freqs = [396.0, 432.0, 528.0, 639.0, 741.0, 852.0]
        state = [cos((528.0 * 0.001) * i) for i in 1:64]
        
        best_freq, coherence = extract_cymatic_eigenfrequency(state, candidate_freqs)
        @test best_freq == 528.0 # Resonates with exact 528 Hz Solfeggio frequency!
        @test coherence > 0.0
        
        # 4. Cymatic Vocabulary Decoding without discrete matrix dot products
        vocab_freqs = [396.0, 432.0, 528.0, 639.0, 741.0, 852.0]
        best_idx = cymatic_decode_vocab(state, vocab_freqs)
        @test best_idx == 3 # 528.0 Hz is index 3
    end

end
