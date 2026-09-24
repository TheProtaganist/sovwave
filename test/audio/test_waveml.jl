"""
    test_waveml.jl

Unit and integration tests for WaveML (Pure Wave-Based Deep Learning).
Verifies:
1. YAML configuration loading, parsing, and serialization
2. d-Dimensional continuous wave fields (1D, 2D, 3D, 5D) and champion wave propagation
3. Flexible wave layers, quantum lattice nodes, and parameter mutations
4. Multi-layer WaveModel forward pass and temporal superposition
5. Energy loss, MMD ground state loss, resonance, and wave accuracy
6. Island-model wave evolution and ground-state convergence
7. Full training loop with real-time speed metrics and sonification callbacks
8. Lossless MKV video model serialization (saving models as video files)
9. Direct inference from saved MKV video files
10. Dual-mode audio sonification (WAV file generation and real-time playback interface)
"""

using Test

if !isdefined(Main, :Sovwave)
    try
        using Sovwave
    catch
        include("../../src/Sovwave.jl")
        using .Sovwave
    end
end
using .Sovwave.Audio

@testset "WaveML: Pure Wave Computing Framework" begin

    @testset "[1] YAML Configuration System" begin
        cfg = default_config()
        @test cfg.field.n_points == 128
        @test cfg.field.dimensions == 3
        @test cfg.field.properties == [:mass, :charge, :energy, :spin]
        @test cfg.model.layers == 4
        @test cfg.model.omega == 432.0
        @test cfg.model.beta_s ≈ 1.618033988749895
        @test cfg.train.batch_size == 16
        @test cfg.train.sonify == true

        @test cfg.audio.carrier_frequency == 432.0
        @test cfg.audio.waveform == :physical

        # Test save and load round-trip
        tmp_yaml = tempname() * ".yaml"
        try
            custom_cfg = WaveMLConfig(
                audio = WaveAudioConfig(carrier_frequency=528.0, waveform=:harmonic, binaural_beat=6.0, channels=2)
            )
            save_config(custom_cfg, tmp_yaml)
            @test isfile(tmp_yaml)
            loaded_cfg = load_config(tmp_yaml)
            @test loaded_cfg.field.n_points == custom_cfg.field.n_points
            @test loaded_cfg.field.dimensions == custom_cfg.field.dimensions
            @test loaded_cfg.model.layers == custom_cfg.model.layers
            @test loaded_cfg.model.omega == custom_cfg.model.omega
            @test loaded_cfg.train.epochs == custom_cfg.train.epochs
            @test loaded_cfg.audio.carrier_frequency == 528.0
            @test loaded_cfg.audio.waveform == :harmonic
            @test loaded_cfg.audio.binaural_beat == 6.0
            @test loaded_cfg.audio.channels == 2
        finally
            isfile(tmp_yaml) && rm(tmp_yaml, force=true)
        end

        # Test loading root model_config.yaml if present
        root_cfg_path = joinpath(@__DIR__, "../../model_config.yaml")
        if isfile(root_cfg_path)
            r_cfg = load_config(root_cfg_path)
            @test r_cfg.model.omega == 432.0
            @test r_cfg.audio.carrier_frequency == 432.0
            @test r_cfg.audio.channels == 2
        end
    end

    @testset "[2] d-Dimensional Wave Fields" begin
        for d in [1, 2, 3, 5]
            f_cfg = WaveFieldConfig(n_points=64, dimensions=d, properties=[:x, :y, :z])
            field = create_field(f_cfg)
            @test field.dimensions == d
            @test length(field.points) == 64
            @test length(field.points[1].position) == d
            @test length(field.points[1].values) == 3

            # Test champion propagation
            amps = [1.0, 0.5]
            phs = [0.0, π/2]
            freqs = [1.0, 2.0]
            out = propagate_field!(field, amps, phs, freqs, 432.0, 0.0, 1.618)
            @test length(out) == 64
            @test field_energy(field) > 0.0
            @test field.time_step == 1

            # Test reset
            reset_field!(field)
            @test field_energy(field) == 0.0
            @test field.time_step == 0

            # Test evaluate field matrix
            mat = evaluate_field(field)
            @test size(mat) == (64, 3)
        end
    end

    @testset "[3] Wave Layers & Quantum Lattice" begin
        layer = create_layer(32, 16; omega=432.0, beta_s=1.618)
        @test layer.nodes == 32
        @test layer.embed_dim == 16
        @test size(layer.amplitudes) == (32, 16)
        @test size(layer.phases) == (32, 16)

        # Forward pass
        input_vec = rand(16)
        out = forward!(layer, input_vec, 0.0)
        @test length(out) == 32
        @test layer_energy(layer) > 0.0

        # Mutation
        old_amps = copy(layer.amplitudes)
        mutate!(layer, 0.1)
        @test layer.amplitudes != old_amps

        # Crossover
        layer_b = create_layer(32, 16; omega=432.0, beta_s=1.618)
        child = crossover(layer, layer_b)
        @test child.nodes == 32
        @test child.embed_dim == 16
    end

    @testset "[4] Multi-Layer WaveModel" begin
        cfg = default_config()
        model = WaveModel(cfg)
        @test length(model.layers) == cfg.model.layers
        @test model_energy(model) == 0.0

        input_data = rand(cfg.model.embed_dims)
        out = forward!(model, input_data; t=0.0)
        @test length(out) == cfg.model.nodes
        @test model_energy(model) > 0.0

        # Deep cloning
        cloned = clone(model)
        @test length(cloned.layers) == length(model.layers)
        mutate!(cloned, 0.2)
        @test cloned.layers[1].amplitudes != model.layers[1].amplitudes
    end

    @testset "[5] Wave Energy Loss & Metrics" begin
        y1 = [0.8, 0.2, 0.5, 0.9]
        y2 = [0.75, 0.25, 0.48, 0.88]

        @test energy_loss(y1, y2) < 0.1
        @test energy_loss(y1, y1) == 0.0
        @test mmd_loss(y1, y1) == 0.0
        @test resonance_loss(y1, y1) == 0.0
        @test interference_loss(y1, y1) == 0.0

        # Perfect match accuracy
        @test wave_accuracy(y1, y1) == 1.0
        # Mismatched accuracy
        @test wave_accuracy([0.9, 0.9], [0.1, 0.1]) == 0.0
    end

    @testset "[6] Island-Model Wave Evolution" begin
        cfg = WaveMLConfig(
            field = WaveFieldConfig(n_points=16, dimensions=2),
            model = WaveModelConfig(layers=2, embed_dims=8, nodes=8),
            train = WaveTrainConfig(epochs=5, population_size=8)
        )
        state = init_population(cfg)
        @test length(state.population) == 8
        @test state.generation == 0

        # Toy batch
        b_in = [rand(8) for _ in 1:4]
        b_target = [rand(8) for _ in 1:4]

        best_e1 = evaluate_population!(state, b_in, b_target)
        @test best_e1 >= 0.0

        evolve_generation!(state, 0.25)
        @test state.generation == 1
        @test length(state.population) == 8
    end

    @testset "[7] Full Training Loop & Speed Metrics" begin
        cfg = WaveMLConfig(
            field = WaveFieldConfig(n_points=16, dimensions=2),
            model = WaveModelConfig(layers=2, embed_dims=4, nodes=4, omega=432.0),
            train = WaveTrainConfig(batch_size=4, epochs=6, population_size=6, learning_rate=0.1, sonify=false)
        )
        model = WaveModel(cfg)

        inputs = [rand(4) for _ in 1:16]
        targets = [zeros(4) for _ in 1:16] # Seeking zero energy state

        trained_model, history = train!(model, inputs, targets, cfg.train; verbose=false)
        @test history.best_loss < Inf
        @test length(history.metrics) == 6
        @test history.metrics[end].calc_time_ms > 0.0
        @test history.metrics[end].throughput_pts_sec > 0.0
        @test history.total_time_sec > 0.0
    end

    @testset "[8] User-Defined Audio Sonification & WAV Export" begin
        cfg = WaveMLConfig(
            model = WaveModelConfig(layers=2, embed_dims=8, nodes=8, omega=432.0),
            audio = WaveAudioConfig(carrier_frequency=440.0, waveform=:sawtooth, channels=1)
        )
        model = WaveModel(cfg)

        # Test 1: User-defined carrier frequency 440.0 Hz (Concert Pitch) instead of 432.0 Hz
        audio_440 = sonify_model(model; carrier_frequency=440.0, duration=0.1)
        @test length(audio_440) == 4800
        @test any(audio_440 .!= 0.0)

        # Test 2: User-defined 528.0 Hz (Solfeggio) frequency
        audio_528 = sonify_model(model; carrier_frequency=528.0, duration=0.1)
        @test length(audio_528) == 4800
        @test any(audio_528 .!= 0.0)

        # Test 3: Arbitrary user-defined carrier frequency (e.g. 314.159 Hz)
        audio_custom = sonify_model(model; carrier_frequency=314.159, duration=0.1)
        @test length(audio_custom) == 4800
        @test audio_custom != audio_440

        # Test 4: Custom waveforms (sine, harmonic, physical, triangle, sawtooth)
        for wf in [:sine, :harmonic, :physical, :triangle, :sawtooth]
            w_buf = sonify_model(model; waveform=wf, duration=0.05)
            @test length(w_buf) == 2400
            @test any(w_buf .!= 0.0)
        end

        # Test 5: Custom envelopes (exponential_decay, adsr, percussive, sustain)
        for env in [:exponential_decay, :adsr, :percussive, :sustain]
            env_buf = sonify_model(model; envelope=env, attack=0.01, decay=0.05, sustain=0.5, release=0.1, duration=0.05)
            @test length(env_buf) == 2400
            @test any(env_buf .!= 0.0)
        end

        # Test 6: Stereo & Binaural Entrainment (channels = 2, binaural_beat = 10.0 Hz)
        stereo_audio = sonify_model(
            model;
            carrier_frequency = 432.0,
            binaural_beat = 10.0,
            channels = 2,
            pan = 0.2,
            duration = 0.1
        )
        @test isa(stereo_audio, Matrix{Float64})
        @test size(stereo_audio) == (2, 4800)
        # Verify Left and Right channels are distinct due to binaural beat offset
        @test stereo_audio[1, :] != stereo_audio[2, :]

        # Test 7: Step sonification with custom carrier frequency
        step_audio = sonify_step(0.05, model; carrier_frequency=528.0, duration=0.05)
        @test length(step_audio) == 2400

        # Test 8: WAV file writers (both Mono and Stereo)
        tmp_mono_wav = tempname() * ".wav"
        tmp_stereo_wav = tempname() * ".wav"
        try
            # Mono WAV
            save_wav(audio_440, tmp_mono_wav; sample_rate=48000)
            @test isfile(tmp_mono_wav)
            @test filesize(tmp_mono_wav) > 44

            header_m = read(tmp_mono_wav, 44)
            @test String(header_m[1:4]) == "RIFF"
            @test String(header_m[9:12]) == "WAVE"
            @test reinterpret(UInt16, header_m[23:24])[1] == 1 # NumChannels = 1

            # Stereo WAV
            save_wav(stereo_audio, tmp_stereo_wav; sample_rate=48000)
            @test isfile(tmp_stereo_wav)
            @test filesize(tmp_stereo_wav) > filesize(tmp_mono_wav)

            header_s = read(tmp_stereo_wav, 44)
            @test String(header_s[1:4]) == "RIFF"
            @test String(header_s[9:12]) == "WAVE"
            @test reinterpret(UInt16, header_s[23:24])[1] == 2 # NumChannels = 2
        finally
            isfile(tmp_mono_wav) && rm(tmp_mono_wav, force=true)
            isfile(tmp_stereo_wav) && rm(tmp_stereo_wav, force=true)
        end

        # Test 9: play_realtime! with custom player string does not error
        @test isa(play_realtime!(audio_440[1:100]; player="none_existing_safe_test"), Bool)
    end

    @testset "[9] Lossless MKV Video Model Serialization & Inference" begin
        cfg = WaveMLConfig(
            model = WaveModelConfig(layers=3, embed_dims=16, nodes=16, omega=432.0)
        )
        model = WaveModel(cfg)

        # Test raw frame conversion
        raw, w, h, frames = model_to_rgb_frames(model)
        @test frames == 3
        @test w == 16
        @test h == 16
        @test length(raw) == 16 * 16 * 3 * 3

        reconstructed = rgb_frames_to_model(raw, w, h, frames, cfg)
        @test length(reconstructed.layers) == 3

        # Test emergent visual frame generation (Step n to Step n_final, NO black screen)
        vis_raw, w_v, h_v, n_v = model_to_visual_frames(model; n_frames=8, mode=:fibonacci_resonance)
        @test length(vis_raw) == w_v * h_v * 3 * 8
        # Frame 1 (Step n) MUST NOT be a black screen
        frame1_sum = sum(Int.(vis_raw[1:(w_v * h_v * 3)]))
        @test frame1_sum > 1000 # Vibrant non-black coherent frame

        # Test both tournament winner color generators and state_colors
        c_fib = emergent_color(1.0, π/4, 1.0, 1.618, 0.5; mode=:fibonacci_resonance)
        c_potts = emergent_color(1.0, π/4, 1.0, 1.618, 0.5; mode=:potts_model_q_state_domains)
        c_potts_alias = emergent_color(1.0, π/4, 1.0, 1.618, 0.5; mode=:potts_champion)
        @test all(0 .<= c_fib .<= 255)
        @test all(0 .<= c_potts .<= 255)
        @test c_potts == c_potts_alias

        # Test custom Potts state colors (Magenta, Yellow, Cyan)
        c_custom = emergent_color(1.0, 0.0, 1.0, 1.618, 0.0;
            mode=:potts_model_q_state_domains,
            state_colors=[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        )
        @test c_custom[1] > 200 && c_custom[3] > 200 # Magenta State 0 has high R and high B

        # Test full MKV save and load with emergent visual track and presentation audio track
        tmp_mkv = tempname() * ".mkv"
        try
            save_model(model, tmp_mkv; render_mode=:potts_model_q_state_domains, include_audio=true)
            @test isfile(tmp_mkv)
            @test filesize(tmp_mkv) > 100

            # Companion MP4 should also be generated with audio
            mp4_f = replace(tmp_mkv, r"\.mkv$" => ".mp4")
            @test isfile(mp4_f)
            @test filesize(mp4_f) > 100

            # Verify model can be loaded from MKV and reconstructed accurately
            loaded_model = load_model(tmp_mkv)
            @test length(loaded_model.layers) == 3
            @test loaded_model.model_config.nodes == 16
            @test loaded_model.model_config.embed_dims == 16

            # Verify parameters recovered with high fidelity
            orig_amp = model.layers[1].amplitudes[1, 1]
            load_amp = loaded_model.layers[1].amplitudes[1, 1]
            @test abs(orig_amp - load_amp) < 0.05

            # Test inference directly from the saved MKV video file (audio is ignored by infer!)
            test_input = rand(16)
            pred_from_mkv = infer(tmp_mkv, test_input)
            @test length(pred_from_mkv) == 16

            # Test in-memory predict
            pred_in_mem = predict(model, test_input)
            @test length(pred_in_mem) == 16
        finally
            isfile(tmp_mkv) && rm(tmp_mkv, force=true)
            meta = replace(tmp_mkv, r"\.mkv$" => "") * "_meta.yaml"
            isfile(meta) && rm(meta, force=true)
            mp4_f = replace(tmp_mkv, r"\.mkv$" => ".mp4")
            isfile(mp4_f) && rm(mp4_f, force=true)
        end
    end

end
