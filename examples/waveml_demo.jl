"""
    waveml_demo.jl

End-to-End Demonstration of WaveML: Pure Wave-Based Deep Learning in Julia.
Demonstrates:
1. YAML configuration loading (`model_config.yaml`)
2. Multi-dimensional wave field setup (d=3 spatial dimensions, x=4 properties)
3. WaveModel architecture (4 layers, 32 embeddings, 64 nodes, 432Hz harmonic ω)
4. Evolution-based training seeking ground state energy (no backprop, no Markov chains)
5. Real-time calculation metrics: calculation time (ms), speed per point (ns/pt), throughput (pts/s)
6. Training sonification: hearing the model evolve as sound + saving to WAV
7. Lossless MKV video model serialization (saving the "brain" of the model as video)
8. Running direct inference from the MKV video file!
"""

using Printf
using Random
using Sovwave
using Sovwave.Audio

"""
    run_waveml_demo()

Runs complete end-to-end WaveML training, sonification, and video serialization demonstration.
"""
function run_waveml_demo()
    println("="^80)
    println(" 🌟 WAVEML: PURE WAVE-BASED DEEP LEARNING DEMO 🌟")
    println("="^80)

    # 1. Load YAML Configuration
    config_file = joinpath(@__DIR__, "../model_config.yaml")
    println("\n[1] Loading Configuration from: $config_file")
    cfg = if isfile(config_file)
        load_config(config_file)
    else
        default_config()
    end
    @printf("  • Field: %d data points in %dD manifold, properties = %s\n",
            cfg.field.n_points, cfg.field.dimensions, cfg.field.properties)
    @printf("  • Model: %d layers, %d embeddings, %d quantum nodes\n",
            cfg.model.layers, cfg.model.embed_dims, cfg.model.nodes)
    @printf("  • Quantum Driving Frequency: ω = %.1f Hz | Fractal Scaling β_s = %.4f\n",
            cfg.model.omega, cfg.model.beta_s)
    @printf("  • User-Defined Audio: Carrier = %.2f Hz | Waveform = %s | Binaural Beat = %.1f Hz | Channels = %d\n",
            cfg.audio.carrier_frequency, cfg.audio.waveform, cfg.audio.binaural_beat, cfg.audio.channels)
    @printf("  • Emergent Video: %s (Tournament Champion) | Pixel Square Size: %s | Target Height: %dpx | %d fps\n",
            cfg.video.render_mode,
            cfg.video.pixel_scale == 0 ? "auto" : string(cfg.video.pixel_scale, "px"),
            cfg.video.target_height, cfg.video.fps)


    # 2. Build d-Dimensional Wave Field
    println("\n[2] Instantiating Continuous d-Dimensional Wave Field")
    field = create_field(cfg.field)
    println("  ✓ Continuous field initialized with $(length(field.points)) spatial points in $(field.dimensions)D space")

    # 3. Build WaveModel
    println("\n[3] Instantiating WaveModel Lattice")
    model = WaveModel(cfg)
    println("  ✓ $(length(model.layers)) wave layers initialized over quantum lattice nodes")

    # 4. Generate Synthetic Wave Ground-State Learning Dataset
    println("\n[4] Generating Training Data for Wave Energy Ground-State Task")
    n_samples = 200
    embed_dim = cfg.model.embed_dims
    inputs = [rand(embed_dim) for _ in 1:n_samples]
    # Ground-state target: destructive interference / minimal energy response
    targets = [0.2 .* sin.(2π .* (1:cfg.model.nodes) ./ cfg.model.nodes) for _ in 1:n_samples]
    println("  ✓ Prepared $n_samples data samples ($embed_dim dimensions each)")

    # 5. Train Model via Pure Wave Evolution with User-Defined Audio
    println("\n[5] Training via Pure Wave Evolution (Seeking Lowest Energy Ground State)...")
    train_cfg = WaveTrainConfig(
        batch_size = 16,
        learning_rate = 0.08,
        epochs = 25,
        population_size = 16,
        energy_target = 0.005,
        sonify = true,
        sonify_realtime = true, # Stream to speakers in real-time
        audio_sample_rate = cfg.audio.sample_rate
    )

    trained_model, history = train!(
        model,
        inputs,
        targets,
        train_cfg;
        audio_cfg = cfg.audio,
        audio_save_dir = joinpath(@__DIR__, "../audio_output"),
        verbose = true
    )

    # 6. Audio Sonification: Listen to the Trained Model with User-Defined Sound Properties
    println("\n[6] Audio Sonification of Trained Model (User-Defined Carrier & Binaural Sound)")
    # Render with user-defined 528Hz Solfeggio frequency and Alpha binaural beat in Stereo!
    custom_audio_cfg = WaveAudioConfig(
        carrier_frequency = 528.0, # User-defined (e.g. 528Hz Solfeggio or 440Hz or 432Hz)
        waveform = :harmonic,      # User-defined waveform
        binaural_beat = 10.0,      # 10 Hz Alpha wave entrainment
        channels = 2,              # Stereo
        envelope = :adsr,          # Full ADSR envelope
        volume = 0.90
    )
    stereo_buffer = sonify_model(trained_model; audio_cfg=custom_audio_cfg, duration=1.0)
    wav_path = joinpath(@__DIR__, "../trained_wave_model_528hz_stereo.wav")
    save_wav(stereo_buffer, wav_path; sample_rate=custom_audio_cfg.sample_rate)
    @printf("  ✓ Rendered model to 528 Hz Stereo Binaural Audio: %s (%.1f KB)\n",
            wav_path, filesize(wav_path)/1024)
    play_realtime!(stereo_buffer; sample_rate=custom_audio_cfg.sample_rate)
    println("  ✓ Played sonified model through speakers")


    # 7. Model Serialization as MKV Video File
    println("\n[7] Serializing Model as MKV Video File (Visualizing the Model's Brain)")
    mkv_path = joinpath(@__DIR__, "../wave_model_brain.mkv")
    save_model(trained_model, mkv_path; video_cfg = cfg.video, audio_cfg = cfg.audio)
    @printf("  ✓ Model saved as MKV video with presentation audio: %s (%.1f KB)\n",
            mkv_path, filesize(mkv_path)/1024)
    println("    (Frames: Step n → Step n_final | Mode: $(cfg.video.render_mode) | Pixel size: $(cfg.video.pixel_scale == 0 ? "auto" : string(cfg.video.pixel_scale, "px")))")


    # 8. Direct Inference from the MKV Video File!
    println("\n[8] Running Direct Inference from MKV Video File...")
    test_input = inputs[1]
    pred_from_mkv = infer(mkv_path, test_input)
    pred_in_mem = predict(trained_model, test_input)

    diff_norm = sum(abs, pred_from_mkv .- pred_in_mem)
    @printf("  ✓ Loaded model directly from MKV and generated %d output node activations\n",
            length(pred_from_mkv))
    @printf("  ✓ Maximum discrepancy vs in-memory model: %.6f\n", diff_norm)
    @printf("  ✓ First 5 output activations: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
            pred_from_mkv[1], pred_from_mkv[2], pred_from_mkv[3], pred_from_mkv[4], pred_from_mkv[5])

    println("\n" * "="^80)
    println(" 🎉 WAVEML DEMO COMPLETED SUCCESSFULLY! 🎉")
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_waveml_demo()
end
