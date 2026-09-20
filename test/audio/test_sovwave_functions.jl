# test/audio/test_sovwave_functions.jl — Unit tests for SovwaveFunctions.jl
using Test
using Sovwave
using Sovwave.WaveML

@testset "SovwaveFunctions API Suite" begin
    # 1. Model creation & Interactive editing
    model = create_model(nodes=16, embed_dims=16, layers=2, omega=432.0)
    @test length(model.layers) == 2
    @test model.model_config.nodes == 16

    # Test scale_amplitudes!
    orig_amp = copy(model.layers[1].amplitudes)
    scale_amplitudes!(model, 1.5; layer_idx=1)
    @test isapprox(model.layers[1].amplitudes, orig_amp .* 1.5; atol=1e-6)

    # Test shift_phases!
    orig_phs = copy(model.layers[1].phases)
    shift_phases!(model, π / 4; layer_idx=1)
    @test isapprox(model.layers[1].phases, mod2pi.(orig_phs .+ (π / 4)); atol=1e-6)

    # Test modulate_frequencies!
    orig_freqs = copy(model.layers[2].frequencies)
    modulate_frequencies!(model, 1.25; layer_idx=2)
    @test isapprox(model.layers[2].frequencies, orig_freqs .* 1.25; atol=1e-6)

    # Test edit_layer!
    edit_layer!(model, 1; beta=2.0, speed=1.5)
    @test all(model.layers[1].fractal_scales .== 2.0)
    @test all(model.layers[1].wave_speeds .== 1.5)

    # Test edit_model!
    edit_model!(model; omega=528.0)
    @test model.model_config.omega == 528.0

    # Test inspect_harmonics
    info = inspect_harmonics(model; layer_idx=1)
    @test info["layer"] == 1
    @test info["nodes"] == 16
    @test info["embed_dim"] == 16
    @test info["mean_amplitude"] > 0.0

    # 2. Dataset loading, validation, and wave transform
    tmp_csv = tempname() * ".csv"
    open(tmp_csv, "w") do io
        println(io, "f1,f2,f3,label")
        for i in 1:20
            println(io, "$(0.1*i),$(0.05*i),$(0.02*i),$(i % 2)")
        end
    end

    ds = load_dataset(tmp_csv; label_col="label")
    @test size(ds.features, 1) == 20
    @test size(ds.features, 2) == 3
    @test length(ds.labels) == 20

    val_res = validate_dataset(ds)
    @test val_res["valid"] == true
    @test val_res["num_samples"] == 20

    # Tournament 9 Data-to-Wave transform
    waves = process_to_waves(ds.features; target_dim=16)
    @test size(waves) == (20, 16)
    @test all(isfinite.(waves))
    rm(tmp_csv, force=true)

    # 3. Custom Wave Layers
    resonator = WaveResonator(16, 16; resonance_freq=432.0, q_factor=5.0)
    @test resonator.in_dim == 16
    @test resonator.out_dim == 16
    in_vec = randn(16)
    out_res = resonator(in_vec)
    @test length(out_res) == 16
    @test all(isfinite.(out_res))

    chamber = WaveChamber(16, 16; num_standing_modes=4, damping=0.05)
    @test chamber.in_dim == 16
    @test chamber.out_dim == 16
    out_ch = chamber(in_vec)
    @test length(out_ch) == 16
    @test all(isfinite.(out_ch))

    # 4. Mechanics-based Optimizer (Tournament 10 Grand Champion)
    opt = WaveMechanicsOptimizer(lr=0.05, gamma=0.55, beta_viscosity=0.01)
    @test opt.lr == 0.05
    @test opt.gamma == 0.55
    step_mechanics!(opt, model, 0.25)
    @test opt.step_count == 1

    # 5. Fast Training run
    tiny_cfg = WaveMLConfig(
        model = WaveModelConfig(nodes=8, embed_dims=8, layers=2),
        train = WaveTrainConfig(epochs=2, population_size=4, batch_size=4, sonify=false)
    )

    train_x = randn(8, 8)
    train_y = randn(8, 8)
    trained_model = train_wave(train_x, train_y; cfg=tiny_cfg, play_sound=false)
    @test length(trained_model.layers) == 2
    @test trained_model.model_config.nodes == 8
end
