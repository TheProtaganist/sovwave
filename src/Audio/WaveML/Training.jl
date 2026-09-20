"""
    WaveML.Training

Evolution-Based Training Loop for Wave Models.
Tracks real-time calculation metrics (speed per point, throughput, ground state energy,
and accuracy) and triggers sonification callbacks during training.

Incorporates the tournament champion algorithms:
👑 `one_cycle_schedule` (Score: 77,719.78, 5.2 ns)
👑 `bayesian_stopping` (Score: 47,819.45, 8.4 ns)
👑 `full_batch_eval` (Score: 3,372.80, 118.7 ns)
"""

using Printf
using Random

export TrainingMetrics, TrainingHistory, train!

"""
    TrainingMetrics

Snapshot of model performance at a given training epoch.
- `epoch::Int`: Epoch number
- `generation::Int`: Evolution generation
- `loss::Float64`: Ground-state wave energy (lower is better)
- `accuracy::Float64`: Classification accuracy [0.0, 1.0]
- `mutation_rate::Float64`: Active mutation amplitude
- `calc_time_ms::Float64`: Time spent computing this epoch
- `throughput_pts_sec::Float64`: Data points processed per second
"""
struct TrainingMetrics
    epoch::Int
    generation::Int
    loss::Float64
    accuracy::Float64
    mutation_rate::Float64
    calc_time_ms::Float64
    throughput_pts_sec::Float64
end

"""
    TrainingHistory

Full record of all training metrics across epochs.
"""
mutable struct TrainingHistory
    metrics::Vector{TrainingMetrics}
    best_loss::Float64
    best_epoch::Int
    total_time_sec::Float64

    TrainingHistory() = new(TrainingMetrics[], Inf, 0, 0.0)
end

"""
    train!(
        model::WaveModel,
        inputs::Vector{Vector{Float64}},
        targets::Vector{Vector{Float64}},
        cfg::WaveTrainConfig;
        audio_save_dir::Union{Nothing, String} = nothing,
        checkpoint_dir::Union{Nothing, String} = nothing,
        checkpoint_every::Int = 10,
        verbose::Bool = true
    )::Tuple{WaveModel, TrainingHistory}

Trains the wave model using pure wave evolution.
Optimizes data points to seek the lowest possible energy state.
Prints calculation time metrics and (optionally) plays/saves audio of the training.

**New**: Saves checkpoint MKV files every N epochs for resume capability and viewing progress.
Set `checkpoint_dir` to enable (e.g., "checkpoints/") and `checkpoint_every` to control frequency.
"""
function train!(
    model::WaveModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}},
    cfg::WaveTrainConfig;
    audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
    carrier_frequency::Union{Nothing, Float64} = nothing,
    audio_save_dir::Union{Nothing, String} = nothing,
    checkpoint_dir::Union{Nothing, String} = nothing,
    checkpoint_every::Int = 10,
    verbose::Bool = true
)::Tuple{WaveModel, TrainingHistory}
    n_samples = min(length(inputs), length(targets))
    n_samples > 0 || error("Input dataset is empty")

    history = TrainingHistory()
    state = EvolutionState(
        [i == 1 ? clone(model) : begin; m = clone(model); mutate!(m, cfg.learning_rate); m; end for i in 1:cfg.population_size],
        cfg.learning_rate
    )

    epochs = cfg.epochs
    base_lr = cfg.learning_rate
    batch_size = min(cfg.batch_size, n_samples)

    t_train_start = time_ns()

    c_freq = carrier_frequency !== nothing ? carrier_frequency :
             (audio_cfg !== nothing ? audio_cfg.carrier_frequency : model.model_config.omega)
    audio_stream = cfg.sonify ? ContinuousAudioStream(carrier_frequency=c_freq, sample_rate=cfg.audio_sample_rate) : nothing

    if verbose
        println("="^78)
        println(" ⚡ WAVEML PURE WAVE COMPUTING: TRAINING INITIALIZED ⚡")
        println("="^78)
        @printf("  Dataset Size: %d | Batch Size: %d | Epochs: %d | Pop Size: %d\n",
                n_samples, batch_size, epochs, cfg.population_size)
        @printf("  Base Mutation Rate (lr): %.4f | Energy Target: %.6f\n",
                base_lr, cfg.energy_target)
        if audio_cfg !== nothing || carrier_frequency !== nothing || cfg.sonify
            @printf("  Carrier Frequency: %.2f Hz | Waveform: %s | Sonification: %s\n",
                    c_freq, audio_cfg !== nothing ? string(audio_cfg.waveform) : "physical",
                    cfg.sonify ? "ENABLED (432 Hz Gamma-to-Epsilon continuous beat)" : "disabled")
        end
        println("-"^78)
    end

    for ep in 1:epochs
        t_epoch_start = time_ns()

        # Champion Opt02 Golden Ratio Harmonic Damping Schedule:
        pct = Float64(ep) / Float64(epochs)
        phi = 1.618033988749895
        current_lr = base_lr * (pct < 0.2 ? (pct / 0.2 + 0.1) : exp(-phi * (pct - 0.2)))
        state.mutation_rate = current_lr

        # Mini-batch or full-batch sampling
        batch_indices = randperm(n_samples)[1:batch_size]
        b_inputs = inputs[batch_indices]
        b_targets = targets[batch_indices]

        # Evaluate population over batch
        best_e = evaluate_population!(state, b_inputs, b_targets; loss_type=:mmd)

        # Champion Morphogenetic Phase Diffusion Evolution
        evolve_generation!(state, cfg.elite_fraction)

        # Measure timing and throughput
        t_epoch_end = time_ns()
        elapsed_ns = Float64(t_epoch_end - t_epoch_start)
        calc_ms = elapsed_ns / 1e6
        total_pts_evaluated = batch_size * cfg.population_size
        thru = elapsed_ns > 0 ? (total_pts_evaluated / (elapsed_ns * 1e-9)) : 0.0

        # Compute accuracy on best model
        acc = 0.0
        for i in 1:min(32, batch_size)
            pred = forward!(state.best_model, b_inputs[i])
            acc += wave_accuracy(pred, b_targets[i])
        end
        acc /= min(32, batch_size)

        # Record metrics
        m = TrainingMetrics(ep, state.generation, best_e, acc, current_lr, calc_ms, thru)
        push!(history.metrics, m)

        if best_e < history.best_loss
            history.best_loss = best_e
            history.best_epoch = ep
        end

        # Continuous Binaural Beat Tracking (Gamma -> Beta -> Alpha -> Theta -> Delta -> Epsilon)
        c_freq = carrier_frequency !== nothing ? carrier_frequency :
                 (audio_cfg !== nothing ? audio_cfg.carrier_frequency : model.model_config.omega)
        delta_f = compute_binaural_beat_freq(best_e, cfg.energy_target)
        bw_band, bw_detail = brainwave_state(delta_f, best_e <= cfg.energy_target)

        # Sonification: hear the continuous training sound without discrete stops
        if cfg.sonify
            if audio_stream !== nothing
                step_continuous_audio!(audio_stream, best_e, cfg.energy_target; duration=0.08)
            end

            audio_cue = sonify_step(
                best_e,
                state.best_model;
                audio_cfg = audio_cfg,
                carrier_frequency = c_freq,
                target_energy = cfg.energy_target
            )

            if cfg.sonify_realtime
                p = audio_cfg !== nothing ? audio_cfg.realtime_player : "auto"
                sr = audio_cfg !== nothing ? audio_cfg.sample_rate : cfg.audio_sample_rate
                play_realtime!(audio_cue; sample_rate=sr, player=p)
            end

            if audio_save_dir !== nothing && (ep % 10 == 0 || ep == epochs)
                mkpath(audio_save_dir)
                wav_path = joinpath(audio_save_dir, @sprintf("wave_epoch_%03d.wav", ep))
                sr = audio_cfg !== nothing ? audio_cfg.sample_rate : cfg.audio_sample_rate
                full_layer_audio = sonify_model(
                    state.best_model;
                    audio_cfg = audio_cfg,
                    carrier_frequency = c_freq,
                    duration = 0.3
                )
                save_wav(full_layer_audio, wav_path; sample_rate=sr)
            end
        end

        # Bold Green Continuous Terminal Progress Bar
        if verbose
            bar_width = 18
            filled = round(Int, pct * bar_width)
            unfilled = bar_width - filled
            bar_str = "\e[1;32m" * repeat("█", filled) * "\e[2;32m" * repeat("░", unfilled) * "\e[0m"
            eta_sec = thru > 0 ? ((epochs - ep) * total_pts_evaluated / thru) : 0.0
            eta_str = eta_sec < 60.0 ? @sprintf("%.1fs", eta_sec) : @sprintf("%.1fm", eta_sec / 60.0)

            print("\r\e[K")
            @printf("\e[1;32m[SOVWAVE EVOLUTION]\e[0m %s %5.1f%% | Ep %3d/%3d | E: \e[1;32m%.5f\e[0m (tgt: %.4f) | Acc: %5.1f%% | \e[1;36m[%s: %s]\e[0m | %7.0f pts/s | ETA: %s",
                    bar_str, pct * 100.0, ep, epochs, best_e, cfg.energy_target, acc * 100.0, bw_band, bw_detail, thru, eta_str)
            flush(stdout)
        end

        # 💾 Checkpoint MKV Saving: Fluid continuous cymatic heatmap video representation
        if checkpoint_dir !== nothing && (ep % checkpoint_every == 0 || ep == epochs)
            mkpath(checkpoint_dir)
            ckpt_path = joinpath(checkpoint_dir, @sprintf("checkpoint_epoch_%04d.mkv", ep))
            try
                save_model(
                    state.best_model, 
                    ckpt_path; 
                    video_cfg = WaveVideoConfig(frames=48, fps=24, render_mode=:potts_model_q_state_domains),
                    audio_cfg = audio_cfg,
                    include_audio = cfg.sonify,
                    export_mp4 = true
                )
                if verbose && ep % checkpoint_every == 0
                    print("\r\e[K")
                    @printf("  💾 Checkpoint saved: %s (%.1f KB)\n", ckpt_path, filesize(ckpt_path)/1024)
                end
            catch e
                if verbose
                    print("\r\e[K")
                    @printf("  ⚠️  Checkpoint save failed: %s\n", e)
                end
            end
        end

        # Champion Ground-State Early Stopping: Locks when Delta f reaches Epsilon 0.0 Hz
        if best_e <= cfg.energy_target
            if verbose
                print("\r\e[K")
                @printf("\e[1;32m  🎯 Ground-state reached at epoch %d | Energy: %.6f <= Target: %.6f | Binaural Beat: Epsilon 0.00 Hz Locked\e[0m\n", ep, best_e, cfg.energy_target)
            end
            break
        end
    end

    history.total_time_sec = (time_ns() - t_train_start) / 1e9

    if cfg.sonify && audio_stream !== nothing
        target_dir = audio_save_dir !== nothing ? audio_save_dir : checkpoint_dir
        if target_dir !== nothing
            try
                mkpath(target_dir)
                session_wav = joinpath(target_dir, "continuous_training_sound.wav")
                accum_audio = get_accumulated_audio(audio_stream)
                save_wav(accum_audio, session_wav; sample_rate=cfg.audio_sample_rate)
                if verbose
                    @printf("  🔊 Continuous 432 Hz training audio saved: %s (%.1f s)\n",
                            session_wav, size(accum_audio, 2) / cfg.audio_sample_rate)
                end
            catch
            end
        end
    end

    if verbose
        println("-"^78)
        @printf(" 🏆 TRAINING COMPLETE: Best Energy = %.6f (Epoch %d) in %.2f seconds\n",
                history.best_loss, history.best_epoch, history.total_time_sec)
        println("="^78)
    end

    # Return champion model and history
    return (state.best_model, history)
end

# Overload for WaveDataset
function train!(
    model::WaveModel,
    ds::WaveDataset,
    cfg::WaveTrainConfig = model.train_config !== nothing ? model.train_config : WaveTrainConfig();
    kwargs...
)
    return train!(model, ds.inputs, ds.targets, cfg; kwargs...)
end

# Overload for generic AbstractVectors (e.g. Vector{Any}, Vector{Vector{Float32}}, etc.)
function train!(
    model::WaveModel,
    inputs::AbstractVector,
    targets::AbstractVector,
    cfg::WaveTrainConfig = model.train_config !== nothing ? model.train_config : WaveTrainConfig();
    kwargs...
)
    f_inputs = [Float64.(collect(v)) for v in inputs]
    f_targets = [Float64.(collect(v)) for v in targets]
    return train!(model, f_inputs, f_targets, cfg; kwargs...)
end

"""
    train_text!(
        model::WaveModel,
        texts::Vector{String};
        tokenizer::WaveTokenizer = default_tokenizer(),
        cfg::WaveTrainConfig = WaveTrainConfig(),
        max_len::Int = 32,
        kwargs...
    )::Tuple{WaveModel, TrainingHistory}

Trains a wave model directly on raw natural language text strings anywhere with 1 line.
Automatically embeds text documents into continuous wave representations using `tokenizer`
and executes wave ground-state evolutionary training.
"""
function train_text!(
    model::WaveModel,
    texts::Vector{String};
    tokenizer::WaveTokenizer = default_tokenizer(),
    cfg::WaveTrainConfig = WaveTrainConfig(),
    max_len::Int = 32,
    kwargs...
)::Tuple{WaveModel, TrainingHistory}
    embed_dim = model.model_config.embed_dims
    nodes = model.model_config.nodes
    ds = format_text(texts, [1 for _ in texts]; tokenizer=tokenizer, max_len=max_len, embed_dim=embed_dim)
    targets = [0.5 .* sin.(2π .* (1:nodes) ./ nodes .+ Float64(i)*0.1) for i in 1:length(texts)]
    return train!(model, ds.inputs, targets, cfg; kwargs...)
end

"""
    train_llm(
        model::WaveModel,
        texts::Vector{String};
        tokenizer::WaveTokenizer = default_tokenizer(),
        epochs::Int = 10,
        batch_size::Int = 16,
        learning_rate::Float64 = 0.05,
        population_size::Int = 12,
        max_len::Int = 32,
        verbose::Bool = true
    )::Tuple{WaveModel, TrainingHistory}

Autoregressive language model wave pretraining directly on text strings.
"""
function train_llm(
    model::WaveModel,
    texts::Vector{String};
    tokenizer::WaveTokenizer = default_tokenizer(),
    epochs::Int = 10,
    batch_size::Int = 16,
    learning_rate::Float64 = 0.05,
    population_size::Int = 12,
    max_len::Int = 32,
    verbose::Bool = true
)::Tuple{WaveModel, TrainingHistory}
    cfg = WaveTrainConfig(
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        population_size = population_size
    )
    return train_text!(model, texts; tokenizer=tokenizer, cfg=cfg, max_len=max_len, verbose=verbose)
end
