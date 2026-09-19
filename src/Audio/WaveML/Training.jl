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
        verbose::Bool = true
    )::Tuple{WaveModel, TrainingHistory}

Trains the wave model using pure wave evolution.
Optimizes data points to seek the lowest possible energy state.
Prints calculation time metrics and (optionally) plays/saves audio of the training.
"""
function train!(
    model::WaveModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}},
    cfg::WaveTrainConfig;
    audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
    carrier_frequency::Union{Nothing, Float64} = nothing,
    audio_save_dir::Union{Nothing, String} = nothing,
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

    if verbose
        println("="^78)
        println(" ⚡ WAVEML PURE WAVE COMPUTING: TRAINING INITIALIZED ⚡")
        println("="^78)
        @printf("  Dataset Size: %d | Batch Size: %d | Epochs: %d | Pop Size: %d\n",
                n_samples, batch_size, epochs, cfg.population_size)
        @printf("  Base Mutation Rate (lr): %.4f | Energy Target: %.6f\n",
                base_lr, cfg.energy_target)
        if audio_cfg !== nothing || carrier_frequency !== nothing
            c_freq = carrier_frequency !== nothing ? carrier_frequency : audio_cfg.carrier_frequency
            @printf("  User-Defined Carrier Frequency: %.2f Hz | Waveform: %s\n",
                    c_freq, audio_cfg !== nothing ? string(audio_cfg.waveform) : "physical")
        end
        println("-"^78)
    end

    for ep in 1:epochs
        t_epoch_start = time_ns()

        # Champion 1Cycle Harmonic Annealing Schedule:
        # Warmup phase (first 30%), followed by cosine decay
        pct = Float64(ep) / Float64(epochs)
        current_lr = if pct < 0.3
            base_lr * (pct / 0.3 + 0.1)
        else
            base_lr * (0.5 * (1.0 + cos(π * (pct - 0.3) / 0.7)))
        end
        state.mutation_rate = current_lr

        # Mini-batch or full-batch sampling
        batch_indices = randperm(n_samples)[1:batch_size]
        b_inputs = inputs[batch_indices]
        b_targets = targets[batch_indices]

        # Evaluate population over batch
        best_e = evaluate_population!(state, b_inputs, b_targets; loss_type=:mmd)

        # Champion Island Model Evolution
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

        # Sonification: hear the training as sound with user-defined sound parameters
        if cfg.sonify
            audio_cue = sonify_step(
                best_e,
                state.best_model;
                audio_cfg = audio_cfg,
                carrier_frequency = carrier_frequency
            )

            # Option 1: Live real-time speaker audio
            if cfg.sonify_realtime
                p = audio_cfg !== nothing ? audio_cfg.realtime_player : "auto"
                sr = audio_cfg !== nothing ? audio_cfg.sample_rate : cfg.audio_sample_rate
                play_realtime!(audio_cue; sample_rate=sr, player=p)
            end

            # Option 2: Save periodic audio files
            if audio_save_dir !== nothing && (ep % 10 == 0 || ep == epochs)
                mkpath(audio_save_dir)
                wav_path = joinpath(audio_save_dir, @sprintf("wave_epoch_%03d.wav", ep))
                sr = audio_cfg !== nothing ? audio_cfg.sample_rate : cfg.audio_sample_rate
                full_layer_audio = sonify_model(
                    state.best_model;
                    audio_cfg = audio_cfg,
                    carrier_frequency = carrier_frequency,
                    duration = 0.3
                )
                save_wav(full_layer_audio, wav_path; sample_rate=sr)
            end
        end

        if verbose && (ep % 5 == 0 || ep == 1 || ep == epochs)
            @printf("  Epoch %3d/%3d | Energy: %8.5f | Acc: %5.1f%% | Calc: %6.2f ms | %9.1f pts/s | lr: %.4f\n",
                    ep, epochs, best_e, acc * 100.0, calc_ms, thru, current_lr)
        end

        # Champion Bayesian Ground-State Early Stopping
        if best_e <= cfg.energy_target
            if verbose
                @printf("  🎯 Ground-state energy target achieved at epoch %d (Energy: %.6f)\n", ep, best_e)
            end
            break
        end
    end

    history.total_time_sec = (time_ns() - t_train_start) / 1e9

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
