"""
    WaveML.Config

YAML-based configuration for WaveML. Defines schema for:
- `WaveFieldConfig`: Data points n, properties x, dimensions d
- `WaveModelConfig`: Layers l, embeddings d, nodes n, omega ω (Hz), fractal β_s, t_frames
- `WaveTrainConfig`: Batch size b, learning rate lr, epochs, population, sonify settings
- `WaveMLConfig`: Root configuration unifying field, model, and training specs
"""

using YAML

export WaveFieldConfig, WaveModelConfig, WaveTrainConfig, WaveAudioConfig, WaveVideoConfig, WaveMLConfig
export load_config, save_config, default_config, parse_color_rgb

"""
    WaveFieldConfig

Configuration for the physical wave field where data points live.
- `n_points::Int`: Number of data points n in the wave field (0 to n)
- `properties::Vector{Symbol}`: x property names shared by all points in the field
- `dimensions::Int`: d dimensions of the wave space (1D curve, 2D plane, 3D surface, 4D, 5D+)
- `distribution::Symbol`: Spatial point distribution (:uniform, :fibonacci, :random, :lattice)
"""
struct WaveFieldConfig
    n_points::Int
    properties::Vector{Symbol}
    dimensions::Int
    distribution::Symbol

    function WaveFieldConfig(;
        n_points::Int = 128,
        properties::Vector{Symbol} = [:mass, :charge, :energy, :spin],
        dimensions::Int = 3,
        distribution::Symbol = :uniform
    )
        n_points > 0 || error("n_points must be > 0, got $n_points")
        dimensions > 0 || error("dimensions must be > 0, got $dimensions")
        !isempty(properties) || error("properties list must not be empty")
        new(n_points, properties, dimensions, distribution)
    end
end

"""
    WaveModelConfig

Configuration for the multi-layer wave model architecture.
- `layers::Int`: l layers of the wave model
- `embed_dims::Int`: d embedding dimensions across l layers
- `nodes::Int`: n quantum lattice nodes per layer
- `omega::Float64`: Harmonic frequency ω in Hz driving temporal superposition across t frames
- `beta_s::Float64`: Fractal scaling parameter β_s
- `t_frames::Int`: Number of temporal superposition time frames
"""
struct WaveModelConfig
    layers::Int
    embed_dims::Int
    nodes::Int
    omega::Float64
    beta_s::Float64
    t_frames::Int

    function WaveModelConfig(;
        layers::Int = 4,
        embed_dims::Int = 32,
        nodes::Int = 64,
        omega::Float64 = 432.0,
        beta_s::Float64 = 1.618033988749895,
        t_frames::Int = 8
    )
        layers > 0 || error("layers must be > 0, got $layers")
        embed_dims > 0 || error("embed_dims must be > 0, got $embed_dims")
        nodes > 0 || error("nodes must be > 0, got $nodes")
        omega > 0.0 || error("omega must be > 0.0 Hz, got $omega")
        beta_s > 0.0 || error("beta_s must be > 0.0, got $beta_s")
        t_frames > 0 || error("t_frames must be > 0, got $t_frames")
        new(layers, embed_dims, nodes, omega, beta_s, t_frames)
    end
end

"""
    WaveTrainConfig

Configuration for wave evolution training.
- `batch_size::Int`: Batch size b
- `learning_rate::Float64`: lr (mutation amplitude scale)
- `epochs::Int`: Number of training epochs
- `population_size::Int`: Number of wave models in evolution population
- `elite_fraction::Float64`: Proportion of elite models preserved per generation
- `mutation_decay::Float64`: Annealing factor for mutation amplitude
- `energy_target::Float64`: Target ground-state energy for early stopping
- `sonify::Bool`: Option to hear training as sound (saved as WAV)
- `sonify_realtime::Bool`: Option to also play training audio through system speakers in real-time
- `audio_sample_rate::Int`: Sample rate for sonification (Hz)
"""
struct WaveTrainConfig
    batch_size::Int
    learning_rate::Float64
    epochs::Int
    population_size::Int
    elite_fraction::Float64
    mutation_decay::Float64
    energy_target::Float64
    sonify::Bool
    sonify_realtime::Bool
    audio_sample_rate::Int

    function WaveTrainConfig(;
        batch_size::Int = 16,
        learning_rate::Float64 = 0.05,
        epochs::Int = 50,
        population_size::Int = 24,
        elite_fraction::Float64 = 0.15,
        mutation_decay::Float64 = 0.995,
        energy_target::Float64 = 0.001,
        sonify::Bool = true,
        sonify_realtime::Bool = false,
        audio_sample_rate::Int = 48000
    )
        batch_size > 0 || error("batch_size must be > 0, got $batch_size")
        learning_rate > 0.0 || error("learning_rate must be > 0, got $learning_rate")
        epochs > 0 || error("epochs must be > 0, got $epochs")
        population_size >= 2 || error("population_size must be >= 2, got $population_size")
        new(batch_size, learning_rate, epochs, population_size, elite_fraction,
            mutation_decay, energy_target, sonify, sonify_realtime, audio_sample_rate)
    end
end

"""
    WaveAudioConfig

Comprehensive user-defined audio and sonification configuration.
All sound properties are fully controllable by the user (no forced defaults):
- `carrier_frequency::Float64`: User-defined carrier frequency in Hz (e.g. 432.0, 440.0, 528.0, 108.0, or any custom value)
- `tuning_standard::Float64`: Base tuning reference (e.g. 432.0, 440.0, 528.0)
- `waveform::Symbol`: Sound waveform (:sine, :harmonic, :physical, :triangle, :sawtooth, :binaural)
- `binaural_beat::Float64`: Binaural beat frequency offset in Hz (e.g. 10.0 for Alpha, 6.0 for Theta, 0.0 for mono)
- `envelope::Symbol`: Amplitude envelope (:exponential_decay, :adsr, :percussive, :sustain)
- `attack::Float64`: ADSR Attack time (seconds)
- `decay::Float64`: ADSR Decay time (seconds)
- `sustain::Float64`: ADSR Sustain amplitude level [0.0, 1.0]
- `release::Float64`: ADSR Release time (seconds)
- `harmonic_richness::Float64`: Overtone intensity factor [0.0, 2.0]
- `volume::Float64`: Master volume gain [0.0, 1.0]
- `pan::Float64`: Stereo pan position [-1.0 (left) to +1.0 (right)]
- `sample_rate::Int`: Audio sample rate in Hz (e.g. 44100, 48000, 96000)
- `channels::Int`: Audio channels (1 = mono, 2 = stereo)
- `realtime_player::String`: Real-time playback player ("auto", "paplay", "aplay", "pw-play", "ffplay", or custom command)
"""
struct WaveAudioConfig
    carrier_frequency::Float64
    tuning_standard::Float64
    waveform::Symbol
    binaural_beat::Float64
    envelope::Symbol
    attack::Float64
    decay::Float64
    sustain::Float64
    release::Float64
    harmonic_richness::Float64
    volume::Float64
    pan::Float64
    sample_rate::Int
    channels::Int
    realtime_player::String

    function WaveAudioConfig(;
        carrier_frequency::Float64 = 432.0,
        tuning_standard::Float64 = 432.0,
        waveform::Symbol = :physical,
        binaural_beat::Float64 = 0.0,
        envelope::Symbol = :exponential_decay,
        attack::Float64 = 0.01,
        decay::Float64 = 0.1,
        sustain::Float64 = 0.7,
        release::Float64 = 0.2,
        harmonic_richness::Float64 = 1.0,
        volume::Float64 = 0.8,
        pan::Float64 = 0.0,
        sample_rate::Int = 48000,
        channels::Int = 1,
        realtime_player::String = "auto"
    )
        carrier_frequency > 0.0 || error("carrier_frequency must be > 0, got $carrier_frequency")
        sample_rate > 0 || error("sample_rate must be > 0, got $sample_rate")
        channels in (1, 2) || error("channels must be 1 or 2, got $channels")
        new(carrier_frequency, tuning_standard, waveform, binaural_beat, envelope,
            attack, decay, sustain, release, harmonic_richness, volume, pan,
            sample_rate, channels, realtime_player)
    end
end

"""
    WaveVideoConfig

Configuration for WaveML video model serialization and rendering.
- `render_mode::Symbol`: Algorithm used for visual rendering (`:potts_champion` [Grand Champion], `:fibonacci_resonance` [Honorable Mention])
- `pixel_scale::Int`: Screen pixel square size per quantum lattice node (1, 2, 4, 8, 16, or 0 for auto-scale)
- `target_height::Int`: Target screen display height in pixels for auto-scaling (e.g. 480, 720, 1080)
- `fps::Int`: Playback frame rate
- `frames::Int`: Number of post-convergence temporal evolution frames (step n to n_final)
"""
struct WaveVideoConfig
    render_mode::Symbol
    pixel_scale::Int
    target_height::Int
    fps::Int
    frames::Int
    state_colors::Vector{Vector{Float64}}

    function WaveVideoConfig(;
        render_mode::Symbol = :potts_model_q_state_domains,
        pixel_scale::Int = 2,
        target_height::Int = 480,
        fps::Int = 4,
        frames::Int = 12,
        state_colors::Vector{Vector{Float64}} = [
            [1.0, 0.0, 1.0], # State 0: FF00FF (Magenta)
            [1.0, 1.0, 0.0], # State 1: FFFF00 (Yellow)
            [0.0, 1.0, 1.0]  # State 2: 00FFFF (Cyan)
        ]
    )
        actual_mode = render_mode == :potts_champion ? :potts_model_q_state_domains : render_mode
        new(actual_mode, pixel_scale, target_height, fps, frames, state_colors)
    end
end

"""
    WaveMLConfig

Unified configuration root combining field, model, training, audio, and video specifications.
"""
struct WaveMLConfig
    field::WaveFieldConfig
    model::WaveModelConfig
    train::WaveTrainConfig
    audio::WaveAudioConfig
    video::WaveVideoConfig

    function WaveMLConfig(;
        field::WaveFieldConfig = WaveFieldConfig(),
        model::WaveModelConfig = WaveModelConfig(),
        train::WaveTrainConfig = WaveTrainConfig(),
        audio::WaveAudioConfig = WaveAudioConfig(),
        video::WaveVideoConfig = WaveVideoConfig()
    )
        new(field, model, train, audio, video)
    end
end

"""
    default_config()::WaveMLConfig

Returns the canonical default configuration for WaveML.
"""
function default_config()::WaveMLConfig
    return WaveMLConfig()
end

"""
    parse_color_rgb(c)::Vector{Float64}

Parses a color value from hex string (e.g. "FF00FF", "#00FFFF") or RGB numeric vector into [r, g, b] in [0.0, 1.0].
"""
function parse_color_rgb(c)::Vector{Float64}
    if c isa AbstractString
        s = strip(c)
        if startswith(s, "#")
            s = s[2:end]
        end
        if length(s) == 6
            r = parse(Int, s[1:2], base=16) / 255.0
            g = parse(Int, s[3:4], base=16) / 255.0
            b = parse(Int, s[5:6], base=16) / 255.0
            return [clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)]
        end
    elseif c isa AbstractVector && length(c) >= 3
        r = Float64(c[1])
        g = Float64(c[2])
        b = Float64(c[3])
        if r > 1.0 || g > 1.0 || b > 1.0
            r /= 255.0
            g /= 255.0
            b /= 255.0
        end
        return [clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)]
    end
    return [1.0, 1.0, 1.0]
end

"""
    load_config(path::String)::WaveMLConfig

Loads and parses a YAML configuration file into a `WaveMLConfig` object.
"""
function load_config(path::String)::WaveMLConfig
    isfile(path) || error("Configuration file not found: $path")
    raw = YAML.load_file(path)

    # Parse Field config
    f_dict = get(raw, "field", Dict{String, Any}())
    props_raw = get(f_dict, "properties", ["mass", "charge", "energy", "spin"])
    props = Symbol[Symbol(p) for p in props_raw]
    field = WaveFieldConfig(
        n_points = Int(get(f_dict, "n_points", 128)),
        properties = props,
        dimensions = Int(get(f_dict, "dimensions", 3)),
        distribution = Symbol(get(f_dict, "distribution", "uniform"))
    )

    # Parse Model config
    m_dict = get(raw, "model", Dict{String, Any}())
    model = WaveModelConfig(
        layers = Int(get(m_dict, "layers", 4)),
        embed_dims = Int(get(m_dict, "embed_dims", 32)),
        nodes = Int(get(m_dict, "nodes", 64)),
        omega = Float64(get(m_dict, "omega", 432.0)),
        beta_s = Float64(get(m_dict, "beta_s", 1.618033988749895)),
        t_frames = Int(get(m_dict, "t_frames", 8))
    )

    # Parse Train config
    t_dict = get(raw, "train", Dict{String, Any}())
    train = WaveTrainConfig(
        batch_size = Int(get(t_dict, "batch_size", 16)),
        learning_rate = Float64(get(t_dict, "learning_rate", 0.05)),
        epochs = Int(get(t_dict, "epochs", 50)),
        population_size = Int(get(t_dict, "population_size", 24)),
        elite_fraction = Float64(get(t_dict, "elite_fraction", 0.15)),
        mutation_decay = Float64(get(t_dict, "mutation_decay", 0.995)),
        energy_target = Float64(get(t_dict, "energy_target", 0.001)),
        sonify = Bool(get(t_dict, "sonify", true)),
        sonify_realtime = Bool(get(t_dict, "sonify_realtime", false)),
        audio_sample_rate = Int(get(t_dict, "audio_sample_rate", 48000))
    )

    # Parse Audio config (User-defined carrier frequency, sound properties, waveforms)
    a_dict = get(raw, "audio", Dict{String, Any}())
    carrier_f = Float64(get(a_dict, "carrier_frequency", model.omega))
    tuning = Float64(get(a_dict, "tuning_standard", carrier_f))
    audio = WaveAudioConfig(
        carrier_frequency = carrier_f,
        tuning_standard = tuning,
        waveform = Symbol(get(a_dict, "waveform", "physical")),
        binaural_beat = Float64(get(a_dict, "binaural_beat", 0.0)),
        envelope = Symbol(get(a_dict, "envelope", "exponential_decay")),
        attack = Float64(get(a_dict, "attack", 0.01)),
        decay = Float64(get(a_dict, "decay", 0.1)),
        sustain = Float64(get(a_dict, "sustain", 0.7)),
        release = Float64(get(a_dict, "release", 0.2)),
        harmonic_richness = Float64(get(a_dict, "harmonic_richness", 1.0)),
        volume = Float64(get(a_dict, "volume", 0.8)),
        pan = Float64(get(a_dict, "pan", 0.0)),
        sample_rate = Int(get(a_dict, "sample_rate", train.audio_sample_rate)),
        channels = Int(get(a_dict, "channels", 1)),
        realtime_player = String(get(a_dict, "realtime_player", "auto"))
    )

    # Parse Video config (Grand Champion potts_model_q_state_domains, pixel square size setting, target screen height)
    v_dict = get(raw, "video", Dict{String, Any}())
    px_raw = get(v_dict, "pixel_scale", 2)
    px_scale = if px_raw isa AbstractString && lowercase(px_raw) == "auto"
        0
    elseif px_raw isa Number
        Int(px_raw)
    else
        2
    end

    raw_mode = Symbol(get(v_dict, "render_mode", "potts_model_q_state_domains"))
    actual_mode = (raw_mode == :potts_champion) ? :potts_model_q_state_domains : raw_mode

    default_colors = [
        [1.0, 0.0, 1.0], # State 0: FF00FF (Magenta)
        [1.0, 1.0, 0.0], # State 1: FFFF00 (Yellow)
        [0.0, 1.0, 1.0]  # State 2: 00FFFF (Cyan)
    ]
    raw_colors = get(v_dict, "state_colors", nothing)
    colors = if raw_colors isa AbstractVector && length(raw_colors) >= 3
        [parse_color_rgb(raw_colors[1]), parse_color_rgb(raw_colors[2]), parse_color_rgb(raw_colors[3])]
    else
        default_colors
    end

    video = WaveVideoConfig(
        render_mode = actual_mode,
        pixel_scale = px_scale,
        target_height = Int(get(v_dict, "target_height", 480)),
        fps = Int(get(v_dict, "fps", 4)),
        frames = Int(get(v_dict, "frames", 12)),
        state_colors = colors
    )

    return WaveMLConfig(field=field, model=model, train=train, audio=audio, video=video)
end

"""
    save_config(cfg::WaveMLConfig, path::String)::Nothing

Serializes a `WaveMLConfig` struct into a YAML file at `path`.
"""
function save_config(cfg::WaveMLConfig, path::String)::Nothing
    dict = Dict{String, Any}(
        "field" => Dict{String, Any}(
            "n_points" => cfg.field.n_points,
            "properties" => [String(p) for p in cfg.field.properties],
            "dimensions" => cfg.field.dimensions,
            "distribution" => String(cfg.field.distribution)
        ),
        "model" => Dict{String, Any}(
            "layers" => cfg.model.layers,
            "embed_dims" => cfg.model.embed_dims,
            "nodes" => cfg.model.nodes,
            "omega" => cfg.model.omega,
            "beta_s" => cfg.model.beta_s,
            "t_frames" => cfg.model.t_frames
        ),
        "train" => Dict{String, Any}(
            "batch_size" => cfg.train.batch_size,
            "learning_rate" => cfg.train.learning_rate,
            "epochs" => cfg.train.epochs,
            "population_size" => cfg.train.population_size,
            "elite_fraction" => cfg.train.elite_fraction,
            "mutation_decay" => cfg.train.mutation_decay,
            "energy_target" => cfg.train.energy_target,
            "sonify" => cfg.train.sonify,
            "sonify_realtime" => cfg.train.sonify_realtime,
            "audio_sample_rate" => cfg.train.audio_sample_rate
        ),
        "audio" => Dict{String, Any}(
            "carrier_frequency" => cfg.audio.carrier_frequency,
            "tuning_standard" => cfg.audio.tuning_standard,
            "waveform" => String(cfg.audio.waveform),
            "binaural_beat" => cfg.audio.binaural_beat,
            "envelope" => String(cfg.audio.envelope),
            "attack" => cfg.audio.attack,
            "decay" => cfg.audio.decay,
            "sustain" => cfg.audio.sustain,
            "release" => cfg.audio.release,
            "harmonic_richness" => cfg.audio.harmonic_richness,
            "volume" => cfg.audio.volume,
            "pan" => cfg.audio.pan,
            "sample_rate" => cfg.audio.sample_rate,
            "channels" => cfg.audio.channels,
            "realtime_player" => cfg.audio.realtime_player
        ),
        "video" => Dict{String, Any}(
            "render_mode" => String(cfg.video.render_mode),
            "pixel_scale" => cfg.video.pixel_scale == 0 ? "auto" : cfg.video.pixel_scale,
            "target_height" => cfg.video.target_height,
            "fps" => cfg.video.fps,
            "frames" => cfg.video.frames,
            "state_colors" => cfg.video.state_colors
        )
    )
    YAML.write_file(path, dict)
    return nothing
end
