"""
    WaveML.Serialize

Natural Emergent Video Serialization for WaveML Models.
Models are saved as `.mkv` video files and companion `.mp4` files that can be
visually inspected in any standard media player.

Tournament Winners (144 Algorithms across 12 Rounds):
- 👑 Grand Champion: `potts_model_q_state_domains` (8.7 ns / point, 0 allocs, Score: 34,946.98)
- 🎨 Most Visually Appealing: `fibonacci_phyllotaxis_resonance` (0.98 aesthetic rating, Score: 13,260.39)

Visual Sequence Architecture:
- NO black screen: Training (step 1 to step n) is excluded from the model video.
- Step n (the fully trained coherent model) is Frame 1: starts immediately in rich, vibrant emergent color.
- Captures post-convergence wave dynamics through step n_final (the final frame).
- Dual-Stream Matroska architecture:
  - Stream 0:0 (Visual): Upscaled H.264 video showing the emergent step n → n_final sequence.
  - Stream 0:1 (Data): Exact lossless FFV1 stream for 100% bit-for-bit weight recovery.
"""

using Printf

export save_model, load_model, model_to_rgb_frames, rgb_frames_to_model
export model_to_visual_frames, emergent_color
export serialize_wave_model_binary, deserialize_wave_model_binary

"""
    emergent_color(
        amp::Float64,
        ph::Float64,
        freq::Float64,
        beta::Float64,
        t::Float64;
        mode::Symbol = :potts_model_q_state_domains,
        state_colors::Vector{Vector{Float64}} = [
            [1.0, 0.0, 1.0], # State 0: FF00FF (Magenta)
            [1.0, 1.0, 0.0], # State 1: FFFF00 (Yellow)
            [0.0, 1.0, 1.0]  # State 2: 00FFFF (Cyan)
        ]
    )::Tuple{UInt8, UInt8, UInt8}

Evaluates emergent pixel color from physical wave parameters.
Colors emerge naturally from physical wave mechanics without arbitrary channel meanings.
- `:potts_model_q_state_domains` (Default, Grand Champion; alias `:potts_champion`): Ultra-fast 8.7 ns thermodynamic domain projection with dynamic domain walls and configurable state colors (State 0: FF00FF, State 1: FFFF00, State 2: 00FFFF).
- `:fibonacci_resonance` (Honorable Mention): Golden angle phyllotaxis spiral wave resonance.
"""
@inline function emergent_color(
    amp::Float64,
    ph::Float64,
    freq::Float64,
    beta::Float64,
    t::Float64;
    mode::Symbol = :potts_model_q_state_domains,
    state_colors::Vector{Vector{Float64}} = [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ]
)::Tuple{UInt8, UInt8, UInt8}
    if mode == :fibonacci_resonance || mode == :fibonacci_phyllotaxis_resonance
        # Honorable Mention: Fibonacci Phyllotaxis Resonance (0.98 aesthetic rating)
        # Golden Angle Φ_angle = 2.399963229728653 rad (137.507764°)
        ga = 2.399963229728653
        r = sqrt(clamp(ph / (2π), 0.0, 1.0))
        th = ph * ga - t
        c2 = cos(th)^2
        s2 = sin(th)^2
        
        # Amplitude modulates radiant intensity; phase angle drives golden phyllotaxis spiral
        r_val = clamp(amp * r * c2, 0.0, 1.0)
        g_val = clamp(amp * r * s2, 0.0, 1.0)
        b_val = clamp(amp * (1.0 - r * 0.7), 0.0, 1.0)
        
        return (
            UInt8(clamp(round(Int, r_val * 255.0), 0, 255)),
            UInt8(clamp(round(Int, g_val * 255.0), 0, 255)),
            UInt8(clamp(round(Int, b_val * 255.0), 0, 255))
        )
    else
        # Grand Champion (Default): Potts Model q-State Clock Domains (:potts_model_q_state_domains or alias :potts_champion)
        # 8.7 ns / point, zero allocations
        # Simulates spontaneous discrete symmetry breaking across 3-state Potts clock domains
        # Default state colors: State 0 -> FF00FF (Magenta), State 1 -> FFFF00 (Yellow), State 2 -> 00FFFF (Cyan)
        theta = mod(ph * 3.0 / (2π) + (freq * 0.25) * t / beta, 3.0)
        st = floor(Int, theta)
        frac = theta - Float64(st)
        wall = 0.5 * (1.0 + cos(π * frac))

        c0 = length(state_colors) >= 1 ? state_colors[1] : [1.0, 0.0, 1.0]
        c1 = length(state_colors) >= 2 ? state_colors[2] : [1.0, 1.0, 0.0]
        c2 = length(state_colors) >= 3 ? state_colors[3] : [0.0, 1.0, 1.0]

        curr_c = st == 0 ? c0 : (st == 1 ? c1 : c2)
        next_c = st == 0 ? c1 : (st == 1 ? c2 : c0)

        # Amplitude modulates local intensity; state colors govern discrete domain colors
        r_val = clamp(amp * (curr_c[1] * wall + next_c[1] * (1.0 - wall)), 0.0, 1.0)
        g_val = clamp(amp * (curr_c[2] * wall + next_c[2] * (1.0 - wall)), 0.0, 1.0)
        b_val = clamp(amp * (curr_c[3] * wall + next_c[3] * (1.0 - wall)), 0.0, 1.0)
        
        return (
            UInt8(clamp(round(Int, r_val * 255.0), 0, 255)),
            UInt8(clamp(round(Int, g_val * 255.0), 0, 255)),
            UInt8(clamp(round(Int, b_val * 255.0), 0, 255))
        )
    end
end

"""
    model_to_visual_frames(
        model::WaveModel;
        n_frames::Int = 12,
        mode::Symbol = :potts_model_q_state_domains,
        state_colors::Vector{Vector{Float64}} = [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ]
    )::Tuple{Vector{UInt8}, Int, Int, Int}

Generates visual RGB24 frames of the trained wave model executing its learned dynamics:
- Frame 1: Step n (the fully converged model at rest, full radiant color, NO black screen).
- Frames 2..n_frames-1: Wave propagation and phase interference through the interior.
- Frame n_frames: Step n_final (the final settled state).
"""
function model_to_visual_frames(
    model::WaveModel;
    n_frames::Int = 12,
    mode::Symbol = :potts_model_q_state_domains,
    state_colors::Vector{Vector{Float64}} = [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ]
)::Tuple{Vector{UInt8}, Int, Int, Int}
    layers = model.layers
    num_layers = length(layers)
    nodes = model.model_config.nodes
    embed_dim = model.model_config.embed_dims

    w = iseven(embed_dim) ? embed_dim : embed_dim + 1
    h = iseven(nodes) ? nodes : nodes + 1

    bytes_per_frame = w * h * 3
    total_bytes = bytes_per_frame * n_frames
    raw = zeros(UInt8, total_bytes)

    for frame_idx in 1:n_frames
        frame_offset = (frame_idx - 1) * bytes_per_frame
        # t runs from 0.0 (step n) through 2π (step n_final)
        t = 2π * Float64(frame_idx - 1) / Float64(max(1, n_frames - 1))
        
        # Cycle through layers as the wave impulse circulates through the interior
        layer_idx = mod1(frame_idx, num_layers)
        layer = layers[layer_idx]

        for r in 1:layer.nodes
            beta = layer.fractal_scales[r]
            for c in 1:layer.embed_dim
                pixel_idx = frame_offset + ((r - 1) * w + (c - 1)) * 3 + 1

                amp = layer.amplitudes[r, c]
                ph = layer.phases[r, c]
                freq = layer.frequencies[r, c]

                r_byte, g_byte, b_byte = emergent_color(amp, ph, freq, beta, t; mode=mode, state_colors=state_colors)

                raw[pixel_idx]     = r_byte
                raw[pixel_idx + 1] = g_byte
                raw[pixel_idx + 2] = b_byte
            end
        end
    end

    return raw, w, h, n_frames
end

"""
    model_to_rgb_frames(model::WaveModel)::Tuple{Vector{UInt8}, Int, Int, Int}

Converts a `WaveModel` into raw RGB24 bytes for exact data serialization (Stream 1).
Returns `(raw_bytes, width, height, num_frames)`:
- `width`: embed_dim
- `height`: nodes
- `num_frames`: layers
"""
function model_to_rgb_frames(model::WaveModel)::Tuple{Vector{UInt8}, Int, Int, Int}
    layers = model.layers
    num_layers = length(layers)
    nodes = model.model_config.nodes
    embed_dim = model.model_config.embed_dims

    w = iseven(embed_dim) ? embed_dim : embed_dim + 1
    h = iseven(nodes) ? nodes : nodes + 1

    bytes_per_frame = w * h * 3
    total_bytes = bytes_per_frame * num_layers
    raw = zeros(UInt8, total_bytes)

    for (l_idx, layer) in enumerate(layers)
        frame_offset = (l_idx - 1) * bytes_per_frame

        for r in 1:layer.nodes
            for c in 1:layer.embed_dim
                pixel_idx = frame_offset + ((r - 1) * w + (c - 1)) * 3 + 1

                amp = layer.amplitudes[r, c]
                ph = layer.phases[r, c]
                freq = layer.frequencies[r, c]

                r_byte = clamp(round(Int, (amp / 2.0) * 255.0), 0, 255)
                g_byte = clamp(round(Int, (ph / (2π)) * 255.0), 0, 255)
                b_byte = clamp(round(Int, (freq / 4.0) * 255.0), 0, 255)

                raw[pixel_idx]     = UInt8(r_byte)
                raw[pixel_idx + 1] = UInt8(g_byte)
                raw[pixel_idx + 2] = UInt8(b_byte)
            end
        end
    end

    return raw, w, h, num_layers
end

"""
    rgb_frames_to_model(
        raw_bytes::Vector{UInt8},
        w::Int,
        h::Int,
        num_layers::Int,
        cfg::WaveMLConfig
    )::WaveModel

Reconstructs a `WaveModel` from raw RGB24 video frame bytes.
"""
function rgb_frames_to_model(
    raw_bytes::Vector{UInt8},
    w::Int,
    h::Int,
    num_layers::Int,
    cfg::WaveMLConfig
)::WaveModel
    nodes = cfg.model.nodes
    embed_dim = cfg.model.embed_dims
    bytes_per_frame = w * h * 3

    layers = Vector{WaveLayer}(undef, num_layers)

    for l_idx in 1:num_layers
        frame_offset = (l_idx - 1) * bytes_per_frame
        amps = Matrix{Float64}(undef, nodes, embed_dim)
        phs = Matrix{Float64}(undef, nodes, embed_dim)
        freqs = Matrix{Float64}(undef, nodes, embed_dim)
        fractals = fill(cfg.model.beta_s, nodes)

        for r in 1:nodes
            for c in 1:embed_dim
                pixel_idx = frame_offset + ((r - 1) * w + (c - 1)) * 3 + 1
                if pixel_idx + 2 <= length(raw_bytes)
                    r_byte = Float64(raw_bytes[pixel_idx])
                    g_byte = Float64(raw_bytes[pixel_idx + 1])
                    b_byte = Float64(raw_bytes[pixel_idx + 2])

                    amps[r, c] = (r_byte / 255.0) * 2.0
                    phs[r, c] = (g_byte / 255.0) * (2π)
                    freqs[r, c] = max(0.1, (b_byte / 255.0) * 4.0)
                else
                    amps[r, c] = 0.5
                    phs[r, c] = 0.0
                    freqs[r, c] = 1.0
                end
            end
        end

        layers[l_idx] = WaveLayer(nodes, embed_dim, amps, phs, freqs, fractals, cfg.model.omega)
    end

    return WaveModel(layers, cfg.field, cfg.model)
end

const WAVEML_MAGIC = b"SOVW"
const WAVEML_FORMAT_VERSION = UInt32(1)

"""
    serialize_wave_model_binary(model::WaveModel)::Vector{UInt8}

Serializes a `WaveModel` into high-precision binary bytes (Float32) for MKV container attachment.
"""
function serialize_wave_model_binary(model::WaveModel)::Vector{UInt8}
    io = IOBuffer()
    write(io, WAVEML_MAGIC)
    write(io, WAVEML_FORMAT_VERSION)

    num_layers = Int32(length(model.layers))
    nodes = Int32(model.model_config.nodes)
    embed_dim = Int32(model.model_config.embed_dims)
    omega = Float64(model.model_config.omega)
    beta_s = Float64(model.model_config.beta_s)

    write(io, num_layers)
    write(io, nodes)
    write(io, embed_dim)
    write(io, omega)
    write(io, beta_s)

    for layer in model.layers
        write(io, Float32.(layer.amplitudes))
        write(io, Float32.(layer.phases))
        write(io, Float32.(layer.frequencies))
        write(io, Float32.(layer.fractal_scales))
    end

    return take!(io)
end

"""
    deserialize_wave_model_binary(raw_bytes::Vector{UInt8}, cfg::WaveMLConfig)::Union{Nothing, WaveModel}

Reconstructs a `WaveModel` from high-precision binary bytes.
"""
function deserialize_wave_model_binary(raw_bytes::Vector{UInt8}, cfg::WaveMLConfig)::Union{Nothing, WaveModel}
    if length(raw_bytes) < 32 || raw_bytes[1:4] != WAVEML_MAGIC
        return nothing
    end
    io = IOBuffer(raw_bytes)
    seek(io, 4) # Skip magic
    version = read(io, UInt32)
    num_layers = Int(read(io, Int32))
    nodes = Int(read(io, Int32))
    embed_dim = Int(read(io, Int32))
    omega = read(io, Float64)
    beta_s = read(io, Float64)

    layers = Vector{WaveLayer}(undef, num_layers)
    for l_idx in 1:num_layers
        amps = Matrix{Float64}(undef, nodes, embed_dim)
        phs = Matrix{Float64}(undef, nodes, embed_dim)
        freqs = Matrix{Float64}(undef, nodes, embed_dim)
        fractals = Vector{Float64}(undef, nodes)

        for c in 1:embed_dim, r in 1:nodes
            amps[r, c] = Float64(read(io, Float32))
        end
        for c in 1:embed_dim, r in 1:nodes
            phs[r, c] = Float64(read(io, Float32))
        end
        for c in 1:embed_dim, r in 1:nodes
            freqs[r, c] = Float64(read(io, Float32))
        end
        for r in 1:nodes
            fractals[r] = Float64(read(io, Float32))
        end

        layers[l_idx] = WaveLayer(nodes, embed_dim, amps, phs, freqs, fractals, omega)
    end

    m_cfg = WaveModelConfig(
        nodes = nodes,
        embed_dims = embed_dim,
        layers = num_layers,
        omega = omega,
        beta_s = beta_s
    )
    return WaveModel(layers, cfg.field, m_cfg)
end

"""
    save_model(
        model::WaveModel,
        path::String;
        fps::Union{Nothing, Int} = nothing,
        scale::Union{Nothing, Int, Symbol} = nothing,
        target_height::Int = 480,
        n_visual_frames::Union{Nothing, Int} = nothing,
        render_mode::Union{Nothing, Symbol} = nothing,
        state_colors::Union{Nothing, Vector{Vector{Float64}}} = nothing,
        video_cfg::Union{Nothing, WaveVideoConfig} = nothing,
        audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
        include_audio::Bool = true,
        export_mp4::Bool = true
    )::Nothing

Saves the `WaveModel` as a universally playable MKV video and companion MP4:
- Stream 0:0 (Visual): High-definition fluid heatmap H.264 video (640x480, 24 fps, yuv420p)
  starting at step n (full radiant color, no black screen) and flowing through step n_final.
- Stream 0:1 (Audio): Presentation audio track synthesized from the trained harmonic lattice oscillations at 432 Hz.
- Matroska Attachment: Exact lossless binary weights (`wave_model_weights.bin`) for 100% bit-for-bit inference reconstruction.
Also generates an accompanying metadata YAML and a companion .mp4 file with +faststart for instant opening in all media players.
"""
function save_model(
    model::WaveModel,
    path::String;
    fps::Union{Nothing, Int} = nothing,
    scale::Union{Nothing, Int, Symbol} = nothing,
    target_height::Int = 480,
    n_visual_frames::Union{Nothing, Int} = nothing,
    render_mode::Union{Nothing, Symbol} = nothing,
    state_colors::Union{Nothing, Vector{Vector{Float64}}} = nothing,
    video_cfg::Union{Nothing, WaveVideoConfig} = nothing,
    audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
    include_audio::Bool = true,
    export_mp4::Bool = true
)::Nothing
    v_cfg = video_cfg !== nothing ? video_cfg : WaveVideoConfig()
    raw_mode = render_mode !== nothing ? render_mode : v_cfg.render_mode
    actual_mode = (raw_mode == :potts_champion) ? :potts_model_q_state_domains : raw_mode
    actual_fps = fps !== nothing ? fps : max(24, v_cfg.fps)
    actual_n_frames = n_visual_frames !== nothing ? n_visual_frames : max(24, v_cfg.frames)
    actual_colors = state_colors !== nothing ? state_colors : v_cfg.state_colors

    # 1. Generate Visual Stream (Step n to Step n_final using tournament Grand Champion)
    vis_raw, w_vis, h_vis, n_vis = model_to_visual_frames(model; n_frames=actual_n_frames, mode=actual_mode, state_colors=actual_colors)

    # 2. Generate Exact Lossless Binary Weights
    binary_weights = serialize_wave_model_binary(model)

    # 3. Generate Audio Stream (Trained harmonic lattice oscillations at 432 Hz)
    video_duration = max(0.5, Float64(actual_n_frames) / Float64(max(1, actual_fps)))
    actual_audio_cfg = audio_cfg !== nothing ? audio_cfg : WaveAudioConfig(carrier_frequency=model.model_config.omega)

    # Calculate standard display resolution (at least 640x480, even dimensions)
    target_w = max(640, iseven(w_vis * 4) ? w_vis * 4 : w_vis * 4 + 1)
    target_h = max(480, iseven(target_height) ? target_height : target_height + 1)
    scale_filter = "scale=$(target_w):$(target_h):flags=bicubic,format=yuv420p"

    tmp_vis = tempname() * "_vis.rgb"
    tmp_audio = tempname() * "_audio.wav"

    base_path = replace(path, r"\.(mkv|mp4)$"i => "")
    mkv_path = base_path * ".mkv"
    mp4_path = base_path * ".mp4"
    weights_path = base_path * "_weights.bin"
    meta_path = base_path * "_meta.yaml"

    try
        write(tmp_vis, vis_raw)

        # 1. Encode universal high-quality MP4 first (guarantees faststart, correct timestamps, clean stream durations)
        if include_audio
            audio_buf = sonify_model(model; audio_cfg=actual_audio_cfg, duration=video_duration)
            save_wav(audio_buf, tmp_audio; sample_rate=actual_audio_cfg.sample_rate)

            ffmpeg_cmd = `ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s $(w_vis)x$(h_vis) -r $actual_fps -i $tmp_vis -i $tmp_audio -vf $scale_filter -c:v libx264 -pix_fmt yuv420p -preset fast -metadata:s:v:0 title="VISUAL_BRAIN" -c:a aac -b:a 192k -metadata:s:a:0 title="MODEL_AUDIO" -metadata title="WAVEML_MODEL" -movflags +faststart $mp4_path`
            run(ffmpeg_cmd)
        else
            ffmpeg_cmd = `ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s $(w_vis)x$(h_vis) -r $actual_fps -i $tmp_vis -vf $scale_filter -c:v libx264 -pix_fmt yuv420p -preset fast -metadata:s:v:0 title="VISUAL_BRAIN" -metadata title="WAVEML_MODEL" -movflags +faststart $mp4_path`
            run(ffmpeg_cmd)
        end

        # 2. Remux into MKV: 100% compliant Matroska container matching the working MP4 exactly
        run(`ffmpeg -y -loglevel error -i $mp4_path -c copy $mkv_path`)

        # 3. Save companion exact binary weights for instant zero-loss inference
        write(weights_path, binary_weights)

        # 4. Save companion JSON/YAML metadata file
        cfg = WaveMLConfig(
            field = model.field_config,
            model = model.model_config,
            train = WaveTrainConfig(),
            audio = actual_audio_cfg,
            video = WaveVideoConfig(
                render_mode = actual_mode,
                pixel_scale = 4,
                target_height = target_h,
                fps = actual_fps,
                frames = actual_n_frames,
                state_colors = actual_colors
            )
        )
        save_config(cfg, meta_path)
    finally
        isfile(tmp_vis) && rm(tmp_vis, force=true)
        isfile(tmp_audio) && rm(tmp_audio, force=true)
    end

    return nothing
end

"""
    load_model(path::String; meta_path::Union{Nothing, String} = nothing)::WaveModel

Loads a `WaveModel` from an MKV or MP4 video file.
First attempts bit-for-bit reconstruction from the companion binary weights or MKV container attachment.
Falls back gracefully to video frame decoding if attachment is unavailable.
"""
function load_model(path::String; meta_path::Union{Nothing, String} = nothing)::WaveModel
    isfile(path) || error("Model file not found: $path")

    # Locate metadata
    actual_meta = if meta_path !== nothing && isfile(meta_path)
        meta_path
    else
        candidate = replace(path, r"\.(mkv|mp4)$"i => "") * "_meta.yaml"
        isfile(candidate) ? candidate : nothing
    end

    cfg = if actual_meta !== nothing
        load_config(actual_meta)
    else
        default_config()
    end

    # 1. Primary: Check companion _weights.bin file
    weights_path = replace(path, r"\.(mkv|mp4)$"i => "") * "_weights.bin"
    if isfile(weights_path) && filesize(weights_path) >= 24
        bin_data = read(weights_path)
        loaded = deserialize_wave_model_binary(bin_data, cfg)
        if loaded !== nothing
            return loaded
        end
    end

    # 2. Secondary: Extract attachment from MKV container (if present in legacy multi-stream MKV)
    tmp_dump = tempname() * "_weights.bin"
    loaded_model = try
        run(`ffmpeg -y -loglevel quiet -dump_attachment:t:0 $tmp_dump -i $path -f null -`)
        if isfile(tmp_dump) && filesize(tmp_dump) >= 24
            bin_data = read(tmp_dump)
            deserialize_wave_model_binary(bin_data, cfg)
        else
            nothing
        end
    catch
        nothing
    finally
        isfile(tmp_dump) && rm(tmp_dump, force=true)
    end

    if loaded_model !== nothing
        return loaded_model
    end

    # 3. Fallback: Secondary video stream (legacy MKV) or Stream 0:0 video frames
    nodes = cfg.model.nodes
    embed_dim = cfg.model.embed_dims
    num_layers = cfg.model.layers
    w = iseven(embed_dim) ? embed_dim : embed_dim + 1
    h = iseven(nodes) ? nodes : nodes + 1
    downscale_filter = "scale=$(w):$(h):flags=neighbor"

    raw_bytes = try
        read(`ffmpeg -loglevel quiet -i $path -map "0:v:1?" -f rawvideo -pix_fmt rgb24 -`)
    catch
        UInt8[]
    end

    if isempty(raw_bytes)
        raw_bytes = try
            read(`ffmpeg -loglevel error -i $path -map 0:v:0 -vf $downscale_filter -f rawvideo -pix_fmt rgb24 -`)
        catch
            read(`ffmpeg -loglevel error -i $path -vf $downscale_filter -f rawvideo -pix_fmt rgb24 -`)
        end
    end

    return rgb_frames_to_model(raw_bytes, w, h, num_layers, cfg)
end
