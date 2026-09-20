"""
    WaveML.Sonify

User-Defined Training Sonification and Audio Rendering for WaveML.
Provides complete user control over carrier frequency, tuning reference, waveforms,
binaural beats, ADSR envelopes, harmonic richness, volume, stereo spatial panning,
and real-time playback drivers.

No sound properties are hardcoded:
- Carrier frequency can be 432 Hz, 440 Hz, 528 Hz (Solfeggio), 108 Hz, or ANY user-defined float
- Waveforms: `:sine`, `:harmonic`, `:physical`, `:triangle`, `:sawtooth`, `:binaural`
- Envelopes: `:exponential_decay`, `:adsr`, `:percussive`, `:sustain`
- Channels: 1 (mono) or 2 (stereo / binaural entrainment)
- Dual output: Lossless 16-bit PCM WAV export and live real-time speaker playback
"""

using Printf

export sonify_model, sonify_step, save_wav, play_realtime!
export ContinuousAudioStream, step_continuous_audio!, get_accumulated_audio, compute_binaural_beat_freq, brainwave_state

"""
    compute_envelope(t::Float64, duration::Float64, env_type::Symbol, a::Float64, d::Float64, s::Float64, r::Float64)::Float64

Computes the instantaneous amplitude envelope at time `t` for a sound of total `duration`.
"""
function compute_envelope(
    t::Float64,
    duration::Float64,
    env_type::Symbol,
    attack::Float64,
    decay::Float64,
    sustain::Float64,
    release::Float64
)::Float64
    t_clamped = clamp(t, 0.0, duration)
    
    if env_type == :exponential_decay
        return exp(-3.0 * (t_clamped / duration))
    elseif env_type == :percussive
        rem = max(0.0, 1.0 - t_clamped / duration)
        return rem * rem
    elseif env_type == :sustain
        # Smooth Hann ramp for first and last 5ms
        ramp = min(0.005, duration * 0.1)
        if t_clamped < ramp
            return 0.5 * (1.0 - cos(π * t_clamped / ramp))
        elseif t_clamped > duration - ramp
            return 0.5 * (1.0 - cos(π * (duration - t_clamped) / ramp))
        else
            return 1.0
        end
    else # :adsr
        tot_adr = attack + decay + release
        scale = tot_adr > duration ? (duration / tot_adr) : 1.0
        att_t = attack * scale
        dec_t = decay * scale
        rel_t = release * scale
        sus_t = max(0.0, duration - att_t - dec_t - rel_t)

        if t_clamped < att_t
            return t_clamped / max(att_t, 1e-6)
        elseif t_clamped < att_t + dec_t
            frac = (t_clamped - att_t) / max(dec_t, 1e-6)
            return 1.0 - frac * (1.0 - sustain)
        elseif t_clamped < att_t + dec_t + sus_t
            return sustain
        else
            frac = (t_clamped - att_t - dec_t - sus_t) / max(rel_t, 1e-6)
            return max(0.0, sustain * (1.0 - frac))
        end
    end
end

"""
    synthesize_oscillator(waveform::Symbol, phase::Float64, overtone_factor::Float64)::Float64

Synthesizes a single sample value for a given wave phase [0, 2π) and waveform type.
"""
function synthesize_oscillator(waveform::Symbol, phase::Float64, overtone_factor::Float64)::Float64
    ph = mod(phase, 2π)
    
    if waveform == :sine
        return sin(ph)
    elseif waveform == :triangle
        # Continuous triangle wave
        return asin(sin(ph)) * (2.0 / π)
    elseif waveform == :sawtooth
        # Continuous sawtooth wave
        return 2.0 * (ph / (2π)) - 1.0
    elseif waveform == :harmonic
        # Fundamental plus overtone harmonics scaled by overtone_factor
        s = sin(ph)
        if overtone_factor > 0.0
            s += 0.5 * overtone_factor * sin(2.0 * ph)
            s += 0.25 * overtone_factor * sin(3.0 * ph)
            s += 0.125 * overtone_factor * sin(4.0 * ph)
        end
        return s / (1.0 + 0.875 * overtone_factor)
    else # :physical (spring-mass damped harmonic resonator)
        s = sin(ph)
        s += 0.35 * sin(1.618 * ph) * overtone_factor
        s += 0.15 * sin(2.618 * ph) * overtone_factor
        return s / (1.0 + 0.5 * overtone_factor)
    end
end

"""
    sonify_model(
        model::WaveModel;
        audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
        carrier_frequency::Union{Nothing, Float64} = nothing,
        waveform::Union{Nothing, Symbol} = nothing,
        binaural_beat::Union{Nothing, Float64} = nothing,
        envelope::Union{Nothing, Symbol} = nothing,
        attack::Union{Nothing, Float64} = nothing,
        decay::Union{Nothing, Float64} = nothing,
        sustain::Union{Nothing, Float64} = nothing,
        release::Union{Nothing, Float64} = nothing,
        harmonic_richness::Union{Nothing, Float64} = nothing,
        volume::Union{Nothing, Float64} = nothing,
        pan::Union{Nothing, Float64} = nothing,
        sample_rate::Union{Nothing, Int} = nothing,
        channels::Union{Nothing, Int} = nothing,
        duration::Float64 = 0.5
    )::Union{Vector{Float64}, Matrix{Float64}}

Renders the wave model's parameter lattice into audio according to full user specifications.
Users can supply a `WaveAudioConfig` or individual keyword arguments for complete customization.
"""
function sonify_model(
    model::WaveModel;
    audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
    carrier_frequency::Union{Nothing, Float64} = nothing,
    waveform::Union{Nothing, Symbol} = nothing,
    binaural_beat::Union{Nothing, Float64} = nothing,
    envelope::Union{Nothing, Symbol} = nothing,
    attack::Union{Nothing, Float64} = nothing,
    decay::Union{Nothing, Float64} = nothing,
    sustain::Union{Nothing, Float64} = nothing,
    release::Union{Nothing, Float64} = nothing,
    harmonic_richness::Union{Nothing, Float64} = nothing,
    volume::Union{Nothing, Float64} = nothing,
    pan::Union{Nothing, Float64} = nothing,
    sample_rate::Union{Nothing, Int} = nothing,
    channels::Union{Nothing, Int} = nothing,
    duration::Float64 = 0.5
)::Union{Vector{Float64}, Matrix{Float64}}

    # Resolve settings: priority = explicit kwarg > audio_cfg > model_config / defaults
    c_freq = carrier_frequency !== nothing ? carrier_frequency :
             (audio_cfg !== nothing ? audio_cfg.carrier_frequency : model.model_config.omega)
    w_form = waveform !== nothing ? waveform :
             (audio_cfg !== nothing ? audio_cfg.waveform : :physical)
    b_beat = binaural_beat !== nothing ? binaural_beat :
             (audio_cfg !== nothing ? audio_cfg.binaural_beat : 0.0)
    env_type = envelope !== nothing ? envelope :
               (audio_cfg !== nothing ? audio_cfg.envelope : :exponential_decay)
    att = attack !== nothing ? attack : (audio_cfg !== nothing ? audio_cfg.attack : 0.02)
    dec = decay !== nothing ? decay : (audio_cfg !== nothing ? audio_cfg.decay : 0.15)
    sus = sustain !== nothing ? sustain : (audio_cfg !== nothing ? audio_cfg.sustain : 0.60)
    rel = release !== nothing ? release : (audio_cfg !== nothing ? audio_cfg.release : 0.25)
    h_rich = harmonic_richness !== nothing ? harmonic_richness : (audio_cfg !== nothing ? audio_cfg.harmonic_richness : 1.0)
    vol = volume !== nothing ? volume : (audio_cfg !== nothing ? audio_cfg.volume : 0.85)
    pan_pos = pan !== nothing ? pan : (audio_cfg !== nothing ? audio_cfg.pan : 0.0)
    s_rate = sample_rate !== nothing ? sample_rate : (audio_cfg !== nothing ? audio_cfg.sample_rate : 48000)
    n_chan = channels !== nothing ? channels : (audio_cfg !== nothing ? audio_cfg.channels : (b_beat > 0.0 ? 2 : 1))

    n_samples = max(128, round(Int, s_rate * duration))
    t_step = 1.0 / s_rate
    num_layers = length(model.layers)

    # Stereo pan gains (equal power panning)
    pan_clamped = clamp(pan_pos, -1.0, 1.0)
    pan_angle = (pan_clamped + 1.0) * (π / 4.0)
    left_gain = cos(pan_angle)
    right_gain = sin(pan_angle)

    if n_chan == 1
        mono_buf = zeros(Float64, n_samples)
        for (l_idx, layer) in enumerate(model.layers)
            nodes = layer.nodes
            embed_dim = layer.embed_dim

            for n_i in 1:min(nodes, 8)
                amp_mean = sum(layer.amplitudes[n_i, :]) / embed_dim
                freq_mult = layer.frequencies[n_i, 1]
                ph_0 = layer.phases[n_i, 1]
                beta = layer.fractal_scales[n_i]

                # User-defined carrier frequency scaling
                f = c_freq * freq_mult * (beta / 1.618033988749895)
                f = clamp(f, 20.0, 16000.0)
                layer_weight = amp_mean / (num_layers + 1e-6)

                @inbounds for s in 1:n_samples
                    t = (s - 1) * t_step
                    env = compute_envelope(t, duration, env_type, att, dec, sus, rel)
                    osc = synthesize_oscillator(w_form, 2π * f * t + ph_0, h_rich)
                    mono_buf[s] += layer_weight * env * osc
                end
            end
        end

        max_v = maximum(abs, mono_buf)
        if max_v > 1e-6
            mono_buf .*= (vol / max_v)
        end
        return mono_buf

    else # Stereo / Binaural channels == 2
        stereo_buf = zeros(Float64, 2, n_samples)

        for (l_idx, layer) in enumerate(model.layers)
            nodes = layer.nodes
            embed_dim = layer.embed_dim

            for n_i in 1:min(nodes, 8)
                amp_mean = sum(layer.amplitudes[n_i, :]) / embed_dim
                freq_mult = layer.frequencies[n_i, 1]
                ph_0 = layer.phases[n_i, 1]
                beta = layer.fractal_scales[n_i]

                base_f = c_freq * freq_mult * (beta / 1.618033988749895)
                # Left and Right frequencies for binaural entrainment
                f_left = clamp(base_f - b_beat * 0.5, 20.0, 16000.0)
                f_right = clamp(base_f + b_beat * 0.5, 20.0, 16000.0)

                layer_weight = amp_mean / (num_layers + 1e-6)

                @inbounds for s in 1:n_samples
                    t = (s - 1) * t_step
                    env = compute_envelope(t, duration, env_type, att, dec, sus, rel)
                    osc_l = synthesize_oscillator(w_form, 2π * f_left * t + ph_0, h_rich)
                    osc_r = synthesize_oscillator(w_form, 2π * f_right * t + ph_0, h_rich)

                    stereo_buf[1, s] += layer_weight * env * osc_l * left_gain
                    stereo_buf[2, s] += layer_weight * env * osc_r * right_gain
                end
            end
        end

        max_v = maximum(abs, stereo_buf)
        if max_v > 1e-6
            stereo_buf .*= (vol / max_v)
        end
        return stereo_buf
    end
end

"""
    compute_binaural_beat_freq(energy::Float64, target_energy::Float64; max_beat::Float64 = 60.0)::Float64

Computes continuous binaural beat frequency Δf(E).
The carrier stays strictly at 432 Hz (or user-defined frequency).
Δf smoothly transitions from chaotic Gamma (40-60 Hz) down through Beta, Alpha, Theta, Delta,
and asymptotically approaches Epsilon (0.5 Hz) while energy > target_energy.
It ONLY reaches 0.0 Hz (true ground state phase-lock) if energy <= target_energy.
"""
function compute_binaural_beat_freq(energy::Float64, target_energy::Float64; max_beat::Float64 = 60.0)::Float64
    if energy <= target_energy
        return 0.0 # Pure ground-state phase lock at 432 Hz
    end
    e_rel = max(0.0, energy - target_energy)
    epsilon_min = 0.5
    delta_f = epsilon_min + (max_beat - epsilon_min) * (e_rel / (0.5 + e_rel))
    return delta_f
end

"""
    brainwave_state(delta_f::Float64, reached_ground::Bool)::Tuple{String, String}

Returns the brainwave classification and frequency label for a given binaural beat frequency.
"""
function brainwave_state(delta_f::Float64, reached_ground::Bool)::Tuple{String, String}
    if reached_ground || delta_f <= 0.05
        return ("Epsilon", "0.00 Hz (Ground State Locked 🎯)")
    elseif delta_f <= 0.5
        return ("Epsilon", @sprintf("%.2f Hz (Asymptotic)", delta_f))
    elseif delta_f <= 4.0
        return ("Delta", @sprintf("%.1f Hz", delta_f))
    elseif delta_f <= 8.0
        return ("Theta", @sprintf("%.1f Hz", delta_f))
    elseif delta_f <= 14.0
        return ("Alpha", @sprintf("%.1f Hz", delta_f))
    elseif delta_f <= 35.0
        return ("Beta", @sprintf("%.1f Hz", delta_f))
    else
        return ("Gamma", @sprintf("%.1f Hz (Chaotic)", delta_f))
    end
end

"""
    ContinuousAudioStream

Accumulates seamless, continuous audio samples without discrete start/stop gaps.
Preserves phase continuity across evolutionary training steps.
"""
mutable struct ContinuousAudioStream
    sample_rate::Int
    carrier_frequency::Float64
    phase_l::Float64
    phase_r::Float64
    buffer_l::Vector{Float64}
    buffer_r::Vector{Float64}

    function ContinuousAudioStream(;
        sample_rate::Int = 48000,
        carrier_frequency::Float64 = 432.0
    )
        new(sample_rate, carrier_frequency, 0.0, 0.0, Float64[], Float64[])
    end
end

"""
    step_continuous_audio!(stream::ContinuousAudioStream, energy::Float64, target_energy::Float64; duration=0.08, volume=0.8)

Appends continuous audio samples to the stream without phase clicks or silence gaps.
"""
function step_continuous_audio!(
    stream::ContinuousAudioStream,
    energy::Float64,
    target_energy::Float64;
    duration::Float64 = 0.08,
    volume::Float64 = 0.8
)::Matrix{Float64}
    delta_f = compute_binaural_beat_freq(energy, target_energy)
    fc = stream.carrier_frequency
    sr = stream.sample_rate
    n_samples = max(1, round(Int, sr * duration))
    dt = 1.0 / sr

    f_left = fc - delta_f * 0.5
    f_right = fc + delta_f * 0.5

    # Subtle chaotic phase jitter in high Gamma band far from ground state
    chaos = delta_f > 35.0 ? (0.02 * sin(2π * 7.83 * (length(stream.buffer_l) * dt))) : 0.0

    chunk = Matrix{Float64}(undef, 2, n_samples)
    @inbounds for s in 1:n_samples
        stream.phase_l += 2π * (f_left + chaos) * dt
        stream.phase_r += 2π * f_right * dt

        if stream.phase_l > 2π; stream.phase_l -= 2π; end
        if stream.phase_r > 2π; stream.phase_r -= 2π; end

        val_l = volume * sin(stream.phase_l)
        val_r = volume * sin(stream.phase_r)

        chunk[1, s] = val_l
        chunk[2, s] = val_r

        push!(stream.buffer_l, val_l)
        push!(stream.buffer_r, val_r)
    end
    return chunk
end

"""
    get_accumulated_audio(stream::ContinuousAudioStream)::Matrix{Float64}

Returns the full contiguous 2xN stereo matrix of accumulated training audio.
"""
function get_accumulated_audio(stream::ContinuousAudioStream)::Matrix{Float64}
    n = length(stream.buffer_l)
    mat = Matrix{Float64}(undef, 2, n)
    @inbounds for i in 1:n
        mat[1, i] = stream.buffer_l[i]
        mat[2, i] = stream.buffer_r[i]
    end
    return mat
end

"""
    sonify_step(
        energy::Float64,
        model::WaveModel;
        audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
        carrier_frequency::Union{Nothing, Float64} = nothing,
        target_energy::Float64 = 0.001,
        duration::Float64 = 0.08,
        stream::Union{Nothing, ContinuousAudioStream} = nothing
    )::Union{Vector{Float64}, Matrix{Float64}}

Generates an audio cue during training reflecting the model's ground-state convergence.
Carrier stays strictly at 432 Hz (or user-defined frequency).
Binaural beat Δf transitions from Gamma (chaotic) down through Beta, Alpha, Theta, Delta,
and asymptotically approaches Epsilon (0.5 Hz) until ground state is reached (0.0 Hz).
"""
function sonify_step(
    energy::Float64,
    model::WaveModel;
    audio_cfg::Union{Nothing, WaveAudioConfig} = nothing,
    carrier_frequency::Union{Nothing, Float64} = nothing,
    target_energy::Float64 = 0.001,
    duration::Float64 = 0.08,
    stream::Union{Nothing, ContinuousAudioStream} = nothing
)::Union{Vector{Float64}, Matrix{Float64}}

    c_freq = carrier_frequency !== nothing ? carrier_frequency :
             (audio_cfg !== nothing ? audio_cfg.carrier_frequency : model.model_config.omega)
    vol = audio_cfg !== nothing ? audio_cfg.volume : 0.8

    if stream !== nothing
        return step_continuous_audio!(stream, energy, target_energy; duration=duration, volume=vol)
    end

    s_rate = audio_cfg !== nothing ? audio_cfg.sample_rate : 48000
    n_chan = audio_cfg !== nothing ? audio_cfg.channels : 1

    delta_f = compute_binaural_beat_freq(energy, target_energy)
    n_samples = max(1, round(Int, s_rate * duration))
    t_step = 1.0 / s_rate

    if n_chan == 1
        buf = Vector{Float64}(undef, n_samples)
        # In mono, modulate carrier with subtle beating that vanishes smoothly at 0 Hz beat
        beat_mod_depth = min(1.0, delta_f / 60.0)
        @inbounds for s in 1:n_samples
            t = (s - 1) * t_step
            beat_factor = (1.0 + beat_mod_depth * cos(2π * delta_f * t)) / (1.0 + beat_mod_depth)
            buf[s] = vol * sin(2π * c_freq * t) * beat_factor
        end
        return buf
    else
        buf = Matrix{Float64}(undef, 2, n_samples)
        f_left = c_freq - delta_f * 0.5
        f_right = c_freq + delta_f * 0.5
        chaos = delta_f > 35.0 ? (0.02 * sin(2π * 7.83 * t_step)) : 0.0
        @inbounds for s in 1:n_samples
            t = (s - 1) * t_step
            buf[1, s] = vol * sin(2π * (f_left + chaos) * t)
            buf[2, s] = vol * sin(2π * f_right * t)
        end
        return buf
    end
end

"""
    save_wav(buffer::Union{Vector{Float64}, Matrix{Float64}}, path::String; sample_rate::Int = 48000)::Nothing

Writes an audio buffer to a standard 16-bit PCM WAV file on disk.
Automatically supports both Mono (1D Vector) and Stereo (2xN Matrix) audio.
"""
function save_wav(buffer::Union{Vector{Float64}, Matrix{Float64}}, path::String; sample_rate::Int = 48000)::Nothing
    is_stereo = isa(buffer, Matrix) && size(buffer, 1) == 2
    num_channels = is_stereo ? 2 : 1
    n_samples = is_stereo ? size(buffer, 2) : length(buffer)

    bits_per_sample = 16
    bytes_per_sample = bits_per_sample ÷ 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    subchunk2_size = n_samples * block_align
    chunk_size = 36 + subchunk2_size

    open(path, "w") do io
        # RIFF Header
        write(io, "RIFF")
        write(io, UInt32(chunk_size))
        write(io, "WAVE")

        # fmt subchunk
        write(io, "fmt ")
        write(io, UInt32(16))               # Subchunk1Size (16 for PCM)
        write(io, UInt16(1))                # AudioFormat (1 = PCM)
        write(io, UInt16(num_channels))     # NumChannels (1 or 2)
        write(io, UInt32(sample_rate))      # SampleRate
        write(io, UInt32(byte_rate))        # ByteRate
        write(io, UInt16(block_align))      # BlockAlign
        write(io, UInt16(bits_per_sample))  # BitsPerSample

        # data subchunk
        write(io, "data")
        write(io, UInt32(subchunk2_size))

        # Sample data (interleaved 16-bit signed integer PCM)
        if !is_stereo
            vec = buffer::Vector{Float64}
            for s in vec
                s_clamped = clamp(s, -1.0, 1.0)
                write(io, round(Int16, s_clamped * 32767.0))
            end
        else
            mat = buffer::Matrix{Float64}
            for col in 1:n_samples
                left_s = clamp(mat[1, col], -1.0, 1.0)
                right_s = clamp(mat[2, col], -1.0, 1.0)
                write(io, round(Int16, left_s * 32767.0))
                write(io, round(Int16, right_s * 32767.0))
            end
        end
    end

    return nothing
end

"""
    play_realtime!(
        buffer::Union{Vector{Float64}, Matrix{Float64}};
        sample_rate::Int = 48000,
        player::String = "auto"
    )::Bool

Plays the audio buffer through the system audio device in real time.
Supports custom user-defined players or auto-detection (`paplay`, `aplay`, `ffplay`, `pw-play`).
"""
function play_realtime!(
    buffer::Union{Vector{Float64}, Matrix{Float64}};
    sample_rate::Int = 48000,
    player::String = "auto"
)::Bool
    tmp_path = tempname() * ".wav"
    try
        save_wav(buffer, tmp_path; sample_rate=sample_rate)

        if player != "auto" && !isempty(player)
            cmd_str = "$player $tmp_path"
            run(pipeline(`bash -c $cmd_str`, stdout=devnull, stderr=devnull))
            return true
        end

        players = ["paplay", "aplay", "pw-play", "ffplay -nodisp -autoexit -loglevel quiet"]
        played = false

        for p in players
            bin_name = split(p)[1]
            if Sys.which(bin_name) !== nothing
                cmd_str = "$p $tmp_path"
                run(pipeline(`bash -c $cmd_str`, stdout=devnull, stderr=devnull))
                played = true
                break
            end
        end

        return played
    catch
        return false
    finally
        isfile(tmp_path) && rm(tmp_path, force=true)
    end
end
