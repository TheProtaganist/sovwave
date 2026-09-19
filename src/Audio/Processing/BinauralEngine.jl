export BinauralBeat, generate_binaural!, create_brainwave_beat

const DELTA_RANGE = (0.5, 4.0)
const THETA_RANGE = (4.0, 8.0)
const ALPHA_RANGE = (8.0, 13.0)
const BETA_RANGE = (13.0, 30.0)
const GAMMA_RANGE = (30.0, 40.0)

"""
    BinauralBeat

A structure to hold parameters for binaural beat generation.
"""
struct BinauralBeat
    carrier_freq::Float64
    beat_freq::Float64
    left_phase::Ref{Float64}
    right_phase::Ref{Float64}
    sample_rate::Int
    trig_func::Symbol

    function BinauralBeat(carrier_freq::Float64, beat_freq::Float64, left_phase::Ref{Float64}, right_phase::Ref{Float64}, sample_rate::Int, trig_func::Symbol)
        if beat_freq < 0.5 || beat_freq > 40.0
            throw(ArgumentError("Beat frequency must be between 0.5 and 40 Hz"))
        end
        if carrier_freq <= 0.0
            throw(ArgumentError("Carrier frequency must be greater than 0"))
        end
        new(carrier_freq, beat_freq, left_phase, right_phase, sample_rate, trig_func)
    end
end

function BinauralBeat(carrier_freq::Float64, beat_freq::Float64, sample_rate::Int=48000, trig_func::Symbol=:sin)
    BinauralBeat(carrier_freq, beat_freq, Ref(0.0), Ref(0.0), sample_rate, trig_func)
end

"""
    generate_binaural!(buffer::Matrix{Float64}, bb::BinauralBeat)::Nothing

Generates binaural beats into the provided buffer.
`buffer` should be a 2xN matrix where row 1 is left and row 2 is right.
"""
function generate_binaural!(buffer::Matrix{Float64}, bb::BinauralBeat)::Nothing
    num_samples = size(buffer, 2)
    dt = BigFloat(1.0) / BigFloat(bb.sample_rate)
    
    carrier = BigFloat(bb.carrier_freq)
    beat = BigFloat(bb.beat_freq)
    
    l_phase = BigFloat(bb.left_phase[])
    r_phase = BigFloat(bb.right_phase[])
    
    for i in 1:num_samples
        t = BigFloat(i-1) * dt
        # Left channel: carrier frequency
        buffer[1, i] = Float64(sin(2 * BigFloat(pi) * carrier * t + l_phase))
        # Right channel: carrier + beat frequency
        buffer[2, i] = Float64(sin(2 * BigFloat(pi) * (carrier + beat) * t + r_phase))
    end
    
    # Advance both phases using BigFloat for precision
    bb.left_phase[] = Float64(mod(l_phase + 2 * BigFloat(pi) * carrier * BigFloat(num_samples) * dt, 2 * BigFloat(pi)))
    bb.right_phase[] = Float64(mod(r_phase + 2 * BigFloat(pi) * (carrier + beat) * BigFloat(num_samples) * dt, 2 * BigFloat(pi)))
    
    return nothing
end

"""
    create_brainwave_beat(type::Symbol; carrier::Float64=432.0, sample_rate::Int=48000)::BinauralBeat

Creates a BinauralBeat instance configured for a specific brainwave type.
Presets: :delta (2Hz), :theta (6Hz), :alpha (10Hz), :beta (20Hz), :gamma (35Hz)
"""
function create_brainwave_beat(type::Symbol; carrier::Float64=432.0, sample_rate::Int=48000)::BinauralBeat
    beat_freq = if type == :delta
        2.0
    elseif type == :theta
        6.0
    elseif type == :alpha
        10.0
    elseif type == :beta
        20.0
    elseif type == :gamma
        35.0
    else
        throw(ArgumentError("Unknown brainwave type: \$type"))
    end
    
    return BinauralBeat(carrier, beat_freq, sample_rate, :sin)
end
