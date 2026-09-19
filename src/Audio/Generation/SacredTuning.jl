export SacredNote, midi_to_freq_432, freq_440_to_432, freq_432_to_440, golden_harmonic, fibonacci_harmonics, generate_chromatic_scale

using .AudioConstants

"""
    SacredNote

Represents a musical note in the 432Hz sacred tuning system.
"""
struct SacredNote
    name::String
    midi_number::Int
    freq_432::Float64
    freq_440::Float64
end

"""
    midi_to_freq_432(midi_note::Int)::Float64

Converts a MIDI note number to its frequency in 432Hz tuning.
Formula: f = 432 * 2^((n-69)/12)
"""
function midi_to_freq_432(midi_note::Int)::Float64
    f_432 = BigFloat(432.0) * (BigFloat(2.0) ^ ((BigFloat(midi_note) - 69.0) / 12.0))
    return Float64(f_432)
end

"""
    freq_440_to_432(freq::Float64)::Float64

Converts a frequency from 440Hz tuning to 432Hz tuning.
Multiplies by 432/440.
"""
function freq_440_to_432(freq::Float64)::Float64
    return Float64(BigFloat(freq) * (BigFloat(432.0) / BigFloat(440.0)))
end

"""
    freq_432_to_440(freq::Float64)::Float64

Converts a frequency from 432Hz tuning to 440Hz tuning.
Multiplies by 440/432.
"""
function freq_432_to_440(freq::Float64)::Float64
    return Float64(BigFloat(freq) * (BigFloat(440.0) / BigFloat(432.0)))
end

"""
    golden_harmonic(base_freq::Float64, order::Int)::Float64

Calculates a harmonic based on the golden ratio (PHI).
Formula: base_freq * PHI^order
"""
function golden_harmonic(base_freq::Float64, order::Int)::Float64
    return Float64(BigFloat(base_freq) * (BigFloat(PHI) ^ order))
end

"""
    fibonacci_harmonics(base_freq::Float64, max_order::Int)::Vector{Float64}

Calculates a series of harmonics based on Fibonacci ratios.
Formula: base_freq * FIBONACCI[n+1]/FIBONACCI[n]
"""
function fibonacci_harmonics(base_freq::Float64, max_order::Int)::Vector{Float64}
    harmonics = Vector{Float64}()
    b_freq = BigFloat(base_freq)
    for n in 1:max_order
        if n+1 <= length(FIBONACCI)
            ratio = BigFloat(FIBONACCI[n+1]) / BigFloat(FIBONACCI[n])
            push!(harmonics, Float64(b_freq * ratio))
        end
    end
    return harmonics
end

"""
    generate_chromatic_scale()::Vector{SacredNote}

Generates the full chromatic scale for the entire MIDI range (0-127) with both 432Hz and 440Hz tunings.
"""
function generate_chromatic_scale()::Vector{SacredNote}
    notes = Vector{SacredNote}()
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in 0:127
        octave = div(i, 12) - 1
        note_idx = (i % 12) + 1
        name = note_names[note_idx] * string(octave)
        f_432 = midi_to_freq_432(i)
        f_440 = Float64(BigFloat(440.0) * (BigFloat(2.0) ^ ((BigFloat(i) - 69.0) / 12.0)))
        push!(notes, SacredNote(name, i, f_432, f_440))
    end
    return notes
end
