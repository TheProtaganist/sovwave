"""
    AudioConstants

System-wide constants for the Quantum Audio Foundation, grounded in 432Hz sacred
tuning and golden ratio harmonics. All frequency calculations reference these
immutable values to ensure mathematical consistency across the entire audio engine.

# Sacred Tuning Philosophy

The module uses A4 = 432Hz instead of the standard concert pitch A4 = 440Hz.
This tuning aligns with:
- Golden ratio (φ ≈ 1.618) harmonic relationships
- Fibonacci sequence natural scaling
- Claimed resonance with natural phenomena and human biology

Mathematical precision is maintained to 12+ decimal places for all calculations
to preserve phase coherence in long-running audio streams.

# References

- Horowitz, L. (1998). "Healing Codes for the Biological Apocalypse"
- Livio, M. (2002). "The Golden Ratio: The Story of Phi"
- Equal temperament tuning: f(n) = 432 · 2^((n-69)/12) for MIDI note n
"""
module AudioConstants

export A4_SACRED, A4_STANDARD, TUNING_RATIO
export PHI, FIBONACCI
export SAMPLE_RATE_44_1, SAMPLE_RATE_48, SAMPLE_RATE_96, SAMPLE_RATE_192
export MIN_BUFFER_SIZE, DEFAULT_BUFFER_SIZE, MAX_BUFFER_SIZE
export PHASE_PRECISION, AMPLITUDE_PRECISION, TARGET_LATENCY_MS

# ============================================================================
# Sacred Tuning Constants
# ============================================================================

"""
    A4_SACRED::Float64

Sacred tuning reference frequency: A4 = 432Hz

This is the foundational frequency for the entire audio system. Unlike the
standard concert pitch of 440Hz, 432Hz is claimed to resonate with natural
phenomena including the Schumann resonance (~7.83Hz × 55 = 430.65Hz).

All frequency calculations derive from this value via equal temperament:
    f(n) = 432 · 2^((n-69)/12)
where n is the MIDI note number (A4 = 69).
"""
const A4_SACRED = 432.0  # Hz

"""
    A4_STANDARD::Float64

Standard concert pitch: A4 = 440Hz

Provided for conversion and compatibility with standard tuning systems.
Most modern music uses this reference, standardized by ISO 16 in 1975.
"""
const A4_STANDARD = 440.0  # Hz

"""
    TUNING_RATIO::Float64

Conversion ratio between 432Hz and 440Hz tuning systems.

    TUNING_RATIO = 432/440 = 0.981818181818...

Use this to convert frequencies:
- 440Hz → 432Hz: multiply by TUNING_RATIO
- 432Hz → 440Hz: divide by TUNING_RATIO
"""
const TUNING_RATIO = A4_SACRED / A4_STANDARD  # 0.9818181818181818

# ============================================================================
# Mathematical Constants
# ============================================================================

"""
    PHI::Float64

The golden ratio: φ = (1 + √5) / 2 ≈ 1.618033988749895

This irrational constant appears throughout nature and sacred geometry:
- Fibonacci ratio: lim(F(n+1)/F(n)) = φ as n→∞
- Self-similar scaling: φ^n generates harmonic series
- Pentagonal symmetry: cos(π/5) = φ/2

Used for generating golden ratio harmonics: f_harmonic = f_base · φ^n

Precision: 15 decimal places (Float64 limit)

Computed via Newton-Raphson method for φ² - φ - 1 = 0 using functional
list comprehension (no loops!) for maximum elegance and performance.
"""
const PHI = let
    # Newton-Raphson: x_new = x - f(x)/f'(x) where f(x) = x² - x - 1
    # Using Julia list comprehension to iterate without explicit loops
    # Starting with x₀ = 1.5, apply transformation 10 times
    last([foldl((x, _) -> x - (x*x - x - 1.0) / (2.0*x - 1.0), 1:10, init=1.5)])
end  # 1.618033988749895

"""
    FIBONACCI::Vector{Int}

First 12 Fibonacci numbers: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

The Fibonacci sequence is defined recursively:
    F(0) = 1, F(1) = 1, F(n) = F(n-1) + F(n-2)

Properties:
- Ratios F(n+1)/F(n) converge to φ (golden ratio)
- Natural scaling for harmonic series
- Appears in phyllotaxis, spiral galaxies, DNA molecules

Used for Fibonacci harmonic generation where harmonic n has frequency
f_n = f_base · F(n)/F(n-1), approximating golden ratio scaling.
"""
const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# ============================================================================
# Sample Rate Constants
# ============================================================================

"""
    SAMPLE_RATE_44_1::Int

CD-quality sample rate: 44,100 Hz

This rate was chosen for CDs to satisfy Nyquist theorem for human hearing
(20-20kHz range) with margin: 44100/2 = 22050 Hz > 20000 Hz.

Time resolution: 1/44100 ≈ 22.68 microseconds per sample
"""
const SAMPLE_RATE_44_1 = 44100

"""
    SAMPLE_RATE_48::Int

Professional audio sample rate: 48,000 Hz

Standard for professional video and audio post-production. Divides evenly
by common frame rates (24, 25, 30, 48, 50, 60).

Time resolution: 1/48000 ≈ 20.83 microseconds per sample
"""
const SAMPLE_RATE_48 = 48000

"""
    SAMPLE_RATE_96::Int

High-resolution audio sample rate: 96,000 Hz

Double the professional standard (2× 48kHz). Used for high-quality recording
and mastering, though benefits above 48kHz are debated.

Time resolution: 1/96000 ≈ 10.42 microseconds per sample
"""
const SAMPLE_RATE_96 = 96000

"""
    SAMPLE_RATE_192::Int

Ultra-high-resolution sample rate: 192,000 Hz

Quadruple the professional standard (4× 48kHz). Used in specialized audio
research and archival recordings. Benefits above 96kHz are questionable
for human perception.

Time resolution: 1/192000 ≈ 5.21 microseconds per sample
"""
const SAMPLE_RATE_192 = 192000

# ============================================================================
# Buffer Configuration Constants
# ============================================================================

"""
    MIN_BUFFER_SIZE::Int

Minimum audio buffer size: 64 samples

Below this size, context switching and system overhead dominate performance.
At 48kHz: 64 samples = 1.33ms latency.
"""
const MIN_BUFFER_SIZE = 64

"""
    DEFAULT_BUFFER_SIZE::Int

Default audio buffer size: 128 samples

Balances latency and computational efficiency for most applications.
Latency at various sample rates:
- 44.1kHz: 2.90 ms
- 48kHz: 2.67 ms
- 96kHz: 1.33 ms
"""
const DEFAULT_BUFFER_SIZE = 128

"""
    MAX_BUFFER_SIZE::Int

Maximum audio buffer size: 2048 samples

Larger buffers improve efficiency but increase latency. Beyond this size,
latency becomes perceptible for interactive applications.

Latency at 48kHz: 2048 samples = 42.67 ms (perceptible delay threshold)
"""
const MAX_BUFFER_SIZE = 2048

# ============================================================================
# Precision Constants
# ============================================================================

"""
    PHASE_PRECISION::Float64

Phase tracking precision: 1e-12 radians

Phase is accumulated across millions of samples in long audio streams.
This precision ensures phase drift remains negligible even after hours
of continuous generation.

At 432Hz and 48kHz sample rate:
- Phase advance per sample: 2π × 432/48000 ≈ 0.0565 radians
- Accumulated error after 1 hour: < 1e-12 × 3600 × 48000 ≈ 0.17 radians
- Resulting frequency drift: negligible
"""
const PHASE_PRECISION = 1e-12  # Radians

"""
    AMPLITUDE_PRECISION::Float64

Amplitude precision: 1e-9 (linear scale)

Ensures amplitude scaling and normalization maintain precision well below
the noise floor of any practical audio system.

For reference:
- 16-bit PCM quantization: 1/65536 ≈ 1.5e-5
- 24-bit PCM quantization: 1/16777216 ≈ 6.0e-8
- 1e-9 is ~60 bits of precision (well beyond audio hardware)
"""
const AMPLITUDE_PRECISION = 1e-9  # Linear scale

"""
    TARGET_LATENCY_MS::Float64

Target end-to-end audio latency: 10.0 milliseconds

This is the goal for total latency from wave generation through audio device
output. Breakdown at 48kHz with 128-sample buffers:
- Buffer generation: < 1ms
- Audio callback processing: < 2ms
- OS audio driver: ~ 2-5ms
- Hardware output: ~ 1-2ms
- Total: < 10ms

Perceptual thresholds:
- < 10ms: Imperceptible for music playback
- 10-20ms: Perceptible but acceptable for most users
- > 20ms: Noticeable delay (problematic for live instruments)
"""
const TARGET_LATENCY_MS = 10.0  # Milliseconds

end # module AudioConstants
