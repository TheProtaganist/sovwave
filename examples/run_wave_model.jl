"""
    Wave Computing Model Runner
    Aetheria Quantum Audio Foundation

Demonstrates:
1. Wave continuous computation with data points carrying multi-channel information
2. Parameter modulation: frequency, amplitude, speed, phase shift, fractal dimension & depth
3. All 12 trigonometric wave kernels
4. Binaural beats synthesis across 5 brainwave presets (Delta, Theta, Alpha, Beta, Gamma)
5. Wave superposition & phase-preserving normalization
6. Quantum processor state collapse & probability amplitude interference
7. Real-time calculation performance metrics (ns, ns/point, throughput, memory)
8. 64-Algorithm Tournament in 8 rounds of 8 algorithms to determine optimal algorithm
"""

using Printf

# Activate project
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "../src/Aetheria.jl"))
using .Aetheria.Audio
using .Aetheria.Audio.AudioConstants

include(joinpath(@__DIR__, "../test/audio/wave_computation_competition.jl"))

"""
    run_full_wave_model()

Demonstrates full end-to-end wave computation, tuning, harmonics, and audio synthesis.
"""
function run_full_wave_model()
    println("="^75)
    println(" 🌟 AETHERIA QUANTUM WAVE COMPUTATION ENGINE & AUDIO MODEL 🌟 ")
    println("="^75)
    println()

    # ------------------------------------------------------------------------
    # 1. SACRED TUNING & HARMONICS
    # ------------------------------------------------------------------------
    println("🎵 [1/6] SACRED TUNING (432Hz) & GOLDEN HARMONICS")
    println("-"^75)
    f_sacred = A4_SACRED
    f_std = A4_STANDARD
    println(@sprintf("  Reference Sacred Pitch (A4):    %.2f Hz", f_sacred))
    println(@sprintf("  Standard ISO Pitch (A4):        %.2f Hz", f_std))
    println(@sprintf("  Sacred/Standard Tuning Ratio:   %.12f", TUNING_RATIO))
    println(@sprintf("  Golden Ratio (PHI):             %.15f", PHI))
    println("  Fibonacci Sequence:             ", FIBONACCI)

    midi_c4 = midi_to_freq_432(60)
    midi_a4 = midi_to_freq_432(69)
    println(@sprintf("  MIDI C4 (Middle C in 432Hz):    %.4f Hz", midi_c4))
    println(@sprintf("  MIDI A4 (432Hz concert pitch):  %.4f Hz", midi_a4))

    println("  Golden Ratio Harmonics (f₀ · φⁿ):")
    for order in 0:4
        h = golden_harmonic(f_sacred, order)
        println(@sprintf("    Order %d (432 · φ^%d):             %.4f Hz", order, order, h))
    end

    fib_h = fibonacci_harmonics(f_sacred, 5)
    println("  Fibonacci Ratio Harmonics:")
    for (i, fh) in enumerate(fib_h)
        println(@sprintf("    Level %d (F(%d)/F(%d)):            %.4f Hz", i, i+1, i, fh))
    end
    println()

    # ------------------------------------------------------------------------
    # 2. CONTINUOUS WAVE REPRESENTATION & 12 TRIG FUNCTIONS
    # ------------------------------------------------------------------------
    println("🌊 [2/6] CONTINUOUS WAVE ENGINE & 12 TRIGONOMETRIC KERNELS")
    println("-"^75)
    sample_rate = 48000.0
    buffer_len = 1024
    buf = zeros(Float64, buffer_len)

    println("  Evaluating 12 trigonometric wave kernels (1024 samples @ 48kHz):")
    trig_kernels = [:sin, :cos, :tan, :cot, :sec, :csc, :sinh, :cosh, :tanh, :coth, :sech, :csch]
    
    for tk in trig_kernels
        wf_tk = WaveFunction(frequency=432.0, amplitude=0.8, trig_func=tk, sample_rate=48000)
        t_start = time_ns()
        generate_buffer!(buf, wf_tk, sample_rate)
        elapsed_us = (time_ns() - t_start) / 1000.0
        
        # summary stats
        clean_buf = filter(isfinite, buf)
        min_v = isempty(clean_buf) ? 0.0 : minimum(clean_buf)
        max_v = isempty(clean_buf) ? 0.0 : maximum(clean_buf)
        println(@sprintf("    %-6s | Time: %6.1f μs | Min: %8.3f | Max: %8.3f | Phase: %.4f rad", 
                         tk, elapsed_us, min_v, max_v, wf_tk.current_phase[]))
    end
    println()

    # ------------------------------------------------------------------------
    # 3. BINAURAL BEATS ENGINE (Brainwave Presets)
    # ------------------------------------------------------------------------
    println("🧠 [3/6] BINAURAL BEATS ENGINE (Brainwave Entrainment)")
    println("-"^75)
    stereo_buf = zeros(Float64, 2, buffer_len)
    presets = [:delta, :theta, :alpha, :beta, :gamma]

    for pr in presets
        bb = create_brainwave_beat(pr, carrier=432.0, sample_rate=48000)
        t_start = time_ns()
        generate_binaural!(stereo_buf, bb)
        dur_us = (time_ns() - t_start) / 1000.0
        
        f_left = bb.carrier_freq - bb.beat_freq / 2.0
        f_right = bb.carrier_freq + bb.beat_freq / 2.0
        println(@sprintf("    Preset %-7s | Beat: %4.1f Hz | Left: %5.1f Hz | Right: %5.1f Hz | Calc: %5.1f μs",
                         string(pr), bb.beat_freq, f_left, f_right, dur_us))
    end
    println()

    # ------------------------------------------------------------------------
    # 4. WAVE SUPERPOSITION & QUANTUM PROCESSOR
    # ------------------------------------------------------------------------
    println("⚛️  [4/6] WAVE SUPERPOSITION & QUANTUM STATE COLLAPSE")
    println("-"^75)
    
    # 16-wave harmonic superposition
    waves = [WaveFunction(frequency=432.0 * (1.0 + 0.05 * i), amplitude=1.0/(i+1), sample_rate=48000) for i in 1:16]
    super_out = zeros(Float64, buffer_len)
    t_start = time_ns()
    superpose!(super_out, waves, sample_rate)
    norm_factor = normalize_preserve_phase!(super_out)
    super_time_us = (time_ns() - t_start) / 1000.0
    println(@sprintf("  16-Wave Superposition: %d samples in %.2f μs (Norm factor: %.4f)", buffer_len, super_time_us, norm_factor))

    # Quantum Wave States
    q_states = [
        QuantumWaveState(WaveFunction(frequency=432.0), 0.6 + 0.0im, false),
        QuantumWaveState(WaveFunction(frequency=528.0), 0.5 + 0.3im, false),
        QuantumWaveState(WaveFunction(frequency=639.0), 0.2 + 0.1im, false),
    ]
    normalize_probabilities!(q_states)
    collapsed_wave = quantum_superpose(q_states, 0.7)
    println(@sprintf("  Quantum Wave Collapse: Selected freq = %.1f Hz (Collapse threshold = 0.7)", collapsed_wave.frequency))
    println()

    # ------------------------------------------------------------------------
    # 5. DATA POINT COMPUTATION (N points x Properties, Sweeping Parameters)
    # ------------------------------------------------------------------------
    println("📊 [5/6] WAVE COMPUTING ON DATA POINTS (Parameter Modulation)")
    println("-"^75)
    
    # Create N data points with X properties
    n_points_list = [64, 256, 1024, 4096]
    interactions = [:multiply, :add, :resonate, :interference, :energy, :quantum]

    for np in n_points_list
        points = create_data_points(np, distribution=:fibonacci)
        
        # Modulate wave parameters
        freq = 432.0
        amp = 0.85
        speed = 1.2
        phase_shift = π / 4
        fd = 1.618
        depth = 3

        wf = WaveFunction(frequency=freq, amplitude=amp, phase=phase_shift, 
                          fractal_depth=depth, fractal_dimension=fd, sample_rate=48000)

        t_start = time_ns()
        res = evaluate_wave_at_points(wf, points, interaction=:multiply)
        t_elapsed = time_ns() - t_start
        
        ns_per_pt = t_elapsed / np
        pts_per_sec = np / (t_elapsed / 1e9)
        
        println(@sprintf("  N = %-4d points | Calc: %8.2f μs | %6.1f ns/pt | %10.2e pts/sec | Inter: :multiply",
                         np, t_elapsed / 1000.0, ns_per_pt, pts_per_sec))
    end

    println("\n  Testing Wave Interaction Operations on 1024 Points:")
    pts1024 = create_data_points(1024, distribution=:uniform)
    wf_test = WaveFunction(frequency=432.0, amplitude=0.9, phase=0.0, sample_rate=48000)
    for inter in interactions
        reset_data_points!(pts1024)
        t_start = time_ns()
        res = evaluate_wave_at_points(wf_test, pts1024, interaction=inter)
        dur_us = (time_ns() - t_start) / 1000.0
        stats = data_point_summary(pts1024)
        println(@sprintf("    %-13s | Time: %6.2f μs | Mean Val: %8.4f | Max: %8.4f | Energy: %8.2f",
                         string(inter), dur_us, stats[:mean_value], stats[:max_value], stats[:total_energy]))
    end
    println()

    # ------------------------------------------------------------------------
    # 6. 64-ALGORITHM COMPETITION (8 Rounds x 8 Algorithms)
    # ------------------------------------------------------------------------
    println("🏆 [6/6] 64-ALGORITHM WAVE COMPUTATION TOURNAMENT")
    println("-"^75)
    champ, champ_score = run_competition()

    println("\n" * "="^75)
    println(" ✅ ALL WAVE COMPUTING MODEL COMPONENTS EXECUTED SUCCESSFULLY! ")
    println("="^75)
    return champ, champ_score
end

# Run model
if abspath(PROGRAM_FILE) == @__FILE__
    run_full_wave_model()
end
