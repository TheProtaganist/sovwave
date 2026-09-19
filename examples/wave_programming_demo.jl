"""
    Wave-Based Programming Demo in Julia
    Aetheria Quantum Audio System

Demonstrates how to write wave-based algorithms in Julia using `WaveProgram`:
1. Defining multi-channel variables/registers as spatial data points in wave space
2. Reading and writing data point values and properties like normal variables
3. Emitting continuous wave pulses with parameter modulation (frequency, amplitude, phase, speed, fractal dimensions)
4. Computing using binaural beat interference patterns
5. Executing wave logic and arithmetic gates
6. Getting real-time nanosecond calculation metrics and throughput
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Aetheria.Audio

function main()
    println("="^72)
    println(" 🌊 AETHERIA: WAVE-BASED COMPUTING IN JULIA 🌊")
    println("="^72)
    println()

    # ------------------------------------------------------------------------
    # Step 1: Initialize a Wave Program
    # ------------------------------------------------------------------------
    println("1. Initializing WaveProgram environment...")
    prog = WaveProgram(sample_rate=48000.0)

    # ------------------------------------------------------------------------
    # Step 2: Define Variables / Data Points in the Wave Field
    # ------------------------------------------------------------------------
    println("2. Defining data points (registers in wave space):")
    # Define points with positions in [0, 2π] and physical/computational properties
    def_point!(prog, :input_a, 0.0,       value=12.0, mass=1.0, charge=0.5)
    def_point!(prog, :input_b, π/2,       value=8.0,  mass=1.5, charge=-0.5)
    def_point!(prog, :carrier, π,         value=1.0,  frequency=432.0)
    def_point!(prog, :binaural_state, 3π/2, value=0.0, frequency=432.0)
    def_point!(prog, :result, 2π,         value=0.0)

    # Variable read/write works naturally with dictionary/indexing syntax:
    println("   • prog[:input_a] = ", prog[:input_a])
    println("   • prog[:input_b] = ", prog[:input_b])
    println("   • Mass of input_b = ", prog[:input_b, :mass])
    println()

    # ------------------------------------------------------------------------
    # Step 3: Define Wave-Based Instructions
    # ------------------------------------------------------------------------
    println("3. Scheduling wave computation operations:")

    # Operation A: Harmonic Wave Pulse with 432Hz Sacred Frequency & Fractal Depth
    println("   [A] Emit 432Hz wave pulse with fractal harmonics (depth=2, dim=1.618)")
    emit_wave!(prog, 432.0, 0.95; phase=0.0, speed=1.0, fractal_depth=2, fractal_dim=1.618, op=:multiply, target=[:input_a, :input_b])

    # Operation B: Binaural Beat Entrainment (Theta range 6.0 Hz for cognitive relaxation)
    println("   [B] Emit 6.0Hz theta binaural beat (carrier=432Hz) onto :binaural_state")
    emit_binaural!(prog, 6.0; carrier=432.0, op=:resonate, target=:binaural_state)

    # Operation C: Wave Logic Gate (Coherent Addition through wave superposition)
    println("   [C] Wave Gate: Coherent ADD(:input_a, :input_b) -> :result")
    wave_gate!(prog, :ADD, :input_a, :input_b, :result)

    # Operation D: Nonlinear Wave Activation (Sigmoidal wave response)
    println("   [D] Nonlinear Wave Gate: SIGMOID(:result) -> :result")
    wave_unary_gate!(prog, :SIGMOID, :result, :result)
    println()

    # ------------------------------------------------------------------------
    # Step 4: Execute using the Championship Winner Kernel (simd_fma_wave_hit!)
    # ------------------------------------------------------------------------
    println("4. Executing wave program (using tournament winner SIMD-FMA kernel)...")
    run!(prog)
    println()

    # ------------------------------------------------------------------------
    # Step 5: Inspect Computed Values
    # ------------------------------------------------------------------------
    println("5. Computed Values After Wave Interactions:")
    for name in [:input_a, :input_b, :carrier, :binaural_state, :result]
        pt = get_point(prog, name)
        println("   • :$name = $(round(pt.value, digits=6)) | hits: $(pt.computation_count) | energy: $(round(pt.accumulated_energy, digits=3)) J")
    end
    println()

    # ------------------------------------------------------------------------
    # Step 6: Calculation Time & Throughput Metrics
    # ------------------------------------------------------------------------
    println("6. Performance & Speed Metrics:")
    metrics(prog)
    println()

    # ------------------------------------------------------------------------
    # Step 7: Batch Wave Computing on N Data Points Grid
    # ------------------------------------------------------------------------
    println("7. High-Throughput Batch Computing (N = 10,000 points):")
    batch_prog = WaveProgram()
    def_points!(batch_prog, 10000, distribution=:fibonacci)
    emit_wave!(batch_prog, 528.0, 1.0; op=:multiply)
    emit_binaural!(batch_prog, 10.0; carrier=432.0, op=:resonate)
    run!(batch_prog)
    metrics(batch_prog)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
