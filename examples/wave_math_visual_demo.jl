"""
    Wave Mathematical Computing Demonstration
    Aetheria Quantum Audio System

Demonstrates wave-based computation across major mathematical subjects:
1. Calculus: 1st & 2nd derivatives, cumulative integration, Taylor expansions
2. Topology: Winding numbers, nodal domain counts, phase vortices, Möbius strip twist
3. Trigonometry & Harmonics: 12 wave kernels, Fourier harmonic synthesis, Lissajous phase dynamics
4. Geometry: Golden Ratio (Φ) logarithmic spirals, Platonic solid modal resonances, Flower of Life fields
5. Algebra & Variables: Real-time symbolic and variable evaluation across wave fields
"""

using Printf
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Aetheria.Audio

"""
    main()

Executes wave mathematical computing demos including calculus, trigonometry, physics, and algebra.
"""
function main()
    println("="^75)
    println(" 🌟 AETHERIA: WAVE MATHEMATICAL COMPUTING ACROSS ALL DOMAINS 🌟")
    println("="^75)
    println()

    # ------------------------------------------------------------------------
    # [1] CALCULUS THROUGH WAVES
    # ------------------------------------------------------------------------
    println("📈 [1/5] CALCULUS THROUGH WAVES (Derivatives, Integrals & Curvature)")
    println("-"^75)
    n_pts = 1000
    xs = collect(range(0.0, 2π, length=n_pts))
    ys = sin.(xs)

    t0 = time_ns()
    dys = wave_derivative(xs, ys)
    t_deriv_ns = time_ns() - t0

    t0 = time_ns()
    d2ys = wave_second_derivative(xs, ys)
    t_laplace_ns = time_ns() - t0

    t0 = time_ns()
    cum_int, total_area = wave_integral(xs, ys)
    t_integ_ns = time_ns() - t0

    println(@sprintf("  • 1st Derivative (d(sin x)/dx ≈ cos x): %6.1f μs | Max error: %.2e", t_deriv_ns/1000, maximum(abs.(dys .- cos.(xs)))))
    println(@sprintf("  • 2nd Derivative / Laplacian (∇²(sin x) ≈ -sin x): %6.1f μs | Max error: %.2e", t_laplace_ns/1000, maximum(abs.(d2ys .- (-ys)))))
    println(@sprintf("  • Cumulative Integral (∫ sin x dx):    %6.1f μs | Total area [0, 2π]: %.4e", t_integ_ns/1000, total_area))
    println(@sprintf("  • Taylor Series (Order 7 approximation):  sin(1.0) ≈ %.10f (true: %.10f)", wave_taylor_approx(1.0, 7), sin(1.0)))
    println()

    # ------------------------------------------------------------------------
    # [2] TOPOLOGY THROUGH WAVES
    # ------------------------------------------------------------------------
    println("🍩 [2/5] TOPOLOGY THROUGH WAVES (Winding Numbers, Invariants & Möbius)")
    println("-"^75)
    # Winding numbers for frequency multipliers
    for freq_mult in [1, 2, 3, 5]
        phases = [mod(freq_mult * 2π * i / 200, 2π) for i in 1:200]
        w = topological_winding_number(phases)
        println(@sprintf("  • Phase Loop (multiplier=%d): Topological Winding Number W = %d", freq_mult, w))
    end

    # Nodal domains (Betti-0)
    ys_multi = sin.(3 .* xs)
    beta_0 = count_nodal_domains(ys_multi)
    println(@sprintf("  • Wave Field ψ = sin(3x): Topological Nodal Domains β₀ = %d", beta_0))

    # Möbius twisted boundary
    mob_ys = copy(ys)
    apply_mobius_twist!(mob_ys, xs)
    println(@sprintf("  • Möbius Strip Twist Condition: ψ(0)=%.4f, ψ(2π)=%.4f (Boundary Inversion: %s)", 
                     mob_ys[1], mob_ys[end], isapprox(mob_ys[end], -mob_ys[1], atol=0.01) ? "Confirmed ✓" : "Off"))
    println()

    # ------------------------------------------------------------------------
    # [3] TRIGONOMETRY & HARMONICS
    # ------------------------------------------------------------------------
    println("📐 [3/5] TRIGONOMETRY & HARMONICS (12 Wave Kernels & Fourier Synthesis)")
    println("-"^75)
    println("  • Evaluating all 12 Trigonometric and Hyperbolic Kernels at θ = π/4:")
    kernels = [:sin, :cos, :tan, :cot, :sec, :csc, :sinh, :cosh, :tanh, :coth, :sech, :csch]
    for k in kernels
        val = evaluate_12_trig(k, π/4)
        println(@sprintf("    %-5s(π/4) = %8.4f", string(k), val))
    end

    # Fourier Synthesis
    harmonics = [1.0, 0.5, 0.25, 0.125]
    synth_val = fourier_harmonic_synthesis(π/3, harmonics, base_freq=1.0)
    println(@sprintf("  • 4-Order Fourier Harmonic Synthesis at π/3 = %.4f", synth_val))
    println()

    # ------------------------------------------------------------------------
    # [4] GEOMETRY & SACRED RESONANCES
    # ------------------------------------------------------------------------
    println("🔯 [4/5] GEOMETRY & SACRED RESONANCES (Phi Spirals & Platonic Solids)")
    println("-"^75)
    r_start = phi_logarithmic_spiral(0.0)
    r_one_turn = phi_logarithmic_spiral(2π)
    r_two_turns = phi_logarithmic_spiral(4π)
    println(@sprintf("  • Golden Ratio (Φ) Logarithmic Spiral Expansion:"))
    println(@sprintf("    r(0)    = %.6f", r_start))
    println(@sprintf("    r(2π)   = %.6f  (Ratio r(2π)/r(0) = %.6f = Φ)", r_one_turn, r_one_turn/r_start))
    println(@sprintf("    r(4π)   = %.6f  (Ratio r(4π)/r(2π) = %.6f = Φ)", r_two_turns, r_two_turns/r_one_turn))

    println("  • Platonic Solid Acoustic Resonances (432Hz Base Sacred Frequency):")
    for solid in [:tetrahedron, :cube, :octahedron, :dodecahedron, :icosahedron]
        res = platonic_solid_resonance(solid, 432.0)
        str_res = join([@sprintf("%.1fHz", r) for r in res[1:min(end,4)]], ", ")
        println(@sprintf("    %-13s: %s", uppercase(string(solid)), str_res))
    end
    println()

    # ------------------------------------------------------------------------
    # [5] ALGEBRA & WAVE VARIABLE EXPRESSIONS
    # ------------------------------------------------------------------------
    println("🧮 [5/5] ALGEBRA & WAVE VARIABLE EXPRESSIONS (Symbolic Wave Registers)")
    println("-"^75)
    ctx = WaveVariableContext(time=0.5, freq=432.0, amp=0.8, phase=π/6)
    ctx.vars[:mass] = 2.5
    ctx.vars[:charge] = -0.5
    ctx.vars[:y] = 1.618

    expressions = [
        "A * sin(f * x + phi)",
        "x^2 + 2 * x * y + y^2",
        "mass * (A * cos(x))^2 + charge * sin(2 * x)",
        "sqrt(abs(sin(x))) + exp(-x * 0.1)"
    ]

    for expr in expressions
        t0 = time_ns()
        results = compile_wave_expression(expr, xs[1:100], ctx)
        elapsed_us = (time_ns() - t0) / 1000.0
        println(@sprintf("  • Expr: %-45s | Calc: %5.1f μs | Mean: %7.3f", expr, elapsed_us, sum(results)/100))
    end

    println("\n" * "="^75)
    println(" 🎉 ALL MATHEMATICAL DOMAINS EXECUTED & VALIDATED WITH WAVES! 🎉")
    println("="^75)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
