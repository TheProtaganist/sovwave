"""
Unit tests for Audio.Processing.WaveMath module.
Validates:
1. Calculus through waves: 1st derivative, 2nd derivative/Laplacian, cumulative integral, Taylor series
2. Topology through waves: Winding number, nodal domains, phase vortex, Möbius twist
3. Trigonometry & Harmonics: 12 kernels, Fourier synthesis, Lissajous coordinates
4. Geometry & Sacred: Golden ratio spiral, Platonic solid modal resonances, Flower of Life
5. Algebra & Variables: Expression parser & evaluator across wave registers (x, y, z, t)
"""

using Test

if !isdefined(Main, :Aetheria)
    include("../../src/Aetheria.jl")
    using .Aetheria
end
using .Aetheria.Audio

@testset "WaveMath Engine" begin

    @testset "Calculus Operations" begin
        # Test 1st derivative of sin(x) -> cos(x)
        xs = collect(range(0.0, 2π, length=500))
        ys = sin.(xs)
        dys = wave_derivative(xs, ys)
        
        # Check interior points
        for i in 50:50:450
            @test isapprox(dys[i], cos(xs[i]), atol=0.01)
        end

        # Test 2nd derivative of sin(x) -> -sin(x)
        d2ys = wave_second_derivative(xs, ys)
        for i in 50:50:450
            @test isapprox(d2ys[i], -sin(xs[i]), atol=0.01)
        end

        # Test cumulative integral of cos(x) -> sin(x)
        cos_ys = cos.(xs)
        cum_int, total_int = wave_integral(xs, cos_ys)
        @test isapprox(cum_int[1], 0.0, atol=1e-10)
        # Integral over [0, 2π] of cos(x) is 0
        @test isapprox(total_int, 0.0, atol=0.02)
        # Integral over [0, π/2] is sin(π/2) = 1.0
        idx_half_pi = round(Int, 500 * (π/2) / (2π))
        @test isapprox(cum_int[idx_half_pi], 1.0, atol=0.02)

        # Test Taylor approximation of sin(x)
        @test isapprox(wave_taylor_approx(0.5, 5), sin(0.5), atol=1e-5)
    end

    @testset "Topology Operations" begin
        # Closed loop phase with 1 full rotation (Winding = 1)
        phases_w1 = [mod(2π * i / 100, 2π) for i in 1:100]
        @test topological_winding_number(phases_w1) == 1

        # Closed loop phase with 2 full rotations (Winding = 2)
        phases_w2 = [mod(4π * i / 100, 2π) for i in 1:100]
        @test topological_winding_number(phases_w2) == 2

        # Flat phase (Winding = 0)
        phases_w0 = fill(π/4, 100)
        @test topological_winding_number(phases_w0) == 0

        # Nodal domains: sin(2x) over [0, 2π] has 2 positive domains
        xs = collect(range(0.0, 2π, length=200))
        ys_2x = sin.(2 .* xs)
        @test count_nodal_domains(ys_2x) == 2

        # Phase vortex
        q1 = phase_vortex_charge(1.0, 0.0, charge=1)
        @test isapprox(q1, 0.0, atol=1e-10)
        q2 = phase_vortex_charge(0.0, 1.0, charge=1)
        @test isapprox(q2, π/2, atol=1e-10)

        # Möbius twist boundary
        mob_ys = copy(ys_2x)
        apply_mobius_twist!(mob_ys, xs)
        @test length(mob_ys) == length(ys_2x)
    end

    @testset "Trigonometric Kernels & Harmonics" begin
        # Check all 12 kernels produce finite numbers
        test_kernels = [:sin, :cos, :tan, :cot, :sec, :csc, :sinh, :cosh, :tanh, :coth, :sech, :csch]
        for k in test_kernels
            val = evaluate_12_trig(k, 0.5)
            @test isfinite(val)
        end

        # Fourier synthesis
        harmonics = [1.0, 0.5, 0.25]
        f_synth = fourier_harmonic_synthesis(π/2, harmonics, base_freq=1.0)
        @test isapprox(f_synth, 1.0 * sin(π/2) + 0.5 * sin(π) + 0.25 * sin(3π/2), atol=1e-10)

        # Lissajous
        lx, ly = lissajous_coordinate(0.0, 1.0, 2.0, phase_delta=0.0)
        @test isapprox(lx, 0.0, atol=1e-10)
        @test isapprox(ly, 0.0, atol=1e-10)
    end

    @testset "Geometry & Sacred Resonances" begin
        # Phi spiral
        r0 = phi_logarithmic_spiral(0.0, a=2.0)
        @test isapprox(r0, 2.0, atol=1e-10)
        r2pi = phi_logarithmic_spiral(2π, a=1.0)
        @test isapprox(r2pi, 1.618033988749895, atol=1e-10)

        # Platonic solid modal resonances
        solids = [:tetrahedron, :cube, :octahedron, :dodecahedron, :icosahedron]
        for s in solids
            res = platonic_solid_resonance(s, 432.0)
            @test length(res) >= 4
            @test res[1] == 432.0
            @test all(r -> r >= 432.0, res)
        end

        # Flower of Life
        fol = flower_of_life_field(0.0, rings=6)
        @test isfinite(fol)
    end

    @testset "Algebra & Variable Expressions" begin
        ctx = WaveVariableContext(time=1.0, freq=2.0, amp=3.0, phase=0.5)
        ctx.vars[:x1] = 4.0
        ctx.vars[:x2] = 5.0

        # Basic arithmetic
        @test evaluate_algebraic_expression("x1 + x2", 0.0, ctx) == 9.0
        @test evaluate_algebraic_expression("x1 * x2", 0.0, ctx) == 20.0
        @test evaluate_algebraic_expression("x2 - x1", 0.0, ctx) == 1.0
        @test evaluate_algebraic_expression("x2 / 2", 0.0, ctx) == 2.5
        @test evaluate_algebraic_expression("x1 ^ 2", 0.0, ctx) == 16.0

        # Mathematical wave expressions
        val = evaluate_algebraic_expression("A * sin(f * x + phi)", 0.0, ctx)
        @test isapprox(val, 3.0 * sin(0.5), atol=1e-10)

        # Nonlinear functions
        @test evaluate_algebraic_expression("sqrt(x1)", 0.0, ctx) == 2.0
        @test evaluate_algebraic_expression("abs(-x2)", 0.0, ctx) == 5.0

        # Vectorized compile
        xs = [0.0, 1.0, 2.0]
        results = compile_wave_expression("x^2 + 1", xs, ctx)
        @test results == [1.0, 2.0, 5.0]
    end

end

println("All WaveMath tests passed! ✓")
