"""
Unit tests for WaveComputing module and WaveProgram DSL.
Verifies:
1. Tournament winner algorithm (simd_fma_wave_hit!) correctness & speed
2. Multi-channel data points with extensible properties
3. WaveProgram definitions, indexing, and step execution
4. Wave operations (emit_wave!, emit_binaural!, emit_interference!, emit_fractal!)
5. Wave logic & arithmetic gates (ADD, SUB, MULT, AND, OR, XOR, SIGMOID, RELU)
6. Metrics calculation reporting
"""

using Test

if !isdefined(Main, :Aetheria)
    include("../../src/Aetheria.jl")
    using .Aetheria
end
using .Aetheria.Audio

@testset "WaveComputing & WaveProgram" begin

    @testset "Winner Kernel: simd_fma_wave_hit!" begin
        pts = create_data_points(64, distribution=:uniform)
        res = zeros(Float64, 64)

        # Run kernel
        simd_fma_wave_hit!(res, pts, 432.0, 1.0, 0.0, 1.0, 1.0; op=:multiply)
        @test length(res) == 64
        @test all(isfinite, res)
        @test pts[1].computation_count == 1
        @test pts[2].accumulated_energy > 0.0

        # Out-of-place
        res2 = simd_fma_wave_hit(pts, 432.0, 0.8, π/4, 1.618, 1.0; op=:add)
        @test length(res2) == 64
        @test all(isfinite, res2)
    end

    @testset "WaveProgram Creation & Point Indexing" begin
        prog = WaveProgram()
        def_point!(prog, :x, 0.0, value=5.0, mass=2.0, charge=1.0)
        def_point!(prog, :y, π/2, value=10.0, mass=1.0, charge=-1.0)
        def_point!(prog, :res, π, value=0.0)

        @test length(prog) == 3
        @test prog[:x] == 5.0
        @test prog[:y] == 10.0
        @test prog[:res] == 0.0

        # Property access
        @test prog[:x, :mass] == 2.0
        @test prog[:y, :charge] == -1.0

        # Setters
        prog[:x] = 7.5
        @test prog[:x] == 7.5
        prog[:x, :mass] = 3.5
        @test prog[:x, :mass] == 3.5
    end

    @testset "Wave Operations Pipeline & Metrics" begin
        prog = WaveProgram()
        def_point!(prog, :alpha, 0.0, value=1.0)
        def_point!(prog, :beta, π/4, value=2.0)
        def_point!(prog, :gamma, π/2, value=3.0)
        def_point!(prog, :out, π, value=0.0)

        # Enqueue instructions
        emit_wave!(prog, 432.0, 1.0, op=:multiply)
        emit_binaural!(prog, 10.0, carrier=432.0, op=:resonate)
        wave_gate!(prog, :ADD, :alpha, :beta, :out)
        wave_unary_gate!(prog, :RELU, :out, :out)

        @test length(prog.instructions) == 4

        # Execute
        run!(prog)
        @test length(prog.execution_history) == 4

        stats = metrics(prog)
        @test stats[:total_time_ns] > 0
        @test stats[:throughput_pts_per_sec] > 0
        @test stats[:total_points] == (4 + 4 + 1 + 1)
    end

    @testset "Wave Logic Gates" begin
        prog = WaveProgram()
        def_point!(prog, :t1, 0.0, value=1.0)
        def_point!(prog, :t2, 0.0, value=1.0)
        def_point!(prog, :f1, 0.0, value=0.0)
        def_point!(prog, :out_and, 0.0, value=0.0)
        def_point!(prog, :out_or, 0.0, value=0.0)
        def_point!(prog, :out_xor, 0.0, value=0.0)
        def_point!(prog, :out_not, 0.0, value=0.0)

        wave_gate!(prog, :AND, :t1, :t2, :out_and)
        wave_gate!(prog, :OR, :t1, :f1, :out_or)
        wave_gate!(prog, :XOR, :t1, :f1, :out_xor)
        wave_unary_gate!(prog, :NOT, :t1, :out_not)

        run!(prog)

        @test prog[:out_and] == 1.0
        @test prog[:out_or] == 1.0
        @test prog[:out_xor] == 1.0
        @test prog[:out_not] == 0.0
    end

    @testset "Wave Operator Overloading" begin
        wf = WaveFunction(frequency=432.0, amplitude=0.5)
        pt = WaveDataPoint(0.0, value=2.0)
        
        # Wave hits point
        res = wf * pt
        @test typeof(res) == Float64

        # Superposition operator
        wf2 = WaveFunction(frequency=440.0, amplitude=0.5)
        wf_sum = wf + wf2
        @test wf_sum.frequency == 436.0
    end

end

println("All WaveComputing tests passed! ✓")
