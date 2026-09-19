"""
Unit tests and property tests for WaveFunction

Tests:
- Constructor validation
- Phase tracking correctness
- Maximum precision helpers
- Phase coherence across buffers
"""

using Test

if !isdefined(Main, :Aetheria)
    include("../../src/Aetheria.jl")
    using .Aetheria
end
using .Aetheria.Audio

@testset "WaveFunction" begin
    
    @testset "Constructor and Validation" begin
        # Valid construction
        wf = WaveFunction(frequency=432.0)
        @test wf.frequency == 432.0
        @test wf.amplitude == 1.0
        @test wf.phase == 0.0
        @test wf.trig_func == :sin
        @test wf.sample_rate == 48000
        @test wf.fractal_depth == 0
        @test wf.fractal_dimension == 2.0
        @test wf.current_phase[] == 0.0
        
        # Custom parameters
        wf2 = WaveFunction(
            frequency=440.0,
            amplitude=0.5,
            phase=π/4,
            trig_func=:cos,
            sample_rate=96000,
            fractal_depth=5,
            fractal_dimension=1.618
        )
        @test wf2.frequency == 440.0
        @test wf2.amplitude == 0.5
        @test wf2.phase ≈ π/4
        @test wf2.trig_func == :cos
        @test wf2.fractal_depth == 5
        @test wf2.fractal_dimension ≈ 1.618
        
        # Phase normalization
        wf3 = WaveFunction(frequency=432.0, phase=3π)
        @test wf3.phase ≈ π atol=1e-15  # 3π mod 2π = π
        
        # Validation errors
        @test_throws ErrorException WaveFunction(frequency=-1.0)  # Negative frequency
        @test_throws ErrorException WaveFunction(frequency=0.0)   # Zero frequency
        @test_throws ErrorException WaveFunction(frequency=30000.0, sample_rate=48000)  # Above Nyquist
        @test_throws ErrorException WaveFunction(frequency=432.0, amplitude=-0.1)  # Negative amplitude
        @test_throws ErrorException WaveFunction(frequency=432.0, amplitude=1.5)   # Amplitude > 1
        @test_throws ErrorException WaveFunction(frequency=432.0, trig_func=:invalid)  # Bad trig func
        @test_throws ErrorException WaveFunction(frequency=432.0, fractal_depth=-1)    # Negative depth
        @test_throws ErrorException WaveFunction(frequency=432.0, fractal_depth=11)    # Too deep
        @test_throws ErrorException WaveFunction(frequency=432.0, fractal_dimension=0.5)  # Too small
        @test_throws ErrorException WaveFunction(frequency=432.0, fractal_dimension=3.5)  # Too large
    end
    
    @testset "Helper Functions - Maximum Precision" begin
        wf = WaveFunction(frequency=432.0, sample_rate=48000)
        
        # Angular frequency: ω = 2πf
        ω = angular_frequency(wf)
        expected_ω = 2 * π * 432.0
        @test ω ≈ expected_ω atol=1e-15
        
        # Period: T = 1/f
        T = period(wf)
        expected_T = 1.0 / 432.0
        @test T ≈ expected_T atol=1e-15
        
        # Wavelength: λ = v/f
        λ = wavelength(wf, 343.0)
        expected_λ = 343.0 / 432.0
        @test λ ≈ expected_λ atol=1e-15
        
        # Samples per cycle
        spc = samples_per_cycle(wf)
        expected_spc = 48000.0 / 432.0
        @test spc ≈ expected_spc atol=1e-15
        
        # Phase advance per sample
        Δφ = phase_advance_per_sample(wf)
        expected_Δφ = 2 * π * 432.0 / 48000.0
        @test Δφ ≈ expected_Δφ atol=1e-15
    end
    
    @testset "Phase Tracking - update_phase!" begin
        wf = WaveFunction(frequency=432.0, sample_rate=48000)
        @test wf.current_phase[] == 0.0
        
        # Advance by 1 sample
        update_phase!(wf, 1)
        expected_phase_1 = 2 * π * 432.0 / 48000.0
        @test wf.current_phase[] ≈ expected_phase_1 atol=1e-15
        
        # Advance by more samples
        reset_phase!(wf)
        update_phase!(wf, 1024)
        expected_phase_1024 = mod(2 * π * 432.0 * 1024 / 48000.0, 2π)
        @test wf.current_phase[] ≈ expected_phase_1024 atol=1e-15
        
        # Test normalization (phase should wrap at 2π)
        reset_phase!(wf)
        update_phase!(wf, 10000)  # Large number to cause wrap
        @test 0.0 <= wf.current_phase[] < 2π
    end
    
    @testset "Phase Tracking - mod2pi_precise" begin
        # Test normalization
        @test mod2pi_precise(0.0) ≈ 0.0 atol=1e-15
        @test mod2pi_precise(Float64(π)) ≈ Float64(π) atol=1e-15
        @test mod2pi_precise(Float64(2π)) ≈ 0.0 atol=1e-14  # 2π wraps to 0
        @test mod2pi_precise(Float64(3π)) ≈ Float64(π) atol=1e-15
        @test mod2pi_precise(Float64(-π)) ≈ Float64(π) atol=1e-15
        @test mod2pi_precise(7.5) ≈ mod(7.5, 2π) atol=1e-15
        
        # Test precision
        large_angle = Float64(100π)
        normalized = mod2pi_precise(large_angle)
        @test 0.0 <= normalized < 2π
        @test normalized ≈ 0.0 atol=1e-14  # 100π is even multiple, should be 0
    end
    
    @testset "Phase Tracking - phase_distance" begin
        # Adjacent phases
        @test phase_distance(0.0, 0.1) ≈ 0.1 atol=1e-15
        @test phase_distance(0.1, 0.0) ≈ 0.1 atol=1e-15
        
        # Opposite phases
        @test phase_distance(0.0, Float64(π)) ≈ Float64(π) atol=1e-15
        
        # Wrap-around case (important!)
        @test phase_distance(0.1, Float64(2π) - 0.1) ≈ 0.2 atol=1e-15
        @test phase_distance(Float64(2π) - 0.1, 0.1) ≈ 0.2 atol=1e-15
        
        # Identical phases
        @test phase_distance(1.5, 1.5) ≈ 0.0 atol=1e-15
        
        # Maximum distance (π)
        for phase in [0.0, Float64(π)/2, Float64(π), Float64(3π)/2]
            opposite = mod2pi_precise(phase + Float64(π))
            @test phase_distance(phase, opposite) ≈ Float64(π) atol=1e-15
        end
    end
    
    @testset "Phase Tracking - reset_phase!" begin
        wf = WaveFunction(frequency=432.0)
        
        # Advance then reset
        update_phase!(wf, 1000)
        @test wf.current_phase[] != 0.0
        reset_phase!(wf)
        @test wf.current_phase[] ≈ 0.0 atol=1e-15
        
        # Reset to custom phase
        reset_phase!(wf, π/2)
        @test wf.current_phase[] ≈ π/2 atol=1e-15
        
        # Reset with normalization
        reset_phase!(wf, 3π)
        @test wf.current_phase[] ≈ π atol=1e-15
    end
    
    @testset "Property Test: Phase Continuity Across Buffers" begin
        # Test that phase advances correctly across multiple buffers
        frequencies = [100.0, 432.0, 1000.0, 5000.0]
        sample_rates = [44100, 48000, 96000]
        buffer_sizes = [64, 128, 256, 512, 1024]
        
        for freq in frequencies
            for sr in sample_rates
                # Skip if above Nyquist
                if freq >= sr / 2
                    continue
                end
                
                wf = WaveFunction(frequency=freq, sample_rate=sr)
                
                for buf_size in buffer_sizes
                    # Generate multiple buffers
                    num_buffers = 10
                    initial_phase = wf.current_phase[]
                    
                    for _ in 1:num_buffers
                        update_phase!(wf, buf_size)
                    end
                    
                    total_samples = num_buffers * buf_size
                    
                    # Verify final phase matches expected
                    expected_phase = mod(initial_phase + 2π * freq * total_samples / sr, 2π)
                    
                    # Use phase_distance to handle wrap-around correctly
                    distance = phase_distance(wf.current_phase[], expected_phase)
                    @test distance < 1e-10  # Very small phase error
                end
                
                # Reset for next sample rate
                reset_phase!(wf)
            end
        end
    end
    
    @testset "Property Test: Phase Advance Formula" begin
        # Property: Δφ = 2πf·N/sr for any frequency, sample count, and sample rate
        
        test_cases = [
            (freq=432.0, sr=48000, samples=1024),
            (freq=440.0, sr=44100, samples=512),
            (freq=1000.0, sr=96000, samples=256),
            (freq=100.0, sr=48000, samples=2048),
        ]
        
        for tc in test_cases
            wf = WaveFunction(frequency=tc.freq, sample_rate=tc.sr)
            reset_phase!(wf)
            
            update_phase!(wf, tc.samples)
            
            expected = mod(2π * tc.freq * tc.samples / tc.sr, 2π)
            @test wf.current_phase[] ≈ expected atol=1e-12
        end
    end
    
end

println("All WaveFunction tests passed! ✓")
