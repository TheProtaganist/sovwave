"""
Unit tests for Audio.Core.AudioConstants module

Tests verify:
- 432Hz/440Hz ratio calculation
- PHI (golden ratio) value precision
- Fibonacci sequence correctness
"""

using Test

if !isdefined(Main, :Aetheria)
    include("../../src/Aetheria.jl")
    using .Aetheria
end
using .Aetheria.Audio.AudioConstants

@testset "AudioConstants" begin
    
    @testset "Sacred Tuning Constants" begin
        # Test A4 sacred frequency
        @test A4_SACRED == 432.0
        @test typeof(A4_SACRED) == Float64
        
        # Test A4 standard frequency
        @test A4_STANDARD == 440.0
        @test typeof(A4_STANDARD) == Float64
        
        # Test tuning ratio calculation
        expected_ratio = 432.0 / 440.0
        @test TUNING_RATIO ≈ expected_ratio atol=1e-15
        @test TUNING_RATIO ≈ 0.9818181818181818 atol=1e-15
        
        # Test ratio properties
        @test TUNING_RATIO < 1.0  # 432Hz is lower than 440Hz
        @test TUNING_RATIO > 0.9  # Reasonable range
        
        # Test ratio conversion round-trip
        freq_440 = 440.0
        freq_432 = freq_440 * TUNING_RATIO
        freq_440_back = freq_432 / TUNING_RATIO
        @test freq_440_back ≈ freq_440 atol=1e-12
    end
    
    @testset "Mathematical Constants" begin
        # Test PHI (golden ratio) to 12 decimal places (requirement)
        expected_phi = (1.0 + sqrt(5.0)) / 2.0
        @test PHI ≈ expected_phi atol=1e-12
        @test PHI ≈ 1.618033988749895 atol=1e-12
        
        # Test PHI properties
        @test PHI > 1.6 && PHI < 1.7  # Reasonable range
        @test PHI^2 ≈ PHI + 1.0 atol=1e-12  # Golden ratio property: φ² = φ + 1
        @test 1.0/PHI ≈ PHI - 1.0 atol=1e-12  # φ⁻¹ = φ - 1
        
        # Test Fibonacci sequence correctness
        @test length(FIBONACCI) == 12
        @test FIBONACCI[1] == 1
        @test FIBONACCI[2] == 1
        @test FIBONACCI[3] == 2
        @test FIBONACCI[4] == 3
        @test FIBONACCI[5] == 5
        @test FIBONACCI[6] == 8
        @test FIBONACCI[7] == 13
        @test FIBONACCI[8] == 21
        @test FIBONACCI[9] == 34
        @test FIBONACCI[10] == 55
        @test FIBONACCI[11] == 89
        @test FIBONACCI[12] == 144
        
        # Test Fibonacci recurrence relation: F(n) = F(n-1) + F(n-2)
        for i in 3:12
            @test FIBONACCI[i] == FIBONACCI[i-1] + FIBONACCI[i-2]
        end
        
        # Test Fibonacci convergence to PHI
        # For large n, F(n+1)/F(n) ≈ φ
        ratio_11_10 = FIBONACCI[11] / FIBONACCI[10]  # 89/55
        @test ratio_11_10 ≈ PHI atol=0.01  # Close approximation
        
        ratio_12_11 = FIBONACCI[12] / FIBONACCI[11]  # 144/89
        @test ratio_12_11 ≈ PHI atol=0.005  # Even closer
    end
    
    @testset "Sample Rate Constants" begin
        # Test sample rate values
        @test SAMPLE_RATE_44_1 == 44100
        @test SAMPLE_RATE_48 == 48000
        @test SAMPLE_RATE_96 == 96000
        @test SAMPLE_RATE_192 == 192000
        
        # Test types
        @test typeof(SAMPLE_RATE_44_1) == Int
        @test typeof(SAMPLE_RATE_48) == Int
        @test typeof(SAMPLE_RATE_96) == Int
        @test typeof(SAMPLE_RATE_192) == Int
        
        # Test relationships
        @test SAMPLE_RATE_96 == 2 * SAMPLE_RATE_48
        @test SAMPLE_RATE_192 == 4 * SAMPLE_RATE_48
        @test SAMPLE_RATE_192 == 2 * SAMPLE_RATE_96
        
        # Test Nyquist frequency coverage (> 20kHz for human hearing)
        @test SAMPLE_RATE_44_1 / 2 > 20000
        @test SAMPLE_RATE_48 / 2 > 20000
    end
    
    @testset "Buffer Configuration Constants" begin
        # Test buffer size values
        @test MIN_BUFFER_SIZE == 64
        @test DEFAULT_BUFFER_SIZE == 128
        @test MAX_BUFFER_SIZE == 2048
        
        # Test types
        @test typeof(MIN_BUFFER_SIZE) == Int
        @test typeof(DEFAULT_BUFFER_SIZE) == Int
        @test typeof(MAX_BUFFER_SIZE) == Int
        
        # Test size relationships
        @test MIN_BUFFER_SIZE < DEFAULT_BUFFER_SIZE
        @test DEFAULT_BUFFER_SIZE < MAX_BUFFER_SIZE
        
        # Test power-of-2 (optimal for FFT and ring buffers)
        @test ispow2(MIN_BUFFER_SIZE)
        @test ispow2(DEFAULT_BUFFER_SIZE)
        @test ispow2(MAX_BUFFER_SIZE)
        
        # Test practical ranges
        @test MIN_BUFFER_SIZE >= 32  # Below this is too inefficient
        @test MAX_BUFFER_SIZE <= 8192  # Above this is too much latency
    end
    
    @testset "Precision Constants" begin
        # Test precision values
        @test PHASE_PRECISION == 1e-12
        @test AMPLITUDE_PRECISION == 1e-9
        @test TARGET_LATENCY_MS == 10.0
        
        # Test types
        @test typeof(PHASE_PRECISION) == Float64
        @test typeof(AMPLITUDE_PRECISION) == Float64
        @test typeof(TARGET_LATENCY_MS) == Float64
        
        # Test precision is stricter than audio hardware
        @test PHASE_PRECISION < 1e-10  # Very tight phase tracking
        @test AMPLITUDE_PRECISION < 1e-6  # Well below 16-bit quantization (1/65536)
        
        # Test latency target is reasonable
        @test TARGET_LATENCY_MS > 0.0
        @test TARGET_LATENCY_MS < 50.0  # Human perception threshold
    end
    
    @testset "Constant Immutability" begin
        # Verify all constants are indeed constant (cannot be reassigned)
        # This is enforced by Julia's const keyword, but we document the intent
        @test isdefined(AudioConstants, :A4_SACRED)
        @test isdefined(AudioConstants, :PHI)
        @test isdefined(AudioConstants, :FIBONACCI)
    end
    
end
