export QuantumWaveState, quantum_superpose, sample_categorical, quantum_interference!, normalize_probabilities!

"""
    QuantumWaveState

A quantum-inspired wave state with probability amplitude.
The probability of selecting this state is |amplitude|².

# Fields
- `wave::WaveFunction` — the underlying wave function
- `probability_amplitude::ComplexF64` — complex probability amplitude
- `collapsed::Bool` — whether this state has been selected/collapsed
"""
mutable struct QuantumWaveState
    wave::WaveFunction
    probability_amplitude::ComplexF64
    collapsed::Bool
end

"""
    normalize_probabilities!(states::Vector{QuantumWaveState})::Nothing

Ensures sum of |amplitude|² = 1 across all states.
"""
function normalize_probabilities!(states::Vector{QuantumWaveState})::Nothing
    total_prob = sum(abs2(s.probability_amplitude) for s in states)
    if total_prob > 0.0
        norm_factor = sqrt(total_prob)
        for s in states
            s.probability_amplitude /= norm_factor
        end
    end
    return nothing
end

"""
    sample_categorical(probabilities::Vector{Float64})::Int

Weighted random selection based on an array of probabilities.
Returns index of selected element.
"""
function sample_categorical(probabilities::Vector{Float64})::Int
    r = rand()
    cumulative = 0.0
    for (i, p) in enumerate(probabilities)
        cumulative += p
        if r <= cumulative
            return i
        end
    end
    return length(probabilities)
end

"""
    quantum_superpose(states::Vector{QuantumWaveState}, collapse_threshold::Float64=0.5)::WaveFunction

Probabilistic state selection based on |amplitude|².
If the maximum probability exceeds the collapse threshold, the state with
highest probability is deterministically selected. Otherwise, a weighted
random selection is performed.
"""
function quantum_superpose(states::Vector{QuantumWaveState}, collapse_threshold::Float64=0.5)::WaveFunction
    normalize_probabilities!(states)
    probs = [abs2(s.probability_amplitude) for s in states]
    
    max_prob = maximum(probs)
    if max_prob >= collapse_threshold
        idx = argmax(probs)
    else
        idx = sample_categorical(probs)
    end
    
    for i in eachindex(states)
        states[i].collapsed = (i == idx)
    end
    return states[idx].wave
end

"""
    quantum_interference!(output::Vector{Float64}, states::Vector{QuantumWaveState}, 
                           sample_rate::Float64)::Nothing

Combines states weighted by probability amplitudes into a single output buffer.
Each state's wave is generated and scaled by the real part of its amplitude.
"""
function quantum_interference!(output::Vector{Float64}, states::Vector{QuantumWaveState}, 
                                sample_rate::Float64)::Nothing
    fill!(output, 0.0)
    num_samples = length(output)
    temp_buffer = Vector{Float64}(undef, num_samples)
    
    for state in states
        weight = real(state.probability_amplitude)
        generate_buffer!(temp_buffer, state.wave, sample_rate)
        @inbounds for i in 1:num_samples
            output[i] += weight * temp_buffer[i]
        end
    end
    
    return nothing
end
