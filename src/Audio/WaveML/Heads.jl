"""
    WaveML.Heads

Task-Specific Wave Heads for Sovwave.
Provides projection and decoding layers for mapping continuous wave field states
to diverse domain outputs:
- :classification (multinomial probabilities via harmonic softmax)
- :regression (continuous value predictions)
- :generation (next-wave packet / token distribution)
- :decision (type-safe structured categorical/boolean/rubric probabilistic decisions for Jev)
- :embedding (unit-sphere continuous wave embedding vectors)
- :reconstruction (multi-dimensional wave state reconstruction)
"""

using LinearAlgebra
using Statistics
using Printf
using Random

export WaveHead, create_head, apply_head, head_loss, mutate_head!

"""
    WaveHead

Task head projecting latent continuous wave representations to task-specific outputs.
"""
mutable struct WaveHead
    head_type::Symbol
    input_dim::Int
    output_dim::Int
    weights::Matrix{Float64}
    bias::Vector{Float64}
    frequencies::Vector{Float64}
    phases::Vector{Float64}
    temperature::Float64

    # Primary constructor initializing weights with Xavier harmonic distribution and carrier phases
    function WaveHead(
        head_type::Symbol,
        input_dim::Int,
        output_dim::Int;
        weights::Union{Nothing, Matrix{Float64}} = nothing,
        bias::Union{Nothing, Vector{Float64}} = nothing,
        frequencies::Union{Nothing, Vector{Float64}} = nothing,
        phases::Union{Nothing, Vector{Float64}} = nothing,
        temperature::Float64 = 1.0
    )
        w = if weights !== nothing
            weights
        else
            # Xavier harmonic initialization
            scale = sqrt(2.0 / Float64(input_dim + output_dim))
            randn(output_dim, input_dim) .* scale
        end

        b = if bias !== nothing
            bias
        else
            zeros(Float64, output_dim)
        end

        freqs = if frequencies !== nothing
            frequencies
        else
            432.0 .* (1.618033988749895 .^ ((0:(output_dim - 1)) ./ 12.0))
        end

        phs = if phases !== nothing
            phases
        else
            [mod(2π * i / output_dim, 2π) for i in 1:output_dim]
        end

        new(head_type, input_dim, output_dim, w, b, freqs, phs, temperature)
    end
end

"""
    create_head(head_type, input_dim, output_dim; kwargs...)

Factory function to construct a `WaveHead` for a given task.
Supported head types:
- `:classification`
- `:regression`
- `:generation`
- `:decision`
- `:embedding`
- `:reconstruction`
"""
function create_head(head_type::Symbol, input_dim::Int, output_dim::Int; kwargs...)::WaveHead
    supported = (:classification, :regression, :generation, :decision, :embedding, :reconstruction)
    if !(head_type in supported)
        throw(ArgumentError("Unsupported head_type :$(head_type). Supported: $(supported)"))
    end
    return WaveHead(head_type, input_dim, output_dim; kwargs...)
end

"""
    apply_head(head::WaveHead, x::AbstractVector{Float64}; kwargs...)

Project continuous wave vector `x` through `head` to produce task output.
"""
function apply_head(head::WaveHead, x::AbstractVector{Float64}; kwargs...)
    # Ensure dimension compatibility
    in_vec = if length(x) == head.input_dim
        x
    elseif length(x) > head.input_dim
        x[1:head.input_dim]
    else
        padded = zeros(Float64, head.input_dim)
        padded[1:length(x)] .= x
        padded
    end

    # Linear harmonic projection
    linear = head.weights * in_vec .+ head.bias

    # Apply phase interference modulation
    for i in 1:head.output_dim
        linear[i] += 0.05 * sin(head.phases[i])
    end

    if head.head_type == :classification || head.head_type == :decision
        # Harmonic Softmax
        t = max(1e-4, head.temperature)
        scaled = linear ./ t
        max_s = maximum(scaled)
        exp_s = exp.(clamp.(scaled .- max_s, -50.0, 50.0))
        probs = exp_s ./ (sum(exp_s) + 1e-12)
        return probs

    elseif head.head_type == :regression
        return linear

    elseif head.head_type == :embedding
        # L2-normalized continuous unit sphere embedding
        nrm = norm(linear)
        return nrm > 1e-12 ? (linear ./ nrm) : linear

    elseif head.head_type == :generation
        # Continuous wave distribution or log-probabilities
        t = max(1e-4, head.temperature)
        scaled = linear ./ t
        max_s = maximum(scaled)
        exp_s = exp.(clamp.(scaled .- max_s, -50.0, 50.0))
        return exp_s ./ (sum(exp_s) + 1e-12)

    elseif head.head_type == :reconstruction
        # Wave field reconstruction with hyperbolic tangent bounding
        return tanh.(linear)

    else
        return linear
    end
end

"""
    head_loss(head::WaveHead, predictions::AbstractVector{Float64}, targets::AbstractVector{Float64})

Compute task-appropriate loss between `predictions` and `targets`.
"""
function head_loss(head::WaveHead, predictions::AbstractVector{Float64}, targets::AbstractVector{Float64})::Float64
    if head.head_type == :classification || head.head_type == :decision || head.head_type == :generation
        # Cross entropy loss
        eps = 1e-12
        loss = 0.0
        n = min(length(predictions), length(targets))
        for i in 1:n
            loss -= targets[i] * log(clamp(predictions[i], eps, 1.0))
        end
        return loss
    elseif head.head_type == :embedding
        # Cosine distance loss: 1 - cos(theta)
        np = norm(predictions)
        nt = norm(targets)
        if np > 1e-12 && nt > 1e-12
            cos_sim = dot(predictions, targets) / (np * nt)
            return 1.0 - cos_sim
        else
            return 1.0
        end
    else
        # Mean Squared Error
        n = min(length(predictions), length(targets))
        return mean((predictions[1:n] .- targets[1:n]) .^ 2)
    end
end

"""
    mutate_head!(head::WaveHead; mutation_rate::Float64 = 0.1, mutation_scale::Float64 = 0.05)

Mutate the weights, biases, frequencies, and phases of the head.
"""
function mutate_head!(head::WaveHead; mutation_rate::Float64 = 0.1, mutation_scale::Float64 = 0.05)
    for i in eachindex(head.weights)
        if rand() < mutation_rate
            head.weights[i] += randn() * mutation_scale
        end
    end
    for i in eachindex(head.bias)
        if rand() < mutation_rate
            head.bias[i] += randn() * mutation_scale
        end
    end
    for i in eachindex(head.phases)
        if rand() < mutation_rate
            head.phases[i] = mod(head.phases[i] + randn() * mutation_scale, 2π)
        end
    end
    return head
end
