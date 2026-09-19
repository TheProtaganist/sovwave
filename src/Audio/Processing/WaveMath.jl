"""
    WaveMath

Comprehensive Wave-Based Mathematical Computing Engine for Aetheria.
Implements mathematical operations computed directly through waves and data points:
1. **Calculus**: Wave derivatives (1st & 2nd order/Laplacian), wave integrals (Riemann/Trapezoid), Taylor series.
2. **Topology**: Winding number, phase singularities/vortices, nodal domains (Betti-0), Möbius twisted boundary conditions.
3. **Trigonometry & Hyperbolic**: 12 trigonometric/hyperbolic kernels, Fourier harmonic synthesis, Lissajous phase dynamics.
4. **Geometry & Sacred**: Golden ratio spiral wave geometry, Platonic solid resonances, Vesica Piscis & Flower of Life.
5. **Algebra & Variable Expressions**: Dynamic mathematical expression evaluator over wave registers and data points (x, y, z, t).
"""

using Printf
if isdefined(@__MODULE__, :AudioConstants)
    using .AudioConstants
end

export WaveMathDomain
export wave_derivative, wave_second_derivative, wave_integral, wave_laplacian, wave_taylor_approx
export topological_winding_number, count_nodal_domains, phase_vortex_charge, apply_mobius_twist!
export evaluate_12_trig, fourier_harmonic_synthesis, lissajous_coordinate
export phi_logarithmic_spiral, platonic_solid_resonance, flower_of_life_field
export evaluate_algebraic_expression, compile_wave_expression, WaveVariableContext

# ============================================================================
# [1] CALCULUS THROUGH WAVES
# ============================================================================

"""
    wave_derivative(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Vector{Float64}

Computes the 1st derivative of a wave field across spatial data points:
    ψ'(x_i) ≈ [ψ(x_{i+1}) - ψ(x_{i-1})] / [x_{i+1} - x_{i-1}]
Uses central difference for internal points and forward/backward differences at edges.
"""
function wave_derivative(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Vector{Float64}
    n = length(y_vals)
    n < 2 && return zeros(Float64, n)
    dy = Vector{Float64}(undef, n)

    # Forward difference at left boundary
    dx0 = x_vals[2] - x_vals[1]
    dy[1] = dx0 > 0 ? (y_vals[2] - y_vals[1]) / dx0 : 0.0

    # Central difference in interior (O(dx²) accuracy)
    @inbounds for i in 2:n-1
        dx = x_vals[i+1] - x_vals[i-1]
        dy[i] = dx > 0 ? (y_vals[i+1] - y_vals[i-1]) / dx : 0.0
    end

    # Backward difference at right boundary
    dxn = x_vals[n] - x_vals[n-1]
    dy[n] = dxn > 0 ? (y_vals[n] - y_vals[n-1]) / dxn : 0.0

    return dy
end

"""
    wave_second_derivative(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Vector{Float64}

Computes 2nd derivative / 1D Laplacian (curvature / kinetic energy operator):
    ∇²ψ(x_i) ≈ [ψ(x_{i+1}) - 2ψ(x_i) + ψ(x_{i-1})] / Δx²
"""
function wave_second_derivative(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Vector{Float64}
    n = length(y_vals)
    n < 3 && return zeros(Float64, n)
    d2y = Vector{Float64}(undef, n)

    @inbounds for i in 2:n-1
        dx_left = x_vals[i] - x_vals[i-1]
        dx_right = x_vals[i+1] - x_vals[i]
        dx_avg = 0.5 * (dx_left + dx_right)
        if dx_avg > 0
            d2y[i] = (y_vals[i+1] - 2.0 * y_vals[i] + y_vals[i-1]) / (dx_avg * dx_avg)
        else
            d2y[i] = 0.0
        end
    end

    d2y[1] = d2y[2]
    d2y[n] = d2y[n-1]
    return d2y
end

const wave_laplacian = wave_second_derivative

"""
    wave_integral(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Tuple{Vector{Float64}, Float64}

Computes cumulative Riemann-Trapezoid wave integral:
    I(x) = ∫₀ˣ ψ(t) dt
Returns (cumulative_integral_vector, total_integral).
"""
function wave_integral(x_vals::Vector{Float64}, y_vals::Vector{Float64})::Tuple{Vector{Float64}, Float64}
    n = length(y_vals)
    cum_int = zeros(Float64, n)
    total = 0.0

    @inbounds for i in 2:n
        dx = x_vals[i] - x_vals[i-1]
        trap_area = 0.5 * (y_vals[i] + y_vals[i-1]) * dx
        total += trap_area
        cum_int[i] = total
    end

    return cum_int, total
end

"""
    wave_taylor_approx(x::Float64, order::Int = 5; freq::Float64 = 1.0, phase::Float64 = 0.0)::Float64

Computes Taylor polynomial approximation of the wave at point x:
    sin(u) = u - u³/3! + u⁵/5! - u⁷/7! + ...
where u = mod2π(freq * x + phase) - π.
"""
function wave_taylor_approx(x::Float64, order::Int = 5; freq::Float64 = 1.0, phase::Float64 = 0.0)::Float64
    u = mod(freq * x + phase + π, 2π) - π
    u2 = u * u
    term = u
    sum_val = u

    # Alternating Taylor expansion
    for k in 1:order
        term = -term * u2 / ((2k) * (2k + 1))
        sum_val += term
    end
    return sum_val
end

# ============================================================================
# [2] TOPOLOGY THROUGH WAVES
# ============================================================================

"""
    topological_winding_number(phases::Vector{Float64})::Int

Computes the topological winding number (degree) of the phase trajectory along a closed loop:
    W = (1 / 2π) ∮ dθ = (1 / 2π) ∑ mod2π(θ_{i+1} - θ_i)
"""
function topological_winding_number(phases::Vector{Float64})::Int
    n = length(phases)
    n < 2 && return 0
    total_delta = 0.0

    @inbounds for i in 1:n-1
        dtheta = phases[i+1] - phases[i]
        # Map difference to [-π, π]
        dtheta = mod(dtheta + π, 2π) - π
        total_delta += dtheta
    end

    # Close the loop
    dtheta_close = mod(phases[1] - phases[n] + π, 2π) - π
    total_delta += dtheta_close

    return round(Int, total_delta / (2π))
end

"""
    count_nodal_domains(y_vals::Vector{Float64})::Int

Counts the topological 0th Betti number β₀ of positive nodal domains:
    Count of contiguous spatial regions where ψ(x) > 0.
"""
function count_nodal_domains(y_vals::Vector{Float64})::Int
    n = length(y_vals)
    count = 0
    in_positive_domain = false

    @inbounds for i in 1:n
        if y_vals[i] > 0.0
            if !in_positive_domain
                count += 1
                in_positive_domain = true
            end
        else
            in_positive_domain = false
        end
    end
    return count
end

"""
    phase_vortex_charge(x::Float64, y::Float64; charge::Int = 1)::Float64

Computes the phase field around a topological vortex core with quantized topological charge Q:
    θ(x, y) = Q · atan(y, x)
"""
function phase_vortex_charge(x::Float64, y::Float64; charge::Int = 1)::Float64
    angle = atan(y, x)
    return mod(charge * angle, 2π)
end

"""
    apply_mobius_twist!(vals::Vector{Float64})::Vector{Float64}

Applies a non-orientable Möbius strip boundary topology to a wave:
    ψ(x) transforms under half-twist: ψ(x) ← ψ(x) · cos(x / 2)
causing sign inversion at 2π: ψ(2π) = -ψ(0).
"""
function apply_mobius_twist!(vals::Vector{Float64}, x_vals::Vector{Float64})::Vector{Float64}
    @inbounds for i in eachindex(vals)
        twist_factor = cos(0.5 * x_vals[i])
        vals[i] *= twist_factor
    end
    return vals
end

# ============================================================================
# [3] TRIGONOMETRY & HARMONICS
# ============================================================================

"""
    evaluate_12_trig(sym::Symbol, angle::Float64)::Float64

Evaluates any of the 12 trigonometric/hyperbolic kernels with pole clamping for numerical stability:
[:sin, :cos, :tan, :cot, :sec, :csc, :sinh, :cosh, :tanh, :coth, :sech, :csch]
"""
function evaluate_12_trig(sym::Symbol, angle::Float64)::Float64
    sym == :sin  && return sin(angle)
    sym == :cos  && return cos(angle)
    sym == :tan  && return clamp(tan(angle), -50.0, 50.0)
    sym == :cot  && return clamp(cot(angle), -50.0, 50.0)
    sym == :sec  && return clamp(sec(angle), -50.0, 50.0)
    sym == :csc  && return clamp(csc(angle), -50.0, 50.0)
    sym == :sinh && return clamp(sinh(angle), -50.0, 50.0)
    sym == :cosh && return clamp(cosh(angle), 1.0, 50.0)
    sym == :tanh && return tanh(angle)
    sym == :coth && return clamp(coth(angle), -50.0, 50.0)
    sym == :sech && return sech(angle)
    sym == :csch && return clamp(csch(angle), -50.0, 50.0)
    return sin(angle)
end

"""
    fourier_harmonic_synthesis(x::Float64, harmonics::Vector{Float64}; base_freq::Float64 = 1.0, phase::Float64 = 0.0)::Float64

Synthesizes a wave from arbitrary Fourier coefficients:
    ψ(x) = ∑_{n=1}^N a_n · sin(n · base_freq · x + phase)
"""
function fourier_harmonic_synthesis(x::Float64, harmonics::Vector{Float64}; base_freq::Float64 = 1.0, phase::Float64 = 0.0)::Float64
    val = 0.0
    @inbounds for (n, amp) in enumerate(harmonics)
        val += amp * sin(n * base_freq * x + phase)
    end
    return val
end

"""
    lissajous_coordinate(t::Float64, freq_x::Float64, freq_y::Float64; phase_delta::Float64 = π/2)::Tuple{Float64, Float64}

Computes 2D parametric wave trajectory (Lissajous phase figure):
    (x(t), y(t)) = (sin(ω_x · t), sin(ω_y · t + δ))
"""
function lissajous_coordinate(t::Float64, freq_x::Float64, freq_y::Float64; phase_delta::Float64 = π/2)::Tuple{Float64, Float64}
    x = sin(freq_x * t)
    y = sin(freq_y * t + phase_delta)
    return (x, y)
end

# ============================================================================
# [4] GEOMETRY & SACRED RESONANCES
# ============================================================================

"""
    phi_logarithmic_spiral(theta::Float64; a::Float64 = 1.0)::Float64

Computes radius of the Golden Ratio (Spira Mirabilis) logarithmic spiral:
    r(θ) = a · Φ^{(θ / 2π)}
where Φ = (1 + √5)/2 ≈ 1.6180339887...
"""
function phi_logarithmic_spiral(theta::Float64; a::Float64 = 1.0)::Float64
    phi = 1.618033988749895
    return a * (phi ^ (theta / (2π)))
end

"""
    platonic_solid_resonance(solid::Symbol, base_freq::Float64 = 432.0)::Vector{Float64}

Returns modal acoustic resonance harmonics for the 5 Platonic solids:
- `:tetrahedron` (4 faces, 4 vertices): [1.0, 1.333, 2.0, 4.0]
- `:cube` (6 faces, 8 vertices): [1.0, 1.5, 2.0, 3.0, 6.0]
- `:octahedron` (8 faces, 6 vertices): [1.0, 1.414, 2.0, 2.828, 4.0]
- `:dodecahedron` (12 faces, 20 vertices): [1.0, 1.2, 1.618, 2.618, 5.0]
- `:icosahedron` (20 faces, 12 vertices): [1.0, 1.618, 2.618, 4.236, 6.854]
"""
function platonic_solid_resonance(solid::Symbol, base_freq::Float64 = 432.0)::Vector{Float64}
    ratios = if solid == :tetrahedron
        [1.0, 4/3, 2.0, 4.0]
    elseif solid == :cube
        [1.0, 1.5, 2.0, 3.0, 6.0]
    elseif solid == :octahedron
        [1.0, sqrt(2), 2.0, 2*sqrt(2), 4.0]
    elseif solid == :dodecahedron
        phi = 1.618033988749895
        [1.0, 1.2, phi, phi^2, 5.0]
    elseif solid == :icosahedron
        phi = 1.618033988749895
        [1.0, phi, phi^2, phi^3, phi^4]
    else
        [1.0, 2.0, 3.0, 4.0]
    end
    return [base_freq * r for r in ratios]
end

"""
    flower_of_life_field(theta::Float64; rings::Int = 6, phase::Float64 = 0.0)::Float64

Computes 6-fold sacred geometric hexagonal interference pattern (Flower of Life):
    ψ_{FOL}(θ) = (1/N) ∑_{k=0}^{rings-1} cos(θ - 2πk / 6 + phase)
"""
function flower_of_life_field(theta::Float64; rings::Int = 6, phase::Float64 = 0.0)::Float64
    val = 0.0
    for k in 0:rings-1
        angle_k = theta - (2π * k / 6.0) + phase
        val += cos(angle_k)
    end
    return val / max(rings, 1)
end

# ============================================================================
# [5] ALGEBRA & WAVE VARIABLE EXPRESSION EVALUATOR
# ============================================================================

"""
    WaveVariableContext

Execution context holding named registers/variables and wave parameters
for evaluating algebraic expressions across a wave field.
"""
mutable struct WaveVariableContext
    vars::Dict{Symbol, Float64}
    time::Float64
    frequency::Float64
    amplitude::Float64
    phase::Float64

    function WaveVariableContext(; time::Float64 = 0.0, freq::Float64 = 432.0, amp::Float64 = 1.0, phase::Float64 = 0.0)
        ctx = new(Dict{Symbol, Float64}(), time, freq, amp, phase)
        ctx.vars[:t] = time
        ctx.vars[:f] = freq
        ctx.vars[:A] = amp
        ctx.vars[:phi] = phase
        ctx.vars[:pi] = π
        ctx.vars[:PHI] = 1.618033988749895
        return ctx
    end
end

"""
    evaluate_algebraic_expression(expr_str::String, x::Float64, ctx::WaveVariableContext)::Float64

Safely parses and evaluates an algebraic expression involving spatial variable `x` and context variables:
Supports:
- Operations: `+`, `-`, `*`, `/`, `^`
- Functions: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`
- Variables: `x`, `t`, `f`, `A`, `phi`, `pi`, `PHI`, or custom keys in `ctx.vars`.
"""
function evaluate_algebraic_expression(expr_str::String, x::Float64, ctx::WaveVariableContext)::Float64
    clean_str = strip(expr_str)
    isempty(clean_str) && return 0.0

    # Build evaluation environment
    env = Dict{Symbol, Any}(
        :x => x,
        :t => ctx.time,
        :f => ctx.frequency,
        :A => ctx.amplitude,
        :phi => ctx.phase,
        :pi => π,
        :π => π,
        :PHI => 1.618033988749895,
        :sin => sin,
        :cos => cos,
        :tan => (u -> clamp(tan(u), -50.0, 50.0)),
        :sinh => sinh,
        :cosh => cosh,
        :tanh => tanh,
        :exp => (u -> clamp(exp(u), 0.0, 1e6)),
        :log => (u -> log(max(abs(u), 1e-12))),
        :sqrt => (u -> sqrt(max(u, 0.0))),
        :abs => abs
    )
    for (k, v) in ctx.vars
        env[k] = v
    end

    try
        parsed = Meta.parse(clean_str)
        return Float64(_eval_ast(parsed, env))
    catch
        # Fallback to simple wave on parse error
        return ctx.amplitude * sin(ctx.frequency * x + ctx.phase)
    end
end

"""Internal recursive AST evaluator for safe mathematical expression execution."""
function _eval_ast(node::Any, env::Dict{Symbol, Any})
    if node isa Number
        return Float64(node)
    elseif node isa Symbol
        return get(env, node, 0.0)
    elseif node isa Expr
        if node.head == :call
            fn_sym = node.args[1]
            args = [_eval_ast(arg, env) for arg in node.args[2:end]]

            # Standard binary operators
            if fn_sym == :+
                return length(args) == 1 ? args[1] : sum(args)
            elseif fn_sym == :-
                return length(args) == 1 ? -args[1] : args[1] - args[2]
            elseif fn_sym == :*
                return prod(args)
            elseif fn_sym == :/
                return abs(args[2]) > 1e-15 ? args[1] / args[2] : 0.0
            elseif fn_sym == :^
                return args[1] ^ args[2]
            elseif haskey(env, fn_sym) && env[fn_sym] isa Function
                fn = env[fn_sym]
                return fn(args...)
            else
                return 0.0
            end
        end
    end
    return 0.0
end

"""
    compile_wave_expression(expr_str::String, x_vals::Vector{Float64}, ctx::WaveVariableContext)::Vector{Float64}

Vectorized evaluation of an algebraic expression over an array of data points.
"""
function compile_wave_expression(expr_str::String, x_vals::Vector{Float64}, ctx::WaveVariableContext)::Vector{Float64}
    n = length(x_vals)
    results = Vector{Float64}(undef, n)
    for i in 1:n
        results[i] = evaluate_algebraic_expression(expr_str, x_vals[i], ctx)
    end
    return results
end

# ============================================================================
# [6] MATHEMATICAL DOMAIN ENUM & DESCRIPTORS
# ============================================================================

@enum WaveMathDomain begin
    DOMAIN_CALCULUS = 1
    DOMAIN_TOPOLOGY = 2
    DOMAIN_TRIGONOMETRY = 3
    DOMAIN_GEOMETRY = 4
    DOMAIN_ALGEBRA = 5
end
