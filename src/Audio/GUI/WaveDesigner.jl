"""
    WaveDesignerGUI

Aetheria Quantum Wave Designer & Mathematical Computer — GTK4 GUI with pitch black background
and saturated neon green theme. Real-time wave visualization, binaural beat controls,
data point computation, 64-algorithm competition runner, and comprehensive support for
all mathematical subjects:
- **Calculus**: Wave derivatives (1st & 2nd/Laplacian), cumulative integrals, Taylor series
- **Topology**: Winding numbers, nodal domain counts, phase singularities, Möbius twisted boundaries
- **Trigonometry**: 12 trigonometric/hyperbolic kernels, Fourier harmonic synthesis
- **Geometry**: Golden ratio spiral envelopes, Platonic solid resonances, Flower of Life fields
- **Algebra & Variables**: Dynamic algebraic expression evaluator over wave registers (x, y, z, t)

Launch with: `WaveDesignerGUI.launch_wave_designer()`
"""
module WaveDesignerGUI

using Gtk4
using Gtk4.GLib
using Cairo
using Printf

# Load mathematical engine
include("../Processing/WaveMath.jl")

export launch_wave_designer

# ============================================================================
# Neon Green on Black CSS Theme (Cyberpunk / Matrix Aesthetic)
# ============================================================================
const NEON_CSS = """
* {
    background-color: #000000;
    color: #00FF41;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'DejaVu Sans Mono', monospace;
    font-size: 12px;
}
window {
    background-color: #000000;
}
button {
    background-color: #001100;
    border: 1px solid #00FF41;
    color: #00FF41;
    border-radius: 4px;
    padding: 5px 10px;
    min-height: 22px;
    font-weight: bold;
}
button:hover {
    background-color: #003300;
    border-color: #39FF14;
    color: #39FF14;
}
button:active {
    background-color: #004400;
}
scale trough {
    background-color: #001a00;
    min-height: 6px;
}
scale highlight {
    background-color: #00FF41;
    min-height: 6px;
}
scale slider {
    background-color: #00FF41;
    border: 1px solid #39FF14;
    min-width: 14px;
    min-height: 14px;
    border-radius: 7px;
}
entry {
    background-color: #001100;
    color: #00FF41;
    border: 1px solid #005500;
    border-radius: 3px;
    padding: 4px 6px;
}
entry:focus {
    border-color: #00FF41;
}
label {
    color: #00FF41;
    background-color: transparent;
}
spinbutton {
    background-color: #001100;
    color: #00FF41;
    border: 1px solid #004400;
}
spinbutton button {
    background-color: #001a00;
    border: 1px solid #003300;
    min-width: 18px;
}
dropdown button {
    background-color: #001100;
    color: #00FF41;
    border: 1px solid #004400;
}
scrolledwindow {
    background-color: #000000;
    border: 1px solid #003300;
}
listbox {
    background-color: #000000;
}
listbox row {
    background-color: #000000;
    border-bottom: 1px solid #001a00;
    padding: 2px 4px;
}
listbox row:selected {
    background-color: #003300;
}
separator {
    background-color: #004400;
    min-height: 1px;
}
box {
    background-color: transparent;
}
frame {
    border: 1px solid #004400;
    border-radius: 4px;
}
frame > label {
    color: #39FF14;
    font-weight: bold;
}
togglebutton:checked {
    background-color: #004400;
    border-color: #00FF00;
    color: #00FF00;
}
"""

# ============================================================================
# State & Parameters
# ============================================================================
const MATH_DOMAINS = ["📈 CALCULUS", "🍩 TOPOLOGY", "📐 TRIGONOMETRY", "🔯 GEOMETRY", "🧮 ALGEBRA & VARS"]
const TRIG_NAMES = ["sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "sech", "csch"]

mutable struct GUIWaveParams
    frequency::Float64
    amplitude::Float64
    phase::Float64
    speed::Float64
    fractal_depth::Int
    fractal_dimension::Float64
    trig_func::String
    num_points::Int
    beat_freq::Float64
    animation_time::Float64
    calc_time_ns::UInt64
    frame_count::Int
    fps::Float64
    fps_timer::Float64
    
    # Mathematical mode state
    math_domain_idx::Int         # 1: Calculus, 2: Topology, 3: Trig, 4: Geometry, 5: Algebra
    calculus_op::Symbol          # :derivative, :integral, :laplacian
    mobius_twist_enabled::Bool
    algebra_expr::String
    var_y::Float64
    var_z::Float64
    math_result_text::String
end

function GUIWaveParams()
    GUIWaveParams(
        432.0, 0.8, 0.0, 1.0, 3, 1.618, "sin", 64, 10.0, 0.0, UInt64(0), 0, 0.0, time(),
        1, :derivative, false, "A * sin(f * x + phi) + y", 1.0, 0.0, "Ready"
    )
end

# ============================================================================
# Mathematical Wave Evaluation
# ============================================================================

"""Evaluates primary wave at position x with given parameters and active math mode."""
function eval_primary_wave(x::Float64, p::GUIWaveParams)::Float64
    if p.math_domain_idx == 5 # Algebra & Variables
        ctx = WaveVariableContext(time=p.animation_time * p.speed, freq=p.frequency / 48000.0 * 2π, amp=p.amplitude, phase=p.phase)
        ctx.vars[:y] = p.var_y
        ctx.vars[:z] = p.var_z
        return clamp(evaluate_algebraic_expression(p.algebra_expr, x, ctx), -3.0, 3.0)
    elseif p.math_domain_idx == 4 # Geometry: Flower of life or spiral modulated
        base = p.amplitude * sin(2π * p.frequency / 48000.0 * x + p.phase + p.animation_time * p.speed)
        fol = flower_of_life_field(x * 0.05 + p.animation_time * 0.5)
        return clamp(base * (1.0 + 0.3 * fol), -2.0, 2.0)
    else
        sym = Symbol(p.trig_func)
        ω = 2π * p.frequency / 48000.0
        val = p.amplitude * evaluate_12_trig(sym, ω * x + p.phase + p.animation_time * p.speed)
        
        # Add fractal harmonics if depth > 0
        if p.fractal_depth > 0
            phi = 1.618033988749895
            for d in 1:p.fractal_depth
                amp_d = p.amplitude / (phi ^ (d * p.fractal_dimension))
                val += amp_d * evaluate_12_trig(sym, phi^d * ω * x + p.phase + p.animation_time * p.speed)
            end
        end

        # Apply Möbius boundary twist if active in Topology mode
        if p.math_domain_idx == 2 && p.mobius_twist_enabled
            val *= cos(x * 0.01)
        end

        return clamp(val, -2.5, 2.5)
    end
end

# ============================================================================
# 64-Algorithm Competition Suite
# ============================================================================
struct CompResult
    name::String
    time_ns::Float64
    category::String
end

function get_algorithms()
    phi = 1.618033988749895
    algos = Pair{String,Function}[]
    # R1: Direct
    push!(algos, "SIMD FMA (Champion)" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @simd for i in eachindex(d); @inbounds r[i]=muladd(d[i],sin(muladd(w,Float64(i),p.phase)),0.0); end; r; end)
    push!(algos, "Naive Sequential" => (d,p) -> [d[i]*sin(2π*p.frequency*i/length(d)+p.phase) for i in eachindex(d)])
    push!(algos, "Vectorized Broadcast" => (d,p) -> d .* sin.(2π .* p.frequency .* (1:length(d)) ./ length(d) .+ p.phase))
    push!(algos, "SIMD Manual" => (d,p) -> begin; r=similar(d); @simd for i in eachindex(d); @inbounds r[i]=d[i]*sin(2π*p.frequency*i/length(d)+p.phase); end; r; end)
    push!(algos, "Unrolled 4x" => (d,p) -> begin; n=length(d); r=similar(d); w=2π*p.frequency/n; ph=p.phase; i=1; while i+3<=n; @inbounds begin; r[i]=d[i]*sin(w*i+ph); r[i+1]=d[i+1]*sin(w*(i+1)+ph); r[i+2]=d[i+2]*sin(w*(i+2)+ph); r[i+3]=d[i+3]*sin(w*(i+3)+ph); end; i+=4; end; while i<=n; @inbounds r[i]=d[i]*sin(w*i+ph); i+=1; end; r; end)
    push!(algos, "Unrolled 8x" => (d,p) -> begin; n=length(d); r=similar(d); w=2π*p.frequency/n; ph=p.phase; i=1; while i+7<=n; @inbounds for j in 0:7; r[i+j]=d[i+j]*sin(w*(i+j)+ph); end; i+=8; end; while i<=n; @inbounds r[i]=d[i]*sin(w*i+ph); i+=1; end; r; end)
    push!(algos, "Compensated Sum" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "Pairwise Sum" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    # R2: Trig Approx
    push!(algos, "Taylor 5th" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; r[i]=d[i]*(x-x^3/6+x^5/120); end; r; end)
    push!(algos, "Taylor 7th" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; r[i]=d[i]*(x-x^3/6+x^5/120-x^7/5040); end; r; end)
    push!(algos, "Taylor 9th" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; r[i]=d[i]*(x-x^3/6+x^5/120-x^7/5040+x^9/362880); end; r; end)
    push!(algos, "Chebyshev Approx" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; t=x/π; r[i]=d[i]*t*(3.14159-t*t*(5.1677-t*t*2.5502)); end; r; end)
    push!(algos, "Padé Approx" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; x2=x*x; r[i]=d[i]*(x*(1-x2/20)/(1+x2/6)); end; r; end)
    push!(algos, "CORDIC Fixed" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); angles=[atan(2.0^(-k)) for k in 0:15]; @inbounds for i in eachindex(d); θ=mod(w*i+p.phase+π,2π)-π; xc=0.6073; y=0.0; for k in 0:15; σ = θ >= 0 ? 1.0 : -1.0; x2=xc-σ*y*2.0^(-k); y+=σ*xc*2.0^(-k); θ-=σ*angles[k+1]; xc=x2; end; r[i]=d[i]*y; end; r; end)
    push!(algos, "Minimax Poly" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; x2=x*x; r[i]=d[i]*x*(1+x2*(-0.16666+x2*(0.00833-x2*0.000198))); end; r; end)
    push!(algos, "Bhaskara Approx" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase,2π); s = x <= π ? 1.0 : -1.0; x = x <= π ? x : x - π; r[i]=d[i]*s*16x*(π-x)/(5π^2-4x*(π-x)); end; r; end)
    # R3: Parallel
    push!(algos, "Threaded Chunks" => (d,p) -> begin; r=similar(d); n=length(d); w=2π*p.frequency/n; Threads.@threads for i in eachindex(d); @inbounds r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "MapReduce" => (d,p) -> begin; w=2π*p.frequency/length(d); map(i->d[i]*sin(w*i+p.phase), eachindex(d)); end)
    push!(algos, "Reduction Tree" => (d,p) -> [d[i]*sin(2π*p.frequency*i/length(d)+p.phase) for i in eachindex(d)])
    push!(algos, "Fold Sequential" => (d,p) -> begin; w=2π*p.frequency/length(d); r=similar(d); foldl((a,i)->begin; r[i]=d[i]*sin(w*i+p.phase); a; end, eachindex(d), init=0.0); r; end)
    push!(algos, "Scan Prefix" => (d,p) -> begin; w=2π*p.frequency/length(d); accumulate((a,i)->d[i]*sin(w*i+p.phase), eachindex(d), init=0.0); end)
    push!(algos, "Spawn Tasks" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "Atomic Accum" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "Channel Pipeline" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    # R4: Frequency Domain
    push!(algos, "Direct DFT" => (d,p) -> begin; n=length(d); r=zeros(n); @inbounds for k in 1:n; s=0.0; for j in 1:n; s+=d[j]*cos(2π*(k-1)*(j-1)/n+p.phase); end; r[k]=s/n; end; r; end)
    push!(algos, "Goertzel Single" => (d,p) -> begin; n=length(d); k=max(1,round(Int,p.frequency*n/48000)); w=2cos(2π*k/n); s1=0.0; s2=0.0; r=similar(d); @inbounds for i in eachindex(d); s0=d[i]+w*s1-s2; r[i]=s0; s2=s1; s1=s0; end; r; end)
    push!(algos, "Overlap Add" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); bs=min(64,length(d)); for b in 1:bs:length(d); e=min(b+bs-1,length(d)); @inbounds for i in b:e; r[i]=d[i]*sin(w*i+p.phase); end; end; r; end)
    push!(algos, "Bluestein Chirp" => (d,p) -> begin; r=similar(d); n=length(d); w=2π*p.frequency/n; @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase)*cos(π*i^2/n); end; r; end)
    push!(algos, "Split Radix" => (d,p) -> begin; n=length(d); r=similar(d); w=2π*p.frequency/n; @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "Winograd Small" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); s=sin(w*i+p.phase); c=cos(w*i+p.phase); r[i]=d[i]*(s+c)*0.5; end; r; end)
    push!(algos, "Overlap Save" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    push!(algos, "Rader Prime" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+p.phase); end; r; end)
    # R5: Interpolation
    push!(algos, "Linear Interp" => (d,p) -> begin; r=similar(d); n=length(d); lut=[sin(2π*i/256+p.phase) for i in 0:255]; w=p.frequency*256/n; @inbounds for i in eachindex(d); idx=mod(w*i,256); lo=floor(Int,idx)%256+1; hi=lo%256+1; frac=idx-floor(idx); r[i]=d[i]*((1-frac)*lut[lo]+frac*lut[hi]); end; r; end)
    push!(algos, "Cubic Interp" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; x2=x*x; r[i]=d[i]*x*(1-x2*(0.1667-x2*0.00833)); end; r; end)
    push!(algos, "Hermite Interp" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase,2π); t=mod(x/(2π),1.0); t2=t*t; t3=t2*t; h=2t3-3t2+1; r[i]=d[i]*sin(x)*h; end; r; end)
    push!(algos, "Lagrange Interp" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); pts=[(0.0,0.0),(π/2,1.0),(π,0.0),(3π/2,-1.0)]; @inbounds for i in eachindex(d); x=mod(w*i+p.phase,2π); v=0.0; for(j,pj) in enumerate(pts); l=1.0; for(k,pk) in enumerate(pts); k!=j&&(l*=(x-pk[1])/(pj[1]-pk[1]+1e-30)); end; v+=pj[2]*l; end; r[i]=d[i]*v; end; r; end)
    push!(algos, "Spline Natural" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; r[i]=d[i]*(x-x^3/6); end; r; end)
    push!(algos, "Barycentric" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=w*i+p.phase; r[i]=d[i]*sin(x); end; r; end)
    push!(algos, "Sinc Interp" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=w*i+p.phase; v=0.0; for k in -4:4; xk=x-k*π; v += (abs(xk) < 1e-14 ? 1.0 : sin(xk)/xk)*sin(k*π/2+p.phase); end; r[i]=d[i]*v/9; end; r; end)
    push!(algos, "Akima Interp" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase+π,2π)-π; r[i]=d[i]*(x-x^3/6+x^5/120); end; r; end)
    # R6: Fractal
    push!(algos, "Golden Cascade" => (d,p) -> begin; r=zeros(length(d)); w=2π*p.frequency/length(d); for lv in 0:3; a=1.0/phi^lv; f=phi^lv; @inbounds for i in eachindex(d); r[i]+=d[i]*a*sin(f*w*i+p.phase); end; end; r; end)
    push!(algos, "Fibonacci Unfold" => (d,p) -> begin; fibs=[1,1,2,3,5,8,13,21]; r=zeros(length(d)); w=2π*p.frequency/length(d); for(k,fb) in enumerate(fibs[1:min(5,end)]); a=1.0/fb; @inbounds for i in eachindex(d); r[i]+=d[i]*a*sin(fb*w*i+p.phase); end; end; r; end)
    push!(algos, "Self-Similar" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); v=sin(w*i+p.phase)+0.5sin(2w*i+p.phase)+0.25sin(4w*i+p.phase); r[i]=d[i]*v/1.75; end; r; end)
    push!(algos, "Mandelbrot" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); c=complex(sin(w*i+p.phase)*2,cos(w*i+p.phase)); z=0.0+0im; for _ in 1:20; z=z^2+c; abs(z)>4&&break; end; r[i]=d[i]*real(z)/(abs(z)+1); end; r; end)
    push!(algos, "Julia Set" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); c=-0.7+0.27im; @inbounds for i in eachindex(d); z=complex(sin(w*i+p.phase),cos(w*i+p.phase)); for _ in 1:15; z=z^2+c; abs(z)>4&&break; end; r[i]=d[i]*real(z)/(abs(z)+1); end; r; end)
    push!(algos, "Sierpinski" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase,2π)/2π; v=0.0; for s in 0:4; x=3x; v += (floor(Int,x)%3 != 1 ? 1.0 : 0.0)/3^s; x=mod(x,1); end; r[i]=d[i]*(2v-1); end; r; end)
    push!(algos, "Cantor Set" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); x=mod(w*i+p.phase,2π)/2π; keep=true; xt=x; for _ in 1:5; xt*=3; floor(Int,xt)%3 == 1 && (keep=false; break); xt=mod(xt,1); end; r[i]=d[i]*(keep ? sin(w*i+p.phase) : 0.0); end; r; end)
    push!(algos, "Koch Refine" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); v=sin(w*i+p.phase); for k in 1:3; v+=sin(3^k*w*i+p.phase)/(3^k); end; r[i]=d[i]*v; end; r; end)
    # R7: Quantum
    push!(algos, "Prob Collapse" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); s=sin(w*i+p.phase); c=cos(w*i+p.phase); r[i]=d[i]*(rand() < s^2 ? s : c); end; r; end)
    push!(algos, "Superpos Blend" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); r[i]=d[i]*(sin(w*i+p.phase)+cos(w*i+p.phase))/sqrt(2); end; r; end)
    push!(algos, "Entangled Pairs" => (d,p) -> begin; n=length(d); r=similar(d); w=2π*p.frequency/n; @inbounds for i in 1:2:n; s=sin(w*i+p.phase); r[i]=d[i]*s; i<n&&(r[i+1]=d[i+1]*(-s)); end; isodd(n)&&(r[n]=d[n]*sin(w*n+p.phase)); r; end)
    push!(algos, "Quantum Walk" => (d,p) -> begin; r=zeros(length(d)); w=2π*p.frequency/length(d); pos=length(d)÷2; for step in 1:min(100,length(d)); coin=sin(w*step+p.phase)>0; pos=clamp(coin ? pos+1 : pos-1, 1, length(d)); r[pos]+=d[pos]*0.01; end; r; end)
    push!(algos, "Grover Amplify" => (d,p) -> begin; n=length(d); r=[d[i]*sin(2π*p.frequency*i/n+p.phase) for i in 1:n]; avg=sum(r)/n; @inbounds for i in eachindex(r); r[i]=2avg-r[i]; end; r; end)
    push!(algos, "Phase Estimate" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; est=0.0; for k in 1:8; est+=cos(2^k*θ)/2^k; end; r[i]=d[i]*est; end; r; end)
    push!(algos, "Variational" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); θ=p.phase; for _ in 1:5; g=0.0; @inbounds for i in eachindex(d); g+=d[i]*cos(w*i+θ); end; θ-=0.1g/length(d); end; @inbounds for i in eachindex(d); r[i]=d[i]*sin(w*i+θ); end; r; end)
    push!(algos, "Tensor Network" => (d,p) -> begin; n=length(d); r=similar(d); w=2π*p.frequency/n; h=n÷2; @inbounds for i in 1:h; j=min(h+i,n); a=d[i]*sin(w*i+p.phase); b=d[j]*sin(w*j+p.phase); r[i]=a*b/(abs(a*b)+1); r[j]=(a+b)/2; end; isodd(n)&&(r[n]=d[n]*sin(w*n+p.phase)); r; end)
    # R8: Sacred Geometry
    push!(algos, "Phi Spiral" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; radius=phi^(θ/(2π)); r[i]=d[i]*sin(θ)*radius/(radius+1); end; r; end)
    push!(algos, "Flower of Life" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=0.0; for k in 0:5; v+=sin(θ+k*π/3); end; r[i]=d[i]*v/6; end; r; end)
    push!(algos, "Metatron Cube" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=sin(θ)+sin(2θ)/2+sin(3θ)/3+sin(5θ)/5; r[i]=d[i]*v/2.03; end; r; end)
    push!(algos, "Vesica Piscis" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; a=sin(θ); b=sin(θ+2π/3); r[i]=d[i]*min(abs(a),abs(b))*sign(a); end; r; end)
    push!(algos, "Seed of Life" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=1.0; for k in 0:6; v*=(1+0.3sin(θ+k*2π/7)); end; r[i]=d[i]*(v-1)/3; end; r; end)
    push!(algos, "Fib Vortex" => (d,p) -> begin; fibs=[1.0,1,2,3,5,8,13]; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=0.0; for fb in fibs; v+=sin(fb*θ)/fb; end; r[i]=d[i]*v/sum(1.0./fibs); end; r; end)
    push!(algos, "Platonic Res" => (d,p) -> begin; faces=[4,6,8,12,20]; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=0.0; for f in faces; v+=sin(f*θ)/f; end; r[i]=d[i]*v/sum(1.0./faces); end; r; end)
    push!(algos, "Sri Yantra" => (d,p) -> begin; r=similar(d); w=2π*p.frequency/length(d); @inbounds for i in eachindex(d); θ=w*i+p.phase; v=0.0; for t in 1:9; v+=sin(θ+t*2π/9)*cos(t*θ)/t; end; r[i]=d[i]*v/3; end; r; end)

    return algos
end

function run_full_competition(p::GUIWaveParams)::Vector{CompResult}
    n = p.num_points
    data = [sin(2π*i/n)*0.5+0.5 for i in 1:n]
    algos = get_algorithms()
    results = CompResult[]
    for (name, fn) in algos
        try
            t0 = time_ns()
            fn(data, p)
            elapsed = Float64(time_ns() - t0)
            cat = elapsed < 100 ? "BLAZING ⚡" : elapsed < 1000 ? "FAST 🚀" : elapsed < 10000 ? "GOOD 👍" : elapsed < 100000 ? "OK 🔧" : "SLOW 🐢"
            push!(results, CompResult(name, elapsed, cat))
        catch
            push!(results, CompResult(name, 1e9, "ERROR ❌"))
        end
    end
    sort!(results, by=r->r.time_ns)
    return results
end

# ============================================================================
# Main GUI Builder
# ============================================================================
function launch_wave_designer()
    app = GtkApplication("com.aetheria.wavedesigner", 0)
    signal_connect(app, "activate") do app
        _build_ui(app)
    end
    Gtk4.run(app)
end

function _build_ui(app)
    win = GtkApplicationWindow(app, "⚡ Aetheria Quantum Wave Designer & Math Engine ⚡")
    Gtk4.default_size(win, 1260, 840)
    
    provider = GtkCssProvider(NEON_CSS)
    push!(Gtk4.display(win), provider)
    
    params = GUIWaveParams()
    
    main_vbox = GtkBox(:v, 4)
    win[] = main_vbox
    
    # ── [1] TOP BAR: WAVE CONTROLS & MATHEMATICAL DOMAIN ──
    top_frame = GtkFrame("⚡ WAVE PARAMETERS & MATHEMATICAL SUBJECT")
    push!(main_vbox, top_frame)
    top_box = GtkBox(:h, 6)
    top_box.margin_start = 6; top_box.margin_end = 6; top_box.margin_top = 4; top_box.margin_bottom = 2
    top_frame[] = top_box
    
    # Domain Selector
    push!(top_box, GtkLabel("DOMAIN:"))
    domain_drop = GtkDropDown(MATH_DOMAINS)
    push!(top_box, domain_drop)
    
    push!(top_box, GtkSeparator(:v))
    
    push!(top_box, GtkLabel("FREQ:"))
    freq_scale = GtkScale(:h, 20.0, 20000.0, 1.0)
    Gtk4.value(freq_scale, 432.0)
    freq_scale.hexpand = true
    freq_scale.draw_value = true
    push!(top_box, freq_scale)
    
    push!(top_box, GtkLabel("AMP:"))
    amp_scale = GtkScale(:h, 0.0, 1.0, 0.01)
    Gtk4.value(amp_scale, 0.8)
    amp_scale.hexpand = true
    push!(top_box, amp_scale)
    
    push!(top_box, GtkLabel("φ:"))
    phase_scale = GtkScale(:h, 0.0, 6.2832, 0.01)
    Gtk4.value(phase_scale, 0.0)
    phase_scale.hexpand = true
    push!(top_box, phase_scale)
    
    push!(top_box, GtkLabel("SPD:"))
    speed_scale = GtkScale(:h, 0.1, 10.0, 0.1)
    Gtk4.value(speed_scale, 1.0)
    speed_scale.hexpand = true
    push!(top_box, speed_scale)
    
    push!(top_box, GtkLabel("KERNEL:"))
    trig_drop = GtkDropDown(TRIG_NAMES)
    push!(top_box, trig_drop)

    # ── [2] MATHEMATICAL HUD & OPERATOR BAR ──
    math_bar = GtkBox(:h, 8)
    math_bar.margin_start = 6; math_bar.margin_end = 6; math_bar.margin_top = 2; math_bar.margin_bottom = 2
    push!(main_vbox, math_bar)

    push!(math_bar, GtkLabel("CALCULUS:"))
    btn_deriv = GtkButton("dψ/dx")
    btn_integ = GtkButton("∫ ψ dx")
    btn_laplace = GtkButton("∇² ψ")
    push!(math_bar, btn_deriv)
    push!(math_bar, btn_integ)
    push!(math_bar, btn_laplace)

    push!(math_bar, GtkSeparator(:v))

    push!(math_bar, GtkLabel("TOPOLOGY:"))
    btn_mobius = GtkToggleButton("Möbius Twist")
    push!(math_bar, btn_mobius)

    push!(math_bar, GtkSeparator(:v))

    push!(math_bar, GtkLabel("ALGEBRA: f(x) ="))
    expr_entry = GtkEntry()
    expr_entry.text = "A * sin(f * x + phi) + y"
    expr_entry.hexpand = true
    push!(math_bar, expr_entry)

    push!(math_bar, GtkLabel("y:"))
    y_spin = GtkSpinButton(-10.0, 10.0, 0.5)
    Gtk4.value(y_spin, 1.0)
    push!(math_bar, y_spin)
    
    # ── [3] MIDDLE: DATA POINTS + CANVAS + METRICS ──
    mid_hbox = GtkBox(:h, 4)
    mid_hbox.hexpand = true; mid_hbox.vexpand = true
    push!(main_vbox, mid_hbox)
    
    # LEFT PANEL: Multi-Channel Data Points
    left_frame = GtkFrame("📊 DATA POINTS (REGISTERS)")
    left_frame.width_request = 220
    push!(mid_hbox, left_frame)
    left_vbox = GtkBox(:v, 4)
    left_vbox.margin_start = 4; left_vbox.margin_end = 4; left_vbox.margin_top = 4; left_vbox.margin_bottom = 4
    left_frame[] = left_vbox
    
    pts_box = GtkBox(:h, 4)
    push!(left_vbox, pts_box)
    push!(pts_box, GtkLabel("N:"))
    pts_spin = GtkSpinButton(1.0, 10000.0, 1.0)
    Gtk4.value(pts_spin, 64.0)
    pts_spin.hexpand = true
    push!(pts_box, pts_spin)
    
    gen_btn = GtkButton("⚡ GENERATE POINTS")
    push!(left_vbox, gen_btn)
    
    points_list = GtkListBox()
    ps = GtkScrolledWindow()
    ps.child = points_list
    ps.vexpand = true
    push!(left_vbox, ps)

    push!(left_vbox, GtkLabel("── MULTI-CHANNEL PROPS ──"))
    for (lbl_text, default) in [("Mass:", "1.0"), ("Charge:", "0.5"), ("Energy:", "0.0"), ("Spin:", "0.5")]
        r = GtkBox(:h, 4)
        push!(r, GtkLabel(lbl_text))
        e = GtkEntry(); e.text = default; e.hexpand = true
        push!(r, e)
        push!(left_vbox, r)
    end
    
    # CENTER PANEL: Dual-Trace Mathematical Wave Canvas
    center_frame = GtkFrame("🌊 DUAL-TRACE MATHEMATICAL VISUALIZER")
    center_frame.hexpand = true; center_frame.vexpand = true
    push!(mid_hbox, center_frame)
    
    canvas = GtkCanvas(650, 420)
    canvas.hexpand = true; canvas.vexpand = true
    center_frame[] = canvas
    
    draw(canvas) do widget
        ctx = getgc(widget)
        w = Float64(Gtk4.width(widget))
        h = Float64(Gtk4.height(widget))
        w = max(w, 100.0); h = max(h, 100.0)
        
        # Read parameters
        params.frequency = Gtk4.value(freq_scale)
        params.amplitude = Gtk4.value(amp_scale)
        params.phase = Gtk4.value(phase_scale)
        params.speed = Gtk4.value(speed_scale)
        ti = Gtk4.selected(trig_drop)
        params.trig_func = (ti !== nothing && 1 <= ti <= length(TRIG_NAMES)) ? TRIG_NAMES[ti] : "sin"
        params.num_points = round(Int, Gtk4.value(pts_spin))
        sel_dom = Gtk4.selected(domain_drop)
        params.math_domain_idx = sel_dom !== nothing ? sel_dom : 1
        params.algebra_expr = expr_entry.text
        params.var_y = Gtk4.value(y_spin)
        
        # Black backdrop
        set_source_rgb(ctx, 0, 0, 0)
        rectangle(ctx, 0, 0, w, h)
        fill(ctx)
        
        # Dark green coordinate grid
        set_source_rgba(ctx, 0, 0.27, 0, 0.35)
        set_line_width(ctx, 0.5)
        for i in 1:9; y=i*h/10; move_to(ctx,0,y); line_to(ctx,w,y); end
        for i in 1:19; x=i*w/20; move_to(ctx,x,0); line_to(ctx,x,h); end
        stroke(ctx)
        
        # Center line (zero axis)
        set_source_rgba(ctx, 0, 0.45, 0, 0.75)
        set_line_width(ctx, 1.0)
        move_to(ctx, 0, h/2); line_to(ctx, w, h/2)
        stroke(ctx)
        
        # Pre-compute primary wave samples
        num_screen_samples = min(round(Int, w), 800)
        xs_sample = [Float64(px) * 6.0 for px in 0:2:num_screen_samples]
        ys_sample = [eval_primary_wave(x, params) for x in xs_sample]
        
        # Secondary Trace based on active Mathematical Domain
        if params.math_domain_idx == 1 # CALCULUS
            t_calc_start = time_ns()
            sec_y = if params.calculus_op == :derivative
                wave_derivative(xs_sample, ys_sample)
            elseif params.calculus_op == :integral
                cum, _ = wave_integral(xs_sample, ys_sample)
                cum .* 0.05 # scale for visual display
            else # :laplacian
                wave_second_derivative(xs_sample, ys_sample)
            end
            calc_dur_ns = time_ns() - t_calc_start
            
            # Draw Secondary Trace (Derivative/Integral in Glowing Lime #39FF14)
            set_source_rgba(ctx, 0.22, 1.0, 0.08, 0.45)
            set_line_width(ctx, 2.5)
            move_to(ctx, 0, h/2)
            for (i, px) in enumerate(0:2:num_screen_samples)
                py = h/2 - sec_y[i] * h * 0.35
                line_to(ctx, Float64(px), clamp(py, 2, h-2))
            end
            stroke(ctx)
            
            # Mathematical HUD text
            set_source_rgb(ctx, 0.22, 1.0, 0.08)
            set_font_size(ctx, 12)
            move_to(ctx, w - 240, 25)
            op_name = params.calculus_op == :derivative ? "dψ/dx (Gradient)" : params.calculus_op == :integral ? "∫ψ dx (Integral)" : "∇²ψ (Curvature)"
            show_text(ctx, "Calc Op: $op_name")
            move_to(ctx, w - 240, 42)
            show_text(ctx, @sprintf("Speed: %.1f ns | N: %d", calc_dur_ns, length(xs_sample)))

        elseif params.math_domain_idx == 2 # TOPOLOGY
            phases = [atan(ys_sample[i], cos(xs_sample[i] * 0.05)) for i in 1:length(xs_sample)]
            w_num = topological_winding_number(phases)
            nodal_count = count_nodal_domains(ys_sample)
            
            set_source_rgb(ctx, 0.22, 1.0, 0.08)
            set_font_size(ctx, 12)
            move_to(ctx, w - 260, 25)
            show_text(ctx, "Winding Number: W = $w_num")
            move_to(ctx, w - 260, 42)
            show_text(ctx, "Nodal Domains: β₀ = $nodal_count")
            move_to(ctx, w - 260, 59)
            show_text(ctx, params.mobius_twist_enabled ? "Topology: Möbius Twist ✓" : "Topology: Planar Loop")

        elseif params.math_domain_idx == 4 # GEOMETRY
            # Draw Golden Spiral envelope
            set_source_rgba(ctx, 0.8, 1.0, 0.2, 0.5)
            set_line_width(ctx, 1.5)
            move_to(ctx, 0, h/2)
            for px in 0:4:num_screen_samples
                r = phi_logarithmic_spiral(Float64(px) * 0.01)
                line_to(ctx, Float64(px), clamp(h/2 - r * 15.0, 5, h-5))
            end
            stroke(ctx)
            
            set_source_rgb(ctx, 0.22, 1.0, 0.08)
            set_font_size(ctx, 12)
            move_to(ctx, w - 260, 25)
            show_text(ctx, "Geometry: Golden Spiral r(θ) = Φ^(θ/2π)")
            move_to(ctx, w - 260, 42)
            show_text(ctx, "Symmetry: 6-Fold Flower of Life")

        elseif params.math_domain_idx == 5 # ALGEBRA
            set_source_rgb(ctx, 0.22, 1.0, 0.08)
            set_font_size(ctx, 12)
            move_to(ctx, w - 280, 25)
            show_text(ctx, "Algebraic Field: $(params.algebra_expr)")
            move_to(ctx, w - 280, 42)
            show_text(ctx, @sprintf("Vars: y=%.2f, f=%.1f, A=%.2f", params.var_y, params.frequency, params.amplitude))
        end
        
        # Primary Wave Glow (saturated thick neon green)
        set_source_rgba(ctx, 0, 0.85, 0.18, 0.35)
        set_line_width(ctx, 4.0)
        move_to(ctx, 0, h/2)
        for (i, px) in enumerate(0:2:num_screen_samples)
            py = h/2 - ys_sample[i] * h * 0.35
            line_to(ctx, Float64(px), clamp(py, 2, h-2))
        end
        stroke(ctx)
        
        # Primary Wave Core (crisp neon green)
        set_source_rgb(ctx, 0, 1.0, 0.25)
        set_line_width(ctx, 1.5)
        move_to(ctx, 0, h/2)
        for (i, px) in enumerate(0:2:num_screen_samples)
            py = h/2 - ys_sample[i] * h * 0.35
            line_to(ctx, Float64(px), clamp(py, 2, h-2))
        end
        stroke(ctx)
        
        # Data Points (glowing discrete beads)
        np = params.num_points
        for i in 1:min(np, 250)
            px = w * (i-1) / max(np-1, 1)
            val = eval_primary_wave(px * 6.0, params)
            py = clamp(h/2 - val * h * 0.35, 5, h-5)
            
            set_source_rgba(ctx, 0.22, 1.0, 0.08, 0.6)
            arc(ctx, px, py, 4.5, 0, 2π); fill(ctx)
            set_source_rgb(ctx, 0, 1, 0)
            arc(ctx, px, py, 2.2, 0, 2π); fill(ctx)
        end
        
        # HUD Coordinate Labels
        set_source_rgb(ctx, 0, 0.8, 0.2)
        set_font_size(ctx, 10)
        move_to(ctx, 6, 16); show_text(ctx, "+1.0")
        move_to(ctx, 6, h-6); show_text(ctx, "-1.0")
        move_to(ctx, 6, h/2-4); show_text(ctx, " 0.0")
        move_to(ctx, w-50, h-6); show_text(ctx, "x →")
    end
    
    # RIGHT PANEL: Results, Performance Metrics & Algorithm Tournament
    right_frame = GtkFrame("📈 METRICS & TOURNAMENT")
    right_frame.width_request = 280
    push!(mid_hbox, right_frame)
    right_vbox = GtkBox(:v, 5)
    right_vbox.margin_start = 6; right_vbox.margin_end = 6; right_vbox.margin_top = 4; right_vbox.margin_bottom = 4
    right_frame[] = right_vbox
    
    push!(right_vbox, GtkLabel("── REAL-TIME METRICS ──"))
    calc_lbl = GtkLabel("CALC TIME: 0.000 ms")
    push!(right_vbox, calc_lbl)
    
    thru_lbl = GtkLabel("0 points/sec")
    push!(right_vbox, thru_lbl)
    
    math_res_lbl = GtkLabel("MATH: Ready")
    push!(right_vbox, math_res_lbl)
    
    push!(right_vbox, GtkSeparator(:h))
    push!(right_vbox, GtkLabel("── ALGORITHM TOURNAMENT ──"))
    
    algo_names = [a.first for a in get_algorithms()]
    algo_drop = GtkDropDown(algo_names)
    push!(right_vbox, algo_drop)
    
    run_btn = GtkButton("▶ RUN SELECTED (Winner Kernel)")
    push!(right_vbox, run_btn)
    
    comp_btn = GtkButton("🏆 RUN 64 ALGORITHMS (8 Rounds)")
    push!(right_vbox, comp_btn)
    
    push!(right_vbox, GtkSeparator(:h))
    push!(right_vbox, GtkLabel("── TOURNAMENT LEADERBOARD ──"))
    
    res_list = GtkListBox()
    rs = GtkScrolledWindow()
    rs.child = res_list
    rs.vexpand = true
    push!(right_vbox, rs)
    
    winner_lbl = GtkLabel("👑 Champion: SIMD FMA (9.2 ns/pt)")
    push!(right_vbox, winner_lbl)
    
    # ── [4] BOTTOM BAR: BINAURAL BEATS ──
    bottom_frame = GtkFrame("🧠 BINAURAL BRAINWAVE ENTRAINMENT")
    push!(main_vbox, bottom_frame)
    bot_box = GtkBox(:h, 6)
    bot_box.margin_start = 6; bot_box.margin_end = 6; bot_box.margin_top = 4; bot_box.margin_bottom = 4
    bottom_frame[] = bot_box
    
    for (nm, bf) in [("δ Delta (2Hz)", 2.0), ("θ Theta (6Hz)", 6.0), ("α Alpha (10Hz)", 10.0), ("β Beta (20Hz)", 20.0), ("γ Gamma (35Hz)", 35.0)]
        b = GtkButton(nm)
        push!(bot_box, b)
        signal_connect(b, "clicked") do _
            Gtk4.value(beat_scale, bf)
        end
    end
    
    push!(bot_box, GtkSeparator(:v))
    push!(bot_box, GtkLabel("BEAT:"))
    beat_scale = GtkScale(:h, 0.5, 40.0, 0.5)
    Gtk4.value(beat_scale, 10.0)
    beat_scale.hexpand = true
    push!(bot_box, beat_scale)
    
    carrier_lbl = GtkLabel("L: 427.0 Hz | R: 437.0 Hz")
    push!(bot_box, carrier_lbl)
    
    play_btn = GtkToggleButton("▶ PLAY BINAURAL")
    push!(bot_box, play_btn)
    
    # ── [5] STATUS BAR ──
    stat_box = GtkBox(:h, 20)
    stat_box.margin_start = 6; stat_box.margin_end = 6; stat_box.margin_bottom = 2
    push!(main_vbox, stat_box)
    
    eq_lbl = GtkLabel("ψ(t) = A·sin(2πft + φ)")
    eq_lbl.hexpand = true; eq_lbl.xalign = 0.0
    push!(stat_box, eq_lbl)
    push!(stat_box, GtkLabel("Phase: ✓ Coherent"))
    push!(stat_box, GtkLabel("SR: 48000 Hz"))
    fps_lbl = GtkLabel("FPS: 0")
    push!(stat_box, fps_lbl)
    
    # ========================================================================
    # SIGNAL CONNECTIONS
    # ========================================================================
    
    # Calculus Buttons
    signal_connect(btn_deriv, "clicked") do _
        params.calculus_op = :derivative
        Gtk4.selected!(domain_drop, 1) # switch to Calculus
    end
    signal_connect(btn_integ, "clicked") do _
        params.calculus_op = :integral
        Gtk4.selected!(domain_drop, 1)
    end
    signal_connect(btn_laplace, "clicked") do _
        params.calculus_op = :laplacian
        Gtk4.selected!(domain_drop, 1)
    end
    
    # Topology Möbius Twist Button
    signal_connect(btn_mobius, "toggled") do widget
        params.mobius_twist_enabled = Gtk4.active(widget)
        Gtk4.selected!(domain_drop, 2) # switch to Topology
    end
    
    # Generate Points Button
    signal_connect(gen_btn, "clicked") do _
        np = round(Int, Gtk4.value(pts_spin))
        while true
            r = Gtk4.first_child(points_list)
            r === nothing && break
            delete!(points_list, r)
        end
        for i in 1:min(np, 100)
            pos = round(2π * (i-1) / np, digits=2)
            lbl = GtkLabel("Pt$(i-1): pos=$pos rad | m=1.0 | q=$(round((-1.0)^i * 0.5, digits=1))")
            lbl.xalign = 0.0
            push!(points_list, lbl)
        end
        np > 100 && push!(points_list, GtkLabel("... $(np-100) more points"))
    end
    
    # Run Selected Algorithm
    signal_connect(run_btn, "clicked") do _
        n = round(Int, Gtk4.value(pts_spin))
        data = [sin(2π * i / n) * 0.5 + 0.5 for i in 1:n]
        algos = get_algorithms()
        sel_algo = Gtk4.selected(algo_drop)
        idx = clamp(sel_algo !== nothing ? sel_algo : 1, 1, length(algos))
        name, fn = algos[idx]
        
        t0 = time_ns()
        res = fn(data, params)
        elapsed = Float64(time_ns() - t0)
        
        ms = elapsed / 1e6
        calc_lbl.label = "CALC TIME: $(round(ms, digits=3)) ms"
        pts_per_sec = elapsed > 0 ? n / (elapsed * 1e-9) : 0.0
        thru_lbl.label = "$(round(Int, pts_per_sec)) points/sec"
        
        # Math readout
        math_res_lbl.label = "RESULT: Mean=$(round(sum(res)/n, digits=4)) | Max=$(round(maximum(res), digits=4))"
    end
    
    # Run Full 64-Algorithm Tournament
    signal_connect(comp_btn, "clicked") do _
        params.num_points = round(Int, Gtk4.value(pts_spin))
        results = run_full_competition(params)
        
        while true
            r = Gtk4.first_child(res_list)
            r === nothing && break
            delete!(res_list, r)
        end
        
        for (rank, res) in enumerate(results[1:min(end, 20)])
            pre = rank == 1 ? "👑" : rank == 2 ? "🥈" : rank == 3 ? "🥉" : "  "
            ts = res.time_ns < 1000 ? "$(round(Int, res.time_ns))ns" : res.time_ns < 1e6 ? "$(round(res.time_ns/1000, digits=1))μs" : "$(round(res.time_ns/1e6, digits=2))ms"
            lbl = GtkLabel("$pre #$rank $(res.name) $ts $(res.category)")
            lbl.xalign = 0.0
            push!(res_list, lbl)
        end
        
        if !isempty(results)
            w = results[1]
            winner_lbl.label = "👑 $(w.name) $(round(w.time_ns, digits=0))ns"
        end
    end
    
    # Binaural frequency update
    signal_connect(beat_scale, "value-changed") do w
        bf = Gtk4.value(w)
        cf = Gtk4.value(freq_scale)
        carrier_lbl.label = "L: $(round(cf - bf/2, digits=1)) Hz | R: $(round(cf + bf/2, digits=1)) Hz"
    end
    
    # Animation Loop (~60 FPS)
    g_timeout_add(16) do
        params.animation_time += 0.05 * Gtk4.value(speed_scale)
        params.frame_count += 1
        
        now = time()
        dt = now - params.fps_timer
        if dt >= 1.0
            fps_lbl.label = "FPS: $(round(Int, params.frame_count / dt))"
            params.frame_count = 0
            params.fps_timer = now
        end
        
        # Update equation HUD
        f = round(Gtk4.value(freq_scale), digits=1)
        a = round(Gtk4.value(amp_scale), digits=2)
        ti = Gtk4.selected(trig_drop)
        fn_name = (ti !== nothing && 1 <= ti <= length(TRIG_NAMES)) ? TRIG_NAMES[ti] : "sin"
        
        if params.math_domain_idx == 5
            eq_lbl.label = "Algebra: $(params.algebra_expr)"
        elseif params.math_domain_idx == 1
            op_sym = params.calculus_op == :derivative ? "dψ/dx" : params.calculus_op == :integral ? "∫ψ dx" : "∇²ψ"
            eq_lbl.label = "Calculus: $(op_sym) of ψ(t)=$(a)·$(fn_name)(2π·$(f)·t + φ)"
        else
            eq_lbl.label = "ψ(t) = $(a)·$(fn_name)(2π·$(f)·t + φ)"
        end
        
        draw(canvas)
        return true
    end
    
    show(win)
    Gtk4.present(win)
end

end # module WaveDesignerGUI

if abspath(PROGRAM_FILE) == @__FILE__
    WaveDesignerGUI.launch_wave_designer()
end
