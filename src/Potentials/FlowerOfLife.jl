"""
    Potentials.FlowerOfLife

Sacred Geometry Potential Wells for Continuous Wave Computing.
Implements the Flower of Life potential V_FoL(r) with exact C_6 hexagonal lattice symmetry
as defined in agenda/agenda.md:

    V_FoL(r) = V_0 ∑_{j=1}^6 cos(k_j · r + φ_j)

Where wave vectors k_j are separated by 60° (π/3 radians):
    k_j = k_0 (cos(jπ/3) x̂ + sin(jπ/3) ŷ)
"""

export FlowerOfLifePotential, flower_of_life_potential, compute_flower_of_life_grid!

struct FlowerOfLifePotential
    V0::Float64
    k0::Float64
    lattice_a::Float64
    k_x::Vector{Float64}
    k_y::Vector{Float64}
    
    function FlowerOfLifePotential(; V0::Float64 = 1.0, lattice_a::Float64 = 4.0)
        k0 = 2π / lattice_a
        k_x = [k0 * cos(j * π / 3.0) for j in 1:6]
        k_y = [k0 * sin(j * π / 3.0) for j in 1:6]
        new(V0, k0, lattice_a, k_x, k_y)
    end
end

"""
    flower_of_life_potential(x::Float64, y::Float64; pot::FlowerOfLifePotential = FlowerOfLifePotential())::Float64

Evaluates the Flower of Life potential V_FoL(r) at continuous coordinates (x, y).
"""
@inline function flower_of_life_potential(
    x::Float64,
    y::Float64;
    pot::FlowerOfLifePotential = FlowerOfLifePotential()
)::Float64
    v = 0.0
    @inbounds for j in 1:6
        v += cos(pot.k_x[j] * x + pot.k_y[j] * y)
    end
    return pot.V0 * v
end

"""
    compute_flower_of_life_grid!(grid::Matrix{Float64}; pot::FlowerOfLifePotential = FlowerOfLifePotential(), dx::Float64 = 0.25)::Matrix{Float64}

Computes the 2D Flower of Life potential landscape onto `grid` with zero heap allocation.
"""
function compute_flower_of_life_grid!(
    grid::Matrix{Float64};
    pot::FlowerOfLifePotential = FlowerOfLifePotential(),
    dx::Float64 = 0.25
)::Matrix{Float64}
    nx, ny = size(grid)
    cx = nx / 2.0
    cy = ny / 2.0
    
    @inbounds for j in 1:ny
        y = (j - cy) * dx
        for i in 1:nx
            x = (i - cx) * dx
            v = 0.0
            @simd for w in 1:6
                v += cos(pot.k_x[w] * x + pot.k_y[w] * y)
            end
            grid[i, j] = pot.V0 * v
        end
    end
    return grid
end
