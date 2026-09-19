# Sacred Geometry Potential Wells

## Overview

Sacred geometry potential wells provide structured spatial constraints that guide wave evolution in Aetheria.jl. These potentials are based on ancient geometric patterns—particularly the Flower of Life and Metatron's Cube—which exhibit remarkable mathematical symmetries. By encoding these patterns as potential energy landscapes, we create computational inductive biases that naturally guide wave-based learning toward structured, efficient solutions.

## The Flower of Life Potential

### Mathematical Definition

The Flower of Life potential is constructed from hexagonal lattice harmonics with C6 rotational symmetry:

```
V_FoL(r) = V₀ Σ_{j=1}^6 cos(k_j · r + φ_j)
```

Where:
- **V₀**: Potential depth (energy scale)
- **k_j**: Wave vectors pointing to hexagon vertices
- **r**: Position vector (x, y, z)
- **φ_j**: Phase offsets (typically 0 for symmetric pattern)

### Wave Vector Construction

The six wave vectors are distributed symmetrically in the plane, separated by 60° (π/3 radians):

```
k_j = k₀ (cos(jπ/3) x̂ + sin(jπ/3) ŷ)
```

For j = 1, 2, 3, 4, 5, 6:

```
k₁ = k₀ (1, 0)           →   0°
k₂ = k₀ (1/2, √3/2)      →  60°
k₃ = k₀ (-1/2, √3/2)     → 120°
k₄ = k₀ (-1, 0)          → 180°
k₅ = k₀ (-1/2, -√3/2)    → 240°
k₆ = k₀ (1/2, -√3/2)     → 300°
```

Where k₀ = 2π/a is the fundamental wave number, and a is the lattice spacing.

### Geometric Pattern

The Flower of Life pattern emerges from overlapping circles:
- Each circle has radius a (lattice spacing)
- Circle centers form a hexagonal lattice
- Overlapping regions create the characteristic petal pattern
- The potential minima occur at the petal centers

**Visual Structure**:
```
        ○
      ○   ○
    ○   ●   ○
      ○   ○
        ○
```

Where ○ represents potential maxima (circle edges) and ● represents the central minimum.

### C6 Rotational Symmetry

The Flower of Life potential has C6 symmetry: rotating by 60° leaves the potential unchanged.

**Mathematical Property**:
```
V_FoL(R_60° · r) = V_FoL(r)
```

Where R_60° is the 60° rotation matrix:
```
R_60° = [cos(π/3)  -sin(π/3)]
        [sin(π/3)   cos(π/3)]
```

**Computational Significance**: C6 symmetry creates six equivalent computational pathways, enabling the system to explore multiple solutions simultaneously and naturally handle rotational invariance.

### Potential Landscape Features

1. **Minima (Wells)**: Located at hexagon centers and petal intersections
   - Depth: -6V₀ (when all cosines align)
   - Spacing: a (lattice constant)
   - Attract wave amplitude

2. **Maxima (Barriers)**: Located between minima
   - Height: +6V₀ (when all cosines anti-align)
   - Create energy barriers between wells

3. **Saddle Points**: Transition regions between wells
   - Enable wave tunneling between minima
   - Control computational transitions

### Parameter Selection

**Lattice Spacing (a)**:
- **Smaller a**: More densely packed wells, finer computational granularity
- **Larger a**: Fewer wells, coarser features
- **Typical range**: a ∈ [0.5, 5.0] in normalized units

**Potential Depth (V₀)**:
- **Larger V₀**: Stronger confinement, waves trapped in wells
- **Smaller V₀**: Weaker confinement, waves can move between wells
- **Typical range**: V₀ ∈ [0.1, 10.0] in energy units

**Tuning Guideline**: 
- Start with a = 1.0 and V₀ = 1.0
- Increase V₀ for stronger structure, decrease for more flexibility
- Adjust a based on input data scale

## Metatron's Cube Potential

### Mathematical Definition

Metatron's Cube is a more complex 3D geometric pattern derived from the Flower of Life. It incorporates the five Platonic solids and exhibits rich symmetry.

```
V_MC(r) = V₀ Σ_{vertices} exp(-|r - r_vertex|²/σ²)
```

Where:
- **r_vertex**: Positions of Metatron's Cube vertices
- **σ**: Width of Gaussian wells
- **V₀**: Potential depth

### Vertex Construction

Metatron's Cube vertices include:
1. **Cube vertices**: 8 corners of a cube
2. **Octahedron vertices**: 6 face centers of the cube
3. **Central point**: Origin

Total: 15 vertices forming a highly symmetric 3D structure.

**Cube vertices** (side length 2):
```
(±1, ±1, ±1)  →  8 vertices
```

**Octahedron vertices**:
```
(±1, 0, 0), (0, ±1, 0), (0, 0, ±1)  →  6 vertices
```

**Center**:
```
(0, 0, 0)  →  1 vertex
```

### Symmetry Properties

Metatron's Cube has:
- **Octahedral symmetry** (O_h point group)
- **48 symmetry operations** (rotations and reflections)
- **Incorporates all Platonic solids**:
  - Tetrahedron
  - Cube
  - Octahedron
  - Dodecahedron
  - Icosahedron

**Computational Significance**: The rich symmetry provides multiple equivalent computational pathways and naturally handles 3D rotational invariance.

### Potential Landscape Features

1. **Central Well**: Deepest minimum at origin
   - Attracts global wave amplitude
   - Represents "consensus" or "average" state

2. **Vertex Wells**: 14 surrounding minima
   - Represent distinct computational states
   - Enable multi-stable computation

3. **Edge Connections**: Pathways between vertices
   - Enable transitions between states
   - Form computational graph structure

### Parameter Selection

**Scale (s)**:
- Controls overall size of the structure
- **Typical range**: s ∈ [0.5, 5.0]

**Well Width (σ)**:
- Controls localization of vertex wells
- **Smaller σ**: Sharp, localized wells
- **Larger σ**: Broad, overlapping wells
- **Typical range**: σ ∈ [0.1, 1.0]

**Potential Depth (V₀)**:
- Same role as in Flower of Life
- **Typical range**: V₀ ∈ [0.1, 10.0]

## Comparison: Flower of Life vs Metatron's Cube

| Property | Flower of Life | Metatron's Cube |
|----------|----------------|-----------------|
| Dimensionality | Primarily 2D | Fully 3D |
| Symmetry | C6 (hexagonal) | O_h (octahedral) |
| Number of wells | Infinite (periodic) | 15 (finite) |
| Complexity | Simpler | More complex |
| Computational use | Spatial feature extraction | Multi-state decision making |
| Best for | Image-like data | Graph-like data |

## Implementation in Wave Dynamics

### Incorporation into Schrödinger Equation

The sacred geometry potential appears in the Hamiltonian:

```
iℏ ∂Ψ/∂t = (-ℏ²/2m ∇² + V_FoL(r) + g|Ψ|²)Ψ
```

Or for Metatron's Cube:

```
iℏ ∂Ψ/∂t = (-ℏ²/2m ∇² + V_MC(r) + g|Ψ|²)Ψ
```

### Incorporation into Free Energy Functional

The potential contributes to the Ginzburg-Landau energy:

```
F[Ψ] = ∫ [ (1/2)|∇Ψ|² + (α/2)|Ψ|² + (β/4)|Ψ|⁴ + V(r)|Ψ|² - ΩΨ ] d³r
```

Where V(r) is either V_FoL(r) or V_MC(r).

**Effect on Optimization**: The potential creates preferred regions for wave amplitude, guiding the optimization toward structured solutions.

## Computational Benefits

### 1. Inductive Bias

Sacred geometry potentials encode structural priors:
- **Hexagonal symmetry**: Natural for periodic patterns (crystals, textures)
- **Octahedral symmetry**: Natural for 3D objects and spatial reasoning
- **Multi-well structure**: Enables discrete-like decisions in continuous framework

### 2. Symmetry Exploitation

Symmetries reduce the effective dimensionality of the problem:
- Equivalent configurations under symmetry need not be explored separately
- Faster convergence to solutions
- Natural handling of rotational invariance

### 3. Energy Landscape Structuring

The potential creates a structured energy landscape:
- **Avoids flat regions**: Provides gradient information everywhere
- **Creates attractors**: Stable computational states
- **Enables tunneling**: Transitions between states via saddle points

### 4. Biological Inspiration

Sacred geometry patterns appear in nature:
- **Honeycomb**: Hexagonal packing (optimal for 2D space-filling)
- **Crystal structures**: Lattice symmetries
- **Molecular geometry**: Platonic solids in chemistry

Using these patterns may align computation with natural optimization principles.

## Numerical Implementation

### Grid Discretization

On a spatial grid with points r_i = (x_i, y_i, z_i):

**Flower of Life**:
```julia
function flower_of_life_potential(x, y, k0, V0)
    V = 0.0
    for j in 1:6
        angle = (j-1) * π/3
        kx = k0 * cos(angle)
        ky = k0 * sin(angle)
        V += cos(kx * x + ky * y)
    end
    return V0 * V
end
```

**Metatron's Cube**:
```julia
function metatrons_cube_potential(x, y, z, scale, sigma, V0)
    vertices = [
        # Cube vertices
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        # Octahedron vertices
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
        # Center
        [0,0,0]
    ]
    
    V = 0.0
    for vertex in vertices
        r_vertex = scale * vertex
        dist_sq = (x - r_vertex[1])^2 + (y - r_vertex[2])^2 + (z - r_vertex[3])^2
        V += exp(-dist_sq / sigma^2)
    end
    return -V0 * V  # Negative for attractive wells
end
```

### Visualization

To visualize the potential:
1. **2D Contour Plot**: For Flower of Life in xy-plane
2. **3D Isosurface**: For Metatron's Cube
3. **Slice Plots**: Cross-sections through 3D potentials

### Validation

Test symmetry properties:
```julia
# Test C6 symmetry for Flower of Life
function test_c6_symmetry(V_func, x, y, tolerance=1e-10)
    V_original = V_func(x, y)
    
    # Rotate by 60 degrees
    angle = π/3
    x_rot = x * cos(angle) - y * sin(angle)
    y_rot = x * sin(angle) + y * cos(angle)
    V_rotated = V_func(x_rot, y_rot)
    
    return abs(V_original - V_rotated) < tolerance
end
```

## Advanced Topics

### Hybrid Potentials

Combine multiple geometry types:

```
V_hybrid(r) = w₁ V_FoL(r) + w₂ V_MC(r)
```

Where w₁, w₂ are weights controlling the mixture.

### Adaptive Potentials

Modify the potential during training:
- **Annealing**: Gradually increase V₀ to strengthen structure
- **Morphing**: Interpolate between different geometries
- **Learning**: Adjust parameters based on task performance

### Higher-Dimensional Generalizations

Extend to 4D or higher:
- **4D Flower of Life**: Use 4D hexagonal lattice
- **4D Platonic Solids**: 120-cell, 600-cell, etc.

## Connection to Other Components

- **Schrödinger Equation**: Potential appears in Hamiltonian
- **Ginzburg-Landau Functional**: Potential contributes to free energy
- **Wave Interference**: Potential guides interference patterns
- **Cymatic Extraction**: Standing waves form at potential minima

## Philosophical Note

Sacred geometry has been revered across cultures for millennia. While Aetheria.jl uses these patterns for their mathematical properties (symmetry, structure, efficiency), their appearance in both nature and human design suggests they may represent fundamental principles of organization and computation. Whether this connection is merely aesthetic or reflects deeper truths remains an open question—but the computational benefits are undeniable.

## References

- Weyl, H. (1952). Symmetry. Princeton University Press.
- Lawlor, R. (1982). Sacred Geometry: Philosophy and Practice.
- See `schrodinger_equation.md` for how potential enters dynamics
- See `wave_interference.md` for how potential shapes standing waves
