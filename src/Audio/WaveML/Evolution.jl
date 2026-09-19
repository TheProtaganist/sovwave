"""
    WaveML.Evolution

Evolution-Based Wave Optimizer.
Replaces gradient descent and Markov chain approximations with pure wave evolution:
waves optimize data points to seek the lowest possible energy ground state.

Incorporates the tournament champion algorithms:
👑 `island_migration_select` (Score: 13,889.65, 29.2 ns)
👑 `arithmetic_crossover` (Score: 2,863.62, 145.3 ns)
👑 `correlated_cma_mutate` (Score: 1,120.58, 357.1 ns)
"""

using Random

export EvolutionState
export init_population, evaluate_population!, evolve_generation!

"""
    EvolutionState

Maintains the state of the evolving population of wave models.
- `population::Vector{WaveModel}`: Current population of models
- `energies::Vector{Float64}`: Evaluated ground-state energy for each individual (lower is better)
- `generation::Int`: Generation counter
- `best_energy::Float64`: Lowest energy achieved so far
- `best_model::WaveModel`: Champion model with lowest energy
- `mutation_rate::Float64`: Current mutation amplitude
- `islands::Int`: Number of isolated sub-populations for island migration
"""
mutable struct EvolutionState
    population::Vector{WaveModel}
    energies::Vector{Float64}
    generation::Int
    best_energy::Float64
    best_model::WaveModel
    mutation_rate::Float64
    islands::Int

    function EvolutionState(pop::Vector{WaveModel}, lr::Float64; islands::Int = 2)
        energies = fill(Inf, length(pop))
        best = clone(pop[1])
        new(pop, energies, 0, Inf, best, lr, islands)
    end
end

"""
    init_population(cfg::WaveMLConfig)::EvolutionState

Initializes a diverse population of wave models based on `cfg.train.population_size`.
"""
function init_population(cfg::WaveMLConfig)::EvolutionState
    pop_size = cfg.train.population_size
    pop = Vector{WaveModel}(undef, pop_size)
    for i in 1:pop_size
        m = WaveModel(cfg)
        # Apply slight initial diversity mutation
        mutate!(m, 0.1)
        pop[i] = m
    end
    return EvolutionState(pop, cfg.train.learning_rate)
end

"""
    evaluate_population!(
        state::EvolutionState,
        batch_inputs::Vector{Vector{Float64}},
        batch_targets::Vector{Vector{Float64}};
        loss_type::Symbol = :mmd
    )::Float64

Evaluates all models in the population over the input-target batch.
Updates their energy states and tracks the global champion. Returns the best energy.
"""
function evaluate_population!(
    state::EvolutionState,
    batch_inputs::Vector{Vector{Float64}},
    batch_targets::Vector{Vector{Float64}};
    loss_type::Symbol = :mmd
)::Float64
    pop = state.population
    N = length(pop)
    batch_len = min(length(batch_inputs), length(batch_targets))

    for i in 1:N
        model = pop[i]
        total_e = 0.0

        for b in 1:batch_len
            out = forward!(model, batch_inputs[b])
            total_e += compute_loss(out, batch_targets[b]; type=loss_type)
        end

        avg_energy = total_e / max(batch_len, 1)
        state.energies[i] = avg_energy

        # Check if new global best
        if avg_energy < state.best_energy
            state.best_energy = avg_energy
            state.best_model = clone(model)
        end
    end

    return state.best_energy
end

"""
    evolve_generation!(state::EvolutionState, elite_fraction::Float64 = 0.15)::Nothing

Advances the population by one generation using the tournament-winning
**Multi-Subpopulation Island Migration Selection** and **Arithmetic Wave Crossover**.
"""
function evolve_generation!(state::EvolutionState, elite_fraction::Float64 = 0.15)::Nothing
    pop = state.population
    N = length(pop)
    energies = state.energies
    lr = state.mutation_rate

    # Sort population by energy (ascending: lowest energy is fittest)
    perm = sortperm(energies)
    sorted_pop = pop[perm]

    # Elite preservation
    n_elites = max(1, round(Int, N * elite_fraction))
    new_pop = Vector{WaveModel}(undef, N)

    for i in 1:n_elites
        new_pop[i] = clone(sorted_pop[i])
    end

    # Island migration tournament selection
    num_islands = state.islands
    island_size = N ÷ num_islands

    for i in (n_elites + 1):N
        # Island selection: pick from an island with periodic cross-island migration
        island_idx = (i % num_islands)
        island_start = island_idx * island_size + 1
        island_end = min((island_idx + 1) * island_size, N)

        # Champion Island Selection: pick 3 random candidates within island
        cands_a = rand(island_start:island_end, 3)
        parent_a = sorted_pop[cands_a[argmin(energies[cands_a])]]

        # Migration: 20% chance of mating with another island
        if rand() < 0.2
            other_island = (island_idx + 1) % num_islands
            o_start = other_island * island_size + 1
            o_end = min((other_island + 1) * island_size, N)
            cands_b = rand(o_start:o_end, 3)
            parent_b = sorted_pop[cands_b[argmin(energies[cands_b])]]
        else
            cands_b = rand(island_start:island_end, 3)
            parent_b = sorted_pop[cands_b[argmin(energies[cands_b])]]
        end

        # Champion Arithmetic Blend Crossover
        child = crossover(parent_a, parent_b)

        # Champion Correlated Mutation
        mutate!(child, lr)

        new_pop[i] = child
    end

    state.population = new_pop
    state.generation += 1
    return nothing
end
