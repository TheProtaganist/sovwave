#!/usr/bin/env julia
"""
Run all 12 rounds of the Forward Pass 144-Algorithm Tournament
Collects champions from each round and generates final report
"""

using Printf

include("tournament_forward_pass_144.jl")

println("="^80)
println(" 🏆 FORWARD PASS 144-ALGORITHM TOURNAMENT - ALL ROUNDS")
println("="^80)
println(" Running all 12 rounds to completion...")
println(" Total algorithms: 144 (13 baseline + 131 variants)")
println("="^80)

# Store all champions
all_champions = Dict{Int, Tuple{String, ForwardMetrics, Float64}}()

# Run all 12 rounds
for round_num in 1:12
    println("\n" * "🔄"^40)
    @printf(" STARTING ROUND %d/12\n", round_num)
    println("🔄"^40)
    
    round_functions = [
        run_tournament_round1, run_tournament_round2, run_tournament_round3,
        run_tournament_round4, run_tournament_round5, run_tournament_round6,
        run_tournament_round7, run_tournament_round8, run_tournament_round9,
        run_tournament_round10, run_tournament_round11, run_tournament_round12
    ]
    
    champion_name, champion_metrics, champion_score, results = round_functions[round_num]()
    all_champions[round_num] = (champion_name, champion_metrics, champion_score)
    
    @printf("\n✅ Round %d Champion: %s (Score: %.2f)\n", round_num, champion_name, champion_score)
    
    # Short pause between rounds
    sleep(0.5)
end

# ====================================================================================
# FINAL REPORT
# ====================================================================================

println("\n" * "="^80)
println(" 🎉 ALL 12 ROUNDS COMPLETE!")
println("="^80)
println(" Total Algorithms Tested: 144")
println(" Total Rounds: 12")
println("="^80)

println("\n📊 CHAMPIONS BY ROUND:")
println("-"^80)
@printf("%-8s %-30s %10s %10s %10s\n", "Round", "Algorithm", "Score", "Accuracy", "Throughput")
println("-"^80)

for round_num in 1:12
    name, metrics, score = all_champions[round_num]
    @printf("%-8d %-30s %10.2f %10.6f %10.1f K\n", 
            round_num, name, score, metrics.accuracy, metrics.throughput_nodes_sec / 1000.0)
end

println("-"^80)

# Find grand champion (highest score across all rounds)
grand_champion_round = 0
grand_champion_name = ""
grand_champion_score = 0.0
grand_champion_metrics = ForwardMetrics(0.0, 0.0, 1e9, 0, 0.0)

for (round_num, (name, metrics, score)) in all_champions
    if score > grand_champion_score
        grand_champion_score = score
        grand_champion_name = name
        grand_champion_metrics = metrics
        grand_champion_round = round_num
    end
end

println("\n" * "="^80)
println(" 🏆 GRAND CHAMPION (Overall Best)")
println("="^80)
@printf("  Algorithm: %s\n", grand_champion_name)
@printf("  Won in Round: %d\n", grand_champion_round)
@printf("  Final Score: %.2f\n", grand_champion_score)
@printf("  Accuracy: %.6f (%.2f%%)\n", grand_champion_metrics.accuracy, grand_champion_metrics.accuracy * 100)
@printf("  Wave Fidelity: %.4f (%.0f%% wave-native)\n", 
        grand_champion_metrics.wave_fidelity, grand_champion_metrics.wave_fidelity * 100)
@printf("  Throughput: %.1f K nodes/sec\n", grand_champion_metrics.throughput_nodes_sec / 1000.0)
@printf("  Latency: %.2f µs\n", grand_champion_metrics.latency_ns / 1000.0)
@printf("  Memory Allocations: %d\n", grand_champion_metrics.memory_allocs)
println("="^80)

# Performance improvement vs baseline
baseline_score = 35665.49  # From Round 1
improvement = ((grand_champion_score - baseline_score) / baseline_score) * 100

println("\n📈 IMPROVEMENT OVER BASELINE:")
@printf("  Baseline Score: %.2f\n", baseline_score)
@printf("  Champion Score: %.2f\n", grand_champion_score)
@printf("  Improvement: +%.1f%%\n", improvement)

println("\n" * "="^80)
println(" 🎯 TOURNAMENT COMPLETE - Ready for production deployment!")
println("="^80)
