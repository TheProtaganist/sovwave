using CUDA
using Sovwave

println("Enabling CUDA...")
enable_cuda!()

println("Running benchmark_system_vs_cuda...")
disp = benchmark_system_vs_cuda(; iters=50, verbose=true)

println("\nVerified Hybrid Dispatcher state:")
println(" • Single Forward Winner:   ", disp.single_forward)
println(" • Batched Forward Winner:  ", disp.batched_forward)
println(" • Vocab Projection Winner: ", disp.vocab_projection)
println(" • Evolution Eval Winner:   ", disp.evolution_eval)
println(" • In-Place Mutation Winner:", disp.mutation)
