# runtests.jl — one-command test entry for Sovwave (specs/Guide.md §3.4).
# Includes the evolve harness + its self-test (SECTION-00), then guarded-includes
# every per-module / per-ALG tournament file that exists (sections 01+ append).
using Test

if !isdefined(Main, :Sovwave)
    try
        using Sovwave
    catch
        include(joinpath(@__DIR__, "../src/Sovwave.jl"))
        using .Sovwave
    end
end
const Aetheria = Sovwave

@testset "Sovwave v$(Sovwave.VERSION)" begin

    @testset "SECTION-00 harness" begin
        include("helpers/evolve_harness.jl")

        # load_settings points at specs/settings.json and mirrors it
        cfg = EvolveHarness.load_settings()
        @test cfg["evolve"]["n"] == 12           # currently tuned to 12 (living config)
        @test cfg["evolve"]["x"] == 50
        @test count(k -> !startswith(k, "_"), keys(cfg["metrics"])) == 6
        @test !isempty(cfg["per_alg_overrides"]["ALG-WELL"]["stop_goal"])

        # metric_weights: global six, renormalized sum == 1
        ws, higher = EvolveHarness.metric_weights(cfg, "ALG-ENC")
        @test isapprox(sum(values(ws)), 1.0; atol = 1e-12)
        @test higher["accuracy"] && !higher["speed"]

        # ALG-WELL override applied: accuracy .45 + speed .10 over base
        # -> new sum = .45+.10+.10+.15+.15+.10 = 1.05, normalized .45/1.05 = 3/7
        ws_well, _ = EvolveHarness.metric_weights(cfg, "ALG-WELL")
        @test isapprox(ws_well["accuracy"], 3/7; atol = 1e-12)
        @test isapprox(ws_well["speed"], 0.10/1.05; atol = 1e-12)

        # normalize + composite math (hand-computed):
        # lower-better raws are inverted: speed .4 -> .6, complexity .1 -> .9, energy .3 -> .7
        # composite = .3*.8 + .2*.6 + .1*.9 + .15*.7 + .15*.5 + .1*1.0 = .73
        scores = Dict{String,Float64}(
            "accuracy" => 0.8, "speed" => 0.4, "complexity" => 0.1,
            "energy" => 0.3, "stability" => 0.5, "reproducibility" => 1.0,
        )
        comp = EvolveHarness.composite_score(scores, ws, higher)
        @test isapprox(comp, 0.73; atol = 1e-12)

        # write_round_log writes into logs/theory/<ALG>/ and appends
        path = EvolveHarness.write_round_log(
            "ALG-HARNESS-SELFTEST", 1, "A1",
            ["metric=accuracy raw=0.8", "composite=$comp"],
        )
        @test isfile(path)
        @test occursin("round1-A1.log", path)
        @test occursin("composite=$comp", read(path, String))
        rm(path)
        rm(dirname(path); recursive=true)     # remove empty ALG-HARNESS-SELFTEST dir

        # print_scoreboard renders without throwing
        EvolveHarness.print_scoreboard([
            (variant = "A1", composite = 0.70),
            (variant = "A2", composite = 0.73),
        ])
    end

    # ---------- guarded includes: sections/ALGs add their files here ----------
    for f in (
        # per-ALG tournament files (owned by each section's l-2/l-1 tasks)
        "theory/test_ALG-ENC.jl", "theory/test_ALG-WELL.jl",
        "theory/test_ALG-LAP.jl", "theory/test_ALG-STEP.jl", "theory/test_ALG-FREE.jl",
        "theory/test_ALG-EXT.jl", "theory/test_ALG-PROTO.jl",
        "theory/test_ALG-SOV.jl", "theory/test_ALG-OPT.jl", "theory/test_ALG-NLIN.jl",
        # per-section behaviour files (owned by each section's l gate)
        "encoding_tests.jl", "core_tests.jl", "potentials_tests.jl", "simulation_tests.jl",
        "extraction_tests.jl", "protocols_tests.jl", "networks_tests.jl", "persistence_tests.jl",
        # audio & quantum wave computing suite
        "audio/test_constants.jl", "audio/test_wavefunction.jl", "audio/test_wave_computing.jl", "audio/test_wave_math.jl", "audio/test_waveml.jl", "audio/test_waveml_modules.jl", "audio/test_utf8_multilingual.jl",
        # cross-section suites (final gate)
        "property_tests.jl", "integration_tests.jl",
    )
        path = joinpath(@__DIR__, f)
        if isfile(path)
            include(path)
        else
            @debug "test/$f not implemented yet — skipped (expected until its section lands)"
        end
    end

end
