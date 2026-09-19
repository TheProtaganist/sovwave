# Theory — Alpha Evolve registry (agenda idea kernels only, ONE ALG per entry)

> Implements the IDEA in `agenda/agenda.md`, not the explanation docs. `Implementation/` = reference for math details.
> Config lives in `specs/settings.json` (LIVING — editable anytime except mid-round, see `Guide.md` §4). Defaults `n=12` variants/round, `x=50` rounds, `seeds=[1,2,3]`.
> Lineage: R1 = n fresh variants (parents: none). R2 = n children of R1 WINNER. R3..x = n children of 2nd-winner line. Past x while stop-goal unmet (R4+ with reason). Stop = stop-goal met + one full no-improvement round (saturated).
> Each ALG is owned by its section's l-2 EVOLVE task, judged in l-1, gated in l. Tournament file: `test/theory/test_ALG-<NAME>.jl` (dozens of @tests, all six metrics, all seeds).

## Template (copy per ALG)
```markdown
## ALG-XX — <kernel> (agenda §) | Section: SECTION-XX | Status: open/evolving/satisfied
Goal: ...
Stop-goal (from settings.json per_alg_overrides): ...
Metric weights: <global six unless overridden — cite override>
Tournament file: `test/theory/test_ALG-XX.jl` | Fixture file: `test/<name>_fixtures.jl`
Evolve task: TASK-XX.(l-2) | Score task: TASK-XX.(l-1) | Winner: WIN-XX
R1 (n fresh, parents: none): A1..An + scores table
R2 (n children of winner <name>): B1..Bn + scores table
R3..x (n children of 2nd-winner <name>): C1..Cn + scores table
Saturation: <round with no improvement + reason, or —>
Promoted: <variant, composite, file:line> → WIN-XX
Settings retunes: <changes + reasons, or none>
```

## ALG-ENC — Harmonic Encoder X→Omega (agenda §4.1) | Section: SECTION-01 | Status: open
Goal: spectral encode/decode replacing tensors. Stop-goal: SNR > 60dB + leakage below Hann baseline (settings.json).
Weights: global six. Tournament: `test/theory/test_ALG-ENC.jl` | Fixtures: `test/encoding_fixtures.jl`.
Evolve: TASK-01.6 (l-2) | Score: TASK-01.7 (l-1) | Winner: WIN-ENC.
R1 (fresh): A1 naive DFT / A2 Hann / A3 Gaussian — scores: — (not run).
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-WELL — Wells eval V_FoL (agenda §4.2/§5.2) | Section: SECTION-02 | Status: open
Goal: fast accurate Flower-of-Life potential. Stop-goal: C6 max-err < 1e-10 AND fastest of accurate (settings.json, accuracy weight 0.45).
Weights: ALG-WELL override (accuracy 0.45, speed 0.10). Tournament: `test/theory/test_ALG-WELL.jl` | Fixtures: `test/potentials_fixtures.jl`.
Evolve: TASK-02.10 (l-2) | Score: TASK-02.11 (l-1) | Winner: WIN-WELL.
R1 (fresh): A1 loop-cos6 / A2 sum-exp / A3 lookup+bilinear — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.
Note: Metatron's Cube NOT evolved v1 (FoL is agenda §5.2 default; TASK-02.6).

## ALG-LAP — Laplacian for simulator (agenda §5.1) | Section: SECTION-03 | Status: open
Goal: discrete Laplacian with known error order. Stop-goal: eigen L2 err < 1e-4 + bounded drift (settings.json).
Weights: global six. Tournament: `test/theory/test_ALG-LAP.jl` | Fixtures: `test/laplacian_fixtures.jl`.
Evolve: TASK-03.4 (l-2 LAP) | Score: TASK-03.5 (l-1 LAP) | Winner: WIN-LAP.
R1 (fresh): A1 2nd-order FD / A2 4th-order / A3 naive-DFT spectral — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-STEP — Time stepper to equilibrium (agenda §5.1/§5.3) | Section: SECTION-03 | Status: open
Goal: stable Psi evolution. Stop-goal: max stable dt + drift < 1e-6/1000 steps (settings.json; tie-break stability-first per TASK-03.10).
Weights: global six. Tournament: `test/theory/test_ALG-STEP.jl` | Fixtures: `test/stepper_fixtures.jl`.
Evolve: TASK-03.9 (l-2 STEP) | Score: TASK-03.10 (l-1 STEP) | Winner: WIN-STEP.
R1 (fresh): A1 Euler / A2 RK4 / A3 Crank-Nicolson — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-FREE — Free energy F + dF/dPsi* (agenda §5.3, replaces backprop) | Section: SECTION-03 | Status: open
Goal: the functional as code. Stop-goal: analytic 1e-6 + FD-check 1e-6 + strict monotonicity (settings.json).
Weights: global six. Tournament: `test/theory/test_ALG-FREE.jl` | Fixtures: `test/free_energy_fixtures.jl`.
Evolve: TASK-03.14 (l-2 FREE) | Score: TASK-03.15 (l-1 FREE) | Winner: WIN-FREE.
R1 (fresh): A1 naive loops / A2 vectorized / A3 FFT-gradient — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-EXT — Cymatic extractor (agenda §4.4) | Section: SECTION-04 | Status: open
Goal: standing waves → digital labels. Stop-goal: 100% clean + ≥95% at σ=0.05 (settings.json).
Weights: global six. Tournament: `test/theory/test_ALG-EXT.jl` | Fixtures: `test/extraction_fixtures.jl`.
Evolve: TASK-04.6 (l-2) | Score: TASK-04.7 (l-1) | Winner: WIN-EXT.
R1 (fresh): A1 peak-find / A2 DFT-peak / A3 nodal+DFT hybrid — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-PROTO — Wave protocol winner (agenda §4.5 sim + Phase 2) | Section: SECTION-05 | Status: open
Goal: ONE winner by composite (SIM scores, NO hardware claims). Stop-goal: settings.json (stability weight 0.30).
Weights: ALG-PROTO override (stability 0.30, accuracy 0.25). Tournament: `test/theory/test_ALG-PROTO.jl` | Fixtures: `test/protocol_fixtures.jl`.
Evolve: TASK-05.8 (l-2) | Score: TASK-05.9 (l-1) | Winner: WIN-PROTO.
R1 (fresh): A1 audio-tone / A2 EMF-pulse / A3 radio-carrier — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-SOV — SOVDense forward (Phase 3 basic) | Section: SECTION-06 | Status: open
Goal: single-layer forward matching reference sim. Stop-goal: reference 1e-6 + linear scaling (settings.json).
Weights: global six. Tournament: `test/theory/test_ALG-SOV.jl` | Fixtures: `test/sov_fixtures.jl`.
Evolve: TASK-06.5 (l-2 SOV) | Score: TASK-06.6 (l-1 SOV) | Winner: WIN-SOV.
R1 (fresh): A1 potential-then-nonlinear / A2 split-step / A3 eigenbasis — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-OPT — Relaxation optimizer, free-energy descent (Phase 3 advanced) | Section: SECTION-06 | Status: open
Goal: fewest steps to |dF|<1e-4. Stop-goal: settings.json (tie-break reproducibility-first per TASK-06.12).
Weights: global six. Tournament: `test/theory/test_ALG-OPT.jl` | Fixtures: `test/optimization_fixtures.jl`.
Evolve: TASK-06.11 (l-2 OPT) | Score: TASK-06.12 (l-1 OPT) | Winner: WIN-OPT.
R1 (fresh): A1 fixed-gamma flow / A2 momentum / A3 AdamWave — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## ALG-NLIN — Native nonlinearity g|Psi|² (agenda §7 strength) | Section: SECTION-06 | Status: open
Goal: stable expressive nonlinearity, no synthetics. Stop-goal: no blow-up g∈[0.1,10] + XOR separable (settings.json, stability-weighted).
Weights: global six. Tournament: `test/theory/test_ALG-NLIN.jl` | Fixtures: `test/nlin_fixtures.jl`.
Evolve: TASK-06.19 (l-2 NLIN) | Score: TASK-06.20 (l-1 NLIN) | Winner: WIN-NLIN.
R1 (fresh): A1 cubic GP / A2 saturable / A3 quintic-capped — scores: —.
R2 (children of winner —): —. R3..x: —. Saturation: —. Promoted: —. Retunes: none.

## Rules
- ONE ALG per evolve task — never combine kernels, even sharing a file (Guide §3.3/§4).
- New round MUST state parent (winner or 2nd-winner) explicitly in the l-2 task Result:.
- Stop = stop-goal met + saturation round. Stop-goal unmet after x → keep going (R4+), reason logged. Round count alone never stops a tournament.
- Promote → source `# ALPHA-EVOLVE WINNER` comment FIRST, then `Winners.md` (l-1 task).
- Losses (FreqMSE/PhaseCoherence/FreeEnergy) are OBJECTIVES, not algorithms — compared qualitatively in integration, never evolved (TASK-06.13 note).
