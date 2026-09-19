# Tasks — Section → Task hierarchy with per-section l-2 / l-1 / l tails. Follows agenda §4 components + Instructions phases. See `Guide.md` §3 for the pattern.

> Rule: `SECTION-XX` groups ONE agenda component / phase. `TASK-XX.N` is ONE niche deliverable (one file/behaviour/variant — never a bundle; split on "and").
> Every TASK: `Implements: REQ-XXX` + `Evolves: ALG-XX` (or `—`), full body per Guide §3.2 (Goal/Files/Steps/Metrics/Fixtures/Logs/Accept/Result). Build tasks 10–30 lines; l-2/l-1/l tails 30–100 lines.
> Tail rule: `l` = last task in section. `l-2` = EVOLVE (ONE ALG ONLY, to saturation). `l-1` = SCORE + Winners.md. `l` = TEST GATE (exit forbidden on red). Multi-ALG sections stack one l-2/l-1 pair PER ALG. `settings.json` is living — see Guide §4.
> Status: `- [ ]` open, `- [x]` done. `Implementation/` docs are reference only.

---

## SECTION-00 — Repo bootstrap + skeleton (Instructions Phase 1 fundamentals; NO ALGs → gate only, no evolve/score tail)

### TASK-00.1 — git init + .gitignore
Implements: REQ-BOOT | Evolves: —
Goal: version-controlled repo root with ignore rules so build artefacts never pollute commits.
Files: `.gitignore` (new).
Steps:
1. Run `git init` at repo root (`/home/intender/Desktop/code/Julia/sovwave`).
2. Write `.gitignore`: Julia (`*.jl.cov`, `*.jl.mem`, `/docs/build/`, `.julia/`), Node (`node_modules/`, `dist/`), logs keep dirs via `!logs/*/.gitkeep`, OS (`.DS_Store`, `Thumbs.db`), editor (`.vscode/*` except `settings.json`).
3. Verify each rule: `git check-ignore -v <probe>` for one probe per rule.
4. Stage everything EXCEPT ignored probes; hold first commit until TASK-00.6 done.
Metrics: none (structural).
Fixtures: probe files `x.jl.cov`, `node_modules/y`, `.DS_Store` created then deleted after check.
Logs: none.
Accept: `git status --short` shows only intended files; all probes reported ignored.
- [x] Result: git init on git 2.25.1 (branch stays `master` — rename unsupported pre-first-commit). .gitignore with 5 probe rules verified via `git check-ignore`: *.jl.cov / node_modules/ / .DS_Store / /docs/build/ / Backups/ (all matched). Probes deleted; staging shows only intended files.

### TASK-00.2 — Project.toml + Manifest (zero-deps v1)
Implements: REQ-BOOT, REQ-NODEPS-v1 | Evolves: —
Goal: loadable Julia package identity with zero dependencies (deps promoted only via Theory wins).
Files: `Project.toml` (new), `Manifest.toml` (generated).
Steps:
1. Generate uuid: `julia -e 'using UUIDs; println(uuid4())'`; paste into `Project.toml`.
2. Write `Project.toml`: `name="Aetheria"`, `version="0.1.0"`, `authors=[...]`, `[compat] julia="1.9"`, EMPTY `[deps]`.
3. Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` on Julia 1.10.2; confirm Manifest written.
4. Print `] status` and record output in Result.
Metrics: none (structural).
Fixtures: Julia 1.10.2 (dev machine); compat floor 1.9 documented but not tested v1.
Logs: none (Manifest itself is the lock record).
Accept: instantiate exits 0; `Project.toml` has zero `[deps]` entries; Manifest exists.
- [x] Result: uuid e796aabd-7f07-43f4-9cac-ebbf89ffa53a. Project.toml: name=Aetheria, v0.1.0, julia=1.9 compat. Manifest.toml written; instantiate + Pkg.status exit 0 — status prints "(empty project)": ZERO external deps. Deviations: added stdlib `Dates` dep + `Test` as test-only target (Julia 1.10 requires declared test target); still zero external packages.

### TASK-00.3 — src/Aetheria.jl include-order stub
Implements: REQ-BOOT, REQ-STYLE | Evolves: —
Goal: single module entry point with dependency-ordered includes per Structure.md §2 so later sections only append.
Files: `src/Aetheria.jl` (new).
Steps:
1. Write `module Aetheria` + `const VERSION = v"0.1.0"`.
2. Add guarded includes in order Core→Encoding→Potentials→Simulation→Extraction→Optimization→Loss→Activation→Networks→Training→Inference→Data→Persistence→WaveProtocols (guard pattern: `isfile(...) ? include(...) : @warn ...` until implemented).
3. Add minimal `export` block (VERSION only v1; sections extend).
4. Load check: `julia --project=. -e 'using Aetheria; println(Aetheria.VERSION)'`.
Metrics: none (structural).
Fixtures: none (no src files exist yet — all guards must warn, none must error).
Logs: none.
Accept: loads with warnings only, zero errors; VERSION prints.
- [x] Result: src/Aetheria.jl: module + 45 guarded includes + const VERSION. Loads exit 0; precompile log proves every stub guard warns. Deviation: VERSION NOT exported (colide-s with Base.VERSION on `using`); documented qualified-access pattern `Aetheria.VERSION`.

### TASK-00.4 — test/runtests.jl + helpers/evolve_harness.jl skeleton
Implements: REQ-BOOT, REQ-TEST, REQ-REPRO | Evolves: —
Goal: one-command test entry + shared multi-metric scorer skeleton reading `specs/settings.json`.
Files: `test/runtests.jl` (new), `test/helpers/evolve_harness.jl` (new: `load_settings()`, `composite_score(scores, weights)`, `write_round_log()`, `print_scoreboard()` stubs).
Steps:
1. `runtests.jl`: `@testset "Aetheria"` including `theory/test_ALG-*.jl` + per-section files, each wrapped `isfile(...) ? include(...) : @warn` until its section lands.
2. Harness: implement `load_settings()` (parse `specs/settings.json` with stdlib JSON or manual parse — no new deps), `composite_score` (weighted sum honoring higher_is_better), `write_round_log` (appends `logs/theory/<ALG>/round<R>-variant<V>.log`).
3. Unit-test the harness itself: 3 hand scores → known composite (assert in file).
4. Run `Pkg.test()` empty-green.
Metrics: none (structural; harness self-test asserts exact composite ±1e-12).
Fixtures: synthetic score vectors (no physics yet).
Logs: none yet (harness tested with temp dir).
Accept: `Pkg.test()` passes; harness self-test green; `load_settings()["evolve"]["n"]==3`.
- [x] Result: test/runtests.jl + test/helpers/evolve_harness.jl (zero-dep JSON-subset parser, stdlib only). Pkg.test first green: 12/12. Harness self-tests: settings load (living n=12, x=50), metric_weights + ALG-WELL override (accuracy 3/7), composite hand-math 0.73 ±1e-12, write_round_log append + cleanup, print_scoreboard smoke. Deviation: assert reads living n=12 (not hardcoded 3) per Guide §4.

### TASK-00.5 — web/apps shared shell + gallery index
Implements: REQ-DEMO | Evolves: —
Goal: offline demo shell (shared utils + CSS + gallery) that all 7 section demos plug into.
Files: `web/apps/_shared/wave-utils.js` (new: `SOV` consts placeholder, `assert(cond,msg)`, `fmt()`), `web/apps/_shared/app.css` (new), `web/apps/index.html` (new: 7 cards 01–07).
Steps:
1. Write `wave-utils.js`: `SOV_DEFAULTS` object (hbar,m,g,V0,a,k0 — values finalized in TASK-02.3; placeholder + `TODO(02.3)` comment), `assert` logging to console + on-page badge, `snrDb()` helper.
2. Write `app.css`: dark canvas theme, card grid, readout mono font, status dots (grey pending / green done).
3. Write `index.html`: 7 cards each with title, agenda-§ ref, REQ ref, status dot, link to `../0N-name/index.html`; zero CDN `<script src>` (grep gate).
4. Open via `file://`: screenshot/record console output in Result.
Metrics: none (structural).
Fixtures: none (placeholder consts only).
Logs: none.
Accept: 7 cards render; zero console errors; `grep -r http web/apps` returns only comments/empty.
- [x] Result: _shared/wave-utils.js + _shared/app.css + index.html (7 cards, title/agenda-§/REQ/status-dot). Gates: CDN grep CLEAN (no http refs); node parse OK (new Function); node functional smoke OK (sovSnrDb identical→Infinity, noisy→finite). Demo dirs 01-07 cross-link (land with their sections).

### TASK-00.6 — docs/examples/logs skeleton + section gate (l = 00.6)
Implements: REQ-REPRO | Evolves: —
Goal: SECTION-00 exit gate — dirs, hello example, and proof the skeleton is green end-to-end.
Files: `docs/{api,theory,tutorials}/.gitkeep`, `examples/hello_sov.jl`, `logs/theory/.gitkeep`, `logs/wave-protocols/.gitkeep` (new).
Steps:
1. Create dirs + gitkeeps.
2. `examples/hello_sov.jl`: `using Aetheria; println("SOV ready v", Aetheria.VERSION)`.
3. Run gate suite: `julia --project=. -e 'using Aetheria'` AND `Pkg.test()` AND `julia --project=. examples/hello_sov.jl` — record all three outputs.
4. `git add -A && git commit -m "SECTION-00 skeleton"` (first commit).
Metrics: none (structural).
Fixtures: none.
Logs: none.
Accept (GATE — exit forbidden on red): all three commands exit 0; first commit hash recorded in Result.
- [x] Result: SECTION-00 GATE: G1 `using Aetheria` exit 0 (prints 0.1.0); G2 `Pkg.test()` exit 0 ("Testing Aetheria tests passed"); G3 `examples/hello_sov.jl` exit 0 ("SOV ready v0.1.0"). First commit follows (tag v0.1.0-sov deferred to TASK-08.4).

---

## SECTION-01 — Harmonic Encoder (agenda §4.1: X → Omega spectra, not tensors). ALGs: ALG-ENC.

### TASK-01.1 — FrequencySpectrum struct (immutable spectrum container)
Implements: REQ-ENC | Evolves: —
Goal: canonical spectrum type every later component (sim, extractor, protocols, demos) depends on.
Files: `src/Encoding/FrequencySpectrum.jl` (new).
Steps:
1. Define `struct FrequencySpectrum` with `frequencies::Vector{Float64}`, `amplitudes::Vector{Float64}`, `phases::Vector{Float64}`, `metadata::Dict{Symbol,Any}`; inner constructor asserting equal lengths.
2. Add docstring: agenda §4.1 ref, field units (Hz, linear amplitude, radians), minimal `encode→spectrum→decode` example.
3. Add `Base.show` one-liner (length + freq range) and `Base.length`.
4. Add `export FrequencySpectrum` to `src/Aetheria.jl`.
5. Load check: `julia --project=. -e 'using Aetheria; println(methods(FrequencySpectrum))'`.
Metrics: none (structural).
Fixtures: hand-built 4-bin spectrum (freqs [1,2,3,4], amps [1,0.5,0.25,0.125], phases zeros).
Logs: none.
Accept: constructs; `length` correct; mismatched-length constructor throws `ArgumentError`; load exits 0.
- [ ] Result:

### TASK-01.2 — HarmonicEncoder struct + DFT variant A1 (naive, zero-deps)
Implements: REQ-ENC | Evolves: ALG-ENC
Goal: first working X→Omega path (R1/A1 baseline every later variant is scored against).
Files: `src/Encoding/HarmonicEncoder.jl` (new: struct + A1).
Steps:
1. Define `struct HarmonicEncoder` with `frequency_range::Tuple{Float64,Float64}`, `resolution::Int`, `window::Symbol` (`:none` for A1).
2. Implement `encode(enc, x::AbstractVector{<:Real})::FrequencySpectrum` via naive O(N^2) DFT (no deps); record per-call ms in `metadata[:encode_ms]`.
3. Implement `decode(enc, s::FrequencySpectrum)::Vector{Float64}` via naive inverse DFT.
4. Handle edge inputs explicitly: empty vector → throw; single sample → DC-only spectrum; non-power-of-2 length → must still work (naive DFT has no radix constraint; assert it).
5. Export `HarmonicEncoder, encode, decode`.
Metrics: structural this task (scored in 01.6); record encode_ms per fixture anyway.
Fixtures: sine 5 Hz @64 samples, chirp 2→8 Hz @64, DC constant 32, single-sample, empty (throw-case).
Logs: none yet (first scored in 01.6 R1/A1 row).
Accept: round-trips sine with SNR > 40 dB (weak A1 bar — 60 dB is the 01.6 stop-goal); empty throws; odd lengths work.
- [ ] Result:

### TASK-01.3 — windowed variants A2 (Hann) + A3 (Gaussian)
Implements: REQ-ENC | Evolves: ALG-ENC
Goal: R1 variant inventory complete (n=3: A1 none, A2 Hann, A3 Gaussian) ready for the 01.6 tournament.
Files: `src/Encoding/HarmonicEncoder.jl` (extend: `window=:hann`, `:gaussian` branches + `windowvec()` helper).
Steps:
1. Implement Hann `0.5(1-cos(2πn/(N-1)))` elementwise pre-multiply in `encode` when `window==:hann`.
2. Implement Gaussian `exp(-0.5((n-c)/σ)^2)`, `σ=N/6`, same hook; document σ choice in comment.
3. Windowing MUST apply on encode only; `decode` unchanged (document the resulting amplitude bias in docstring — honest, not hidden).
4. Niche checks: window vector length == N for N in {1,2,3,31,32,33,64}; Hann endpoints ≈ 0; Gaussian peak == 1.0 at center (assert each).
5. Re-run TASK-01.2 fixtures for A2/A3; confirm no throw, spectra sized to resolution.
Metrics: structural (leakage scored in 01.6).
Fixtures: same five as 01.2 + N∈{1,2,3,31,33} window-shape probes.
Logs: none yet.
Accept: A2/A3 construct via `HarmonicEncoder(...; window=:hann/:gaussian)`; shape asserts pass; no API change to `decode`.
- [ ] Result:

### TASK-01.4 — leakage + edge fixture pack (smallest-detail probes)
Implements: REQ-ENC, REQ-TEST | Evolves: ALG-ENC
Goal: fixture granularity that makes the tournament honest — off-bin tones, DC leak, Nyquist edge, noise floors.
Files: `test/encoding_fixtures.jl` (new: pure fixture builders, no asserts).
Steps:
1. Off-bin tones: 5.0 Hz (on-bin) vs 5.3 Hz (off-bin) @64 samples, amplitude 1.0 — exposes leakage differences between A1/A2/A3.
2. DC + Nyquist: constant 1.0 vector; alternating ±1 Nyquist vector — window DC-bias probe.
3. Noise floors: sine + white noise σ ∈ {0.0, 0.01, 0.05}, seeds [1,2,3] from settings.json.
4. Length sweep: N ∈ {8, 16, 32, 64, 128} sine round-trips (scaling data for the speed metric).
5. Each builder deterministic: `StableRNG`-free, `Random.seed!(seed)` stdlib only; document seed in name.
Metrics: none (fixtures only; consumed by 01.6 + 01.8).
Fixtures: THIS task IS fixtures (listed above; 3 noise × 3 seeds × 5 lengths core grid).
Logs: none.
Accept: all builders run; each returns `(x, truth_meta)`; determinism check (same seed twice → identical bits).
- [ ] Result:

### TASK-01.5 — demo 01-encoder (draw → spectrum → decode)
Implements: REQ-DEMO | Evolves: —
Goal: visual proof of §4.1 a human can play with offline; mirrors Julia consts.
Files: `web/apps/01-encoder/index.html` + `web/apps/01-encoder/app.js` (new).
Steps:
1. Canvas: draw-a-signal (pointer) → resample to 64 → naive DFT in JS (mirror of A1, comment `// MIRROR TASK-01.2 A1`) → spectrum bars → inverse DFT overlay + SNR readout.
2. Controls: resolution slider {32,64,128}, window select none/hann/gaussian (same formulas as Julia, comment each).
3. `console.assert` round-trip on built-in 5 Hz sine; on-page PASS/FAIL badge via shared `assert()`.
4. No CDN: `grep -rE "src=\"http|href=\"http" web/apps/01-encoder` must return empty.
5. Cross-check: JS SNR for 5 Hz sine must be within 3 dB of Julia A1 log value (record both in Result).
Metrics: none (visual; numeric mirror tolerance = JS 1e-3 per settings.json).
Fixtures: built-in sine; user-drawn (unasserted, visual only).
Logs: none.
Accept: file:// opens; draw→spectrum→decode live; asserts pass; no network calls.
- [ ] Result:

### TASK-01.6 — EVOLVE ALG-ENC tournament (l-2; ONE ALG ONLY)
Implements: REQ-ENC, REQ-REPRO | Evolves: ALG-ENC
Goal: exhaustive n-per-round evolution of the encoder to saturation per settings.json stop-goal (SNR > 60 dB + leakage below Hann baseline); this SHOULD take a long time — no shortcuts.
Files: `test/theory/test_ALG-ENC.jl` (new: tournament file, dozens of @tests), `src/Encoding/HarmonicEncoder.jl` (variant additions as rounds demand).
Steps:
1. Read `specs/settings.json`: n=3, x=3, seeds=[1,2,3] unless retuned (log any retune + reason here first; living-config rule).
2. R1 (n fresh, parents: none): A1 naive-none (exists, TASK-01.2), A2 Hann (exists, 01.3), A3 Gaussian (exists, 01.3). Run FULL fixture grid from TASK-01.4 (on/off-bin x DC/Nyquist x noise 0/0.01/0.05 x seeds x lengths 8..128) through `test/helpers/evolve_harness.jl` scoring ALL six metrics (accuracy=SNR, speed=encode_ms+scaling slope, complexity=lines, energy=allocs proxy via @allocated, stability=noise degradation slope, reproducibility=seed variance).
3. Score R1 with `composite_score`; record per-variant per-fixture rows to `logs/theory/ALG-ENC/round1-<variant>.log` + print scoreboard.
4. R2 (n children of R1 WINNER): e.g. if A2 wins → B1 Hann+zero-pad x2, B2 Hann+σ-tuned Gaussian hybrid, B3 Hann+DC-restore. Implement each as a `window=` branch (no API break). Same fixture grid + same six metrics. Log `round2-*.log`.
5. R3..x (n children of R2 2ND-winner variants): mutate the runner-up line (e.g. C1/C2/C3 = σ ∈ {N/8, N/6, N/4} sweep if Gaussian-lineage runner-up). Log `round3-*.log`.
6. Past x if stop-goal unmet: continue R4+ with reason logged each round (saturation rule). Stop only on: stop-goal met AND one full round with no composite improvement (saturated), then name the winner + runner-up.
7. Smallest-detail probes inside every round: single-length outliers (N=1,3,33), single-seed reruns, single-noise-level slices — each its own log line, never averaged away silently (report mean ± std).
8. Wire `test/theory/test_ALG-ENC.jl` so `Pkg.test()` runs R-current standings deterministically (tournament full-run guarded by ENV["SOV_FULL_EVOLVE"] for CI speed; default runs smoke subset + asserts log files exist).
Metrics: all six from settings.json; stop-goal = settings.json per_alg_overrides.ALG-ENC.stop_goal.
Fixtures: TASK-01.4 full grid (3 noise x 3 seeds x 5 lengths + on/off-bin + DC/Nyquist + N∈{1,3,33} outliers).
Logs: `logs/theory/ALG-ENC/round<R>-<variant>.log` every round; `logs/theory/ALG-ENC/scoreboard.md` cumulative.
Accept: ≥x rounds logged; stop-goal met + saturation round logged; winner + runner-up named with composites; tournament file runs in Pkg.test().
- [ ] Result:

### TASK-01.7 — SCORE ALG-ENC + WIN-ENC (l-1; same ONE ALG)
Implements: REQ-ENC, REQ-TEST | Evolves: ALG-ENC
Goal: judged scoreboard + documented winner (source comment FIRST, then Winners.md).
Files: winning variant block in `src/Encoding/HarmonicEncoder.jl` (comment), `specs/Winners.md` (WIN-ENC entry), `logs/theory/ALG-ENC/scoreboard.md` (final).
Steps:
1. Assemble final scoreboard: every variant × all six metrics (mean ± std over seeds) + composite; table in scoreboard.md AND in this task Result:.
2. Declare winner + runner-up by composite; tie-break order: accuracy → reproducibility → stability → energy → complexity → speed (document any tie).
3. Per-metric why-won analysis (one line EACH): why winner beat runner-up on accuracy, speed, complexity, energy, stability, reproducibility — no hand-waving, cite numbers.
4. Update source FIRST: `# ALPHA-EVOLVE WINNER: ALG-ENC variant <name> (<date>, fitness <composite>)` + `Why it won:` + `What this does:` atop the winning branch.
5. Write `## WIN-ENC` entry (Chosen, Runner-up, scores table, Why per metric, `Code: src/Encoding/HarmonicEncoder.jl:LINE`, Next-to-try).
6. Pin the winning config as the default `HarmonicEncoder()` kwargs (non-breaking: new default, old values still constructible).
7. Record any settings.json retunes made during 01.6 with reasons.
Metrics: all six (final judging).
Fixtures: same grid as 01.6 (final confirmation run, all seeds).
Logs: `logs/theory/ALG-ENC/scoreboard.md` (final), winner confirmation run log.
Accept: WIN-ENC has zero `pending`; source comment present (grep); default constructor = winner; scoreboard committed.
- [ ] Result:

### TASK-01.8 — SECTION-01 test gate (l = 01.8; exit forbidden on red)
Implements: REQ-ENC, REQ-TEST, REQ-DEMO | Evolves: —
Goal: prove SECTION-01 green end-to-end (unit + tournament standings + demo mirror) before SECTION-02 begins.
Files: `test/encoding_tests.jl` (new: behaviour asserts on WINNER config), `test/theory/test_ALG-ENC.jl` (run), `web/apps/01-encoder` (asserts).
Steps:
1. `test/encoding_tests.jl`: winner-config round-trip SNR asserts (sine/chirp/DC/Nyquist/noise grid, all seeds), edge throws (empty), odd-length ok, window-shape asserts (from 01.3), determinism assert.
2. Run `julia --project=. -e 'using Pkg; Pkg.test()'`: record FULL output; zero failures/errors allowed (tournament file runs standings subset + log-existence asserts).
3. Demo check: open `web/apps/01-encoder/index.html` file://, confirm PASS badge + JS SNR within 3 dB of Julia winner log.
4. Commit `SECTION-01` with gate output pasted in Result:.
Metrics: gate thresholds = stop-goal (SNR > 60 dB) + all asserts green.
Fixtures: full 01.4 grid on winner config.
Logs: gate run output pasted in Result: (no new log files; tournament logs already exist).
Accept (GATE): Pkg.test() exit 0 with encoding files included; demo PASS; commit hash recorded.
- [ ] Result:
---

## SECTION-02 — Sacred Geometry Wells (agenda §4.2) + Core types (§5.1 support). ALGs: ALG-WELL.

### TASK-02.1 — Core Grid struct (1D/2D/3D domains)
Implements: REQ-SIM | Evolves: —
Goal: spatial domain type every field, potential, and simulator builds on.
Files: `src/Core/Grid.jl` (new).
Steps:
1. Define `struct Grid` with `dimensions::Tuple`, `spacing::Tuple`, `origin::Tuple`; inner constructor asserting `all(spacing .> 0)` and `length` consistency (1..3D).
2. Helpers: `create_grid(dims, spacing; origin=zeros)`, `npoints(g)` (product), `coordinates(g)` (per-axis vectors), `cellvolume(g)`.
3. Edge niches: 1-point dim (length-1 axis legal), non-uniform spacing (assert per-axis), negative origin legal, zero/negative spacing throws.
4. Docstring with 1D + 2D examples; export `Grid, create_grid, npoints, coordinates, cellvolume`.
5. Load check via `using Aetheria`.
Metrics: none (structural).
Fixtures: dims (8,), (8,8), (8,8,8), (1,8) degenerate, spacing (0.5,) vs (0.1,0.2) non-uniform, origin (-1.0,).
Logs: none.
Accept: all fixtures construct; zero/negative spacing throws `ArgumentError`; `npoints` exact.
- [ ] Result:

### TASK-02.2 — Core WaveField struct (Psi state container)
Implements: REQ-SIM | Evolves: —
Goal: the computational state `Psi(r,t)` as a typed container (agenda §5.1).
Files: `src/Core/WaveField.jl` (new).
Steps:
1. Define `struct WaveField` (mutable: time evolves) with `spatial_grid::Grid`, `psi::Array{ComplexF64}`, `time::Float64`, `hbar::Float64`, `mass::Float64`; constructor asserts `size(psi)==dimensions`.
2. Constructors: `create_wave_field(grid; kind=:gaussian/:packet/:plane/:zeros, kwargs...)` — gaussian (center, sigma per axis), packet (k0 envelope), plane (k vector), zeros.
3. Niche checks: psi dtype enforced ComplexF64 (convert, never silent Float); time defaults 0.0; hbar/mass > 0 enforced.
4. Helpers: `total_probability(w)=sum(abs2,psi)*cellvolume` (diagnostic, not physics claim), `copy_field`.
5. Docstring: Psi is a CLASSICAL field for compute (not quantum), agenda §5.1 ref.
Metrics: none (structural).
Fixtures: grids from 02.1 × kinds × sigma ∈ {0.5, 1.0, 2.0}, k0 ∈ {0, 1, 5}.
Logs: none.
Accept: mismatched psi size throws; all kinds construct; `total_probability` finite.
- [ ] Result:

### TASK-02.3 — Core Constants (Julia↔JS single source)
Implements: REQ-SIM, REQ-DEMO | Evolves: —
Goal: one constant set mirrored exactly in JS so demos never drift from Julia.
Files: `src/Core/Constants.jl` (new), `web/apps/_shared/wave-utils.js` (mirror update).
Steps:
1. Define `HBAR=1.0, MASS=1.0, G_DEFAULT=1.0, V0_DEFAULT=1.0, A_DEFAULT=1.0, K0_DEFAULT=2π/1.0, SEED=1` with docstring units + agenda-§ refs.
2. Mirror into `SOV_DEFAULTS` in wave-utils.js with `// MIRROR src/Core/Constants.jl — TASK-02.3, change both` header.
3. Add grep gate note: `grep HBAR src/Core/Constants.jl web/apps/_shared/wave-utils.js` must show matching numerals.
4. Export consts from Aetheria.
Metrics: none (structural).
Fixtures: none (values themselves).
Logs: none.
Accept: numerals identical both files; load + node parse (if node works) clean.
- [ ] Result:

### TASK-02.4 — SacredGeometry abstract + evaluate interface
Implements: REQ-WELL | Evolves: —
Goal: extension hook so custom wells plug in without touching the simulator.
Files: `src/Potentials/SacredGeometry.jl` (new).
Steps:
1. `abstract type SacredGeometryPotential end`; `evaluate_potential(::SacredGeometryPotential, pos)::Float64` fallback throwing `ErrorException` naming the missing method.
2. `evaluate_grid(pot, grid)::Array{Float64}` helper (loops positions → matrix, documents ordering x-fastest).
3. Extension example in docstring (custom struct + method, 5 lines).
4. Export both names.
Metrics: none (structural).
Fixtures: dummy subtype in test asserting fallback throws + helper shape == grid dims.
Logs: none.
Accept: fallback error message names type + function; helper output size exact.
- [ ] Result:

### TASK-02.5 — FlowerOfLife struct + variant A1 (loop-cos6)
Implements: REQ-WELL | Evolves: ALG-WELL
Goal: agenda §5.2 potential, first variant (R1/A1) + C6 baseline.
Files: `src/Potentials/FlowerOfLife.jl` (new: struct `V0,k0,phases` + A1 `evaluate_potential` 6-cos loop).
Steps:
1. Struct + docstring (formula, k_j construction j=1..6 at (j-1)*π/3 — note index convention in comment; agenda §5.2 ref).
2. A1 loop: `V0*sum(cos(kx*x+ky*y+φ))` over 6 angles; support 2D pos + 3D pos (z ignored, documented).
3. Niche probes: origin value == 6V0 (all cos align — assert); far-field bounded |V|≤6V0 (100 random pts seed 1); phase-shifted variant differs (sanity).
4. `evaluate_grid` smoke on 8×8 (finite everywhere).
Metrics: structural (scored in 02.8); record C6 max-err + ms on 256² anyway.
Fixtures: origin, 12 C6 sample pts (radius ∈ {0.3,1.0,2.5} × angles), 100-pt bound set seed 1.
Logs: none yet (first scored in 02.8 R1/A1 row).
Accept: origin == 6V0 exact; bound holds; grid finite.
- [ ] Result:

### TASK-02.6 — MetatronsCube struct (13-vertex Gaussian wells)
Implements: REQ-WELL | Evolves: ALG-WELL
Goal: second well family (agenda §4.2) so the simulator has an alternative attractor.
Files: `src/Potentials/MetatronsCube.jl` (new).
Steps:
1. Struct `scale, sigma, V0`; vertex table: cube 8 (±1,±1,±1) + octahedron 6 (axes) + center — 15 entries (document 13-vs-15 counting: 8+6+1=15; name kept per agenda).
2. `evaluate_potential` = `-V0*Σ exp(-|r-s·v|²/σ²)` (attractive wells, negative).
3. Niche checks: center value == -V0*(8e^{-3s²/σ²}+6e^{-s²/σ²}+1) analytic (assert 1e-12); vertex-permutation invariance (swap two vertices → identical); σ→0 locality (far point → ≈0).
4. `evaluate_grid` smoke 8³.
Metrics: structural (not evolved v1; ALG-WELL evolves FoL only — document why: FoL is the agenda default §5.2).
Fixtures: center analytic, permuted table, σ ∈ {0.1, 0.5, 1.0}, scale ∈ {0.5, 1.0, 2.0}.
Logs: none.
Accept: analytic-center assert; permutation-invariant; finite everywhere.
- [ ] Result:

### TASK-02.7 — FoL variants A2 (sum-exp) + A3 (lookup+bilinear)
Implements: REQ-WELL | Evolves: ALG-WELL
Goal: R1 inventory complete (n=3) for the 02.8 tournament.
Files: `src/Potentials/FlowerOfLife.jl` (extend: `mode::Symbol` :loop/:exp/:lookup + precompute table for :lookup).
Steps:
1. A2 sum-exp: `V0*real(Σ exp(i(k·r+φ)))` — algebraically identical, different flop path (comment why: tests trig-vs-complex throughput).
2. A3 lookup: precompute `evaluate_grid` on fine lattice (spacing a/16) at construction; bilinear interp at query; document error-vs-speed trade in comment.
3. Equivalence asserts: A1 vs A2 agree < 1e-12 on 50 pts (seed 2); A3 agrees < 1e-3 (interp tolerance, documented, NOT 1e-12).
4. Table-invalidation rule: mutating V0/k0 rebuilds table (assert stale-table guard throws if bypassed).
5. Re-run 02.5 probes for A2/A3.
Metrics: structural (equivalence + timings recorded for 02.8).
Fixtures: 50-pt equivalence set seed 2 + origin + 12 C6 pts.
Logs: none yet.
Accept: A1/A2 1e-12 agree; A3 within 1e-3; stale-guard works.
- [ ] Result:

### TASK-02.8 — C6 + equivalence fixture pack (smallest-detail probes)
Implements: REQ-WELL, REQ-TEST | Evolves: ALG-WELL
Goal: honest fixture grid for the 02.10 tournament — symmetry, bounds, lattice phases.
Files: `test/potentials_fixtures.jl` (new: builders only).
Steps:
1. C6 rotation set: radii {0.3, 1.0, 2.5} x 12 base angles x 60-degree rotates (exact R60 matrix, not approximate) — err = |V(r)-V(R60 r)| per point, seeds n/a (deterministic geometry).
2. Bound set: 200 pts uniform in [-3a, 3a]^2 seed 1 (|V|<=6V0 assert data); lattice-phase set: phi_j = 0 vs phi_j = j*0.1 (symmetry-break probe, recorded not asserted).
3. Scale sweep: a in {0.5, 1.0, 2.0} x V0 in {0.1, 1.0, 10.0} (9 combos, well-depth sanity).
4. Grid-size timing set: {32^2, 64^2, 128^2, 256^2} evaluate_grid timings (speed-metric scaling slope).
5. Determinism: rerun same seed twice -> identical bits assert.
Metrics: none (fixtures; consumed by 02.10 + 02.12).
Fixtures: THIS task IS fixtures (C6 36 pts + bound 200 + phase 2 + scale 9 + timing 4).
Logs: none.
Accept: builders run; C6 pairs exact-rotated; determinism holds.
- [ ] Result:

### TASK-02.9 — demo 02-wells (heatmap + C6 readout)
Implements: REQ-DEMO | Evolves: —
Goal: see the hex wells + live symmetry proof offline.
Files: `web/apps/02-wells/index.html` + `web/apps/02-wells/app.js` (new).
Steps:
1. Canvas heatmap of V_FoL over [-3a, 3a]^2 (128px, JS mirror of A1 loop, `// MIRROR TASK-02.5 A1`); sliders V0 {0.1..10}, a {0.5..2.0}; click-probe shows V(x,y) numeral.
2. C6 readout: sample 12 pts, rotate 60 deg in JS, max-err display (must read < 1e-3 given Float32 — document tolerance gap vs Julia 1e-10).
3. Mode toggle loop/exp (JS both; lookup omitted with reason comment).
4. console.assert C6 + bound (|V|<=6V0+eps); on-page PASS/FAIL badge.
5. No-CDN grep gate (same as 01.5 step 4).
Metrics: visual; JS tolerance 1e-3 per settings.json.
Fixtures: built-in 12-pt C6 set.
Logs: none.
Accept: hex wells visible; C6 readout < 1e-3; asserts pass; offline clean.
- [ ] Result:

### TASK-02.10 — EVOLVE ALG-WELL tournament (l-2; ONE ALG ONLY)
Implements: REQ-WELL, REQ-REPRO | Evolves: ALG-WELL
Goal: evolve FoL evaluation to saturation (stop-goal: C6 max-err < 1e-10 AND fastest of the accurate variants); long and exhaustive.
Files: `test/theory/test_ALG-WELL.jl` (new, dozens of @tests), `src/Potentials/FlowerOfLife.jl` (round-driven additions).
Steps:
1. Read settings.json (n, x, seeds; log any retune + reason first).
2. R1 (fresh): A1 loop (02.5), A2 sum-exp (02.7), A3 lookup (02.7). Full 02.8 grid through harness: accuracy=C6 max-err, speed=ms on 256^2 + scaling slope, complexity=lines, energy=@allocated x evals, stability=phase-shift + far-field behaviour, reproducibility=rerun variance (must be ~0 — deterministic math; nonzero = bug).
3. Log `logs/theory/ALG-WELL/round1-<variant>.log` + scoreboard; R2 = n children of winner (e.g. winner loop -> B1 loop+@inbounds+@simd, B2 loop+precomputed k table, B3 loop+Float32 staging with error report).
4. R3..x = n children of 2nd-winner line (e.g. lookup-lineage C1/C2/C3 = lattice a/8, a/16, a/32 density sweep).
5. Past x while stop-goal unmet (R4+ logged with reason); stop on stop-goal + one no-improvement round.
6. Smallest-detail slices every round: single-radius C6 rows, single-scale rows, single-size timing rows — own log lines, mean +- std.
7. Wire tournament file into Pkg.test() (full-run behind SOV_FULL_EVOLVE; default = standings + log-existence).
Metrics: all six; accuracy weight 0.45 per settings.json override.
Fixtures: 02.8 full grid.
Logs: `logs/theory/ALG-WELL/round<R>-<variant>.log` + `scoreboard.md`.
Accept: >=x rounds; stop-goal met + saturation logged; winner + runner-up named; file runs in Pkg.test().
- [ ] Result:

### TASK-02.11 — SCORE ALG-WELL + WIN-WELL (l-1; same ONE ALG)
Implements: REQ-WELL, REQ-TEST | Evolves: ALG-WELL
Goal: judged winner with per-metric why + source comment FIRST + Winners.md.
Files: `src/Potentials/FlowerOfLife.jl` (winner comment), `specs/Winners.md` (WIN-WELL), `logs/theory/ALG-WELL/scoreboard.md` (final).
Steps:
1. Final scoreboard: every variant x six metrics (mean +- std) + composite; paste table in Result:.
2. Winner + runner-up by composite; tie-break accuracy -> reproducibility -> stability -> energy -> complexity -> speed.
3. Why-won per metric (six lines, numbers cited): especially accuracy (C6 err) and speed (256^2 ms) trade — e.g. lookup wins speed but loses accuracy: quantify both.
4. Source comment FIRST (`# ALPHA-EVOLVE WINNER: ALG-WELL ...`), then WIN-WELL entry (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `FlowerOfLife()` mode kwarg (non-breaking).
6. Record settings retunes + reasons.
Metrics: all six final.
Fixtures: 02.8 grid confirmation run.
Logs: scoreboard.md final + confirmation log.
Accept: WIN-WELL complete, zero pending; grep finds comment; default = winner.
- [ ] Result:

### TASK-02.12 — SECTION-02 test gate (l = 02.12; exit forbidden on red)
Implements: REQ-WELL, REQ-SIM, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-02 green (Core + wells + tournament + demo) before SECTION-03.
Files: `test/core_tests.jl` (Grid/WaveField/Constants asserts) + `test/potentials_tests.jl` (C6, bounds, equivalence, MC analytic) — new; run `test/theory/test_ALG-WELL.jl`.
Steps:
1. core_tests: construct/throw/size/dtype/mirror-value asserts from 02.1-02.3.
2. potentials_tests: C6 1e-10 (winner), A1/A2 1e-12 agree, A3 1e-3 band, MC analytic + permutation, bound set.
3. Full `Pkg.test()` green; record output. Demo 02-wells PASS badge + C6 < 1e-3.
4. Commit SECTION-02 with output in Result:.
Metrics: gate = stop-goal + all asserts.
Fixtures: 02.8 grid on winner.
Logs: gate output in Result:.
Accept (GATE): exit 0; demo PASS; commit recorded.
- [ ] Result:
---

## SECTION-03 — Morphogenetic Wave Simulator (agenda §4.3 zero-ALU; §5.1+§5.3, NOT backprop). ALGs: ALG-LAP, ALG-STEP, ALG-FREE (ONE evolve + ONE score task EACH — never combined).

### TASK-03.1 — Laplacian variant A1 (2nd-order central FD)
Implements: REQ-SIM | Evolves: ALG-LAP
Goal: baseline discrete Laplacian (R1/A1) with known error order.
Files: `src/Simulation/TimeEvolution.jl` (new: `laplacian!(out, psi, grid)` A1).
Steps:
1. Implement 2nd-order central differences per axis: `(p[i+1]-2p[i]+p[i-1])/h^2`; Dirichlet-zero boundary handling documented (ghost = 0) + Neumann option stub throwing informative error.
2. Niche probes: constant field -> Laplacian == 0 exact (assert 1e-14); linear field -> 0 exact; quadratic x^2 -> 2.0 per axis (assert 1e-9 on interior, document boundary exclusion of 1 ring).
3. Complex support: real + imag parts differenced separately (assert on psi = (1+i)x^2).
4. Non-uniform spacing: per-axis h used (fixture spacing (0.1, 0.2) on 2D quadratic, assert per-axis 2.0).
5. @inbounds policy documented (bounds-checked v1, perf pass belongs to the 03.4 tournament, NOT here).
Metrics: structural (error order scored in 03.4).
Fixtures: constant/linear/quadratic 1D+2D, complex quadratic, non-uniform 2D.
Logs: none yet.
Accept: interior asserts pass; boundary ring excluded + documented; complex correct.
- [ ] Result:

### TASK-03.2 — Laplacian variants A2 (4th-order) + A3 (spectral naive)
Implements: REQ-SIM | Evolves: ALG-LAP
Goal: R1 inventory complete (n=3) for the LAP tournament.
Files: `src/Simulation/TimeEvolution.jl` (extend: `lapmode::Symbol` :o2/:o4/:spectral).
Steps:
1. A2 4th-order 5-point stencil `(-p[i+2]+16p[i+1]-30p[i]+16p[i-1]-p[i-2])/12h^2`; 2-ring boundary exclusion documented.
2. A3 spectral via naive DFT: forward -> multiply -(k^2) -> inverse (zero-deps, periodic BC documented as DIFFERENT from A1/A2 Dirichlet — comparison fixtures use periodic-compatible modes only, stated explicitly).
3. Equivalence: A1 vs A2 on interior of smooth sine (agree < 5e-3 at h=0.1 — different orders, band documented); A3 exact (< 1e-9) on periodic eigenmode.
4. Re-run 03.1 probes for A2/A3 where BC-compatible; record exclusions per probe.
Metrics: structural (scored in 03.4).
Fixtures: 03.1 set + periodic sine eigenmodes k in {1, 2, 4} + h in {0.2, 0.1, 0.05} convergence rows.
Logs: none yet.
Accept: A2 4th-order convergence slope ≈ 4 (log-log, eyeball + recorded); A3 eigen-exact; BC differences documented per probe.
- [ ] Result:

### TASK-03.3 — LAP fixture pack (eigen + convergence + boundary rows)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-LAP
Goal: fixture granularity for an honest Laplacian tournament.
Files: `test/laplacian_fixtures.jl` (new builders).
Steps:
1. Eigen rows: plane waves k in {1,2,4} on periodic grids N in {16, 32, 64} (truth = -k^2 psi).
2. Convergence rows: sine mode, h in {0.2, 0.1, 0.05, 0.025} (slope data for accuracy metric).
3. Boundary rows: Dirichlet quadratic (interior-only asserts), single-point dim edge (N=3 minimum for A2 documented — A2 must throw informative error below 5 pts/axis).
4. Timing rows: N^2 in {32^2, 64^2, 128^2, 256^2} per variant (speed scaling slope).
5. Determinism: builders pure (no RNG except documented seed-9 jitter row for stability).
Metrics: none (fixtures; 03.4 + 03.14).
Fixtures: THIS task IS fixtures (eigen 9 + convergence 4 + boundary 3 + timing 4 + jitter 1).
Logs: none.
Accept: builders run; A2-minimum-size rule throws cleanly; determinism holds.
- [ ] Result:

### TASK-03.4 — EVOLVE ALG-LAP (l-2 LAP; ONE ALG ONLY)
Implements: REQ-SIM, REQ-REPRO | Evolves: ALG-LAP
Goal: evolve the Laplacian to saturation (stop-goal: eigen L2 err < 1e-4 AND bounded drift); exhaustive.
Files: `test/theory/test_ALG-LAP.jl` (new, dozens of @tests), `src/Simulation/TimeEvolution.jl` (round additions).
Steps:
1. Settings read + retune-logging (living-config rule).
2. R1: A1/A2/A3 (exist) on FULL 03.3 grid via harness — accuracy=eigen L2 + convergence slope match, speed=ms + slope, complexity=lines, energy=@allocated, stability=jitter-row variance + Dirichlet/periodic consistency note, reproducibility=rerun variance.
3. Log round1-*.log + scoreboard. R2 = n children of winner (e.g. o4-winner -> B1 o4+@inbounds, B2 o4 compact 4th-order alternative stencil coefficients documented, B3 o4 mixed-precision interior).
4. R3..x = n children of 2nd-winner line (e.g. spectral-lineage C1/C2/C3 = dealias cutoffs {none, 2/3-rule, exponential filter} with error report each).
5. Past x while unmet (R4+ reasoned); stop on stop-goal + saturation round. Smallest-detail slices every round (single-k rows, single-h rows, single-size rows; mean +- std).
6. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-LAP.stop_goal.
Fixtures: 03.3 full grid.
Logs: `logs/theory/ALG-LAP/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-03.5 — SCORE ALG-LAP + WIN-LAP (l-1 LAP; same ONE ALG)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-LAP
Goal: judged LAP winner, comment FIRST, Winners.md second.
Files: `src/Simulation/TimeEvolution.jl` (LAP winner comment), `specs/Winners.md` (WIN-LAP), scoreboard final.
Steps:
1. Final table: variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break accuracy -> reproducibility -> stability -> energy -> complexity -> speed.
3. Six why-lines with numbers (convergence slope achieved? boundary cost? spectral BC caveat quantified?).
4. Source comment FIRST, then WIN-LAP (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `lapmode` (non-breaking).
6. Record retunes.
Metrics: all six final.
Fixtures: 03.3 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-LAP complete; grep comment; default = winner.
- [ ] Result:

### TASK-03.6 — Simulator struct + stepper variant A1 (explicit Euler)
Implements: REQ-SIM | Evolves: ALG-STEP
Goal: agenda §5.1 time evolution container + first stepper (R1/A1).
Files: `src/Simulation/MorphogeneticSimulator.jl` (new: struct + A1).
Steps:
1. Struct: `wavefield::WaveField` (mutable ref), `potential::SacredGeometryPotential`, `g::Float64`, `dt::Float64`, `stepmode::Symbol` (:euler A1); constructor asserts dt > 0, g >= 0.
2. `rhs!(dpsi, sim)` = Schrodinger RHS: kinetic via WIN-LAP Laplacian (-hbar^2/2m factor) + V*psi + g|psi|^2 psi; each term separately unit-probed (zero-potential plane wave -> pure phase rotation rate check).
3. `evolve!(sim, steps::Int)` A1 explicit Euler; `time` advanced by steps*dt; guard `steps >= 0`.
4. Niche: zero-dt throws; negative steps throws; g=0 linear-only path asserted vs analytic phase (1e-6, 10 steps tiny dt).
5. Docstring: zero-ALU direction note (v1 simulates; REQ-PAPER-FIDELITY).
Metrics: structural (stability scored in 03.8).
Fixtures: free particle (V=0,g=0) analytic-phase; harmonic t=0 smoke; dt in {1e-4, 1e-3, 1e-2}.
Logs: none yet.
Accept: analytic-phase assert passes; guards throw; time advances exactly.
- [ ] Result:

### TASK-03.7 — Stepper variants A2 (RK4) + A3 (Crank-Nicolson)
Implements: REQ-SIM | Evolves: ALG-STEP
Goal: R1 inventory complete for the STEP tournament.
Files: `src/Simulation/MorphogeneticSimulator.jl` (extend: :rk4, :cn).
Steps:
1. A2 RK4 on full RHS (4 rhs! evals/step; document 4x cost for the speed metric).
2. A3 Crank-Nicolson: implicit solve on kinetic part (1D tridiagonal Thomas; 2D/3D documented ADI approach or explicit fallback with reason — no silent behaviour change).
3. Per-variant dt-stability probes on free particle (dt doubling ladder until blow-up; record max stable dt each).
4. Cost accounting: rhs!-evals per step recorded per variant (feeds energy metric).
5. Re-run 03.6 fixtures for A2/A3.
Metrics: structural (scored 03.8).
Fixtures: 03.6 set + dt ladder {1e-4 .. 5e-2} x modes.
Logs: none yet.
Accept: max-stable-dt recorded all three; A2 matches analytic tighter than A1 at same dt (assert ratio);
  A3 documented 1D exact-solve vs multi-D approach.
- [ ] Result:

### TASK-03.8 — STEP fixture pack (stability ladders + conservation rows)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-STEP
Goal: fixture granularity for an honest stepper tournament.
Files: `test/stepper_fixtures.jl` (new builders).
Steps:
1. dt ladders: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2} x modes {free particle, gaussian packet, harmonic} x variants (drift + blow-up rows).
2. Conservation rows: probability + energy diagnostics each step (1000-step runs at chosen dt; drift slopes recorded).
3. Nonlinear rows: g in {0, 0.5, 1.0, 5.0} soliton-smoke (boundedness only v1, shape asserted loosely).
4. Cost rows: rhs!-eval counts + @allocated per step per variant (energy/speed inputs).
5. Seeds [1,2,3] on noisy-IC row (reproducibility input).
Metrics: none (fixtures; 03.9 + 03.15).
Fixtures: THIS task IS fixtures (ladders 7 x modes 3 + conservation 3 + nonlinear 4 + cost 3 + seeds 3).
Logs: none.
Accept: builders run; blow-up rows actually blow up for Euler at large dt (record dt_blowup); determinism holds.
- [ ] Result:

### TASK-03.9 — EVOLVE ALG-STEP (l-2 STEP; ONE ALG ONLY)
Implements: REQ-SIM, REQ-REPRO | Evolves: ALG-STEP
Goal: evolve the stepper to saturation (stop-goal: max stable dt recorded AND drift < 1e-6/1000 steps at chosen dt); exhaustive.
Files: `test/theory/test_ALG-STEP.jl` (new, dozens of @tests), `src/Simulation/MorphogeneticSimulator.jl` (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 03.8 grid via harness — accuracy=analytic-phase err + conservation drift, speed=ms/step + rhs!-evals, complexity=lines, energy=allocs x steps, stability=dt_blowup ranking + nonlinear boundedness, reproducibility=seed variance.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. rk4-winner -> B1 rk4+adaptive-dt controller, B2 rk4+@inbounds rhs, B3 rk4-compensated summation).
4. R3..x = n children of 2nd-winner line (e.g. euler-lineage dead? document DEATH explicitly if euler never competitive — killing a line is a result, logged with numbers).
5. Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices every round (single-dt rows, single-mode rows, single-g rows; mean +- std).
6. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-STEP.stop_goal.
Fixtures: 03.8 full grid.
Logs: `logs/theory/ALG-STEP/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-03.10 — SCORE ALG-STEP + WIN-STEP (l-1 STEP; same ONE ALG)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-STEP
Goal: judged STEP winner, comment FIRST, Winners.md second.
Files: `src/Simulation/MorphogeneticSimulator.jl` (STEP winner comment), `specs/Winners.md` (WIN-STEP), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break stability -> accuracy -> reproducibility -> energy -> speed -> complexity (stability first: a stepper must be stable).
3. Six why-lines with numbers (max dt each? drift slopes? cost per step? dead-line documentation?).
4. Source comment FIRST, then WIN-STEP (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `stepmode` (non-breaking).
6. Record retunes.
Metrics: all six final.
Fixtures: 03.8 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-STEP complete; grep comment; default = winner.
- [ ] Result:

### TASK-03.11 — Free energy F variant A1 (naive loops) + derivative
Implements: REQ-SIM | Evolves: ALG-FREE
Goal: agenda §5.3 functional as code (R1/A1) — the object the whole optimizer story descends.
Files: `src/Simulation/EnergyFunctionals.jl` (new: `compute_free_energy(sim)`, `functional_derivative!(out, sim)` A1 loops).
Steps:
1. Terms separately: gradient `1/2|grad|^2` (via WIN-LAP Laplacian/gradient helpers — reuse winner, note dependency), quadratic `a/2|psi|^2`, quartic `b/4|psi|^4`, coupling `-Omega*psi` (Omega passed as array or driving-field fn; document both paths).
2. `functional_derivative!` A1: `-1/2 lap + a/2 psi + b/2 |psi|^2 psi - Omega` per grid point (agenda equilibrium formula, comment each term).
3. Niche: zero-field F == 0 exact; constant-field analytic check (hand-computed, assert 1e-9); coupling-sign check (positive Omega lowers F — assert direction).
4. Params struct `GLParams(a, b)` with b > 0 enforced (stability requirement, agenda §5.3).
5. Export all.
Metrics: structural (scored 03.13).
Fixtures: zero/constant/gaussian fields x Omega {0, uniform, double-well} x (a,b) in {(-0.1,1.0),(0.1,1.0)}.
Logs: none yet.
Accept: analytic asserts pass; sign checks pass; b<=0 throws.
- [ ] Result:

### TASK-03.12 — F variants A2 (vectorized) + A3 (FFT-gradient)
Implements: REQ-SIM | Evolves: ALG-FREE
Goal: R1 inventory complete for the FREE tournament.
Files: `src/Simulation/EnergyFunctionals.jl` (extend: `fmode::Symbol` :loops/:vectorized/:spectral).
Steps:
1. A2 broadcast/vectorized (same math, no loops; comment allocation intent for energy metric).
2. A3 spectral gradient (DFT-based grad; periodic-BC caveat documented like 03.2).
3. Equivalence: A1 vs A2 < 1e-12 on 5 fixtures (assert); A3 within documented band on periodic-compatible fixtures only.
4. Gradient FD-check harness helper (finite-difference of F vs analytic derivative, per-point max-err) — shared with 03.13 scoring.
5. Re-run 03.11 fixtures for A2/A3.
Metrics: structural (scored 03.13).
Fixtures: 03.11 set + periodic-compatible subset flagged.
Logs: none yet.
Accept: A1/A2 1e-12 agree; FD-check helper runs; A3 band documented.
- [ ] Result:

### TASK-03.13 — FREE fixture pack (analytic + FD-check + monotonic rows)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-FREE
Goal: fixtures for the FREE tournament + the relaxation proof.
Files: `test/free_energy_fixtures.jl` (new builders).
Steps:
1. Analytic rows: harmonic-potential ground-state energy (hand formula, assert data) x (a,b) pairs.
2. FD-check rows: random smooth fields seed 1..3, FD epsilon ladder {1e-4..1e-7} (report best-epsilon err per variant).
3. Monotonic rows: double-well Omega + gradient-flow 50 steps (F strictly decreasing assert data; per-step F series recorded).
4. Coupling rows: Omega scale ladder {0, 0.1, 1.0, 10.0} (F-shift direction + magnitude).
5. Timing rows: grid sizes {32^2, 64^2, 128^2} per variant.
Metrics: none (fixtures; 03.14 + 03.16).
Fixtures: THIS task IS fixtures (analytic 4 + FD 3x4eps + monotonic 3 + coupling 4 + timing 3).
Logs: none.
Accept: builders run; analytic rows match hand values (recorded); FD helper converges then diverges with eps (U-shape logged, not hidden).
- [ ] Result:

### TASK-03.14 — EVOLVE ALG-FREE (l-2 FREE; ONE ALG ONLY)
Implements: REQ-SIM, REQ-REPRO | Evolves: ALG-FREE
Goal: evolve F evaluation to saturation (stop-goal: analytic 1e-6 + FD-check 1e-6 + strict monotonicity); exhaustive.
Files: `test/theory/test_ALG-FREE.jl` (new, dozens of @tests), `src/Simulation/EnergyFunctionals.jl` (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 03.13 grid via harness — accuracy=analytic err + FD-check err, speed=ms + scaling, complexity=lines, energy=@allocated, stability=eps-ladder robustness + monotonicity margin, reproducibility=seed variance.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. vectorized-winner -> B1 vectorized+@inbounds+@simd, B2 vectorized fused terms single-pass, B3 vectorized preallocated buffers).
4. R3..x = n children of 2nd-winner line (e.g. spectral-lineage C1/C2/C3 = k-truncations). Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-(a,b) rows, single-eps rows, single-size rows).
5. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-FREE.stop_goal.
Fixtures: 03.13 full grid.
Logs: `logs/theory/ALG-FREE/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-03.15 — SCORE ALG-FREE + WIN-FREE (l-1 FREE; same ONE ALG)
Implements: REQ-SIM, REQ-TEST | Evolves: ALG-FREE
Goal: judged FREE winner, comment FIRST, Winners.md second.
Files: `src/Simulation/EnergyFunctionals.jl` (FREE winner comment), `specs/Winners.md` (WIN-FREE), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break accuracy -> reproducibility -> stability -> energy -> complexity -> speed.
3. Six why-lines with numbers (FD-check errs? monotonic margins? spectral caveat quantified?).
4. Source comment FIRST, then WIN-FREE (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `fmode` (non-breaking).
6. Record retunes.
Metrics: all six final.
Fixtures: 03.13 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-FREE complete; grep comment; default = winner.
- [ ] Result:

### TASK-03.16 — demo 03-simulator (packet + F curve, winner defaults)
Implements: REQ-DEMO | Evolves: —
Goal: watch relaxation happen (uses WIN-LAP/WIN-STEP/WIN-FREE defaults so the demo shows judged code, not arbitrary picks).
Files: `web/apps/03-simulator/index.html` + `web/apps/03-simulator/app.js` (new).
Steps:
1. 1D packet canvas (JS explicit-Euler mirror of A1 + note that Julia default may be the WIN-STEP winner; label which integrator each side uses — honesty).
2. Well overlay (FoL slice), live F-curve plot (JS F mirror of A1 loops + comment), sliders hbar/m/g/dt/V0/a, play/pause/step/reset.
3. Instability demo: dt slider past stability limit shows blow-up (educational; matches 03.8 dt_blowup row qualitatively).
4. console.assert F-series decreases for default dt; PASS/FAIL badge. No-CDN grep gate.
5. Record JS-vs-Julia F qualitative match note (same shape, looser tolerance — documented).
Metrics: visual; JS 1e-3.
Fixtures: built-in packet + double-well Omega preset.
Logs: none.
Accept: F visibly decreases; large dt blows up; asserts pass; offline clean.
- [ ] Result:

### TASK-03.17 — SECTION-03 test gate (l = 03.17; exit forbidden on red)
Implements: REQ-SIM, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-03 green — three tournaments judged, simulator proven, demo honest — before SECTION-04.
Files: `test/simulation_tests.jl` (new: lap-winner interior asserts, stepper analytic-phase + guards, F analytic/sign/monotonic, relaxation dF=0 convergence) + run `test/theory/test_ALG-{LAP,STEP,FREE}.jl`.
Steps:
1. simulation_tests: winner-config asserts from 03.1-03.13 fixture packs (eigen, convergence smoke, analytic-phase, F analytic 1e-9, monotonic strict, b<=0 throws, dt guards).
2. Full `Pkg.test()` green; record output (three tournament files run standings + log-existence).
3. Demo 03-simulator PASS badge check.
4. Commit SECTION-03 with output in Result:.
Metrics: gate = all three stop-goals + asserts.
Fixtures: winner-config subsets of 03.3/03.8/03.13 grids.
Logs: gate output in Result:.
Accept (GATE): exit 0; three WINs complete (LAP/STEP/FREE); demo PASS; commit recorded.
- [ ] Result:
---

## SECTION-04 — Cymatic Eigen-State Extraction (agenda §4.4: standing waves → digital). ALGs: ALG-EXT.

### TASK-04.1 — EigenState struct (decode output container)
Implements: REQ-EXT | Evolves: —
Goal: typed extraction result every downstream consumer (SOV layers, inference, demos) depends on.
Files: `src/Extraction/EigenState.jl` (new).
Steps:
1. Immutable struct: `mode_numbers::Tuple`, `nodes::Vector`, `antinodes::Vector`, `spectrum::FrequencySpectrum`, `label::String`, `energy::Float64`.
2. `Base.show` pretty mode tuple (e.g. `(2,1)`); constructor asserts nodes/antinodes finite, energy finite.
3. Empty-state constructor `EigenState(:empty)` for the no-signal edge (documented, not throwing).
4. Export + docstring with agenda §4.4 ref + extract→label example.
5. Load check.
Metrics: none (structural).
Fixtures: hand states (1,1)/(2,1)/empty; NaN-energy must throw.
Logs: none.
Accept: constructs all three; pretty-prints; NaN throws `ArgumentError`.
- [ ] Result:

### TASK-04.2 — Extractor variant A1 (local-maxima peak-find)
Implements: REQ-EXT | Evolves: ALG-EXT
Goal: first decoder (R1/A1 baseline): peaks → modes.
Files: `src/Extraction/CymaticExtractor.jl` (new: struct `CymaticExtractor(threshold, min_sep)` + A1 `extract(field)::EigenState`).
Steps:
1. A1: local maxima above `threshold` with `min_sep` suppression (document O(P^2) suppression cost for the complexity metric).
2. Mode inference from peak lattice spacings (axis FFT of peak coords? No — A1 is PURE spatial: spacings → integer ratios → mode numbers; document failure mode on noisy fields for the tournament to punish honestly).
3. Nodes as midpoints between antinode rows (document approximation); spectrum via WIN-ENC encoder (reuse winner config if judged, else A1 — note dependency + revisit flag).
4. Niche: single-peak field → mode (1,1)?; flat field → `:empty` (not crash); threshold=0 → every point (guard: assert minimum min_sep > 0).
5. Export.
Metrics: structural (accuracy scored 04.5).
Fixtures: clean synthetic cos modes (1,1),(2,1),(3,2),(4,3) on 64^2; flat; single-peak.
Logs: none yet.
Accept: clean modes decoded (accuracy bar is 04.5's job — here just no-crash + sane output shape); flat → empty.
- [ ] Result:

### TASK-04.3 — Extractor variants A2 (DFT-peak) + A3 (nodal+DFT hybrid)
Implements: REQ-EXT | Evolves: ALG-EXT
Goal: R1 inventory complete (n=3).
Files: `src/Extraction/CymaticExtractor.jl` (extend: `extmode::Symbol` :peaks/:spectral/:hybrid).
Steps:
1. A2 spectral: naive-DFT of field → dominant k-bins → mode numbers (periodic-BC caveat documented; Dirichlet fixtures flagged non-comparable where mismatched).
2. A3 hybrid: nodal-line count (zero-crossing rows/cols) cross-checked against A2 bins; disagree → lower-confidence label suffix `-uncertain` (documented, honest).
3. Confidence field: add `confidence::Float64` to extraction metadata (NOT the struct — metadata dict; struct change needs deprecation note if pursued).
4. Re-run 04.2 fixtures for A2/A3; record per-fixture mode + confidence.
Metrics: structural (scored 04.5).
Fixtures: 04.2 set + mode (5,1) high-order + rotated (2,1) 15° (rotation-robustness probe, expect degradation — recorded).
Logs: none yet.
Accept: all three modes run all fixtures; confidence populated; uncertain-suffix rule triggers on at least one probe (prove the path executes).
- [ ] Result:

### TASK-04.4 — EXT fixture pack (synthetic + noise + edge rows)
Implements: REQ-EXT, REQ-TEST | Evolves: ALG-EXT
Goal: tournament-grade fixtures incl. the noise ladder the stop-goal demands.
Files: `test/extraction_fixtures.jl` (new builders).
Steps:
1. Synthetic grid: modes (m,n) in {1..4}x{1..3} = 12 modes x sizes {32^2, 64^2} (24 clean rows).
2. Noise ladder: σ in {0.0, 0.01, 0.05, 0.1} x seeds [1,2,3] on modes (1,1),(2,1),(3,2) (36 noisy rows; stop-goal judges σ=0.05 slice).
3. Edge rows: flat field (expect :empty), single Gaussian bump (expect low-confidence), two-mode superposition (expect dominant OR uncertain — either accepted IF confidence rule fires correctly).
4. Timing rows: 64^2 vs 128^2 per variant.
5. Determinism: seeded noise reruns identical.
Metrics: none (fixtures; 04.5 + 04.8).
Fixtures: THIS task IS fixtures (24 + 36 + 3 + 2).
Logs: none.
Accept: builders run; all rows finite fields; determinism holds; superposition row documents accepted outcomes.
- [ ] Result:

### TASK-04.5 — demo 04-extraction (nodes/antinodes viz)
Implements: REQ-DEMO | Evolves: —
Goal: SEE standing waves decode (counter-propagating pair → nodes → label).
Files: `web/apps/04-extraction/index.html` + `web/apps/04-extraction/app.js` (new).
Steps:
1. Canvas: analytic (m,n) mode renderer (JS, mirror of fixture builder + comment), node dots (zero lines), antinode glow (maxima), spectrum bars (JS DFT mirror).
2. Controls: mode select (m,n), noise slider σ, variant select peaks/spectral/hybrid (JS mirrors of A1/A2/A3 logic, each commented `// MIRROR TASK-04.x`).
3. Decoded-label readout + confidence; `console.assert` (2,1)-clean decodes (2,1); noisy asserts only warn (honest: JS ≠ judged Julia).
4. PASS/FAIL badge on clean asserts; No-CDN grep gate.
5. Cross-check note: JS variant ranking may differ from Julia tournament — document, do not force agreement.
Metrics: visual; JS 1e-3.
Fixtures: built-in (2,1) clean + σ presets.
Logs: none.
Accept: (2,1) shows correct node count visually + label; asserts on clean pass; offline clean.
- [ ] Result:

### TASK-04.6 — EVOLVE ALG-EXT (l-2; ONE ALG ONLY)
Implements: REQ-EXT, REQ-REPRO | Evolves: ALG-EXT
Goal: evolve the extractor to saturation (stop-goal: 100% clean + >=95% at σ=0.05); exhaustive.
Files: `test/theory/test_ALG-EXT.jl` (new, dozens of @tests), `src/Extraction/CymaticExtractor.jl` (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 04.4 grid via harness — accuracy=mode-hit rate (clean + per-σ slices), speed=ms + scaling, complexity=lines, energy=@allocated, stability=noise-degradation slope + rotation-probe behaviour, reproducibility=seed variance.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. hybrid-winner -> B1 hybrid+adaptive threshold (Otsu-style), B2 hybrid+subpixel peak refine, B3 hybrid+multiscale check).
4. R3..x = n children of 2nd-winner line (e.g. spectral-lineage C1/C2/C3 = zero-pad factors {1,2,4} sweep). Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-mode rows, single-σ rows, single-size rows; mean +- std).
5. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-EXT.stop_goal.
Fixtures: 04.4 full grid.
Logs: `logs/theory/ALG-EXT/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-04.7 — SCORE ALG-EXT + WIN-EXT (l-1; same ONE ALG)
Implements: REQ-EXT, REQ-TEST | Evolves: ALG-EXT
Goal: judged EXT winner, comment FIRST, Winners.md second.
Files: `src/Extraction/CymaticExtractor.jl` (winner comment), `specs/Winners.md` (WIN-EXT), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break accuracy -> stability -> reproducibility -> energy -> complexity -> speed.
3. Six why-lines with numbers (clean-hit deltas? σ=0.05 deltas? rotation behaviour? uncertain-suffix precision/recall?).
4. Source comment FIRST, then WIN-EXT (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `extmode` (non-breaking).
6. Record retunes.
Metrics: all six final.
Fixtures: 04.4 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-EXT complete; grep comment; default = winner.
- [ ] Result:

### TASK-04.8 — SECTION-04 test gate (l = 04.8; exit forbidden on red)
Implements: REQ-EXT, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-04 green (extraction judged + proven + visible) before SECTION-05.
Files: `test/extraction_tests.jl` (new: winner-config mode asserts on 12-mode grid, noise σ=0.05 slice >=95% assert, flat→empty, guard asserts) + run `test/theory/test_ALG-EXT.jl`.
Steps:
1. extraction_tests: full 04.4-grid winner asserts (clean 100%, σ=0.05 slice threshold, edge outcomes incl. uncertain-suffix path).
2. Full `Pkg.test()` green; record output.
3. Demo 04-extraction PASS badge check.
4. Commit SECTION-04 with output in Result:.
Metrics: gate = stop-goal + asserts.
Fixtures: 04.4 grid on winner.
Logs: gate output in Result:.
Accept (GATE): exit 0; WIN-EXT complete; demo PASS; commit recorded.
- [ ] Result:
---

## SECTION-05 — Hardware API simulated + protocol winner (agenda §4.5 + Instructions Phase 2; SIM-ONLY, no SDR/FPGA claims). ALGs: ALG-PROTO.

### TASK-05.1 — WaveProtocol abstract + transmit/receive interface
Implements: REQ-HW | Evolves: —
Goal: common protocol contract so audio/EMF/radio variants are interchangeable in the TestRunner.
Files: `src/WaveProtocols/WaveProtocol.jl` (new).
Steps:
1. `abstract type WaveProtocol end`; interface `transmit(proto, omega::FrequencySpectrum)::Vector{Float64}` + `receive(proto, wave)::FrequencySpectrum` fallbacks throwing named errors.
2. Shared `ProtocolResult(wave, recovered, snr_db, ms, params)` struct (immutable) — every variant returns this (uniform scoring input).
3. Noise-injection helper `add_noise(wave, sigma, seed)` (stdlib RNG only) shared by all sims.
4. Docstring: SIMULATED v1 banner (REQ-PAPER-FIDELITY) + agenda §4.5 ref + Instructions Phase-2 144-algo spirit note (v1 samples n=3/round; scale-up path documented).
5. Export all.
Metrics: none (structural).
Fixtures: dummy proto asserting fallbacks throw; noise helper determinism (same seed twice identical).
Logs: none.
Accept: fallbacks throw with type+fn names; noise deterministic; SIMULATED banner present (grep).
- [ ] Result:

### TASK-05.2 — SignalProcessing synth kit (shared tone/pulse/carrier builders)
Implements: REQ-HW | Evolves: ALG-PROTO
Goal: zero-deps synthesis primitives all three protocol sims share (no duplicated DSP).
Files: `src/WaveProtocols/SignalProcessing.jl` (new).
Steps:
1. `tone(freq, dur, rate)` sine burst; `pulse_train(positions, width, rate)` Gaussian pulses; `carrier(fc, dur, rate)` + `am_modulate(carrier, msg)` sideband builder.
2. Each builder pure + seeded where stochastic (document rate default 8000 Hz sim-rate, NOT audio hardware).
3. Niche: zero-duration → empty (not crash); negative freq → throw; pulse overlap → summed (documented linearity).
4. Unit probes: tone peak at freq bin (DFT check); pulse energy conservation (sum ≈ analytic); AM sidebands at fc±fm (assert bins).
Metrics: structural (scored via protocols in 05.7).
Fixtures: builder self-check set (freqs {100, 440, 1000}, durs {0, 0.01, 0.1}).
Logs: none yet.
Accept: self-checks pass; edge throws documented; no deps beyond stdlib.
- [ ] Result:

### TASK-05.3 — AudioProtocols sim R1/A1 (FSK tone bursts over Omega)
Implements: REQ-HW | Evolves: ALG-PROTO
Goal: first protocol candidate (R1/A1): frequency-shift-keyed tones carrying Omega bins.
Files: `src/WaveProtocols/AudioProtocols.jl` (new: `struct AudioTone <: WaveProtocol` with `base_freq, bin_spacing, tone_dur`).
Steps:
1. `transmit`: map top-K spectrum bins → sequential tone bursts (K param, default 8; document truncation = information loss, scored as accuracy).
2. `receive`: per-burst DFT → peak bin → reconstructed spectrum (top-K only; missing bins zero + flag in metadata).
3. Channel sim: attenuation + ambient-noise σ param (default per settings; sweep in 05.6 fixtures).
4. Niche: K > nbins → clamp + warn (not crash); empty spectrum → silence vector (receive → empty + flag).
5. Log first scores to `logs/wave-protocols/audio-tone-r1.log` (fidelity SNR, stability across seeds, cost ms + samples).
Metrics: structural now (judged 05.7: fidelity→accuracy, stability, cost→speed+energy).
Fixtures: winner-ENC spectra of sine/chirp (K ∈ {4, 8, 16}), noise σ ∈ {0, 0.01, 0.05}.
Logs: `logs/wave-protocols/audio-tone-r1.log`.
Accept: round-trips with SNR logged; clamp/warn path executes; SIMULATED labels in code.
- [ ] Result:

### TASK-05.4 — EMFProtocols sim R1/A2 (pulse-position trains)
Implements: REQ-HW | Evolves: ALG-PROTO
Goal: second candidate (R1/A2): pulse positions encoding bin magnitudes.
Files: `src/WaveProtocols/EMFProtocols.jl` (new: `struct EMFPulse <: WaveProtocol` with `slot_width, slots, amp`).
Steps:
1. `transmit`: quantize top-K magnitudes → pulse positions in slots (document quantization = accuracy loss, scored).
2. `receive`: matched-filter peaks → positions → magnitudes (threshold param; missed pulses flagged).
3. Channel: impulse-noise bursts (sporadic high-σ hits, seed-controlled) + baseline drift probe (document both; drift is the EMF-specific stability test).
4. Niche: slots overflow → saturate + flag; zero magnitude → slot skipped (receive reconstructs zero, asserted).
5. Log `logs/wave-protocols/emf-pulse-r1.log` same rubric as 05.3.
Metrics: structural now (judged 05.7).
Fixtures: same spectra as 05.3 + impulse-burst seeds [1,2,3] + drift slopes {0, 1e-4, 1e-3}.
Logs: `logs/wave-protocols/emf-pulse-r1.log`.
Accept: round-trips logged; overflow/skip paths execute; same rubric fields as audio (harness-mergeable).
- [ ] Result:

### TASK-05.5 — RadioProtocols sim R1/A3 (AM carrier + sidebands)
Implements: REQ-HW | Evolves: ALG-PROTO
Goal: third candidate (R1/A3): AM carrier carrying the spectrum envelope.
Files: `src/WaveProtocols/RadioProtocols.jl` (new: `struct RadioCarrier <: WaveProtocol` with `fc, bandwidth, depth`).
Steps:
1. `transmit`: spectrum envelope → AM modulate carrier (depth param; overmodulation > 1.0 guarded: clamp + flag, documented distortion).
2. `receive`: envelope-detect (rectify + smooth) → resample → spectrum (document smoothing = high-bin loss, scored).
3. Channel: carrier drift Δf + multipath echo (two-tap, seed delays) — the radio-specific stability tests.
4. Niche: fc=0 → baseband path (documented degenerate, still runs); depth=0 → unmodulated (receive → flat + flag).
5. Log `logs/wave-protocols/radio-carrier-r1.log` same rubric.
Metrics: structural now (judged 05.7).
Fixtures: same spectra + drift {0, 5, 20} Hz + echo taps seed-controlled.
Logs: `logs/wave-protocols/radio-carrier-r1.log`.
Accept: round-trips logged; clamp/degenerate paths execute; rubric-mergeable.
- [ ] Result:

### TASK-05.6 — PROTO fixture pack (spectra x channels x seeds grid)
Implements: REQ-HW, REQ-TEST | Evolves: ALG-PROTO
Goal: tournament-grade channel grid — every protocol faces identical conditions.
Files: `test/protocol_fixtures.jl` (new builders).
Steps:
1. Spectra rows: winner-ENC encodings of sine/chirp/DC/Nyquist fixtures (reuse 01.4 builders — dependency noted).
2. Channel rows: clean; AWGN σ ∈ {0, 0.01, 0.05}; impulses (EMF); drift+echo (radio); each x seeds [1,2,3].
3. K sweep rows: top-K ∈ {4, 8, 16} (truncation-loss curve per protocol).
4. Cost rows: message lengths {short 64, long 1024} (speed scaling per protocol).
5. Schema assert: every row carries {proto, spectrum_id, channel, seed, K} (TestRunner joins on this).
Metrics: none (fixtures; 05.7 + 05.10).
Fixtures: THIS task IS fixtures (spectra 4 x channels 5 x seeds 3 x K 3 + cost 2).
Logs: none.
Accept: builders run; schema complete on every row; determinism holds.
- [ ] Result:

### TASK-05.7 — TestRunner (N-variant harness, fixed seeds, rubric-merge)
Implements: REQ-HW, REQ-REPRO | Evolves: ALG-PROTO
Goal: one command scoring ALL protocol variants on the 05.6 grid with the settings.json six metrics.
Files: `src/WaveProtocols/TestRunner.jl` (new: `run_suite(protos, grid)::Vector{ProtocolResult}` + `summarize(results)` table).
Steps:
1. Iterate protos x grid rows: transmit → channel → receive → SNR (accuracy), ms + samples (speed/energy inputs), per-seed variance (reproducibility), per-channel degradation (stability).
2. `summarize` prints markdown table (proto x metric mean +- std) AND returns machine-readable rows for the harness `composite_score`.
3. Determinism: same seeds twice → identical table (assert in test).
4. Cheap-mode flag: `SOV_PROTO_FAST=1` runs clean-channel subset (CI speed; full grid in 05.8 tournament).
5. Export + docstring with usage example.
Metrics: harness itself (judged via 05.8).
Fixtures: 05.6 grid (consumed, not built here).
Logs: `logs/wave-protocols/suite-<date>.log` per run.
Accept: 3 protos x full grid runs; table prints; rerun identical; fast-mode subset documented.
Logs: suite logs per run.
- [ ] Result:

### TASK-05.8 — EVOLVE ALG-PROTO (l-2; ONE ALG ONLY)
Implements: REQ-HW, REQ-REPRO | Evolves: ALG-PROTO
Goal: evolve the protocol choice to saturation (stop-goal: ONE winner by composite across 3 seeds, sim-only); exhaustive.
Files: `test/theory/test_ALG-PROTO.jl` (new, dozens of @tests), protocol files (round additions).
Steps:
1. Settings read + retune-logging (note: PROTO weights stability 0.30 per settings override).
2. R1: A1 audio / A2 EMF / A3 radio (exist, 05.3-05.5) on FULL 05.6 grid via TestRunner+harness — accuracy=recovered-SNR, speed=ms, complexity=lines, energy=samples x ms, stability=channel-degradation slope, reproducibility=seed variance.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. radio-winner -> B1 radio+pre-emphasis, B2 radio+error-repeat K-bins, B3 radio+adaptive depth).
4. R3..x = n children of 2nd-winner line (e.g. audio-lineage C1/C2/C3 = K ∈ {8, 12, 16} + tone_dur sweep). Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-channel rows, single-K rows, single-seed rows).
5. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six (stability-weighted); stop-goal = settings ALG-PROTO.stop_goal.
Fixtures: 05.6 full grid.
Logs: `logs/theory/ALG-PROTO/round<R>-<variant>.log` + `logs/wave-protocols/` channel logs + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-05.9 — SCORE ALG-PROTO + WIN-PROTO (l-1; same ONE ALG) + ProtocolLogger
Implements: REQ-HW, REQ-TEST | Evolves: ALG-PROTO
Goal: judged protocol winner + permanent per-run logs (source comment FIRST, Winners.md second).
Files: `src/WaveProtocols/ProtocolLogger.jl` (new: per-run `.log` + `summary.md` writer), winning protocol file (comment), `specs/Winners.md` (WIN-PROTO), `test/protocols_tests.jl` (log-exists + schema asserts).
Steps:
1. ProtocolLogger: `log_run(result)` appends timestamped row; `write_summary(table)` regenerates `logs/wave-protocols/summary.md`; schema = {proto, variant, spectrum_id, channel, seed, K, snr_db, ms, flags}.
2. Final scoreboard variants x six metrics + composite; paste in Result:.
3. Winner/runner-up by composite; tie-break stability -> accuracy -> reproducibility -> energy -> speed -> complexity.
4. Six why-lines with numbers (which channel separated them? K-sensitivity? cost gap?).
5. Source comment FIRST on winning file, then WIN-PROTO entry (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try) + explicit SIM-ONLY banner in entry.
6. protocols_tests: suite runs fast-mode green; log files exist + schema-valid; rerun deterministic.
7. Record retunes.
Metrics: all six final.
Fixtures: 05.6 confirmation run.
Logs: `logs/wave-protocols/summary.md` final + confirmation.
Accept: WIN-PROTO complete with SIM-ONLY banner; grep comments; tests green; logs schema-valid.
- [ ] Result:

### TASK-05.10 — demo 05-hardware-api (switcher + log viewer, SIM banner)
Implements: REQ-DEMO | Evolves: —
Goal: play with all three protocol sims + see real scores, never mistaking sim for hardware.
Files: `web/apps/05-hardware-api/index.html` + `web/apps/05-hardware-api/app.js` (new).
Steps:
1. Protocol switch (audio/emf/radio, JS mirrors `// MIRROR TASK-05.x`): waveform canvas, recovered-spectrum bars, score table (SNR/stability/cost from Julia logs, pasted constants + date — documented snapshot, not live link).
2. Channel select (clean/noise/impulse/drift) + K slider; big SIMULATED — NO SDR HARDWARE banner (accept-gated: banner text asserted by grep in Result:).
3. console.assert each JS mirror round-trips its built-in sine; PASS/FAIL badge. No-CDN grep gate.
4. Winner badge: marks the 05.9 winner (manual const + date; update task noted).
Metrics: visual; JS 1e-3.
Fixtures: built-in sine per protocol.
Logs: none (displays snapshot of Julia logs).
Accept: switching updates wave + scores; banner present (grep); asserts pass; offline clean.
- [ ] Result:

### TASK-05.11 — SECTION-05 test gate (l = 05.11; exit forbidden on red)
Implements: REQ-HW, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-05 green (protocol judged + logged + visible, zero hardware claims) before SECTION-06.
Files: `test/protocols_tests.jl` (from 05.9) + run `test/theory/test_ALG-PROTO.jl`.
Steps:
1. protocols_tests full-grid (or fast-mode + full-behind-flag documented): asserts on schema, determinism, winner-reproduces-best-SNR smoke.
2. Full `Pkg.test()` green; record output.
3. Honesty grep: `grep -ri "SDR\|FPGA\|hardware" src/WaveProtocols web/apps/05-hardware-api` — every hit must co-occur with SIMULAT* within 3 lines (record hits in Result:).
4. Demo PASS badge. Commit SECTION-05.
Metrics: gate = stop-goal + asserts + honesty-grep clean.
Fixtures: 05.6 grid.
Logs: gate output in Result:.
Accept (GATE): exit 0; WIN-PROTO complete; honesty-grep clean; demo PASS; commit recorded.
- [ ] Result:
---

## SECTION-06 — SOV network + training + inference (Instructions Phase 3: basic THEN advanced; replaces MLP/KAN API shape). ALGs: ALG-SOV, ALG-OPT, ALG-NLIN (ONE evolve + ONE score EACH — never combined).

### TASK-06.1 — SOVLayer abstract + forward contract
Implements: REQ-SOV | Evolves: —
Goal: layer interface every dense/custom layer obeys (forward on wave states, not tensors).
Files: `src/Networks/SOVLayer.jl` (new).
Steps:
1. `abstract type SOVLayer end`; `forward(layer, input::WaveField)::WaveField` fallback error naming type.
2. `forward_batch(layer, inputs::Vector{WaveField})` default (maps forward; documents no batching trick v1).
3. Param-count helper `nparams(layer)::Int` (default 0; dense overrides) — feeds complexity metric honestly.
4. Docstring: SOV-vs-MLP/KAN framing (agenda §2-3) + custom-layer example.
5. Export.
Metrics: none (structural).
Fixtures: dummy layer asserting fallback throws + batch-maps correctly.
Logs: none.
Accept: fallback error names type; batch == map(forward); nparams default 0.
- [ ] Result:

### TASK-06.2 — SOVDense struct + forward variant A1 (potential-then-nonlinear)
Implements: REQ-SOV | Evolves: ALG-SOV
Goal: first SOV layer (R1/A1): encode→well→step→nonlinear→extract INSIDE one forward.
Files: `src/Networks/SOVDense.jl` (new: struct `well, sim_params, nlin, extractor` + A1 forward).
Steps:
1. Struct holds component handles (well::SacredGeometryPotential, stepper config, g, extractor) — composition, not inheritance (Structure.md rule).
2. A1 forward order: input WaveField → apply well potential → WIN-STEP evolve K steps → cubic native nonlinearity (A1 of ALG-NLIN, interim default + revisit flag) → WIN-EXT extract → re-embed label as output WaveField (document re-embed = v1 bridge, honest limitation).
3. Shape asserts: output grid == input grid; output finite; steps param recorded in metadata.
4. Reference-sim check helper: single-layer forward vs hand-rolled sim script agree < 1e-9 (self-consistency, not accuracy — accuracy is 06.4's job).
5. Export.
Metrics: structural (scored 06.4).
Fixtures: gaussian packet input x wells {FoL, MC} x K steps {1, 5, 20} x g {0, 1.0}.
Logs: none yet.
Accept: shapes exact; self-consistency 1e-9; re-embed documented.
- [ ] Result:

### TASK-06.3 — SOVDense variants A2 (split-step) + A3 (eigenbasis project)
Implements: REQ-SOV | Evolves: ALG-SOV
Goal: R1 inventory complete (n=3) for the SOV tournament.
Files: `src/Networks/SOVDense.jl` (extend: `fwdmode::Symbol` :stack/:splitstep/:eigen).
Steps:
1. A2 split-step: alternate kinetic half-step / potential-full / kinetic half-step per sub-step (document 2nd-order splitting error for accuracy metric).
2. A3 eigenbasis: project field onto first M well eigenmodes (M param; eigmodes via WIN-EXT spectral peaks — dependency noted), mix, re-project (document M-truncation = accuracy knob).
3. Cost notes: per-mode op counts recorded (feeds energy metric); M ladder {4, 8, 16} smoke.
4. Re-run 06.2 fixtures for A2/A3; self-consistency per variant (each vs its own hand-roll).
Metrics: structural (scored 06.4).
Fixtures: 06.2 set + M ladder.
Logs: none yet.
Accept: three modes construct via kwarg; self-consistency each < 1e-9; M-truncation documented.
- [ ] Result:

### TASK-06.4 — SOV fixture pack (reference-sim + scaling rows)
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-SOV
Goal: fixtures for the SOV tournament (reference agreement + linear-scaling proof).
Files: `test/sov_fixtures.jl` (new builders).
Steps:
1. Reference rows: 6 hand-rolled sim scripts (packet x wells x K) as ground truth; forward-vs-reference err per variant (accuracy = agreement < 1e-6 stop-goal slice).
2. Scaling rows: grid sizes {16^2, 32^2, 64^2} timings per variant (linear-scaling slope assert data).
3. Mode rows: M ladder {4, 8, 16} for A3 only (truncation curve).
4. g rows: g in {0, 0.5, 1.0} (nonlinearity-on/off slices).
5. Determinism: reruns identical.
Metrics: none (fixtures; 06.5 + 06.19).
Fixtures: THIS task IS fixtures (reference 6 + scaling 3 + M 3 + g 3).
Logs: none.
Accept: builders run; reference scripts independently reviewed (second pair of eyes note in Result:); determinism holds.
- [ ] Result:

### TASK-06.5 — EVOLVE ALG-SOV (l-2 SOV; ONE ALG ONLY)
Implements: REQ-SOV, REQ-REPRO | Evolves: ALG-SOV
Goal: evolve the dense forward to saturation (stop-goal: reference 1e-6 + linear scaling); exhaustive.
Files: `test/theory/test_ALG-SOV.jl` (new, dozens of @tests), `src/Networks/SOVDense.jl` (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 06.4 grid via harness — accuracy=reference err, speed=ms + scaling slope, complexity=nparams + lines, energy=@allocated x steps, stability=g-slice robustness + M-truncation grace, reproducibility=rerun variance.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. splitstep-winner -> B1 splitstep+K-adaptive, B2 splitstep+fused-nonlinear, B3 splitstep half/full mix).
4. R3..x = n children of 2nd-winner line (e.g. eigen-lineage C1/C2/C3 = M {8, 12, 16} + mixing variants). Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-K rows, single-M rows, single-g rows).
5. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-SOV.stop_goal.
Fixtures: 06.4 full grid.
Logs: `logs/theory/ALG-SOV/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-06.6 — SCORE ALG-SOV + WIN-SOV (l-1 SOV; same ONE ALG)
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-SOV
Goal: judged SOV winner, comment FIRST, Winners.md second.
Files: `src/Networks/SOVDense.jl` (SOV winner comment), `specs/Winners.md` (WIN-SOV), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break accuracy -> reproducibility -> stability -> energy -> complexity -> speed.
3. Six why-lines with numbers (reference errs? scaling slopes? M-trade? g-robustness?).
4. Source comment FIRST, then WIN-SOV (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as default `fwdmode` (non-breaking).
6. Record retunes.
Metrics: all six final.
Fixtures: 06.4 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-SOV complete; grep comment; default = winner.
- [ ] Result:

### TASK-06.7 — SOVNetwork composition + batch parity
Implements: REQ-SOV | Evolves: —
Goal: multi-layer nets + batch/single parity (the MLP-replacement API shape).
Files: `src/Networks/SOVNetwork.jl` (new: `layers::Vector{SOVLayer}`, encoder, extractor; `forward(net,x)`, `forward_batch`).
Steps:
1. Chain forwards; per-layer WaveField metadata trace (layer index + mode used — debuggability).
2. Depths {1, 2, 3} smoke on packet input (finite outputs; per-layer energy trace recorded).
3. Batch parity: `forward_batch(net, xs) == map(forward, xs)` < 1e-9 (assert all rows).
4. Depth-scaling note: ms vs depth recorded (feeds demo honesty: deeper = slower, stated).
5. Export.
Metrics: none (structural; judged end-to-end in 06.19).
Fixtures: depths x widths(K steps {1,5}) x batch sizes {1, 4, 8}.
Logs: none.
Accept: parity 1e-9; depths finite; scaling recorded.
- [ ] Result:

### TASK-06.8 — WaveOptimizer abstract + GradientFlow A1 (fixed-gamma)
Implements: REQ-SOV | Evolves: ALG-OPT
Goal: free-energy descent interface + first optimizer (R1/A1): fixed-step gradient flow (agenda §5.3, NOT backprop).
Files: `src/Optimization/WaveOptimizer.jl` (new: abstract + `update!`/`reset!` fallbacks), `src/Optimization/GradientFlow.jl` (new: A1 `gamma` fixed).
Steps:
1. Abstract + fallbacks (error names type, like SOVLayer pattern).
2. A1: `update!(opt, field, grad) = field.psi .-= gamma .* grad` (document: descends F via WIN-FREE derivative; gamma param, default 0.01).
3. Gamma ladder smoke: gamma in {1e-4, 1e-3, 1e-2, 0.1} on double-well fixture (divergence at 0.1 EXPECTED — record, honest).
4. `reset!` no-op default (documented; Adam overrides in 06.9).
5. Export.
Metrics: structural (steps-to-converge scored 06.11).
Fixtures: double-well Omega fixture (shared builder `test/optimization_fixtures.jl` started here: wells depth/positions fixed + seeds).
Logs: none yet.
Accept: descent direction correct (F decreases 1 step, assert); gamma=0.1 diverges (record, not hide); fallbacks throw.
- [ ] Result:

### TASK-06.9 — Optimizer variants A2 (momentum) + A3 (AdamWave)
Implements: REQ-SOV | Evolves: ALG-OPT
Goal: R1 inventory complete (n=3) for the OPT tournament.
Files: `src/Optimization/GradientFlow.jl` (extend: `momentum` mode A2), `src/Optimization/AdamWave.jl` (new: A3 full Adam on WaveField moments).
Steps:
1. A2 momentum: velocity buffer keyed by objectid (document aliasing caveat) `v = mu*v + grad; psi -= gamma*v`, mu default 0.9.
2. A3 AdamWave: first/second moment Dicts + bias correction + eps + weight_decay (defaults lr 1e-3, b1 0.9, b2 0.999, eps 1e-8, wd 0.01 — document provenance: Adam defaults, wave-adapted).
3. State-hygiene niches: `reset!` clears buffers (assert post-reset update == fresh-opt update); two fields interleaved updates do not cross-contaminate (assert).
4. Re-run 06.8 gamma ladder for A2/A3 (lr ladder for A3: {1e-4, 1e-3, 1e-2}).
Metrics: structural (scored 06.11).
Fixtures: 06.8 fixture + lr ladder + interleave pair + reset probe.
Logs: none yet.
Accept: A2/A3 construct; reset-hygiene asserts pass; no cross-contamination.
- [ ] Result:

### TASK-06.10 — OPT fixture pack (wells x gammas x seeds grid)
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-OPT
Goal: tournament-grade optimizer grid (steps-to-|dF|, final F, overshoot rows).
Files: `test/optimization_fixtures.jl` (extend: grids + ladders).
Steps:
1. Well rows: double-well (near + far ICs), harmonic single-well (easy reference), flat+tilt (drift probe) — each with Omega builder + IC field.
2. Rate ladders: gamma/lr per-variant ladders (4 rates each, from 06.8-06.9) — best-rate selection is PART of each variant's score (document: tuning budget equal, 4 rates each, no favoritism).
3. Cap rows: max 2000 steps (|dF|<1e-4 stop); non-converged rows record steps=inf + final F (scored, not dropped).
4. Overshoot rows: F-series recorded per 10 steps (monotonicity-violation counts — stability input).
5. Seeds [1,2,3] on noisy-IC row.
Metrics: none (fixtures; 06.11 + 06.19).
Fixtures: THIS task IS fixtures (wells 4 x rates 4 x seeds 3 + overshoot series).
Logs: none.
Accept: builders run; at least one non-converged row exists at extreme rates (prove the cap executes); determinism holds.
- [ ] Result:

### TASK-06.11 — EVOLVE ALG-OPT (l-2 OPT; ONE ALG ONLY)
Implements: REQ-SOV, REQ-REPRO | Evolves: ALG-OPT
Goal: evolve the descent to saturation (stop-goal: fewest steps to |dF|<1e-4 across seeds); exhaustive.
Files: `test/theory/test_ALG-OPT.jl` (new, dozens of @tests), optimizer files (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 06.10 grid via harness — accuracy=final-F gap to best-known + converge-rate, speed=steps + ms/step, complexity=state-lines, energy=steps x allocs, stability=overshoot counts + far-IC robustness, reproducibility=seed variance of steps.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. adam-winner -> B1 adam+warmup, B2 adam+gradient-clip, B3 adam+gamma-anneal schedule).
4. R3..x = n children of 2nd-winner line (e.g. momentum-lineage C1/C2/C3 = mu {0.8, 0.9, 0.99} sweep). Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-well rows, single-rate rows, single-seed rows).
5. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six; stop-goal = settings ALG-OPT.stop_goal.
Fixtures: 06.10 full grid.
Logs: `logs/theory/ALG-OPT/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-06.12 — SCORE ALG-OPT + WIN-OPT (l-1 OPT; same ONE ALG)
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-OPT
Goal: judged OPT winner, comment FIRST, Winners.md second.
Files: winning optimizer file (comment), `specs/Winners.md` (WIN-OPT), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break reproducibility -> stability -> accuracy -> energy -> speed -> complexity (an optimizer must be reliable first).
3. Six why-lines with numbers (step counts? overshoot diffs? far-IC gaps? rate sensitivity?).
4. Source comment FIRST, then WIN-OPT (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Pin winner as training default (non-breaking; document in TrainingLoop).
6. Record retunes.
Metrics: all six final.
Fixtures: 06.10 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-OPT complete; grep comment; default pinned.
- [ ] Result:

### TASK-06.13 — WaveLoss abstract + FrequencyDomainMSE (ONE formula)
Implements: REQ-SOV | Evolves: —
Goal: first loss (ONE formula, niche): spectral MSE between predicted and target EigenState spectra.
Files: `src/Loss/WaveLoss.jl` (new: abstract + `compute_loss` fallback), `src/Loss/FrequencyDomainMSE.jl` (new).
Steps:
1. Abstract + fallback (names type).
2. FreqMSE: mean(|pred.spec.amps - target.spec.amps|^2) over bins (document bin-alignment requirement: same encoder config, assert equal lengths).
3. Niches: identical inputs -> 0.0 exact; zero-vs-one -> 1.0-scale sane; mismatched lengths -> throw (not silent truncate).
4. Export.
Metrics: none (structural; losses compared qualitatively in 06.19 integration, not evolved — document why: losses are objectives, not algorithms).
Fixtures: hand spectra pairs (identical, shifted, scaled, mismatched).
Logs: none.
Accept: zero-exact; mismatch throws; sane scale.
- [ ] Result:

### TASK-06.14 — PhaseCoherenceLoss (ONE formula)
Implements: REQ-SOV | Evolves: —
Goal: second loss (ONE formula, niche): phase-alignment cost (complements magnitude-only FreqMSE).
Files: `src/Loss/PhaseCoherenceLoss.jl` (new).
Steps:
1. Formula: mean(1 - cos(pred.phase - target.phase)) weighted by mean amplitude (quiet bins do not dominate — document weighting choice).
2. Niches: identical -> 0.0; π-shifted -> 2.0 max (assert); amplitude-zero bins contribute ~0 (assert with zero-amp fixture).
3. Export + docstring contrasting with FreqMSE (when to use which — guidance, not prescription).
Metrics: none (structural).
Fixtures: identical, shifted-π, zero-amp, noisy-phase ladder σ {0, 0.1, 0.5}.
Logs: none.
Accept: bounds [0,2] asserted; weighting behaves; guidance written.
- [ ] Result:

### TASK-06.15 — FreeEnergyLoss (ONE formula)
Implements: REQ-SOV | Evolves: —
Goal: third loss (ONE formula, niche): raw F[Psi] as loss (ties training to the physics objective directly).
Files: `src/Loss/FreeEnergyLoss.jl` (new: wraps WIN-FREE `compute_free_energy`).
Steps:
1. `compute_loss` = F value (document: unbounded-below risk if coupling dominates — assert finite on fixtures, warn in docstring).
2. Niches: relaxed state < perturbed state (assert direction on double-well fixture); zero-field == 0.
3. Export + docstring (when F-loss vs spectral losses: F-loss needs no target spectrum — unsupervised flavor, documented).
Metrics: none (structural).
Fixtures: relaxed vs perturbed pair + zero field.
Logs: none.
Accept: direction assert passes; finite; guidance written.
- [ ] Result:

### TASK-06.16 — WaveActivation abstract + native nonlinearity A1 (cubic GP)
Implements: REQ-SOV | Evolves: ALG-NLIN
Goal: agenda §7 native-nonlinearity story starts here (R1/A1): cubic g|Psi|^2, NO synthetic activations yet.
Files: `src/Activation/WaveActivation.jl` (new: abstract + `activate` fallback), `src/Activation/WaveSigmoid.jl` (new: A1 cubic thin wrapper — honest interim, documented).
Steps:
1. Abstract + fallback (names type).
2. A1 cubic: `activate(field) = g*abs2.(psi).*psi` elementwise (Gross-Pitaevskii form; comment physical meaning: self-focusing intensity potential).
3. Niches: zero field -> zero (assert); g=0 -> identity-ish zero-op (assert no-op); single-spike growth direction sane (sign check).
4. Blow-up smoke: g in {0.1, 1.0, 10.0} x 20 steps (boundedness recorded, NOT asserted yet — 06.18 judges).
5. Export.
Metrics: structural (stability/expressivity scored 06.18).
Fixtures: zero/spike/gaussian fields x g ladder.
Logs: none yet.
Accept: zero/shape asserts pass; smoke runs without crash (bounds recorded).
- [ ] Result:

### TASK-06.17 — Nonlinearity variants A2 (saturable) + A3 (quintic-capped) + ReLU/Tanh wrappers
Implements: REQ-SOV | Evolves: ALG-NLIN
Goal: R1 inventory complete (n=3) + wave-flavoured ReLU/Tanh names for the MLP-migrant API.
Files: `src/Activation/WaveSigmoid.jl` (extend: :cubic/:saturable/:quintic modes), `src/Activation/WaveReLU.jl` + `src/Activation/WaveTanh.jl` (new thin wrappers with documented wave semantics — NOT piecewise clones: ReLU = positive-density gate on |psi|^2, Tanh = saturating phase-preserving squash; formulas in docstrings).
Steps:
1. A2 saturable: `g|psi|^2/(1+s|psi|^2)*psi`, s default 1.0 (document: caps growth → stability candidate).
2. A3 quintic-capped: cubic + `-q|psi|^4 psi` stabilizer, q default 0.1 (document: higher-order brake).
3. Wrappers delegate to a chosen mode (default = interim A1; re-pinned to WIN-NLIN in 06.19 task step).
4. Re-run 06.16 fixtures for A2/A3; record blow-up smoke per variant.
Metrics: structural (scored 06.18).
Fixtures: 06.16 set + s ladder {0.5, 1.0, 2.0} + q ladder {0.05, 0.1, 0.5}.
Logs: none yet.
Accept: three modes construct; wrappers delegate (assert same output as mode); smoke recorded.
- [ ] Result:

### TASK-06.18 — NLIN fixture pack + XOR-wave separability rows
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-NLIN
Goal: tournament fixtures incl. the expressivity proof (linearly-inseparable wave task).
Files: `test/nlin_fixtures.jl` (new builders).
Steps:
1. Stability rows: g in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} x modes {cubic, saturable, quintic} x 100 steps (max|psi| series recorded; blow-up = Inf/NaN flagged, scored not dropped).
2. XOR-wave rows: 4-phase dataset (00/01/10/11 as phase-quadrant packets) → single dense layer → linear readout separability score (accuracy = separability margin; document readout training = 20 fixed-gamma steps, equal budget all variants).
3. s/q ladder rows (from 06.17) on stability slice only (keeps grid affordable).
4. Seeds [1,2,3] on IC jitter row.
5. Timing rows: per-step ms per mode.
Metrics: none (fixtures; 06.19 evolve + 06.23 gate).
Fixtures: THIS task IS fixtures (stability 18 + XOR 4x3seeds + ladders + timing).
Logs: none.
Accept: builders run; at least one cubic blow-up row at high g exists (prove the detector works); XOR rows all finite pre-readout.
- [ ] Result:

### TASK-06.19 — EVOLVE ALG-NLIN (l-2 NLIN; ONE ALG ONLY)
Implements: REQ-SOV, REQ-REPRO | Evolves: ALG-NLIN
Goal: evolve the native nonlinearity to saturation (stop-goal: no blow-up g in [0.1,10] AND XOR separable); exhaustive.
Files: `test/theory/test_ALG-NLIN.jl` (new, dozens of @tests), activation files (round additions).
Steps:
1. Settings read + retune-logging.
2. R1: A1/A2/A3 (exist) on FULL 06.18 grid via harness — accuracy=XOR margin, speed=ms/step, complexity=lines, energy=allocs x steps, stability=blow-up-free g-range width (PRIMARY — a nonlinearity must not explode), reproducibility=seed variance of margin.
3. Log round1 + scoreboard. R2 = n children of winner (e.g. saturable-winner -> B1 saturable s-adaptive, B2 saturable+quintic-brake hybrid, B3 saturable per-mode s).
4. R3..x = n children of 2nd-winner line (e.g. cubic-lineage C1/C2/C3 = g-rescaled + clip hybrids, each blow-up row kept visible — killing cubic honestly if it dies).
5. Past x while unmet (R4+ reasoned); saturation stop. Smallest-detail slices (single-g rows, single-s/q rows, single-seed XOR rows).
6. Wire into Pkg.test() (full behind SOV_FULL_EVOLVE; default standings + log-existence).
Metrics: all six (stability-weighted); stop-goal = settings ALG-NLIN.stop_goal.
Fixtures: 06.18 full grid.
Logs: `logs/theory/ALG-NLIN/round<R>-<variant>.log` + scoreboard.md.
Accept: >=x rounds; stop-goal + saturation logged; winner/runner-up named; runs in Pkg.test().
- [ ] Result:

### TASK-06.20 — SCORE ALG-NLIN + WIN-NLIN (l-1 NLIN; same ONE ALG)
Implements: REQ-SOV, REQ-TEST | Evolves: ALG-NLIN
Goal: judged NLIN winner, comment FIRST, Winners.md second + re-pin wrappers.
Files: winning activation file (comment), `specs/Winners.md` (WIN-NLIN), scoreboard final.
Steps:
1. Final table variants x six metrics + composite; paste in Result:.
2. Winner/runner-up by composite; tie-break stability -> accuracy -> reproducibility -> energy -> complexity -> speed.
3. Six why-lines with numbers (g-range widths? XOR margins? blow-up rows survived?).
4. Source comment FIRST, then WIN-NLIN (Chosen, Runner-up, scores, Why per metric, Code file:line, Next-to-try).
5. Re-pin WaveReLU/WaveTanh/WaveSigmoid defaults to WIN-NLIN mode + SOVDense interim-default flag cleared (06.2 revisit closed here — record).
6. Record retunes.
Metrics: all six final.
Fixtures: 06.18 confirmation run.
Logs: scoreboard.md final + confirmation.
Accept: WIN-NLIN complete; grep comment; wrappers delegate to winner (assert); revisit flag closed.
- [ ] Result:

### TASK-06.21 — Data trio (Loader + Preprocessing + Augmentation, niche-split)
Implements: REQ-SOV | Evolves: —
Goal: dataset support with deterministic batching (three files, three behaviours, one task family — split asserts per file).
Files: `src/Data/DataLoader.jl` (new: `SOVData` vec + `batches(data, bs; shuffle, seed)` iterator), `src/Data/Preprocessing.jl` (new: normalize→Omega-scale via WIN-ENC encoder), `src/Data/Augmentation.jl` (new: freq-shift + noise augment on spectra).
Steps:
1. Loader: shuffle with seed (stdlib), last-batch drop-or-keep flag (documented default keep), empty-data throws.
2. Preprocessing: per-feature z-norm (train stats struct, no leakage: stats from train split only — assert with hand stats), then encode.
3. Augmentation: shift-bins ±k + gaussian noise σ (seeded; document augment-on-spectra not raw — wave-native choice).
4. Tiny synthetic dataset builder (32 samples, 2 classes as phase-quadrant packets — shared with 06.22/06.24): class balance asserted.
5. Export all.
Metrics: none (structural).
Fixtures: 32-sample set x batch sizes {4, 8, 32} x seeds {1,2} + stats-leakage probe.
Logs: none.
Accept: batches cover all samples (count assert); shuffle deterministic per seed; stats-leakage probe passes (train stats ≠ test stats unless identical data).
- [ ] Result:

### TASK-06.22 — TrainingLoop + Validation + EarlyStopping (niche-split trio)
Implements: REQ-SOV | Evolves: —
Goal: train/validate/stop loop using WIN-OPT default + chosen loss (no new optimizer math here — wiring only).
Files: `src/Training/TrainingLoop.jl` (new: `train!(net, data, opt, loss; epochs)` returning trace), `src/Training/Validation.jl` (new: holdout score fn), `src/Training/EarlyStopping.jl` (new: patience struct + `should_stop`).
Steps:
1. Loop: per-epoch forward_batch → loss → WIN-FREE derivative → WIN-OPT update; trace rows {epoch, loss, F} (document: two objectives tracked, loss drives updates, F monitored — honest dual tracking).
2. Validation: split-score on holdout (same metric fns, no grad); overfit-smoke: train tiny 32-set 3 epochs, holdout scored (record gap, no assert direction v1).
3. EarlyStopping: patience on F (default 5, min_delta 1e-6); plateau fixture triggers stop (assert stopped_epoch < max); improving fixture never stops early (assert full run).
4. 5-epoch tiny run: loss AND F decrease epoch0→epoch4 (assert both — the v1 training proof).
5. Export all.
Metrics: none (wiring; end-to-end judged in 06.24 gate via loss/F deltas).
Fixtures: 32-sample set from 06.21 x losses {FreqMSE, FreeEnergy} x patience {2, 5}.
Logs: `logs/training/tiny-run-<date>.log` trace (first training log dir use — create `logs/training/`).
Accept: 5-epoch decreases asserted; early-stop triggers/does-not per fixture; trace logged.
- [ ] Result:

### TASK-06.23 — Inference pair (SingleSample + Batch, parity-gated)
Implements: REQ-SOV | Evolves: —
Goal: prediction API with proven single==batch parity + latency note.
Files: `src/Inference/SingleSample.jl` (new: `predict(net, x)::String` label via forward→WIN-EXT→label), `src/Inference/BatchInference.jl` (new: `predict_batch` + ms/sample log).
Steps:
1. Single: full pipeline on one WaveField (document each stage in trace metadata — debuggable predictions).
2. Batch: maps single (document no fusion v1); parity assert < 1e-9 vs looped singles on 8-sample batch.
3. Latency: ms/sample recorded per net depth {1,2} (feeds demo honesty: real numbers shown).
4. Edge: empty batch → empty result (not crash); malformed input → informative throw.
5. Export.
Metrics: none (structural; latency recorded).
Fixtures: trained tiny net from 06.22 (dependency noted) x batch sizes {1, 8}.
Logs: latency rows in `logs/training/tiny-run` continuation.
Accept: parity 1e-9; edges behave; latency recorded.
- [ ] Result:

### TASK-06.24 — demo 06-sov-network (tiny train viz, honest numbers)
Implements: REQ-DEMO | Evolves: —
Goal: watch a real (tiny) SOV train: dots, curves, eigenstates — with Julia's actual numbers.
Files: `web/apps/06-sov-network/index.html` + `web/apps/06-sov-network/app.js` (new).
Steps:
1. Dataset dots (32-sample 2-class projection), per-epoch F/loss curve (snapshot constants from 06.22 log + date — documented snapshot), eigenstate canvas of current prediction.
2. Train/reset buttons (JS mini-loop mirror, FEWER epochs + note it is illustrative; real proof = Julia log).
3. Latency readout from 06.23 (ms/sample pasted + date).
4. console.assert curve data strictly has final < initial (snapshot assert); PASS/FAIL badge. No-CDN grep gate.
5. Honesty footer: SIMULATED, tiny-data, not a benchmark (accept-gated text present).
Metrics: visual; snapshot numbers cited with dates.
Fixtures: snapshot trace from 06.22.
Logs: none (displays Julia log snapshot).
Accept: curve decreases; footer present (grep); asserts pass; offline clean.
- [ ] Result:

### TASK-06.25 — SECTION-06 test gate (l = 06.25; exit forbidden on red)
Implements: REQ-SOV, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-06 green — three ALGs judged (SOV/OPT/NLIN), full pipeline proven, demo honest — before SECTION-07.
Files: `test/networks_tests.jl` (new: layer/network/loss/activation/data/loader/training/inference asserts) + `test/integration_tests.jl` (new: encode→wells→simulate→extract→loss→update 3-epoch tiny loop, F/loss down) + run `test/theory/test_ALG-{SOV,OPT,NLIN}.jl`.
Steps:
1. networks_tests: winner-config asserts per file family (forward shapes/parity, loss zeros/bounds, activation delegation-to-winner, loader determinism, training decreases, inference parity, early-stop triggers).
2. integration_tests: end-to-end 3 epochs on 32-sample set (loss AND F down vs epoch 0 — THE v1 SOV proof).
3. Full `Pkg.test()` green; record output (three tournament files standings + log-existence).
4. Demo 06 PASS badge. Commit SECTION-06.
Metrics: gate = integration deltas + all asserts + three stop-goals.
Fixtures: 32-sample set + winner configs throughout.
Logs: gate output in Result: (+ `logs/training/` trace).
Accept (GATE): exit 0; WIN-SOV/WIN-OPT/WIN-NLIN complete; integration deltas negative; demo PASS; commit recorded.
- [ ] Result:
---

## SECTION-07 — mp4-of-gifs + gguf compat (Instructions Phase 4; NO ALGs → build + gate only, no evolve/score tail)

### TASK-07.1 — ModelMetadata struct + plain-text save/load round-trip
Implements: REQ-STORE | Evolves: —
Goal: archivable model identity (arch hash, ALG winners used, seeds, F trace, agenda-§ refs) with zero new deps.
Files: `src/Persistence/ModelMetadata.jl` (new: struct + `save_metadata(path, meta)` + `load_metadata(path)`).
Steps:
1. Struct: `arch::String` (layer list + modes), `winners::Dict{String,String}` (ALG→variant, all 10 filled by 07.5 gate time — empty allowed now, asserted later), `seeds::Vector{Int}`, `f_trace::Vector{Float64}`, `refs::Vector{String}` (agenda sections), `julia_version::String`.
2. Serialization: hand-rolled `key=value` + section markers (stdlib only; document why not JSON: zero-deps rule — revisit needs Theory win, logged).
3. Niches: unicode label round-trips (UTF-8 assert); empty trace legal; missing file throws `SystemError` (not silent default); corrupt line → `ArgumentError` naming line number.
4. Arch-hash helper: `arch_hash(net)` deterministic string (layer types + modes; assert same net twice identical, different mode differs).
5. Export all.
Metrics: none (structural).
Fixtures: hand meta (2-layer net, 5-epoch trace) x corrupt variants (bad line, missing field, unicode).
Logs: none.
Accept: save→load identical (== assert); corrupt variants throw named errors; hash deterministic.
- [ ] Result:

### TASK-07.2 — MP4Storage full design doc + stub (gif1..n → mp4 → zip)
Implements: REQ-STORE | Evolves: —
Goal: the mp4-of-gifs design REVIEWABLE in-docstring + stub API that fails cleanly (full encode = v2).
Files: `src/Persistence/MP4Storage.jl` (new).
Steps:
1. Docstring design (20+ lines): per-epoch frequency-frame render (which arrays → which pixels, colormap, resolution) → gif scenes (epochs-per-gif K, default 5) → mp4 container (codec note: needs VideoIO — NOT a v1 dep, stated) → decode path (mp4 → frame extraction → zip of gifs + metadata sidecar).
2. Stubs: `save_model(path, net, trace)` + `load_model(path)` both throw `ErrorException("mp4 encode deferred to v2 — see docstring design")` (exact message asserted in test).
3. Pure helpers that NEED no codec: `frame_spec(trace)` (frame count, dims, K grouping) + `gif_plan(trace)` (scene boundaries) — implemented + tested (real logic, not stubs).
4. Export all three.
Metrics: none (structural; helpers unit-tested).
Fixtures: 5-epoch + 12-epoch traces (scene-boundary edge: 12/5 → scenes [5,5,2] asserted).
Logs: none.
Accept: stubs throw exact message; helpers return exact plans; design doc complete enough to implement v2 from.
- [ ] Result:

### TASK-07.3 — GGUFConverter stub + field-mapping table (mp4 → gguf → quant path)
Implements: REQ-STORE | Evolves: —
Goal: the gguf bridge mapped on paper + stubbed in code (quantization = v2).
Files: `src/Persistence/GGUFConverter.jl` (new).
Steps:
1. Comment mapping table: SOV layer (well params, g, stepper mode, eigen-modes M) → gguf tensor slots (names, dtypes, shapes) — one row per SOV param family; unknown mappings marked `??` honestly (no invented compat).
2. Stubs: `to_gguf(meta, path)` + `quantize(path; bits)` throw `ErrorException("gguf convert deferred to v2")` (exact message asserted).
3. Pure helper: `gguf_plan(meta)` listing convertible vs blocked fields (implemented; blocked reasons cite missing v2 pieces).
4. Export.
Metrics: none (structural).
Fixtures: hand meta from 07.1 (plan output asserted: convertible set exact).
Logs: none.
Accept: stubs throw exact; plan lists blocked with reasons; `??` rows present (honesty checkable by grep).
- [ ] Result:

### TASK-07.4 — demo 07-persistence (frame-strip + metadata viewer, v2 note)
Implements: REQ-DEMO | Evolves: —
Goal: SEE the mp4-of-gifs idea (frame strip from a real trace) + inspect metadata.
Files: `web/apps/07-persistence/index.html` + `web/apps/07-persistence/app.js` (new).
Steps:
1. Frame-strip: per-epoch mini-canvases (freq-bar frames from 06.22 trace snapshot + date), gif-scene boundary markers every K=5 (mirror of 07.2 `gif_plan` logic, commented).
2. Metadata viewer: paste-Julia-metadata-textarea → parse → field table + validity badge (JS mirror of 07.1 format, commented).
3. Big `v2: real mp4 encode + gguf quant live here` note (accept-gated text).
4. console.assert scene-boundary math on 12-epoch fixture ([5,5,2]); PASS/FAIL badge. No-CDN grep gate.
Metrics: visual; snapshot cited with date.
Fixtures: 12-epoch snapshot trace.
Logs: none (displays snapshots).
Accept: strip renders with boundaries; metadata parses + validates; v2 note present; offline clean.
- [ ] Result:

### TASK-07.5 — SECTION-07 test gate (l = 07.5; exit forbidden on red)
Implements: REQ-STORE, REQ-TEST, REQ-DEMO | Evolves: —
Goal: SECTION-07 green (metadata proven, stubs honest, demo visible) before sign-off.
Files: `test/persistence_tests.jl` (new: metadata round-trip incl. unicode, corrupt-line throws with line numbers, stub exact-message asserts, gif_plan boundaries [5,5,2], gguf_plan convertible set).
Steps:
1. persistence_tests: all 07.1-07.3 asserts (round-trip ==, hash determinism, stub messages exact, plans exact).
2. Full `Pkg.test()` green; record output.
3. Demo 07 PASS badge + v2-note grep check.
4. Commit SECTION-07.
Metrics: gate = all asserts + exact-message matches.
Fixtures: 07.1-07.3 sets.
Logs: gate output in Result:.
Accept (GATE): exit 0; stub messages byte-exact; demo PASS; commit recorded.
- [ ] Result:

---

## SECTION-08 — v1 sign-off (agenda §7/§8 honesty + Requirements §4 acceptance; NO ALGs — competition lives in sections, NOT here)

### TASK-08.1 — property fuzz suite (cross-section invariants)
Implements: REQ-TEST | Evolves: —
Goal: randomized invariants tying sections together (the bugs unit tests miss).
Files: `test/property_tests.jl` (new).
Steps:
1. C6 fuzz: 200 random pts (seed 99) on WIN-WELL (err < 1e-10 all — assert max, not mean).
2. Round-trip fuzz: 50 random vectors (lengths 7..130, seeds 100..149) on WIN-ENC (SNR > 60 dB all — assert min).
3. F-monotonic fuzz: 10 random smooth fields + double-well Omega (F decreases every step for 20 steps — assert all series).
4. Seed-variance fuzz: winner configs rerun seeds [7,8,9] (composites within recorded bands — reproducibility proven outside tournament seeds).
5. Determinism: whole file rerun identical (assert by fixed seeds, no wall-clock asserts).
Metrics: gate thresholds = stop-goals (min/max, never mean-only).
Fixtures: THIS task IS fixtures (200 + 50 + 10 + reruns).
Logs: fuzz failures (if any) to `logs/property-failures-<date>.log` with reproducer seed.
Accept: all fuzz asserts pass; no `pending` bands; failure log absent.
- [ ] Result:

### TASK-08.2 — docs + examples (theory per agenda §5, API per module, train-tiny tutorial)
Implements: REQ-REPRO | Evolves: —
Goal: a stranger can learn + run v1 from docs alone.
Files: `docs/theory/waves.md` (agenda §5 retold with file pointers), `docs/api/*.md` (one per src module, generated-by-hand list of exports), `docs/tutorials/train-tiny.md` (copy-paste 32-sample run), `examples/train_tiny.jl` (runnable copy of tutorial).
Steps:
1. Theory page: Schrodinger + FoL + F + extraction, each with equation + `src/...` pointer + winner variant named.
2. API pages: per-module exports + one-line descriptions (assert completeness via script: every `export` appears — record script output).
3. Tutorial + example: identical commands; run example, paste output in Result: (loss AND F decrease visible).
4. Honesty footer on every page (SIMULATED v1, §7 limits quoted once centrally + linked).
Metrics: none (docs; completeness script output recorded).
Fixtures: 32-sample set (same as 06.21).
Logs: example output in Result:.
Accept: example runs exit 0; export-coverage script 100%; footer on all pages (grep).
- [ ] Result:

### TASK-08.3 — honesty pass (sim-vs-physical labels, zero unqualified claims)
Implements: REQ-PAPER-FIDELITY | Evolves: —
Goal: every wave-compute claim in the repo says SIMULATED v1 (agenda §7 SNR/ADC limits quoted, §8 zero-ALU framed as direction).
Files: sweep (demos + docstrings + docs + README if added).
Steps:
1. Grep `zero-ALU|hardware|SDR|FPGA|broadcast|physical|instant` across `src web/apps docs examples specs` — record EVERY hit in Result: with file:line.
2. Each hit must co-occur with `SIMULAT*` within 3 lines OR carry a `v2:`/`direction:` qualifier — fix violators in place.
3. Agenda §7 limits (SNR susceptibility, ADC precision) quoted in `docs/theory/waves.md` + linked from each demo footer (assert links resolve as relative paths).
4. Rerun grep: zero unqualified hits (paste clean output).
Metrics: none (binary gate: clean or not).
Fixtures: none (repo-wide).
Logs: hit list + clean rerun in Result:.
Accept: clean grep output; all demo footers carry honesty text (7/7 asserted by grep count).
- [ ] Result:

### TASK-08.4 — v1 acceptance sign-off (Requirements §4 verbatim; l = 08.4 FINAL GATE)
Implements: REQ-TEST, REQ-REPRO, REQ-PAPER-FIDELITY | Evolves: all
Goal: prove every acceptance line true, then tag.
Files: none new (evidence only).
Steps:
1. `julia --project=. -e 'using Aetheria'` → paste version line.
2. `julia --project=. -e 'using Pkg; Pkg.test()'` FULL → paste tail (0 failures) + counts per test file.
3. Open all 7 demos file:// → 7 PASS badges (list each).
4. Protocol winner: cite WIN-PROTO variant + composite + log path.
5. Tiny train: cite integration-test loss/F deltas (epoch0→end, both negative).
6. Metadata: cite persistence round-trip assert line.
7. Tasks: count `[x]` vs `[ ]` (must be all closed); Winners: grep zero `pending`.
8. `git tag v0.1.0-sov && git log --oneline -3` → paste.
Metrics: acceptance = Requirements §4 lines, each with evidence pointer (no bare claims).
Fixtures: n/a.
Logs: evidence pointers in Result: (test output, log paths, tag).
Accept (FINAL GATE — ship/no-ship): all 8 evidence items present; tag exists; zero open tasks; zero pending WINs.
- [ ] Result:
