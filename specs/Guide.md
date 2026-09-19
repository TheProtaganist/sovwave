# Guide — reuse these specs on ANY project

> Copy `specs/*` to new repo, rename domain terms, keep the loop. Ground-truth rule: specs implement the IDEA paper/agenda, explanation docs are reference only.

## 1. The 7 files

1. `Requirements.md` — DONE definition. EARS style, `REQ-XXX`, editable, never renumber.
2. `Structure.md` — layout bible: every folder/file role + import rules + style + test mirror.
3. `Tasks.md` — all work as `SECTION-XX` groups containing niche `TASK-XX.N` tasks. Each task header: `Implements: REQ-XXX` + `Evolves: ALG-XX` (or `—`).
4. `Theory.md` — Alpha Evolve registry: ONE entry per algorithmic kernel (ONE ALG ONLY per entry). Per ALG: goal, stop-goal, metric weights ref, seeds, round logs, parents explicit (R1 fresh, R2 from winner, R3+ from 2nd-winner).
5. `Winners.md` — final per-ALG winners + why, grouped under section headers. Source `# ALPHA-EVOLVE WINNER` comment FIRST, then entry with `file:line`.
6. `settings.json` — LIVING reference config, NOT a final product. Holds `n` (variants/round), `x` (rounds), `seeds`, metrics + weights + thresholds, per-ALG overrides, tolerances. Editable at ANY time (see §4 protocol). Code reads it; humans tune it.
7. This `Guide.md`.

Defaults: `n=3`, `x=3`, `seeds=[1,2,3]`. Tune anytime.


## 2. Bootstrap a new project

1. Copy `specs/` + keep agenda/idea-paper as ground truth + `Implementation/`-equivalent as explanation reference only.
2. Fill Requirements from agenda components + roadmap phases (functional REQs per component, test/demo/store REQs, out-of-scope, acceptance).
3. Fill Structure from desired stack (root layout, import table, style, langA↔langB contract, deps ladder, `test/` mirror incl. `test/theory/` + `test/helpers/`).
4. Create `settings.json` FIRST (defaults in §3.5), then fill Tasks section by section, each ending in its `l-2 / l-1 / l` tail (see §3.3).
5. Fill Theory with ONE ALG per entry (never bundle two kernels). Leave Winners pending.
6. Work SECTION order; section exit requires its `l` gate green. Log everything; promote winners per section, not at the end.

## 3. Task anatomy — niche tasks + the l-2 / l-1 / l tail

### 3.1 Niche rule (as niche as possible)

- `SECTION-XX` groups ONE agenda component / roadmap phase.
- `TASK-XX.N` is ONE niche deliverable: ONE struct, ONE variant, ONE fixture family, ONE demo panel — never a bundle. If a task contains "and" joining two verifiable behaviours, split it.
- Test the SMALLEST details: single stencil coefficient, single window parameter, single seed, single noise level, single grid size. Each gets its own fixture row and log line. Max accuracy from max granularity.

### 3.2 Mandatory task body template (every task, no exceptions)

```markdown
### TASK-XX.N — <niche name>
Implements: REQ-XXX[, REQ-YYY] | Evolves: ALG-ZZZ (or —)
Goal: <one sentence — what exists after this task that did not before>
Files: <exact paths created/modified>
Steps:
1. <numbered, each step one action with its command>
2. ...
Metrics: <names + thresholds from settings.json, or "none (structural)">
Fixtures: <inputs, seeds, grid sizes, noise levels — exact values>
Logs: <paths written>
Accept: <measurable gate — numbers, file existence, command exit 0>
- [ ] Result: <commit + test output + log path when done>
```

### 3.3 The per-section tail: l-2 evolve, l-1 score, l gate

Let `l` = last task number in a section. Every section owning ≥1 ALG ends with this tail. Sections with NO ALG end with a test gate only.

- `TASK-XX.(l-2)` — EVOLVE (ONE ALG ONLY). Build all `n` variants, run `x` rounds minimum, iterate past `x` until the stop-goal saturates (a full round with no improvement = saturated: log reason, stop). SHOULD take a long time if done right — exhaustive, not the easy way out. Contents (30–100 lines): variant inventory with parent lineage (R1 fresh; R2 children of winner; R3+ children of 2nd-winner), harness command, fixture/seed table to smallest detail, per-round log paths, saturation gate.
- `TASK-XX.(l-1)` — SCORE + DOCUMENT (same ONE ALG). Multi-metric scoreboard (ALL metrics from settings.json, never one), source `# ALPHA-EVOLVE WINNER` comment FIRST, then the `WIN-*` entry: chosen, runner-up, full scores, why-won per metric, `file:line`, next-to-try.
- `TASK-XX.l` — TEST GATE (whole section). Section's `test/` files + demo asserts, all green. Section exit FORBIDDEN on red. Records full test output.
- Multi-ALG sections stack pairs: SECTION-03 (LAP, STEP, FREE) = LAP build → LAP evolve → LAP score → STEP build → STEP evolve → STEP score → FREE build → FREE evolve → FREE score → `03.l` gate. NEVER one combined evolve for two kernels — separate evolve tasks even sharing a file.
- Worked miniature: SECTION-06 ending at `06.14` → `06.12` = NLIN evolve (l-2), `06.13` = NLIN score → WIN-NLIN (l-1), `06.14` = integration gate (l).

- Body length: normal build tasks 10–30 lines; `l-2` evolve, `l-1` score, `l` gate tasks 30–100 lines (inventories, commands, scoreboards take space — required).


### 3.4 The test/ mirror (Julia shown; port per stack)

```text
test/
  runtests.jl                  # includes everything; one command scores all
  helpers/evolve_harness.jl    # shared multi-metric scorer, reads specs/settings.json
  theory/test_ALG-ENC.jl       # per-ALG tournament files: fixtures, seeds,
  theory/test_ALG-WELL.jl      #   per-metric asserts, scoreboard print,
  theory/test_ALG-....jl       #   log writes to logs/theory/<ALG>/
  encoding_tests.jl            # per-section behaviour files (owned by l gates)
  potentials_tests.jl
  ...
  property_tests.jl            # fuzz owned by final section gate
  integration_tests.jl         # end-to-end owned by final section gate
```

Each `test/theory/test_ALG-*.jl` is LONG ENOUGH: fixture tables, all metrics, all seeds, edge cases — dozens of `@test`s, not smoke. The `l` gate runs its section's files; the final `l` runs the full suite.

### 3.5 settings.json shape (living reference)

```jsonc
{
  "evolve": { "n": 3, "x": 3, "seeds": [1, 2, 3], "note": "editable anytime except mid-round" },
  "metrics": {
    "accuracy":        { "weight": 0.30, "higher_is_better": true },
    "speed":           { "weight": 0.20, "higher_is_better": false },
    "complexity":      { "weight": 0.10, "higher_is_better": false },
    "energy":          { "weight": 0.15, "higher_is_better": false },
    "stability":       { "weight": 0.15, "higher_is_better": true },
    "reproducibility": { "weight": 0.10, "higher_is_better": true }
  },
  "per_alg_overrides": { "ALG-WELL": { "metrics": { "accuracy": { "weight": 0.45 } } } },
  "tolerances": { "julia": 1e-6, "c6_symmetry": 1e-10, "js": 1e-3 },
  "extra_metrics": []
}
```

Defaults: `accuracy` (vs fixture/analytic), `speed` (wall ms + scaling), `complexity` (lines/allocs — simpler wins ties), `energy` (allocs × steps proxy), `stability` (noise/drift robustness), `reproducibility` (seed-variance gate — high variance fails). Add via `extra_metrics` (harness picks up by name, no rewrite).

### 3.6 ALG / WIN / source comment templates

ALG entry: `## ALG-XXX — <kernel> (agenda §) | Section: SECTION-XX | Status:` + goal, stop-goal, weights ref, R1/R2/R3+ lineage, promoted line.
WIN entry: `## WIN-XXX — <variant> | Section: SECTION-XX | Date:` + chosen, runner-up, scores, why per metric, `Code: file:line`, next-to-try.
Source comment (Julia; `//` for JS):

```julia
# ALPHA-EVOLVE WINNER: ALG-XXX variant <name> (<date>, fitness <score>)
# Why it won: <per-metric 1-2 lines>
# What this does: <math + steps, inputs->outputs>
```

## 4. Rules that make it work

- No task without REQ. No ALG without owning SECTION + l-2 task. No WIN without l-1 scoreboard + source comment first.
- ONE ALG per evolve task. ONE behaviour per build task. Smallest-detail fixtures always.
- `settings.json` is LIVING: edit `n`, `x`, seeds, weights, thresholds, extra metrics at ANY time — EXCEPT mid-round (finish the running round on old settings, log change + reason in the l-2 task's `Result:`, rerun affected tournaments from R1 if weights changed).
- Tournaments run to SATURATION (stop-goal unmet = keep evolving past `x`, reason logged), never stop at round count alone.
- No section exits with a red `l` gate. No project ships with a `pending` WIN for an owned ALG.
- Only algorithmic kernels evolve; UI/UX excluded. Fixed seeds; 1:1 doc↔code for tests.
- Validate links before coding: `grep Implements:` covers all REQs; every ALG has an l-2 owner; every WIN has an l-1 owner; every ALG section ends in an `l` gate.

## 5. Porting notes (non-Julia)

Replace Julia specifics (`Project.toml`, `Pkg.test()`, `test/*.jl`, Float64 tols) with stack equivalents (`package.json` + jest, `pyproject` + pytest, …). Keep `SECTION/l-2/l-1/l` IDs, `settings.json` shape, and saturation + living-config rules identical.

