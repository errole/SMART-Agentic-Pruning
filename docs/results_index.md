# CAP-Prune results index

Generated 2026-11-23 17:08:54 from `C:\Users\elbas\cap-prune-runs`. Code: commit `9c4891aa64387e72f5d831130b95b85711fc0f09` on `research/cap-prune-experiments` (working tree clean: True). Environment: {'python': '3.12.10', 'platform': 'Windows-11-10.0.26300-SP0', 'torch': '2.11.0+cu128', 'cuda': '12.8', 'gpu': 'NVIDIA GeForce RTX 5090', 'transformers': '5.23.0'}.

Code provenance: results produced before the first research commit came from uncommitted code; for runs with a `registration.json` the SHA-256 (16 hex) of the key source files at registration is listed with the files that have changed since. Headline numbers are copied from each experiment's own JSON outputs.

| experiment | label | configs (sha256/12) | reports | code changed since registration |
|---|---|---|---|---|
| **exp1-adaptive-pilot**: Experiment 1: adaptive candidate evaluation (offline replay, 3 allocation policies) | pilot | `configs/adaptive_pilot.yaml` 6423623ba952 | `C:\Users\elbas\cap-prune-runs\adaptive\pilot\replay\summary.md` | not registered (pre-commit code) |
| **gate-design**: Capability-gate design on development data (MMLU validation) - locked the live gate | development (locks settings) | `configs/gate_dev.yaml` 629edd37b21c | `C:\Users\elbas\cap-prune-runs\adaptive\gate-dev\design\summary.md` | not registered (pre-commit code) |
| **live-gate-gemini**: Experiment 2: capability gate in the live loop, Gemini controller (arms A-D) | pilot, one seed (arm B paused at 7/10 by API quota; arm D not started) | `configs/live_gate.yaml` a57ca023a6f3 | `C:\Users\elbas\cap-prune-runs\live-gate\_live\report.md` | not registered (pre-commit code) |
| **live-gate-ollama**: Experiment 2 with a local controller (Ollama Qwen2.5-7B) | exploratory, one seed | `configs/live_gate_local.yaml` 4e18bc3d6d06 | `C:\Users\elbas\cap-prune-runs\live-gate-ollama\_live\report.md` | not registered (pre-commit code) |
| **gap-diagnostic**: 2x2 gap diagnostic of the greedy checkpoint (pool vs held-out x 0- vs 5-shot) | diagnostic (held-out split inspected) | `configs/live_gate.yaml` a57ca023a6f3 | `C:\Users\elbas\cap-prune-runs\live-gate-diagnostics\greedy_cap_gap\summary.md` | not registered (pre-commit code) |
| **fixed-state**: Fixed-state controller comparison (Qwen2.5-14B vs 7B, free-form vs shortlist) | pilot | `configs/fixed_state.yaml` 97465ad161f6 | `C:\Users\elbas\cap-prune-runs\fixed-state\compare\report.md` | not registered (pre-commit code) |
| **ranker**: Learned action ranking (offline, leave-one-trajectory-out) | offline comparison | `configs/ranker.yaml` 659422b25c3b | `C:\Users\elbas\cap-prune-runs\ranker\data\report.md` | not registered (pre-commit code) |
| **gradient-live**: Gradient-only top-1 live pilot (10 proposals / 30k questions) | fresh MMLU-CF audit (first use of MMLU-CF) | `configs/gradient_live.yaml` c53cc31bbbfa | `C:\Users\elbas\cap-prune-runs\gradient-live\_analysis\report.md`<br>`C:\Users\elbas\cap-prune-runs\gradient-live\_analysis\audit_materialized\audit_materialized.md` | not registered (pre-commit code) |
| **gradient-live-ext**: Extended gradient-only search (arm A: top-1, 30 proposals / 100k questions) | exploratory audit (sets inspected before) | `configs/gradient_live_ext.yaml` 128bd5f3cfac | `C:\Users\elbas\cap-prune-runs\gradient-live-ext\_analysis\report.md`<br>`C:\Users\elbas\cap-prune-runs\gradient-live-ext\_analysis\audit_materialized\audit_materialized.md` | gradient_audit.py, gradient_live.py |
| **gradient-boundary**: Compression boundary of the top-1 trajectory + gate-margin calibration | exploratory (calibration inconclusive) | `configs/gradient_boundary.yaml` 95ff6d7afab7 | `C:\Users\elbas\cap-prune-runs\gradient-boundary\report.md` | not registered (pre-commit code) |
| **gradient-top2**: Arm B: gradient top-2 fallback, current gate | exploratory audit | `configs/gradient_top2.yaml` 8d8b6ce6053a | `C:\Users\elbas\cap-prune-runs\gradient-top2\_analysis\audit_materialized\audit_materialized.md` | none |
| **gradient-top2-strict**: Arm C: gradient top-2 fallback, stricter gate (margin 2.0 pts, exploratory) | exploratory audit, exploratory margin | `configs/gradient_top2_strict.yaml` c98443c15468 | `C:\Users\elbas\cap-prune-runs\gradient-top2-strict\_analysis\audit_materialized\audit_materialized.md` | none |
| **gradient-compare**: A vs B vs C live-policy comparison | exploratory | `configs/gradient_compare.yaml` 9a0c18da0dc2 | `C:\Users\elbas\cap-prune-runs\gradient-compare\report.md` | not registered (pre-commit code) |
| **unstructured**: Unstructured-mode CAP-Prune matrix (started, never progressed past setup) | incomplete | `configs/base.yaml` ef931627011e | - | not registered (pre-commit code) |

## Headline numbers

### exp1-adaptive-pilot: Experiment 1: adaptive candidate evaluation (offline replay, 3 allocation policies)
Folder `adaptive/pilot` (present); docs `docs/adaptive_evaluation.md`; runs: none.
* round_robin: 10,700 questions, accept/reject/inconclusive 0/5/2, false accepts 0
* threshold_aware: 7,700 questions, accept/reject/inconclusive 0/6/1, false accepts 0
* widest_interval: 10,600 questions, accept/reject/inconclusive 0/5/2, false accepts 0

### gate-design: Capability-gate design on development data (MMLU validation) - locked the live gate
Folder `adaptive/gate-dev` (present); docs `docs/live_gate.md`; runs: none.
* locked: {'tolerance': 0.03, 'budget': 5614, 'spending': 'uniform', 'min_sample': 100, 'batch_size': 100, 'alpha_false_accept': 0.05, 'alpha_false_reject': 0.05}

### live-gate-gemini: Experiment 2: capability gate in the live loop, Gemini controller (arms A-D)
Folder `live-gate` (present); docs `docs/live_gate.md`; runs: dense (science b7ea635bd4d2e4b9), greedy_cap__t10__s0 (science e953b3271808439a), llm_cap__t10__s0 (science aaee9401bde68f73), llm_ppl__t10__s0 (science 4e46bc1ccea82d67).
* greedy_cap__t10__s0: MLP/eligible sparsity 3.72 %, stop max_iterations; held-out MMLU (5-shot) worst category STEM +5.18 pts, overall +2.82 pts
* llm_cap__t10__s0: no final summary (run unfinished)
* llm_cap_rr__t10__s0: no final summary (run unfinished)
* llm_ppl__t10__s0: MLP/eligible sparsity 10.42 %, stop target_reached; held-out MMLU (5-shot) worst category STEM +18.23 pts, overall +14.27 pts

### live-gate-ollama: Experiment 2 with a local controller (Ollama Qwen2.5-7B)
Folder `live-gate-ollama` (present); docs `docs/live_gate.md`; runs: dense (science 0ec03024bb4ea634), llm_cap__t10__s0 (science fadff14feeebf704), llm_cap_rr__t10__s0 (science 3e5f4604526b13ed), llm_ppl__t10__s0 (science 2d77cd18140793da).
* llm_cap__t10__s0: MLP/eligible sparsity 1.15 %, stop max_iterations; held-out MMLU (5-shot) worst category Other +1.61 pts, overall +1.25 pts
* llm_cap_rr__t10__s0: MLP/eligible sparsity 1.19 %, stop max_iterations; held-out MMLU (5-shot) worst category Other +1.88 pts, overall +1.33 pts
* llm_ppl__t10__s0: MLP/eligible sparsity 3.94 %, stop max_iterations; held-out MMLU (5-shot) worst category STEM +7.61 pts, overall +3.57 pts

### gap-diagnostic: 2x2 gap diagnostic of the greedy checkpoint (pool vs held-out x 0- vs 5-shot)
Folder `live-gate-diagnostics/greedy_cap_gap` (present); docs `docs/live_gate.md`; runs: none.
* holdout|0: overall +1.84, worst STEM +4.23
* holdout|5: overall +2.82, worst STEM +5.18
* pool|0: overall +1.37, worst STEM +2.78
* pool|5: overall +1.85, worst Humanities +2.34

### fixed-state: Fixed-state controller comparison (Qwen2.5-14B vs 7B, free-form vs shortlist)
Folder `fixed-state/compare` (present); docs `docs/fixed_state.md`; runs: none.
* greedy shortlist: mean worst-category loss +3.97, accepts 1/4
* qwen2.5:14b-instruct free: mean worst-category loss +3.64, accepts 3/12
* qwen2.5:14b-instruct shortlist: mean worst-category loss +3.93, accepts 6/12
* qwen2.5:7b-instruct free: mean worst-category loss +3.70, accepts 6/12
* qwen2.5:7b-instruct shortlist: mean worst-category loss +3.81, accepts 4/12

### ranker: Learned action ranking (offline, leave-one-trajectory-out)
Folder `ranker/data` (present); docs `docs/ranker.md`; runs: none.
* unseen trajectories, mean worst-category loss: random +2.17, greedy +1.65, gradient +1.09, probe +1.17, ridge +1.10, ridge+probe +1.16, gradient sequential @probe budget +1.09, ridge sequential @probe budget +1.10

### gradient-live: Gradient-only top-1 live pilot (10 proposals / 30k questions)
Folder `gradient-live` (present); docs `docs/gradient_live.md`; runs: gradient_top1__t00__s0 (science 2b6007042db5a463).
Provenance note: Ran on uncommitted code. Reproduced exactly with later code: gradient-live-ext replayed attempts 1-10 (shortlists, actions, gate decisions, question counts, per-category counts, final mask) under verify_prefix.
* frozen `03810f5b84f494f6`: 10 accepts / 10 proposals, 20,200 paired gate questions, stop proposal_cap, MLP sparsity 7.44 %
* materialised: 78,643,200 params removed (4.57 %)
* MMLU held-out final split (exploratory (reused questions)): worst Other +0.48, largest simultaneous UB Other +1.45 -> meets the limit; the simultaneous upper bounds support it
* MMLU-CF validation (fresh): worst Social Sciences +0.23, largest simultaneous UB Social Sciences +1.86 -> meets the limit; the simultaneous upper bounds support it

### gradient-live-ext: Extended gradient-only search (arm A: top-1, 30 proposals / 100k questions)
Folder `gradient-live-ext` (present); docs `docs/gradient_live.md`; runs: gradient_top1__t00__s0 (science ee3f94ea2c15507e).
Provenance note: gradient_live.py / gradient_audit.py changed after registration (top-2 fallback, gate margin, logging, reports). Reproduced exactly with the later code: gradient-top2 (arm B) replayed attempts 1-16 under verify_prefix, final mask included.
* frozen `e65e9461315ef207`: 15 accepts / 16 proposals, 36,952 paired gate questions, stop candidate_rejected, MLP sparsity 11.16 %
* materialised: 117,964,800 params removed (6.86 %)
* MMLU-CF validation (exploratory (MMLU-CF validation, inspected in the pilot's audit; comparison only)): worst Other +2.16, largest simultaneous UB Other +3.51 -> point estimates meet the limit, but the intervals do not exclude a loss above it
* MMLU held-out final split (exploratory (reused questions)): worst STEM +3.65, largest simultaneous UB STEM +5.39 -> does NOT meet the limit (a category's point estimate is above it)

### gradient-boundary: Compression boundary of the top-1 trajectory + gate-margin calibration
Folder `gradient-boundary` (present); docs `docs/gradient_live.md`; runs: none.
* attempt 10: 4.57 % params, largest UB MMLU-CF validation +1.86, MMLU held-out final split +1.45, meets audit: True
* attempt 11: 5.03 % params, largest UB MMLU-CF validation +1.51, MMLU held-out final split +1.18, meets audit: True
* attempt 12: 5.48 % params, largest UB MMLU-CF validation +2.67, MMLU held-out final split +2.06, meets audit: True
* attempt 13: 5.94 % params, largest UB MMLU-CF validation +3.48, MMLU held-out final split +1.90, meets audit: False
* attempt 14: 6.40 % params, largest UB MMLU-CF validation +3.95, MMLU held-out final split +5.55, meets audit: False
* attempt 15: 6.86 % params, largest UB MMLU-CF validation +3.51, MMLU held-out final split +5.39, meets audit: False
* attempt 16 rank 2: 7.31 % params, largest UB MMLU-CF validation +3.95, MMLU held-out final split +6.03, meets audit: False
* margin calibration: chosen None pts (conclusive: False)

### gradient-top2: Arm B: gradient top-2 fallback, current gate
Folder `gradient-top2` (present); docs `docs/gradient_live.md`; runs: gradient_top2__t00__s0 (science 08bbcefb5637a4c2).
* frozen `4ac58c5aa8348773`: 17 accepts / 21 proposals, 51,274 paired gate questions, stop fallback_exhausted, MLP sparsity 12.65 %
* materialised: 133,693,440 params removed (7.77 %)
* MMLU-CF validation (exploratory (MMLU-CF validation, inspected in the pilot's audit; comparison only)): worst Other +2.23, largest simultaneous UB Other +3.65 -> point estimates meet the limit, but the intervals do not exclude a loss above it
* MMLU held-out final split (exploratory (reused questions)): worst STEM +4.12, largest simultaneous UB STEM +5.92 -> does NOT meet the limit (a category's point estimate is above it)

### gradient-top2-strict: Arm C: gradient top-2 fallback, stricter gate (margin 2.0 pts, exploratory)
Folder `gradient-top2-strict` (present); docs `docs/gradient_live.md`; runs: gradient_top2__t00__s0 (science 10e532179d07b326).
* frozen `05e68e072909705a`: 7 accepts / 10 proposals, 33,206 paired gate questions, stop fallback_exhausted, MLP sparsity 5.21 %
* materialised: 55,050,240 params removed (3.20 %)
* MMLU-CF validation (exploratory (MMLU-CF validation, inspected in the pilot's audit; comparison only)): worst Other +0.14, largest simultaneous UB Social Sciences +1.05 -> meets the limit; the simultaneous upper bounds support it
* MMLU held-out final split (exploratory (reused questions)): worst Other +0.05, largest simultaneous UB Other +0.86 -> meets the limit; the simultaneous upper bounds support it

### gradient-compare: A vs B vs C live-policy comparison
Folder `gradient-compare` (present); docs `docs/gradient_live.md`; runs: none.
* A (gradient top-1, current gate (the extended run, reused: same settings)): 6.86 % params, 36,952 paired questions, 678 s search; MMLU-CF validation: point estimates meet the limit, but the intervals do not exclude a loss above it; MMLU held-out final split: does NOT meet the limit (a category's point estimate is above it)
* B (gradient top-2, current gate): 7.77 % params, 51,274 paired questions, 962 s search; MMLU-CF validation: point estimates meet the limit, but the intervals do not exclude a loss above it; MMLU held-out final split: does NOT meet the limit (a category's point estimate is above it)
* C (gradient top-2, locked stricter gate): 3.20 % params, 33,206 paired questions, 597 s search; MMLU-CF validation: meets the limit; the simultaneous upper bounds support it; MMLU held-out final split: meets the limit; the simultaneous upper bounds support it

### unstructured: Unstructured-mode CAP-Prune matrix (started, never progressed past setup)
Folder `unstructured` (present); docs `README.md`; runs: cap_prune__t20__s0 (science 10d1da77622d1a41), dense (science eaf17ecc04a66c2d).
* cap_prune__t20__s0: no final summary (run unfinished)

## Checkpoint archive (every accepted state, rebuilt and verified)

* `C:\Users\elbas\cap-prune-runs\gradient-live\gradient_top1__t00__s0`: archived, 10 states
* `C:\Users\elbas\cap-prune-runs\gradient-live-ext\gradient_top1__t00__s0`: archived, 15 states
* `C:\Users\elbas\cap-prune-runs\gradient-top2\gradient_top2__t00__s0`: archived, 17 states
* `C:\Users\elbas\cap-prune-runs\gradient-top2-strict\gradient_top2__t00__s0`: archived, 7 states
* `C:\Users\elbas\cap-prune-runs\live-gate\dense`: nothing to archive (no accepted state), 0 states
* `C:\Users\elbas\cap-prune-runs\live-gate\greedy_cap__t10__s0`: archived, 4 states
* `C:\Users\elbas\cap-prune-runs\live-gate\llm_cap__t10__s0`: archived, 4 states
* `C:\Users\elbas\cap-prune-runs\live-gate\llm_ppl__t10__s0`: archived, 2 states
* `C:\Users\elbas\cap-prune-runs\live-gate-ollama\dense`: nothing to archive (no accepted state), 0 states
* `C:\Users\elbas\cap-prune-runs\live-gate-ollama\llm_cap__t10__s0`: archived, 8 states
* `C:\Users\elbas\cap-prune-runs\live-gate-ollama\llm_cap_rr__t10__s0`: archived, 7 states
* `C:\Users\elbas\cap-prune-runs\live-gate-ollama\llm_ppl__t10__s0`: archived, 10 states
* `C:\Users\elbas\cap-prune-runs\unstructured\cap_prune__t20__s0`: nothing to archive (no accepted state), 0 states
* `C:\Users\elbas\cap-prune-runs\unstructured\dense`: nothing to archive (no accepted state), 0 states

## Test records

* `C:\Users\elbas\cap-prune-runs\_tests\2026-11-23_19caeeb4f434`: 104 passed in 790.08s (0:13:10) (commit `19caeeb4f434`)
* `C:\Users\elbas\cap-prune-runs\_tests\2026-11-23_earlier-session`: see folder (commit `n/a`)

## Backups (verified copies of the runs root)

* 2026-11-23_170822: `cap-prune-runs_2026-11-23_170822_9c4891aa6438.zip` (111 MB, 2398 files, sha256 `ff583044aa138247...`) -> `E:\cap-prune-backups\cap-prune-runs_2026-11-23_170822_9c4891aa6438.zip` verified=True; `C:\Users\elbas\OneDrive\Documents\Education\NYU Tandon - GY\Efficient AI Computing Algorithm and Implementation\project\CAP-Prune-backups\cap-prune-runs_2026-11-23_170822_9c4891aa6438.zip` verified=True
