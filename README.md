# CAP-Prune — implementation

**Capability-Aware, Budgeted Agentic Pruning for Hardware-Efficient LLMs**
(Elbasan, RS, Maheshwari, Thomas — NYU Tandon, Efficient AI Computing project).

This package implements the proposal end to end on **Qwen3-1.7B**: the dense
baseline, Wanda, the Kodathala & Vunnam (2026, arXiv:2601.09694) agent-guided
pruning baseline, CAP-Prune's budgeted, uncertainty-aware domain testing with
domain-loss and deployment acceptance rules, the three comparison allocation
policies and the LLM/no-LLM ablation — plus hardware benchmarking, 3-seed
statistics, and a fault-tolerance layer so that **a job that fails midway, for
any reason, resumes where it stopped and reaches exactly the result an
uninterrupted run would have**.

---

## 1. Proposal → code

| Proposal item | Where |
|---|---|
| Evaluate original Qwen3-1.7B | method `dense` |
| Wanda baseline | method `wanda` · `pruning/wanda.py::sequential_wanda` (layer-sequential, per-output-row) |
| Paper-style agent with fixed LLM controller | method `paper_agent` · `search/controller.py`, `search/loop.py` (z-scored Wanda sensitivity + gradient profiles, JSON decisions, self-reflection feedback, 15 % PPL rollback) |
| Separate calibration / development / pruning-time / final data | `data/prepare.py` (C4-train calibration, C4-val PPL gate, MMLU-validation dev, MMLU-test split 40 % probe / 60 % final, FreebaseQA, WikiText-2) |
| Uncertainty in each domain's accuracy loss | `search/testing.py` — paired Dirichlet posterior on (lost, agree, gained) per domain |
| Adaptive allocation + minimum coverage | `testing.policy: cap` — `sd × (α + 4·P(1−P))` with P = P(domain over its limit), after ≥ `min_per_domain` questions in every domain |
| Accept/reverse rules; inconclusive keeps last model | `sequential_test` → pass (every domain's P(over limit) ≤ 0.20) / fail (some domain ≥ 0.90, early stop) / inconclusive ⇒ rollback |
| Domain-loss limits | absolute limit on a sparsity-proportional schedule + imbalance limit |
| Deployment requirements | `deploy:` — each candidate is materialised and must not be slower (prefill/decode) than its parent, optional memory cap |
| Equal testing / observed-loss / statistical-uncertainty comparisons | methods `cap_balanced`, `cap_loss_ranked`, `cap_neyman` (same budgets, same stopping rule) |
| Contribution of adaptive testing vs LLM guidance | 2×2: `paper_agent`, `cap_prune`, `heuristic_ppl`, `heuristic_cap` |
| Hardware-supported pruning, ~20 % / 40 % | `configs/qwen3_1.7b_nm.yaml` (2:4 sparse tensor cores) and `configs/qwen3_1.7b_structured.yaml` (MLP channels physically removed) |
| Achieved sparsity, largest domain loss, MMLU, factual QA, WikiText-2 PPL | `runner.py` final stages, `report.py` |
| Prefill / token-generation latency, peak GPU memory | `eval/bench.py` on a materialised model (`pruning/materialize.py`) |
| Total GPU time, confidence intervals, early stops, 3 seeds | `report.py` (seed t-intervals, paired bootstrap, paired method differences) |
| Checkpoint recovery | `utils/checkpoint.py`, `runner.py`, `supervisor.py` — see §5 |
| Experiment 1: adaptive candidate evaluation (exact finite-pool bounds, 3 allocation policies, offline replay) | `adaptive/` · `adaptive-build` / `adaptive-replay` — see [docs/adaptive_evaluation.md](docs/adaptive_evaluation.md) |
| Experiment 2: capability gate in the live pruning-and-rollback loop (arms A–D, gate design on dev data, interleaved benchmark) | `gate: {kind: capability}` in `search/loop.py`, `adaptive/live.py` · `gate-design` / `live-bench` / `live-report` — see [docs/live_gate.md](docs/live_gate.md) |
| Fixed-state controller comparison (Qwen2.5-14B vs 7B, free-form vs shortlist, frozen parents) | `fixed_state.py` · `fixed-state` — see [docs/fixed_state.md](docs/fixed_state.md) |
| Learned action ranking (ridge per MMLU category vs random / greedy / gradient / probe selectors, leave-one-trajectory-out) | `ranker.py` · `ranker collect` / `ranker evaluate` — see [docs/ranker.md](docs/ranker.md) |
| Gradient-only selector in the live loop (top-1 per parent, question/proposal caps, stall diagnostic, fresh MMLU-CF audit; materialised verification; extended search replaying the pilot to its first stall; compression boundary, gate-margin calibration, top-2 fallback and stricter-gate arms) | `gradient_live.py` (method kind `ranked_search`, `verify_prefix`, `fallback_k`, `gate_margin`), `gradient_audit.py`, `gradient_boundary.py` · `supervise` + `gradient-live audit-data / register / freeze / diagnose / audit / audit-materialized / report` + `gradient-boundary recover / margin-plan / audit / calibrate / compare` — see [docs/gradient_live.md](docs/gradient_live.md) |
| Population-level re-gating of the A/B/C runs (pre-registered rule, CPU) and continuous paired signals on the gate pool | `population_regate.py`, `pool_signals.py` · `population-regate register / run`, `pool-signals score / analyze` — see [docs/population_gate.md](docs/population_gate.md) |
| Screen-then-confirm protocol (screening on the reused pool; confirmation on fresh items by an intersection-union test with fixed-sequence testing over checkpoints, amendment 1; confirmation sample sizes) and the matched-compression signal tests (KL vs. net flip vs. gradient rank 1 among equal-size actions, split-half on the gate pool; stall parents and the ranker's ordinary parents; policy recommendation) | `screen_confirm.py`, `matched_signals.py` · `screen-confirm sizes`, `matched-signals register / score / analyze / compare / after-rejection / tv-posthoc`; the amendment-2 proposer in the live loop (`ranked_search.fallback_policy: kl_probe`, `ranked_search.counterfactual`, `capprune/signals.py`) — see [docs/screen_confirm_protocol.md](docs/screen_confirm_protocol.md), [docs/matched_signals.md](docs/matched_signals.md), [docs/matched_signals_ordinary.md](docs/matched_signals_ordinary.md) |
| Confirmatory study on new model families (SmolLM3-3B-Base, Granite-3.3-2B-Base): data sealed per model before any scoring, arms (a) perplexity rollback / (b) capability gate / (c) capability gate + KL probe, confirmation by amendment 1; amendment 3 scores every frozen checkpoint and adds S1–S4 | `confirmatory.py`, `confirmatory_study.py` · `confirmatory build-data / admissibility / freeze / confirm / report / study` — plan [docs/confirmatory_study_plan.md](docs/confirmatory_study_plan.md), results [docs/confirmatory_report.md](docs/confirmatory_report.md) |
| Paper notes: claims C1–C6 and the negative LLM-controller result, with their evidence and confirming experiments; related work, each paper read before it was described | [docs/paper/claims.md](docs/paper/claims.md), [docs/paper/related_work.md](docs/paper/related_work.md) |
| Research records: every accepted checkpoint archived and verified, results index, verified backups, commit recorded per run | `archive.py`, `results_index.py`, `configs/results_registry.yaml` · `archive-checkpoints`, `results-index`, `backup` — see [docs/research_records.md](docs/research_records.md) |

## 2. Setup (Windows 11, RTX 5090)

```powershell
cd "<this folder>"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# RTX 50-series (Blackwell) needs a CUDA 12.8+ PyTorch build. Check pytorch.org for the current command, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[test]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Controller API key (the paper used `gemini-3-flash-preview`; check that the model id is still served):

```powershell
setx GEMINI_API_KEY "your-key"      # then open a new terminal
```

Any OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio, Ollama, vLLM) or
Anthropic works — edit `controller.model` / `controller.endpoints` in
`configs/base.yaml`. A local model in LM Studio is a good "frozen" controller.

**Keep run outputs off OneDrive.** Defaults are `~/cap-prune-runs/...`
(`C:\Users\<you>\cap-prune-runs`). OneDrive locks files mid-sync, which fights
checkpoint renames and SQLite.

## 3. Check the install offline (2 minutes, CPU, no keys)

```powershell
python -m pytest -q                                    # 43 tests incl. kill/hang/corruption recovery
python -m capprune make-tiny --out .smoke/tiny-qwen3
python -m capprune prepare --config configs/smoke.yaml
Start-Process python "tests/mock_controller.py --port 8765"   # fake LLM agent
python -m capprune supervise --config configs/smoke.yaml
python -m capprune report    --config configs/smoke.yaml

# experiment 1 (adaptive candidate evaluation) on CPU, no keys
python -m capprune adaptive-fixture --out .smoke/adaptive-fixture
python -m capprune adaptive-replay  --config configs/adaptive_fixture.yaml
python -m capprune adaptive-build   --config configs/adaptive_smoke.yaml
python -m capprune adaptive-replay  --config configs/adaptive_smoke.yaml
```

## 4. Running the experiment

```powershell
# 1) data + model download (once; verified with SHA-256 thereafter)
python -m capprune prepare --config configs/base.yaml

# 2) a first real run to time things and sanity-check (watch the log)
python -m capprune run --config configs/base.yaml --method dense
python -m capprune run --config configs/base.yaml --method cap_prune --target 0.2 --seed 0

# 3) tune the (TUNE) thresholds on DEVELOPMENT data only
python -m capprune supervise --config configs/dev_tuning.yaml
python -m capprune report    --config configs/dev_tuning.yaml

# 4) the full matrices (each resumable, each in its own runs_dir)
powershell -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Config configs\base.yaml
powershell -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Config configs\qwen3_1.7b_nm.yaml
powershell -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Config configs\qwen3_1.7b_structured.yaml

# progress / integrity / results
python -m capprune status --config configs/base.yaml
python -m capprune verify --config configs/base.yaml
python -m capprune report --config configs/base.yaml      # -> runs_dir/_report/{report.md,runs.csv,methods.csv}
```

Each matrix = `dense` + 8 methods × 2 targets × 3 seeds. Override anything
with `--set key=value` (e.g. `--set matrix.seeds=[0]`).

## 5. What happens when something fails

**To resume after any failure, run the same command again.** Nothing needs cleaning up.

| Failure | What protects you |
|---|---|
| Crash, Python exception, OOM-kill, driver reset, power loss | Sealed (checksummed) checkpoint after **every** iteration; masks written content-addressed *before* the checkpoint that references them; the profile, controller reply, every probe answer, PPL and benchmark of an iteration are persisted/cached before use ⇒ the interrupted iteration replays to the **identical** outcome, with **no new LLM call** |
| Crash during final evaluation | Answers cached per chunk (SQLite, WAL, synchronous=FULL); only unanswered questions are re-run. Stage progress lives inside the checkpoint |
| Hung process (stuck kernel / HTTP) | Worker heartbeats from its main thread; supervisor kills and restarts it after `hang_timeout_s` of no progress |
| Repeated crashes | Supervisor restarts with backoff; the restart budget only counts crashes with *no* checkpoint progress; a final retry pass over failed runs; `runs/_matrix/ledger.json` records every attempt and exit code |
| Supervisor itself dies / reboot | `scripts/run_matrix.ps1` restarts the supervisor; `scripts/register_autostart.ps1` resumes at logon |
| Ctrl+C / `Stop-Process` | First signal: checkpoint-safe exit (code 75). Second signal: immediate exit (still safe — resume replays) |
| LLM API outage / rate limits | Retries with exponential backoff + jitter, multiple endpoints, JSON repair re-ask; if all fail the run **pauses** (exit 76) and the supervisor retries later. By default failover may only use the *same* model (`allow_model_switch: false`) so the "fixed controller" stays fixed; `on_unavailable: heuristic` instead uses a labelled fallback |
| Missing key / 401 / 403 | Detected as permanent; that endpoint is skipped immediately |
| CUDA OOM | Batch size halved and the learned size checkpointed |
| Corrupted checkpoint or mask file | Every file is verified; falls back to the newest valid checkpoint, then to an older accepted state, then to `reliability.mirror_dir` (set it to a second disk); if *all* are unusable the run replays deterministically from its caches |
| Corrupted cache database | Quarantined and rebuilt |
| Half-written data / truncated download | Prepared data verified by SHA-256 on every start; downloads checksum-verified and retried |
| Disk full | Checked before every write; old checkpoints pruned; clean exit (74) instead of a corrupt file |
| Two workers / two supervisors on the same run | PID-checked locks (`os.kill(pid, 0)` is never used on Windows — it would terminate the process) |
| Config edited mid-experiment | Resume refused if any *science* setting changed (reliability settings may change freely); `--force-resume` to override |
| GPU time accounting across failures | Committed GPU-seconds live in the checkpoint; time lost to failures is reconstructed from the append-only journal and reported separately |

These behaviours are exercised by `tests/test_recovery.py`, which hard-kills
the worker at 8 critical points (after profiling, after the controller reply,
after applying a candidate, mid-test, between saving masks and checkpointing,
between writing a checkpoint and its pointer, mid-final-MMLU, mid-FreebaseQA)
and asserts the final trajectory and metrics are identical to an uninterrupted
run; plus hang detection, graceful stop, controller outage, all-checkpoints-corrupt
replay and the config-change guard. Fault injection is inert unless
`CAPPRUNE_FAULTS` is set.

## 6. Output layout

```
~/cap-prune-runs/<mode>/
  _shared/                    dense references (probe answers, PPL0, quick bench)
  _matrix/ledger.json         supervisor ledger;  supervisor.log
  _report/report.md|runs.csv|methods.csv
  cap_prune__t20__s0/
    run.json                  resolved config, science hash, environment
    journal.jsonl             append-only log of every iteration, test and restart
    ckpt/  masks/  profiles/  caches/  logs/
    final/  mmlu.json  freebaseqa.json  wikitext2.json  benchmark.json  summary.json
    DONE
```

## 7. Design choices to know about

* **Losses are paired against the dense model** on identical questions (probe and final), which removes between-question variance.
* **Probes are 0-shot, final MMLU is 5-shot** (paper protocol). The final split is never seen during search.
* The domain-loss limit grows linearly with sparsity progress (`loss_limit.schedule: linear`) so damage cannot be front-loaded; the imbalance limit protects the most-affected domain.
* All four CAP variants share minimum coverage, budgets and the stopping rule; they differ only in *where* questions go.
* **The gate's defaults are simulation-checked** (`test_gate_operating_characteristics_at_real_config_defaults`, real per-domain probe-pool sizes): a harmless step passes ≥ 85–90 % of the time and a domain truly 10 points down is rejected ≥ 80 %. This needs ~2,400 probe questions per step and limits ≥ ~0.04; the first version's defaults (800 questions, 0.02 floor, family-wise rule) rejected ~96 % of harmless steps and were replaced. If you tighten limits or cut the budget, re-run that test with your values.
* Reported GPU time excludes time spent waiting on the controller API (`controller_wait_seconds`, also reported).
* Run seeds change: calibration windows (as in Wanda), per-iteration profiling subsets, probe order, controller sampling. The data split (`data.seed`) is fixed so every run is evaluated on the same questions.
* Paper details that were ambiguous and how they are resolved: gradient importance = mean |∇W| over the unit's *active* weights; activation/gradient subsets drawn per iteration from the run's 128 calibration windows; FreebaseQA is reported on its `eval` split (the paper's count matches `train`; set `data.freebaseqa.split`); the 2:4 baseline for 20/40 % converts the least-sensitive units first.
* In `nm` mode a unit is either dense or 2:4 (that is what the kernels execute); in `structured` mode eligible weights are MLP weights.

## 8. Verified here vs. still to verify on your machine

Verified (CPU, tiny random Qwen3 with the real Qwen3 code paths, transformers
5.23 / torch 2.15): all 43 tests, the full matrix → report pipeline, all three
pruning modes, materialised structured models giving identical outputs, and the
recovery scenarios above.

Not run here (no GPU / no internet for model weights in this environment):

* Real Qwen3-1.7B, C4/MMLU/FreebaseQA downloads, and an actual LLM controller. Do step 4.2 first and read the log.
* **2:4 kernels on the RTX 5090.** `torch.sparse` semi-structured support on consumer Blackwell (sm_120) depends on your PyTorch build. If conversion fails, the benchmark records `nm_backend: unavailable: …` and measures the dense equivalent — it never reports a fake speed-up. Structured mode needs no special kernels.
* Threshold values marked `(TUNE)` are starting points; set them with `dev_tuning.yaml` before final runs. The simulation assumes 4–10 % of answers flip per step; check the real flip rate in your first run's `journal.jsonl` (`test.tallies`).
* The per-step deployment check compares latency against the parent with a 5 % tolerance; measure your GPU's run-to-run timing noise first and adjust `deploy.max_latency_ratio_vs_parent`.
* Windows-specific paths (signals, file locking, Task Scheduler scripts) were written for Windows but tested only on Linux.
* Run time: time your first `cap_prune` run; the matrix is ~49 runs per mode.
