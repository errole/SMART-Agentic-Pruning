# Research records: what is kept, where, and how to verify it

Every number we report should trace back to the code, the configuration and the data that produced it.

| record | where | how it is produced / verified |
|---|---|---|
| Code | git branch `research/cap-prune-experiments` (the commits that produced each result) | `run.json` of every new run stores the commit and whether code/configs had uncommitted changes (`git`) plus `config_path`; the gradient-live registrations store SHA-256 (16 hex) of the key source files |
| Experiment runs | `~/cap-prune-runs/<experiment>/<run>/` | `run.json` (resolved config, science hash, environment, commit), `journal.jsonl` (every attempt), sealed checkpoints, profiles, answer caches |
| Analyses and reports | `~/cap-prune-runs/<experiment>/_analysis/` and the boundary / compare folders | JSON next to every Markdown report; per-item answers for every audit |
| Every accepted checkpoint | `~/cap-prune-runs/_archive/checkpoints/<experiment>/<run>/` (+ `index.json`) | `capprune archive-checkpoints`: rebuilds the whole accepted chain from the journal and the run's persisted profiles and checks every state against the journal (mask hash or sparsity), any stored mask, and the final checkpoint. New structured runs keep every mask anyway (`reliability.keep_all_masks: true`) |
| Test results | `~/cap-prune-runs/_tests/<date>_<commit>/` | `junit.xml` (per test), `pytest.log` (`-rA -v --durations=0`), the pytest artifacts (`basetemp/`), `environment.txt` (`pip freeze`, versions), `record.json` (commit, clean tree, summary) |
| Results index | `~/cap-prune-runs/RESULTS_INDEX.md` + `results_index.json` (copy in `docs/results_index.md`) | `capprune results-index`: experiments from `configs/results_registry.yaml`, config SHA-256s, commit, code changed since registration, report paths, headline numbers read from each experiment's JSON |
| Backups | `E:\cap-prune-backups\` (second physical SSD) and the OneDrive project folder `CAP-Prune-backups\` | `capprune backup`: one zip of the whole runs root with a per-file SHA-256 manifest inside and next to it; each copy is re-hashed and zip-tested after copying; ledger in `~/cap-prune-runs/_archive/backups.json` |

## Conventions

* **Write-once, dated outputs.** New experiments get new run and analysis folders; nothing completed is overwritten
  (the gradient pilots were hash-checked before and after each follow-up).
* **Pre-registration.** Caps, margins and decision rules are written to a `registration.json` /
  `calibration_plan.json` before the run or scoring they govern; the code refuses to proceed otherwise.
* **Labels.** Audits on question sets that were inspected before are labelled exploratory in every report.
* **Commit before running.** Start a new experiment from a clean, committed tree; `run.json` records the commit and
  flags uncommitted code/config changes.

## Commands

```powershell
python -m capprune archive-checkpoints                    # -> ~/cap-prune-runs/_archive/checkpoints
python -m capprune results-index                          # -> ~/cap-prune-runs/RESULTS_INDEX.md
python -m capprune backup --stage E:/cap-prune-backups --dest E:/cap-prune-backups --dest "<OneDrive project folder>/CAP-Prune-backups"
python -m pytest -q -rA -v --durations=0 --junitxml=<dir>/junit.xml --basetemp=<dir>/basetemp
```
