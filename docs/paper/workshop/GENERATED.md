# Generated files

Regenerate with:

```
python -m capprune paper-a --config configs/confirmatory.yaml --out docs/paper/workshop
```

Last generated at commit `a469e60` (clean tree before generating: True), 2026-11-26 00:55.

Inputs (run files):

* `~/cap-prune-runs/confirmatory/study/study.json`
* `~/cap-prune-runs/exploratory/frontier.json`
* `~/cap-prune-runs/exploratory/loss_curves.json`
* `~/cap-prune-runs/exploratory/c1_mechanism.json`
* the `ppl_top1` runs' `journal.jsonl` and `<run>.order.json` (for `tables/perplexity_arm_a.md`)

Outputs:

* `figures/loss_curves_smollm3-3b.png`
* `figures/loss_curves_smollm3-3b.pdf`
* `figures/loss_curves_granite-3.3-2b.png`
* `figures/loss_curves_granite-3.3-2b.pdf`
* `figures/certification_frontier.png`
* `figures/certification_frontier.pdf`
* `figures/overshoot_and_cost.png`
* `figures/overshoot_and_cost.pdf`
* `tables/primary.md`
* `tables/secondary_S1_S4.md`
* `tables/c1.md`
* `tables/c6.md`
* `tables/frontier.md`
* `tables/perplexity_arm_a.md`

Not generated: `tables/deviations.md` (transcribed from `docs/confirmatory_report.md`) and `draft.md`.
