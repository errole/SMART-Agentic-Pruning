# Arm (a), the perplexity screen: the figures behind section 5.2

Sources: each `ppl_top1` run's `journal.jsonl` (dense C4 perplexity `ppl0`, each trial's `ppl_trial` and `relative_increase`; the threshold is 15 % over the parent) aligned to `batches/<batch>/<run>.order.json` by mask hash; per-category estimates from `~/cap-prune-runs/exploratory/loss_curves.json`.

## The screen and the confirmed claims

| model | seed | accepted / proposals | largest one-step rise | dense C4 perplexity | claim (checkpoint) | steps from claim to end | removed at the end | overshoot | STEM at the end | Humanities at the end | Social Sciences at the end | Other at the end | worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model 1 | 0 | 30 / 30 | 1.4 % | 14.11 | 3 | 27 | 18.41 % | 16.57 % | +9.92 | +10.50 | +17.97 | +14.43 | Social Sciences |
| model 1 | 1 | 30 / 30 | 1.1 % | 14.11 | 2 | 28 | 18.41 % | 17.19 % | +11.19 | +10.67 | +20.73 | +14.91 | Social Sciences |
| model 1 | 2 | 30 / 30 | 1.5 % | 14.11 | 4 | 26 | 18.41 % | 15.96 % | +12.39 | +10.63 | +20.67 | +13.31 | Social Sciences |
| model 2 | 0 | 30 / 30 | 1.7 % | 8.64 | 3 | 27 | 18.62 % | 16.76 % | +16.83 | +20.13 | +26.23 | +25.69 | Social Sciences |
| model 2 | 1 | 30 / 30 | 1.7 % | 8.64 | 4 | 26 | 18.62 % | 16.14 % | +17.92 | +18.53 | +32.13 | +24.94 | Social Sciences |
| model 2 | 2 | 30 / 30 | 3.0 % | 8.64 | 3 | 27 | 18.62 % | 16.76 % | +14.31 | +14.63 | +24.70 | +19.60 | Social Sciences |

## Perplexity / dense, and a cumulative rule (post hoc: 15 % against the dense model)

The cumulative rule stops before the first checkpoint whose perplexity exceeds the dense model's by more than 15 %. Its stopping checkpoint is also the last one at or below that level in every run (checked; the last column).

| model | seed | at the claim | at the first failing checkpoint (position) | at the screen's end | cumulative rule: stops at checkpoint | parameters removed | worst category there (estimate, points) | perplexity / dense there (next) | last checkpoint <= threshold |
|---|---|---|---|---|---|---|---|---|---|
| model 1 | 0 | 1.016 | 1.023 (4) | 1.311 | 18 | 339.7 M (11.05 %) | Social Sciences +14.1 | 1.147 (1.157) | yes |
| model 1 | 1 | 1.009 | 1.014 (3) | 1.250 | 20 | 377.5 M (12.28 %) | Social Sciences +13.3 | 1.146 (1.158) | yes |
| model 1 | 2 | 1.021 | 1.027 (5) | 1.296 | 18 | 339.7 M (11.05 %) | Social Sciences +13.9 | 1.140 (1.154) | yes |
| model 2 | 0 | 1.021 | 1.033 (4) | 1.398 | 13 | 204.5 M (8.07 %) | Other +17.7 | 1.140 (1.154) | yes |
| model 2 | 1 | 1.027 | 1.037 (5) | 1.386 | 15 | 235.9 M (9.31 %) | Social Sciences +17.5 | 1.141 (1.161) | yes |
| model 2 | 2 | 1.047 | 1.072 (4) | 1.516 | 7 | 110.1 M (4.35 %) | Social Sciences +3.0 | 1.137 (1.155) | yes |

**Ranges quoted in the draft** (computed here; check against the text):

* SmolLM3: removed at the end 18.4 % to 18.4 %; per-category loss at the end +9.9 to +20.7; worst category at the end: Social Sciences; perplexity / dense at the end 1.250 to 1.311; largest one-step rise 1.1 % to 1.5 %.
* Granite: removed at the end 18.6 % to 18.6 %; per-category loss at the end +14.3 to +32.1; worst category at the end: Social Sciences; perplexity / dense at the end 1.386 to 1.516; largest one-step rise 1.7 % to 3.0 %.
* All runs: claims at checkpoints 2 to 4; 26 to 28 steps from claim to the screen's end; overshoot 16.0 % to 17.2 %; worst-category loss at the end +18.0 to +32.1 points.
* Largest one-step rise over all runs 3.0 %; in the other 5 runs at most 1.7 %.
* Perplexity / dense: at the claims 1.009 to 1.047; at the first failing checkpoints 1.014 to 1.072; at the screen's end 1.250 to 1.516.
* Cumulative rule: stops after 7 to 20 steps, 110.1 M to 377.5 M parameters removed; worst-category estimate there +3.0 to +17.7 points; same as the last checkpoint at or below the threshold in 6 of 6 runs.
