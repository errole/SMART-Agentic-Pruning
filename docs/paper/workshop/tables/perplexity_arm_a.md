# Arm (a), the perplexity screen: the figures behind section 5.2

Sources: each `ppl_top1` run's `journal.jsonl` (dense C4 perplexity `ppl0`, each trial's `ppl_trial` and `relative_increase`; the threshold is 15 % over the parent) aligned to `batches/<batch>/<run>.order.json` by mask hash; per-category estimates from `~/cap-prune-runs/exploratory/loss_curves.json`.

| model | seed | accepted / proposals | largest one-step rise | dense C4 perplexity | perplexity / dense at the claim | at the screen's end | claim (checkpoint) | steps from claim to end | removed at the end | overshoot | STEM at the end | Humanities at the end | Social Sciences at the end | Other at the end | worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model 1 | 0 | 30 / 30 | 1.4 % | 14.11 | 1.016 | 1.311 | 3 | 27 | 18.41 % | 16.57 % | +9.92 | +10.50 | +17.97 | +14.43 | Social Sciences |
| model 1 | 1 | 30 / 30 | 1.1 % | 14.11 | 1.009 | 1.250 | 2 | 28 | 18.41 % | 17.19 % | +11.19 | +10.67 | +20.73 | +14.91 | Social Sciences |
| model 1 | 2 | 30 / 30 | 1.5 % | 14.11 | 1.021 | 1.296 | 4 | 26 | 18.41 % | 15.96 % | +12.39 | +10.63 | +20.67 | +13.31 | Social Sciences |
| model 2 | 0 | 30 / 30 | 1.7 % | 8.64 | 1.021 | 1.398 | 3 | 27 | 18.62 % | 16.76 % | +16.83 | +20.13 | +26.23 | +25.69 | Social Sciences |
| model 2 | 1 | 30 / 30 | 1.7 % | 8.64 | 1.027 | 1.386 | 4 | 26 | 18.62 % | 16.14 % | +17.92 | +18.53 | +32.13 | +24.94 | Social Sciences |
| model 2 | 2 | 30 / 30 | 3.0 % | 8.64 | 1.047 | 1.516 | 3 | 27 | 18.62 % | 16.76 % | +14.31 | +14.63 | +24.70 | +19.60 | Social Sciences |

**Ranges quoted in the draft** (computed here; check against the text):

* SmolLM3: removed at the end 18.4 % to 18.4 %; per-category loss at the end +9.9 to +20.7; worst category: Social Sciences; perplexity / dense at the end 1.25 to 1.31; largest one-step rise 1.1 % to 1.5 %.
* Granite: removed at the end 18.6 % to 18.6 %; per-category loss at the end +14.3 to +32.1; worst category: Social Sciences; perplexity / dense at the end 1.39 to 1.52; largest one-step rise 1.7 % to 3.0 %.
* All runs: claims at checkpoints 2 to 4; 26 to 28 steps from claim to the screen's end; overshoot 16.0 % to 17.2 %; worst-category loss at the end +18.0 to +32.1 points; perplexity / dense at the claim 1.009 to 1.047.
