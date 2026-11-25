# Primary endpoint: confirmed parameters removed per model x arm x seed

Source: `~/cap-prune-runs/confirmatory/study/study.json` (`models[].primary`). Seed 0 of model 1 is the pilot.

| model | seed | (a) perplexity | (b) gate | (c) gate + KL | (b) − (a) | (c) − (b) |
|---|---|---|---|---|---|---|
| model 1 | 0 | 56.6 M (1.84 %) | 56.6 M (1.84 %) | 56.6 M (1.84 %) | 0.0 M | 0.0 M |
| model 1 | 1 | 37.7 M (1.23 %) | 37.7 M (1.23 %) | 37.7 M (1.23 %) | 0.0 M | 0.0 M |
| model 1 | 2 | 75.5 M (2.46 %) | 75.5 M (2.46 %) | 75.5 M (2.46 %) | 0.0 M | 0.0 M |
| model 1 | mean [range] | 56.6 M [37.7 M, 75.5 M] | 56.6 M [37.7 M, 75.5 M] | 56.6 M [37.7 M, 75.5 M] | 0.0 M | 0.0 M |
| model 2 | 0 | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 62.9 M (2.48 %) | 0.0 M | 15.7 M |
| model 2 | 1 | 62.9 M (2.48 %) | 62.9 M (2.48 %) | 78.6 M (3.10 %) | 0.0 M | 15.7 M |
| model 2 | 2 | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 0.0 M | 0.0 M |
| model 2 | mean [range] | 52.4 M [47.2 M, 62.9 M] | 52.4 M [47.2 M, 62.9 M] | 62.9 M [47.2 M, 78.6 M] | 0.0 M | 10.5 M |
