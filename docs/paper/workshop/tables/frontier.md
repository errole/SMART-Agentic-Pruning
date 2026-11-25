# Certification frontier (EXPLORATORY)

Source: `~/cap-prune-runs/exploratory/frontier.json`. Certified parameters removed, median [IQR] over the 50 subsamples of every seed (the full set: one value per seed).

## model 1 (smollm3-3b)

| arm | limit | n = 500 | n = 1000 | n = 2000 | n = full |
|---|---|---|---|---|---|
| (a) perplexity | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 18.9 M [18.9 M, 18.9 M] |
| (a) perplexity | 3 pts | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [47.2 M, 66.1 M] |
| (a) perplexity | 4 pts | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [42.5 M, 75.5 M] | 75.5 M [66.1 M, 75.5 M] |
| (a) perplexity | 5 pts | 18.9 M [18.9 M, 37.7 M] | 56.6 M [37.7 M, 75.5 M] | 75.5 M [56.6 M, 75.5 M] | 94.4 M [75.5 M, 94.4 M] |
| (b) gate | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 18.9 M [18.9 M, 18.9 M] |
| (b) gate | 3 pts | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [47.2 M, 66.1 M] |
| (b) gate | 4 pts | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [42.5 M, 75.5 M] | 75.5 M [66.1 M, 75.5 M] |
| (b) gate | 5 pts | 18.9 M [18.9 M, 37.7 M] | 56.6 M [37.7 M, 75.5 M] | 75.5 M [56.6 M, 75.5 M] | 94.4 M [75.5 M, 94.4 M] |
| (c) gate + KL | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 18.9 M [18.9 M, 18.9 M] |
| (c) gate + KL | 3 pts | 0.0 M [0.0 M, 0.0 M] | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [47.2 M, 66.1 M] |
| (c) gate + KL | 4 pts | 18.9 M [0.0 M, 18.9 M] | 37.7 M [18.9 M, 56.6 M] | 56.6 M [42.5 M, 75.5 M] | 75.5 M [66.1 M, 75.5 M] |
| (c) gate + KL | 5 pts | 18.9 M [18.9 M, 37.7 M] | 56.6 M [37.7 M, 75.5 M] | 75.5 M [56.6 M, 75.5 M] | 94.4 M [75.5 M, 103.8 M] |

## model 2 (granite-3.3-2b)

| arm | limit | n = 500 | n = 1000 | n = 2000 | n = full |
|---|---|---|---|---|---|
| (a) perplexity | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 15.7 M [15.7 M, 31.5 M] | 31.5 M [31.5 M, 39.3 M] |
| (a) perplexity | 3 pts | 0.0 M [0.0 M, 0.0 M] | 31.5 M [15.7 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 55.1 M] |
| (a) perplexity | 4 pts | 15.7 M [0.0 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 62.9 M] | 62.9 M [55.1 M, 62.9 M] |
| (a) perplexity | 5 pts | 31.5 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 47.2 M] | 62.9 M [47.2 M, 62.9 M] | 62.9 M [55.1 M, 86.5 M] |
| (b) gate | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 15.7 M [15.7 M, 31.5 M] | 31.5 M [31.5 M, 39.3 M] |
| (b) gate | 3 pts | 0.0 M [0.0 M, 0.0 M] | 31.5 M [15.7 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 55.1 M] |
| (b) gate | 4 pts | 15.7 M [0.0 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 62.9 M] | 62.9 M [55.1 M, 62.9 M] |
| (b) gate | 5 pts | 31.5 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 47.2 M] | 62.9 M [47.2 M, 62.9 M] | 62.9 M [55.1 M, 62.9 M] |
| (c) gate + KL | 2 pts | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 15.7 M [15.7 M, 31.5 M] | 31.5 M [31.5 M, 39.3 M] |
| (c) gate + KL | 3 pts | 0.0 M [0.0 M, 0.0 M] | 31.5 M [15.7 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 62.9 M [55.1 M, 70.8 M] |
| (c) gate + KL | 4 pts | 15.7 M [0.0 M, 31.5 M] | 47.2 M [31.5 M, 47.2 M] | 62.9 M [47.2 M, 78.6 M] | 94.4 M [86.5 M, 102.2 M] |
| (c) gate + KL | 5 pts | 31.5 M [31.5 M, 47.2 M] | 47.2 M [47.2 M, 78.6 M] | 94.4 M [78.6 M, 110.1 M] | 110.1 M [94.4 M, 118.0 M] |

