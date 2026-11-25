# Secondary endpoints S1–S4 (added after the pilot; model 1 seed 0's values are post hoc)

Source: `~/cap-prune-runs/confirmatory/study/study.json` (`models[].secondary`). Mean [range] over seeds; S4 median [IQR] over all draws of an arm's seeds.

## model 1 (smollm3-3b), all seeds

| endpoint | (a) | (b) | (c) |
|---|---|---|---|
| S1 overshoot | 509.6 M [490.7 M, 528.5 M] | 44.0 M [18.9 M, 56.6 M] | 88.1 M [75.5 M, 113.2 M] |
| S2 items per confirmed M | 1,195 [1,041, 1,388] | 2,007 [1,531, 2,572] | 2,828 [2,232, 3,331] |
| S2 GPU s per confirmed M | 19.4 [16.1, 23.4] | 33.6 [24.2, 44.6] | 48.5 [37.7, 58.2] |
| S3 confirmed at 5 points | 81.8 M [56.6 M, 94.4 M] | 81.8 M [56.6 M, 94.4 M] | 88.1 M [56.6 M, 113.2 M] |
| S4, n = 500 | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] |
| S4, n = 1000 | 18.9 M [0.0 M, 18.9 M] | 18.9 M [0.0 M, 18.9 M] | 18.9 M [0.0 M, 18.9 M] |
| S4, n = 2000 | 37.7 M [18.9 M, 56.6 M] | 37.7 M [18.9 M, 56.6 M] | 37.7 M [18.9 M, 56.6 M] |
| S4, n = full | 56.6 M [47.2 M, 66.1 M] | 56.6 M [47.2 M, 66.1 M] | 56.6 M [47.2 M, 66.1 M] |

## model 1 (smollm3-3b), excluding model 1 seed 0

| endpoint | (a) | (b) | (c) |
|---|---|---|---|
| S1 overshoot | 509.6 M [490.7 M, 528.5 M] | 37.7 M [18.9 M, 56.6 M] | 75.5 M [75.5 M, 75.5 M] |
| S2 items per confirmed M | 1,215 [1,041, 1,388] | 2,051 [1,531, 2,572] | 2,781 [2,232, 3,331] |
| S2 GPU s per confirmed M | 19.8 [16.1, 23.4] | 34.4 [24.2, 44.6] | 47.9 [37.7, 58.2] |
| S3 confirmed at 5 points | 75.5 M [56.6 M, 94.4 M] | 75.5 M [56.6 M, 94.4 M] | 84.9 M [56.6 M, 113.2 M] |
| S4, n = 500 | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] |
| S4, n = 1000 | 18.9 M [0.0 M, 18.9 M] | 18.9 M [0.0 M, 18.9 M] | 18.9 M [0.0 M, 18.9 M] |
| S4, n = 2000 | 37.7 M [18.9 M, 56.6 M] | 37.7 M [18.9 M, 56.6 M] | 37.7 M [18.9 M, 56.6 M] |
| S4, n = full | 56.6 M [47.2 M, 66.1 M] | 56.6 M [47.2 M, 66.1 M] | 56.6 M [47.2 M, 66.1 M] |

## model 2 (granite-3.3-2b), all seeds

| endpoint | (a) | (b) | (c) |
|---|---|---|---|
| S1 overshoot | 419.4 M [408.9 M, 424.7 M] | 5.2 M [0.0 M, 15.7 M] | 47.2 M [31.5 M, 62.9 M] |
| S2 items per confirmed M | 1,342 [1,249, 1,388] | 1,596 [1,404, 1,921] | 2,456 [2,152, 2,896] |
| S2 GPU s per confirmed M | 18.6 [16.8, 19.5] | 21.7 [18.9, 26.5] | 37.2 [31.2, 45.3] |
| S3 confirmed at 5 points | 73.4 M [47.2 M, 110.1 M] | 57.7 M [47.2 M, 62.9 M] | 104.9 M [78.6 M, 125.8 M] |
| S4, n = 500 | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] | 0.0 M [0.0 M, 0.0 M] |
| S4, n = 1000 | 31.5 M [15.7 M, 31.5 M] | 31.5 M [15.7 M, 31.5 M] | 31.5 M [15.7 M, 31.5 M] |
| S4, n = 2000 | 47.2 M [31.5 M, 47.2 M] | 47.2 M [31.5 M, 47.2 M] | 47.2 M [31.5 M, 47.2 M] |
| S4, n = full | 47.2 M [47.2 M, 55.1 M] | 47.2 M [47.2 M, 55.1 M] | 62.9 M [55.1 M, 70.8 M] |

