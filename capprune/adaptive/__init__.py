"""Adaptive candidate evaluation (CAP-Prune experiment 1).

Question: can adaptive allocation of benchmark questions across domains find a
pruned checkpoint that meets per-domain accuracy-loss limits with less
evaluation compute than round-robin early stopping?

    bounds.py     exact finite-pool (hypergeometric) loss bounds + error spending
    evaluator.py  plan of looks, shared decision rule, sequential test of one candidate
    policies.py   round_robin, widest_interval, threshold_aware
    cache.py      replay input format and the replay source (reveals items on request only)
    scorer.py     full-pool reference decisions and holdout check (privileged)
    replay.py     runs every policy on every candidate, search metric, report
    fixture.py    deterministic CPU fixture
    build.py      builds a cache from the Qwen3 pipeline (MMLU scorer, structured Wanda, benchmark)

See docs/adaptive_evaluation.md.
"""
