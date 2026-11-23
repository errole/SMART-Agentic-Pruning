"""A small deterministic replay cache for CPU tests (no model, no data download).

Four domains named after the broad MMLU categories.  Every count below is
exact, so the full-pool losses are known in advance:

    candidate       order  what it is                                        full-pool loss
    clear_fail        0    "Other" loses 30 points, the rest exactly 0        Other +28 pts       -> does not qualify
    near_threshold    1    "Humanities" sits exactly at the 5-point limit     Humanities +5.0 pts -> qualifies (at the limit)
    easy_pass         2    every domain breaks and fixes 1 % of items          all 0               -> qualifies

With the settings in ``configs/adaptive_fixture.yaml`` (tolerance 0.05,
budget 1,000 of 2,550 pool items) the expected decisions are: clear_fail
REJECT; near_threshold INCONCLUSIVE after spending its budget (Humanities can
receive at most 1,000 - 3 x 40 = 880 of its 900 items, so its interval around a
loss of exactly 0.05 never drops to 0.05); easy_pass ACCEPT.
"Other" is last in the round-robin order on purpose.  Latencies are synthetic
(decode ms/token).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import write_items, write_model

DOMAINS = {  # name: (pool size, holdout size, dense accuracy)
    "STEM": (600, 900, 0.55),
    "Humanities": (900, 1350, 0.60),
    "Social Sciences": (500, 750, 0.70),
    "Other": (550, 825, 0.65),
}
CANDIDATES = {  # name: (order, decode ms/token, {domain: (broken rate, fixed rate)}, default rates)
    "clear_fail": (0, 6.0, {"Other": (0.30, 0.02)}, (0.01, 0.01)),
    "near_threshold": (1, 6.6, {"Humanities": (0.10, 0.05)}, (0.01, 0.01)),
    "easy_pass": (2, 7.4, {}, (0.01, 0.01)),
}
DENSE_DECODE_MS = 9.0


def _slug(d: str) -> str:
    return d.lower().replace(" ", "_")


def write_fixture(out: str | Path, seed: int = 7) -> Path:
    root = Path(out)
    items, dense = [], {}
    split_ids: dict[tuple[str, str], list[str]] = {}
    for j, (d, (n_pool, n_hold, acc)) in enumerate(DOMAINS.items()):
        for split, n in (("pool", n_pool), ("holdout", n_hold)):
            rng = np.random.default_rng([seed, j, 0 if split == "pool" else 1])
            ids = [f"{_slug(d)}/{split}/{i:04d}" for i in range(n)]
            split_ids[(d, split)] = ids
            correct = np.zeros(n, dtype=int)
            correct[rng.permutation(n)[: int(round(acc * n))]] = 1
            for i, c, t in zip(ids, correct, rng.integers(60, 400, size=n)):
                dense[i] = int(c)
                items.append({"item_id": i, "domain": d, "split": split, "tokens": int(t)})
    write_items(root, items)
    write_model(root, {"name": "dense", "role": "dense", "order": -1, "correct": dense,
                       "latency": {"decode_ms_per_token": DENSE_DECODE_MS}, "timing": None,
                       "info": {"kind": "fixture"}})
    for ci, (name, (order, lat, special, default)) in enumerate(CANDIDATES.items()):
        cand = dict(dense)
        for j, d in enumerate(DOMAINS):
            r_break, r_fix = special.get(d, default)
            for si, split in enumerate(("pool", "holdout")):
                ids = split_ids[(d, split)]
                rng = np.random.default_rng([seed, 100 + ci, j, si])
                right = [i for i in ids if dense[i] == 1]
                wrong = [i for i in ids if dense[i] == 0]
                n = len(ids)
                for i in rng.permutation(right)[: int(round(r_break * n))]:
                    cand[str(i)] = 0
                for i in rng.permutation(wrong)[: int(round(r_fix * n))]:
                    cand[str(i)] = 1
        write_model(root, {"name": name, "role": "candidate", "order": order, "correct": cand,
                           "latency": {"decode_ms_per_token": lat}, "timing": None,
                           "info": {"kind": "fixture", "rates": {d: special.get(d, default) for d in DOMAINS}}})
    return root
