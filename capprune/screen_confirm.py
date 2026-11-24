"""Confirmation sample sizes for the screen-then-confirm protocol (``capprune screen-confirm sizes``).

The confirmation is ONE test of the frozen final checkpoint against the dense model on fresh items: full alpha, Bonferroni
over the categories, and per category the exact one-sided population upper bound of the rule v1 construction
(``population_regate.population_bounds``: Clopper-Pearson on the lost and gained counts, union bound), with no
spending over candidates or looks.  Sample sizes are computed per category from the paired disagreement rates recorded
in ``population-regate/regate.json``.
"""
from __future__ import annotations

import numpy as np

from .config import resolve_path
from .population_regate import population_bounds
from .utils.io import atomic_write_json, atomic_write_text, read_json


def certifies(n: int, kp: int, km: int, delta: float, tol: float) -> bool:
    return population_bounds(n, kp, km, delta, delta)[1] <= tol + 1e-12


def n_needed(q: float, mu: float, delta: float, tol: float, power: float | None = None, draws: int = 2000,
             seed: int = 0, step: int = 50, n_max: int = 1_000_000) -> int | None:
    """Smallest n (multiple of ``step``) whose single-look bound certifies ``tol``: at the expected counts (n50) or in at
    least ``power`` of ``draws`` simulated samples (seeded).  p+ = (q + mu)/2, p- = (q - mu)/2."""
    pp, pm = (q + mu) / 2, (q - mu) / 2
    if pm < 0:
        pp, pm = mu, 0.0

    def ok(n):
        if power is None:
            return certifies(n, round(pp * n), round(pm * n), delta, tol)
        rng = np.random.default_rng([seed, n])
        k = rng.multinomial(n, [pp, pm, 1 - pp - pm], size=draws)
        return np.mean([certifies(n, int(a), int(b), delta, tol) for a, b, _ in k]) >= power

    lo, hi = 1, n_max // step
    if not ok(hi * step):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid * step):
            hi = mid
        else:
            lo = mid + 1
    return lo * step


def sizes(cfg: dict) -> dict:
    sc = cfg["screen_confirm"]
    R = read_json(resolve_path(cfg, sc["regate"]))
    A = R["runs"][sc["run"]]["attempts"]
    final = next(a for a in A if a["attempt"] == int(sc["final_attempt"]))
    cats = list(final["full_pool_projection"])
    alpha, tol = float(sc["alpha"]), float(sc["tolerance"])
    delta = alpha / len(cats)  # one test, Bonferroni over the categories
    out = {"alpha": alpha, "delta_per_category": delta, "tolerance": tol, "source": str(resolve_path(cfg, sc["regate"])),
           "run": sc["run"], "final_attempt": int(sc["final_attempt"]), "categories": {}}
    for c in cats:
        qs = [a["full_pool_projection"][c]["q_disagree"] for a in A]
        qv = {"final": final["full_pool_projection"][c]["q_disagree"], "median": float(np.median(qs)), "max": float(max(qs))}
        out["categories"][c] = {"q": qv, "n_revealed_final": final["full_pool_projection"][c]["n_revealed"], "needed": {
            f"{100 * mu:.1f}": {f"{k}_{which}": n_needed(qv[which], mu, delta, tol, None if k == "n50" else 0.8)
                                for which in ("final", "median", "max") for k in ("n50", "n80")}
            for mu in sc["true_losses"]}}
    return out


def render(S: dict) -> str:
    L = [f"Per-category one-sided bound at delta = {S['delta_per_category']:.4f} (alpha {S['alpha']} / {len(S['categories'])} "
         f"categories), tolerance {100 * S['tolerance']:.1f} points. Rates: run {S['run']}'s final checkpoint (attempt "
         f"{S['final_attempt']}) and the median / maximum over the run's attempts. n50 = bound at the expected counts "
         "certifies (about 50 % power); n80 = 80 % of 2,000 seeded simulated samples certify. Steps of 50 items.", "",
         "| category | q final / median / max | true loss | n50 (final q) | n80 (final q) | n80 (median q) | n80 (max q) |",
         "|---|---|---|---|---|---|---|"]
    for c, v in S["categories"].items():
        for mu, d in v["needed"].items():
            f = lambda x: "> 1,000,000" if x is None else f"{x:,}"  # noqa: E731
            L.append(f"| {c} | {100 * v['q']['final']:.1f} / {100 * v['q']['median']:.1f} / {100 * v['q']['max']:.1f} % | {mu} pts | "
                     f"{f(d['n50_final'])} | {f(d['n80_final'])} | {f(d['n80_median'])} | {f(d['n80_max'])} |")
    return "\n".join(L) + "\n"


def write_sizes(cfg: dict):
    S = sizes(cfg)
    out = resolve_path(cfg, cfg["screen_confirm"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "confirmation_sizes.json", S)
    atomic_write_text(out / "confirmation_sizes.md", render(S))
    return out / "confirmation_sizes.md"
