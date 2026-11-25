"""Bound development (docs/bound_development_plan.md, committed 231c2a6 before any candidate touched real data).

``capprune bound-dev simulate``: validity (section 3), sample sizes (4), the best valid bound (5) and go/no-go
condition 1 (7) - data-free.
``capprune bound-dev apply``: the candidates on the scored confirmation outcomes (6), condition 2 and the verdict (7).
Labelled "method development on seen data: it selects a method, it is not a claim". CPU only.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import bounds as B
from .config import resolve_path
from .confirmatory import CATS, _git, counts_from_outcomes
from .utils.io import atomic_write_json, atomic_write_text, read_json

LABEL = "method development on seen data: it selects a method, it is not a claim"
ALPHA, TOL = 0.05, 0.03
VALIDITY = {"n": (500, 1000, 3000), "q": (0.04, 0.06, 0.08, 0.10, 0.12, 0.15), "delta": (0.0, 0.01, 0.015, 0.02, 0.03)}
SIZE_Q = (0.06, 0.07, 0.08, 0.09)
SIZE_DELTA = (0.0, 0.01, 0.015)
REPS_VALIDITY, REPS_POWER = 10_000, 4_000
SE_VALIDITY = math.sqrt(0.95 * 0.05 / REPS_VALIDITY)


def _out(cfg: dict) -> Path:
    return resolve_path(cfg, (cfg["confirmatory"].get("bound_dev") or {}).get("out_dir", "~/cap-prune-runs/bound-dev"))


# ============================================================================ section 3: validity
def validity() -> dict:
    rows, i = [], 0
    for n in VALIDITY["n"]:
        for q in VALIDITY["q"]:
            for d in VALIDITY["delta"]:
                pl, pg = (q + d) / 2, (q - d) / 2
                row = {"n": n, "q": q, "delta": d, "p_lost": pl, "p_gained": pg}
                for m in B.METHODS:
                    rng = np.random.default_rng([20261125, i])
                    sim = 1 - B.simulate_certify(m, n, pl, pg, ALPHA, d, REPS_VALIDITY, rng)
                    ex = None if m == "betting" else 1 - B.exact_certify_probability(m, n, pl, pg, ALPHA, d)
                    row[m] = {"coverage_simulated": sim, "coverage_exact": ex}
                rows.append(row)
                i += 1
    verdict = {}
    for m in B.METHODS:
        if m == "betting":
            worst = min(r[m]["coverage_simulated"] for r in rows)
            ok = worst >= 0.95 - 3 * SE_VALIDITY
        else:
            worst = min(r[m]["coverage_exact"] for r in rows)
            ok = worst >= 0.95 - 1e-9
        verdict[m] = {"worst_coverage": worst, "meets_requirement": bool(ok),
                      "criterion": "simulated >= 0.95 - 3 SE" if m == "betting" else "exact >= 0.95"}
    # the extra boundary check of the exact test at the real-data sizes (section 3)
    extra = {}
    for n in (3000, 3500, 3600):
        qs = np.round(np.arange(0.040, 0.15001, 0.001), 4)
        sizes = [B.exact_certify_probability("exact_score", n, (q + TOL) / 2, (q - TOL) / 2, ALPHA, TOL) for q in qs]
        extra[str(n)] = {"max_size": float(max(sizes)), "at_q": float(qs[int(np.argmax(sizes))]),
                         "z_critical": B.exact_score_critical(n, TOL, ALPHA)}
    ok_extra = all(v["max_size"] <= ALPHA + 1e-12 for v in extra.values())
    verdict["exact_score"]["boundary_check_real_n"] = ok_extra
    verdict["exact_score"]["meets_requirement"] = bool(verdict["exact_score"]["meets_requirement"] and ok_extra)
    verdict["normal"]["meets_requirement_note"] = "reference only: cannot be adopted whatever its coverage"
    return {"grid": VALIDITY, "reps": REPS_VALIDITY, "rows": rows, "verdict": verdict, "exact_score_boundary_real_n": extra}


# ============================================================================ section 4: sample sizes
def _power(method: str, n: int, q: float, d: float) -> float:
    pl, pg = (q + d) / 2, (q - d) / 2
    if method == "betting":
        rng = np.random.default_rng([20261126, n, round(1000 * q), round(10000 * d)])
        return B.simulate_certify("betting", n, pl, pg, ALPHA, TOL, REPS_POWER, rng)
    return B.exact_certify_probability(method, n, pl, pg, ALPHA, TOL)


def n_needed(method: str, q: float, d: float, target: float = 0.80, lo: int = 50, hi: int = 20000) -> dict:
    cache = {}

    def pw(n):
        if n not in cache:
            cache[n] = _power(method, n, q, d)
        return cache[n]

    if pw(lo) >= target:
        return {"n": lo, "power_at_n": pw(lo), "power_at_n_minus_50": None}
    if pw(hi) < target:
        return {"n": None, "power_at_n": pw(hi), "power_at_n_minus_50": None, "note": f"> {hi}"}
    while hi - lo > 50:
        mid = (lo + hi) // 2 // 50 * 50
        mid = mid if lo < mid < hi else lo + 50
        if pw(mid) >= target:
            hi = mid
        else:
            lo = mid
    return {"n": hi, "power_at_n": pw(hi), "power_at_n_minus_50": pw(hi - 50)}


def sample_sizes() -> dict:
    out = {}
    for m in B.METHODS:
        out[m] = {f"{d:.3f}": {f"{q:.2f}": n_needed(m, q, d) for q in SIZE_Q} for d in SIZE_DELTA}
    return out


# ============================================================================ sections 5 and 7.1
def select_best(valid: dict, sizes: dict) -> dict:
    cands = []
    for m in B.ADOPTABLE:
        if not valid[m]["meets_requirement"]:
            continue
        ns = [sizes[m]["0.010"][f"{q:.2f}"]["n"] for q in SIZE_Q]
        mean = float(np.mean([x if x is not None else math.inf for x in ns]))
        cands.append((mean, m))
    if not cands:
        return {"best": None, "reason": "no candidate meets the validity requirement"}
    mean, m = min(cands)
    return {"best": m, "mean_n_at_1_point": mean, "candidates": {mm: v for v, mm in cands}}


def condition_1(best: str | None, sizes: dict) -> dict:
    if best is None:
        return {"holds": False, "reason": "no valid candidate"}
    per_q = {}
    for q in SIZE_Q:
        nb = sizes[best]["0.010"][f"{q:.2f}"]["n"]
        nc = sizes["cp_union"]["0.010"][f"{q:.2f}"]["n"]
        per_q[f"{q:.2f}"] = {"n_best": nb, "n_cp_union": nc, "ratio": (nb / nc) if (nb and nc) else None,
                             "at_least_25_percent_fewer": bool(nb is not None and nc is not None and nb <= 0.75 * nc)}
    return {"holds": all(v["at_least_25_percent_fewer"] for v in per_q.values()), "per_q": per_q,
            "rule": "n_best <= 0.75 x n_CP at a true loss of 1.0 point for each of q = 6, 7, 8, 9 % (80 % power, one category)"}


def simulate(cfg: dict) -> Path:
    t0 = time.time()
    V = validity()
    S = sample_sizes()
    best = select_best(V["verdict"], S)
    c1 = condition_1(best["best"], S)
    res = {"label": "method development (data-free simulation)", "plan": "docs/bound_development_plan.md", "alpha": ALPHA,
           "tolerance": TOL, "validity": V, "sample_sizes": S, "best_valid_bound": best, "condition_1": c1,
           "commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "", "seconds": time.time() - t0,
           "written_at": time.time()}
    od = _out(cfg)
    od.mkdir(parents=True, exist_ok=True)
    atomic_write_json(od / "simulation.json", res)
    atomic_write_text(od / "simulation.md", render_simulation(res))
    return od / "simulation.md"


# ============================================================================ section 6 and 7.2: the scored data
def _sequences(oc: dict, i: int, cats: np.ndarray, qids: np.ndarray) -> dict[str, np.ndarray]:
    """Per category, the paired differences (lost +1, gained -1, agree 0) of checkpoint ``i`` in the plan's item order."""
    out = {}
    for c in CATS:
        ix = np.flatnonzero(cats == c)
        ix = ix[np.argsort(qids[ix], kind="stable")]
        ix = ix[B.category_order(list(qids[ix]), c)]
        d, t = oc["dense"][i][ix], oc["cand"][i][ix]
        out[c] = (d & ~t).astype(np.float64) - (t & ~d).astype(np.float64)
    return out


def decide(method: str, oc: dict, i: int) -> dict:
    cats, qids = oc["category"], oc["qids"]
    cnt = counts_from_outcomes(oc["dense"][i], oc["cand"][i], cats)
    per = {}
    if method == "betting":
        seqs = _sequences(oc, i, cats, qids)
        for c in CATS:
            per[c] = {"certify": bool(B.betting_certifies(seqs[c][None, :], ALPHA, TOL)[0])}
    else:
        for c, (n, kp, km) in cnt.items():
            ok = bool(B.region(method, n, np.array([kp]), np.array([km]), ALPHA, TOL)[0])
            stat = {"cp_union": lambda: {"upper": B.cp_union_upper(n, kp, km, ALPHA)},
                    "exact_score": lambda: {"z": float(B.score_z(n, kp, km, TOL)), "z_critical": B.exact_score_critical(n, TOL, ALPHA)},
                    "normal": lambda: {"upper": float(B.normal_upper(n, kp, km, ALPHA))}}[method]()
            per[c] = {"certify": ok, **stat}
    return {"pass": all(v["certify"] for v in per.values()), "categories": per,
            "estimate": {c: (kp - km) / n for c, (n, kp, km) in cnt.items()}}


def run_sequence(method: str, oc: dict) -> dict:
    claim, tested = None, []
    for i in range(len(oc["position"])):
        r = decide(method, oc, i)
        tested.append({"position": int(oc["position"][i]), **r})
        if not r["pass"]:
            break
        claim = i
    return {"certified_checkpoints": 0 if claim is None else claim + 1,
            "claim_position": None if claim is None else int(oc["position"][claim]),
            "parameters_removed": 0 if claim is None else int(oc["materialized_parameters_removed"][claim]),
            "tested": tested}


def apply(cfg: dict) -> Path:
    from .exploratory import load_models

    od = _out(cfg)
    sim = read_json(od / "simulation.json")
    best = sim["best_valid_bound"]["best"]
    valid = [m for m in B.ADOPTABLE if sim["validity"]["verdict"][m]["meets_requirement"]]
    methods = ["cp_union", *valid, "normal"]
    models = load_models(cfg)
    rows = []
    for M in models:
        for r in M["runs"]:
            oc = r["oc"]
            res = {m: run_sequence(m, oc) for m in methods}
            cp = res["cp_union"]
            if (cp["claim_position"], cp["parameters_removed"]) != (r["confirm"]["claim_position"], r["confirm"]["confirmed_parameters_removed"]):
                raise SystemExit(f"{r['run_id']}: the CP union reference does not reproduce the recorded claim")
            nxt = cp["certified_checkpoints"]  # index of the first checkpoint past the CP claim
            past = {m: (decide(m, oc, nxt) if nxt < len(oc["position"]) else None) for m in methods}
            rows.append({"model": M["model_key"], "label": M["label"], "run_id": r["run_id"], "method": r["method"],
                         "seed": r["seed"], "frozen_checkpoints": int(len(oc["position"])),
                         "certified": {m: {k: v for k, v in res[m].items() if k != "tested"} for m in methods},
                         "first_checkpoint_past_cp_claim": None if nxt >= len(oc["position"]) else
                         {"position": int(oc["position"][nxt]), **{m: past[m] for m in methods}}})
    # condition 2: arm (a)'s orders
    pairs = []
    for x in rows:
        if x["method"] != "ppl_top1":
            continue
        gain = (x["certified"][best]["certified_checkpoints"] - x["certified"]["cp_union"]["certified_checkpoints"]) if best else None
        pairs.append({"model": x["model"], "seed": x["seed"], "cp_union": x["certified"]["cp_union"]["certified_checkpoints"],
                      "best": None if best is None else x["certified"][best]["certified_checkpoints"], "gain": gain,
                      "one_more": bool(gain is not None and gain >= 1)})
    n_gain = sum(p["one_more"] for p in pairs)
    c2 = {"holds": bool(best is not None and n_gain >= 4), "pairs_with_one_more": n_gain, "of": len(pairs), "ceiling": 5,
          "pairs": pairs, "rule": "on arm (a)'s frozen orders, full n, 3 points: >= 1 more certified checkpoint than the CP "
                                  "union bound in >= 4 of the 6 model x seed pairs (ceiling 5: model 1 seed 2's shared failing step, +3.20)"}
    c1 = sim["condition_1"]
    verdict = {"B_proceeds": bool(c1["holds"] and c2["holds"]), "best_valid_bound": best,
               "condition_1_simulation": c1["holds"], "condition_2_real_data": c2["holds"]}
    res = {"label": LABEL, "plan": "docs/bound_development_plan.md", "methods": methods, "normal_note": "reference only, not valid",
           "rows": rows, "condition_2": c2, "condition_1": c1, "verdict": verdict, "simulation_commit": sim["commit"],
           "commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "", "written_at": time.time(),
           "sources": {M["model_key"]: [r["outcomes_path"] for r in M["runs"]] for M in models}}
    atomic_write_json(od / "real_data.json", res)
    atomic_write_text(od / "real_data.md", render_real(res, sim))
    return od / "real_data.md"


# ============================================================================ rendering
def _ratio(r) -> str:
    return "–" if r is None else f"{r:.2f}"


_ML = {"cp_union": "CP union (reference)", "exact_score": "exact score (ii)", "betting": "betting CS (iii)",
       "normal": "normal (reference only, not valid)"}


def render_simulation(R: dict) -> str:
    V, S = R["validity"], R["sample_sizes"]
    L = ["# Bound development: simulation (data-free)", "", f"Plan: `{R['plan']}`. alpha = {R['alpha']}, limit = 3 points. "
         f"Commit `{R['commit'][:7]}` (clean tree: {R['clean_tree']}).", "",
         "## Validity (coverage of the one-sided 95 % upper bound; 90 grid points)", "",
         "| method | worst coverage | criterion | valid |", "|---|---|---|---|"]
    for m, v in V["verdict"].items():
        L.append(f"| {_ML[m]} | {v['worst_coverage']:.4f} | {v['criterion']} | {v['meets_requirement']}{' (reference only)' if m == 'normal' else ''} |")
    L += ["", "Exact score test at the real-data sizes, boundary size over q = 4.0-15.0 %: " + "; ".join(
        f"n = {n}: max {v['max_size']:.4f} at q = {100 * v['at_q']:.1f} % (z* = {v['z_critical']:.4f})" for n, v in V["exact_score_boundary_real_n"].items()), "",
          "Worst coverage by n (exact for count-based methods, simulated for betting):", "",
          "| n | " + " | ".join(_ML[m] for m in V["verdict"]) + " |", "|---|" + "---|" * len(V["verdict"])]
    for n in V["grid"]["n"]:
        rs = [r for r in V["rows"] if r["n"] == n]
        cells = []
        for m in V["verdict"]:
            key = "coverage_simulated" if m == "betting" else "coverage_exact"
            w = min(rs, key=lambda r: r[m][key])
            cells.append(f"{w[m][key]:.4f} (q {100 * w['q']:.0f} %, loss {100 * w['delta']:.1f})")
        L.append(f"| {n:,} | " + " | ".join(cells) + " |")
    L += ["", "## Sample size per category for 80 % power to certify 3 points (one category)", "",
          "| true loss | q | " + " | ".join(_ML[m] for m in S) + " |", "|---|---|" + "---|" * len(S)]
    for d in SIZE_DELTA:
        for q in SIZE_Q:
            cells = []
            for m in S:
                x = S[m][f"{d:.3f}"][f"{q:.2f}"]
                ncp = S["cp_union"][f"{d:.3f}"][f"{q:.2f}"]["n"]
                rr = f" ({x['n'] / ncp:.2f}x)" if (x["n"] and ncp and m != "cp_union") else ""
                cells.append(f"{x['n']:,}{rr}" if x["n"] else x.get("note", "–"))
            L.append(f"| {100 * d:.1f} points | {100 * q:.0f} % | " + " | ".join(cells) + " |")
    b, c1 = R["best_valid_bound"], R["condition_1"]
    L += ["", f"**Best valid bound (section 5):** {_ML.get(b['best'], b['best'])}"
          + (f", mean n at a 1-point loss {b['mean_n_at_1_point']:,.0f}" if b.get("best") else f" ({b.get('reason')})") + ".", "",
          f"**Condition 1** ({c1.get('rule', '')}): **{'holds' if c1['holds'] else 'does not hold'}**.", ""]
    for q, v in (c1.get("per_q") or {}).items():
        L.append(f"* q = {100 * float(q):.0f} %: best {v['n_best']}, CP union {v['n_cp_union']}, ratio "
                 f"{_ratio(v['ratio'])}: {'yes' if v['at_least_25_percent_fewer'] else 'no'}")
    return "\n".join(L) + "\n"


def render_real(R: dict, sim: dict) -> str:
    M_ = lambda v: f"{v / 1e6:.1f} M"  # noqa: E731
    methods = R["methods"]
    L = ["# Bound development: the scored confirmation data", "", f"**{LABEL}.**", "",
         f"Plan: `{R['plan']}`. IUT and fixed sequence unchanged; only the per-category decision changes. 3-point limit, full "
         f"confirmation sets. Commit `{R['commit'][:7]}` (clean tree: {R['clean_tree']}).", "",
         "## Certified checkpoints (parameters removed) per run", "",
         "| model | run | frozen | " + " | ".join(_ML[m] for m in methods) + " |", "|---|---|---|" + "---|" * len(methods)]
    for x in R["rows"]:
        L.append(f"| {x['label']} | {x['run_id']} | {x['frozen_checkpoints']} | " + " | ".join(
            f"{x['certified'][m]['certified_checkpoints']} ({M_(x['certified'][m]['parameters_removed'])})" for m in methods) + " |")
    L += ["", "## The first checkpoint past the CP claim (per category)", ""]
    for x in R["rows"]:
        p = x["first_checkpoint_past_cp_claim"]
        if not p:
            L.append(f"* {x['run_id']} ({x['model']}): none (the whole frozen order is certified by the CP union bound)")
            continue
        cells = []
        for c in CATS:
            est = p["cp_union"]["estimate"][c]
            parts = [f"est {100 * est:+.2f}", f"CP UB {100 * p['cp_union']['categories'][c]['upper']:+.2f}"]
            if "exact_score" in p:
                e = p["exact_score"]["categories"][c]
                parts.append(f"Z {e['z']:.2f} vs {e['z_critical']:.3f}")
            if "betting" in p:
                parts.append(f"betting {'yes' if p['betting']['categories'][c]['certify'] else 'no'}")
            parts.append(f"normal UB {100 * p['normal']['categories'][c]['upper']:+.2f}")
            cells.append(f"{c}: " + ", ".join(parts))
        L.append(f"* {x['run_id']} ({x['model']}), checkpoint {p['position']}: " + "; ".join(cells))
    c2, c1, v = R["condition_2"], R["condition_1"], R["verdict"]
    L += ["", "## Go/no-go for paper B (plan section 7)", "",
          f"Best valid bound (chosen from the simulation alone): **{_ML.get(v['best_valid_bound'], v['best_valid_bound'])}**.", "",
          f"**Condition 1 (simulation):** {'holds' if c1['holds'] else 'does not hold'}. " + "; ".join(
              f"q {100 * float(q):.0f} %: {w['n_best']} vs {w['n_cp_union']} items ({_ratio(w['ratio'])}x)"
              for q, w in (c1.get('per_q') or {}).items()), "",
          f"**Condition 2 (real data, arm (a) orders):** {'holds' if c2['holds'] else 'does not hold'}: one more certified "
          f"checkpoint in {c2['pairs_with_one_more']} of {c2['of']} pairs (needed 4; ceiling {c2['ceiling']}).", "",
          "| model | seed | CP union | best | gain |", "|---|---|---|---|---|"]
    for p in c2["pairs"]:
        L.append(f"| {p['model']} | {p['seed']} | {p['cp_union']} | {p['best']} | {p['gain']} |")
    L += ["", f"**Verdict: B {'PROCEEDS' if v['B_proceeds'] else 'DOES NOT PROCEED'}.**", ""]
    return "\n".join(L) + "\n"
