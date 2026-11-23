"""Choose the live capability gate's settings on development data (``capprune gate-design``).

1. Profiles.  For each candidate in the development cache (MMLU validation answers), the
   per-category rates of broken (dense right, candidate wrong) and fixed answers.
   Shifted versions keep each category's churn (broken + fixed) but set the worst category's
   net loss to a chosen value (e.g. 2 points = passes a 3-point limit, 4 points = fails it);
   other categories keep min(measured loss, that value).
2. Simulation.  Each profile becomes a synthetic pool with the real pruning-time pool sizes
   (counts only, no pruning-time answers); the gate runs on ``orders`` random question orders
   per option.  Round-robin allocation is used because an ACCEPT, and therefore the acceptance
   rate, is identical under every allocation policy (fixed looks and question orders).
3. Locking rule (see configs/gate_dev.yaml) picks tolerance, budget and spending.
4. Separately reported: the locked gate replayed on the earlier pilot's pruning-time cache
   (how often it accepts the candidate that truly passes), and an exploratory 5-point run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cache import ModelRecord, ReplayCache
from .config import cfg_path
from .evaluator import ACCEPT, INCONCLUSIVE, REJECT, Settings, evaluate_candidate, make_plan
from .policies import make_policy
from .replay import run_replay
from .scorer import reference


def dev_profiles(cache: ReplayCache, domains: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    dense = cache.dense.correct
    out = {}
    for c in cache.candidates:
        prof = {}
        for d in domains:
            ids = cache.pool[d]
            kp = sum(1 for i in ids if dense[i] == 1 and c.correct[i] == 0)
            km = sum(1 for i in ids if dense[i] == 0 and c.correct[i] == 1)
            prof[d] = (kp / len(ids), km / len(ids))
        out[c.name] = prof
    return out


def shifted(prof: dict[str, tuple[float, float]], worst_loss: float) -> dict[str, tuple[float, float]]:
    worst = max(prof, key=lambda d: prof[d][0] - prof[d][1])
    out = {}
    for d, (pb, pf) in prof.items():
        churn = pb + pf
        loss = worst_loss if d == worst else min(pb - pf, worst_loss)
        if churn >= abs(loss):      # same churn, new split between broken and fixed
            out[d] = ((churn + loss) / 2, (churn - loss) / 2)
        else:                       # too little churn to carry that loss: every changed answer is broken (or fixed)
            out[d] = (max(loss, 0.0), max(-loss, 0.0))
    return out


def synthetic_cache(prof: dict[str, tuple[float, float]], sizes: dict[str, int]) -> tuple[ReplayCache, dict]:
    items, dense, cand, truth = [], {}, {}, {}
    for d, N in sizes.items():
        kp, km = int(round(prof[d][0] * N)), int(round(prof[d][1] * N))
        truth[d] = (kp - km) / N
        for i in range(N):
            iid = f"{d}/{i:05d}"
            items.append({"item_id": iid, "domain": d, "split": "pool", "tokens": None})
            dense[iid], cand[iid] = (1, 0) if i < kp else (0, 1) if i < kp + km else (1, 1)
    return ReplayCache(items, ModelRecord("dense", "dense", -1, dense, None, None, {}),
                       [ModelRecord("c", "candidate", 0, cand, None, None, {})]), truth


def simulate(prof, sizes, settings: Settings, orders: int) -> dict:
    from .cache import ReplaySource

    cache, truth = synthetic_cache(prof, sizes)
    plan = make_plan(settings, sizes)
    dec, qs = [], []
    for seed in range(orders):
        run = evaluate_candidate(ReplaySource(cache, "c", seed, settings.budget), make_policy("round_robin"), plan, "c")
        dec.append(run.decision)
        qs.append(run.questions)
    return {"accept": dec.count(ACCEPT) / orders, "reject": dec.count(REJECT) / orders,
            "inconclusive": dec.count(INCONCLUSIVE) / orders, "mean_questions": float(np.mean(qs)),
            "worst_true_loss": max(truth.values()), "qualifies": all(v <= settings.tolerance[d] + 1e-9
                                                                      for d, v in truth.items())}


def _settings(domains, tol, budget, spending, d: dict) -> Settings:
    sp, rho = ("power", 2.0) if spending == "power2" else (spending, 1.0)
    return Settings.from_dict({"domains": domains, "tolerance": tol, "min_sample": d["min_sample"],
                               "batch_size": d["batch_size"], "budget": budget,
                               "alpha_false_accept": d["alpha_false_accept"],
                               "alpha_false_reject": d["alpha_false_reject"], "spending": sp, "spending_rho": rho})


def design(acfg: dict) -> Path:
    from ..config import data_dir, load_config
    from ..data.prepare import PreparedData
    from ..model import tokenizer_id

    d = acfg["design"]
    domains = list(acfg["evaluation"]["domains"])
    if d.get("pool_sizes"):
        sizes = {dm: int(d["pool_sizes"][dm]) for dm in domains}
    else:  # pruning-time pool sizes of the live experiment (item counts only, no answers)
        live = load_config(cfg_path(acfg, d["live_config"]))
        field = live["gate"]["capability"].get("domain_field", "category")
        probe = PreparedData(data_dir(live), tokenizer_id(live)).mmlu_probe
        sizes = {dm: sum(1 for r in probe if r[field] == dm) for dm in domains}
    dev = ReplayCache.load(cfg_path(acfg, acfg["cache_dir"]))
    profs = dev_profiles(dev, domains)
    tol = float(d["tolerance"])
    shifts = [float(x) for x in d["shifted_worst_losses"]]
    pass_loss, fail_loss = min(shifts), max(shifts)
    profiles = {f"{n} (dev, as measured)": p for n, p in profs.items()}
    for n, p in profs.items():
        for s in shifts:
            profiles[f"{n} shifted to {100 * s:.0f} pts"] = shifted(p, s)
    opts = [(b, sp) for b in d["options"]["budget"] for sp in d["options"]["spending"]]

    def table(tolerance):
        res = {}
        for b, sp in opts:
            st = _settings(domains, tolerance, int(b), sp, d)
            res[f"{b}|{sp}"] = {name: simulate(p, sizes, st, int(d["orders"])) for name, p in profiles.items()}
        return res

    main = table(tol)
    pass_names = [n for n in profiles if n.endswith(f"shifted to {100 * pass_loss:.0f} pts")]
    fail_names = [n for n in profiles if n.endswith(f"shifted to {100 * fail_loss:.0f} pts")]

    def min_acc(key):
        return min(main[key][n]["accept"] for n in pass_names)

    # ---- locking rule
    full = f"{max(int(b) for b, _ in opts)}|uniform"
    tolerance_ok = min_acc(full) >= 0.5
    chosen_b = next((int(b) for b in sorted({int(b) for b, _ in opts}) if min_acc(f"{b}|uniform") >= 0.8),
                    max(int(b) for b, _ in opts))
    spending = "uniform"
    if f"{chosen_b}|power2" in main and min_acc(f"{chosen_b}|power2") >= min_acc(f"{chosen_b}|uniform") + 0.10:
        spending = "power2"
    locked = {"tolerance": tol if tolerance_ok else None, "budget": chosen_b,
              "spending": "power" if spending == "power2" else "uniform",
              **({"spending_rho": 2.0} if spending == "power2" else {}),
              "min_sample": d["min_sample"], "batch_size": d["batch_size"],
              "alpha_false_accept": d["alpha_false_accept"], "alpha_false_reject": d["alpha_false_reject"]}

    # ---- separately reported: validation on the pilot's pruning-time replay cache, exploratory 5 points
    validation = None
    if d.get("validation_cache") and cfg_path(acfg, d["validation_cache"]).exists():
        vc = ReplayCache.load(cfg_path(acfg, d["validation_cache"]))
        st = _settings(domains, tol, chosen_b, spending, d)
        ref = reference(vc, st)
        acc = {c.name: 0 for c in vc.candidates}
        n_orders = 50
        for seed in range(n_orders):
            for r in run_replay(vc, Settings.from_dict({**st.to_dict(), "seed": seed}), ["threshold_aware"]):
                acc[r["candidate"]] += r["decision"] == ACCEPT
        validation = {c: {"accept_rate": acc[c] / n_orders, "truly_qualifies": ref[c]["qualifies"],
                          "worst_pool_loss": ref[c]["max_loss"]} for c in acc}
    exploratory = table(0.05)

    out = cfg_path(acfg, acfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pool_sizes": sizes, "dev_pool_sizes": dev.pool_sizes, "profiles": profiles, "tolerance": tol,
               "results": main, "locked": locked, "tolerance_feasible": tolerance_ok,
               "validation_on_pilot_cache": validation, "exploratory_tolerance_0.05": exploratory,
               "pass_profiles": pass_names, "fail_profiles": fail_names}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(render(summary, opts, profs), encoding="utf-8")
    return out / "summary.md"


def render(S: dict, opts, profs) -> str:
    L = ["# Gate design on development data (MMLU validation)", "",
         f"Pruning-time pool sizes used in simulation: {S['pool_sizes']}; development pool sizes: {S['dev_pool_sizes']}.", "",
         "## Development profiles (per-category broken / fixed answer rates, %)", "",
         "| candidate | " + " | ".join(S["pool_sizes"]) + " | worst net loss (pts) |", "|---" * (len(S["pool_sizes"]) + 2) + "|"]
    for n, p in profs.items():
        L.append(f"| {n} | " + " | ".join(f"{100 * p[d][0]:.1f} / {100 * p[d][1]:.1f}" for d in S["pool_sizes"])
                 + f" | {100 * max(p[d][0] - p[d][1] for d in p):+.1f} |")
    L += ["", f"## Simulated gate at tolerance {100 * S['tolerance']:.0f} pts (acceptance rate / mean questions)", "",
          "| profile | true worst loss | " + " | ".join(f"budget {b}, {sp}" for b, sp in opts) + " |",
          "|---|---|" + "---|" * len(opts)]
    for name in S["profiles"]:
        first = S["results"][f"{opts[0][0]}|{opts[0][1]}"][name]
        cells = [f"{100 * S['results'][f'{b}|{sp}'][name]['accept']:.0f}% / {S['results'][f'{b}|{sp}'][name]['mean_questions']:.0f}"
                 for b, sp in opts]
        L.append(f"| {name} | {100 * first['worst_true_loss']:+.1f} | " + " | ".join(cells) + " |")
    lk = S["locked"]
    L += ["", "## Locked settings", "", "```yaml", *[f"{k}: {v}" for k, v in lk.items()], "```", "",
          f"Tolerance feasible at the full-pool budget: {S['tolerance_feasible']}. Rule: smallest budget with >= 80 % "
          f"acceptance for every pass profile ({', '.join(S['pass_profiles'])}); power spending only if it adds >= 10 points.", ""]
    v = S.get("validation_on_pilot_cache")
    if v:
        L += ["## Separately reported: locked gate on the pilot's pruning-time replay cache (50 question orders)", "",
              "| candidate | truly qualifies (pool) | worst pool loss | acceptance rate |", "|---|---|---|---|"]
        for c, r in v.items():
            L.append(f"| {c} | {'yes' if r['truly_qualifies'] else 'no'} | {100 * r['worst_pool_loss']:+.1f} | {100 * r['accept_rate']:.0f}% |")
        L.append("")
    L += ["## Exploratory (not used for locking): tolerance 5 pts", "",
          "| profile | " + " | ".join(f"budget {b}, {sp}" for b, sp in opts) + " |", "|---|" + "---|" * len(opts)]
    for name in S["profiles"]:
        L.append(f"| {name} | " + " | ".join(
            f"{100 * S['exploratory_tolerance_0.05'][f'{b}|{sp}'][name]['accept']:.0f}%" for b, sp in opts) + " |")
    return "\n".join(L) + "\n"
