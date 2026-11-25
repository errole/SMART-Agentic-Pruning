"""EXPLORATORY analyses of the confirmatory study's already-scored outcomes (``capprune exploratory``). CPU only.

Nothing here changes a recorded confirmatory result. Everything reads the per-batch files written by ``confirmatory
confirm`` / ``report`` (``*.confirm.json``, ``*.order.json``, ``*.outcomes.npz``, ``report.json``), each model's
``registration.json`` and the runs' journals, and writes to ``~/cap-prune-runs/exploratory/``:

* ``frontier.json``: certified parameters removed under the pre-registered procedure (Clopper-Pearson union bound,
  per-category IUT, alpha = 0.05, fixed sequence over the frozen order) at tolerances 2-5 points and confirmation
  sizes 500 / 1,000 / 2,000 / full per category (the amendment-3 S4 draws: 50 seeded, category-stratified);
* ``loss_curves.json``: the per-category paired loss with one-sided 95 % lower and upper bounds at every checkpoint of
  every run, with each arm's screen stop and confirmation stop;
* ``c1_mechanism.json``: each category's pool-to-confirmation gap next to its gate-pool size and the gate decisions
  made on that pool (2 models x 4 categories: descriptive only).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .confirmatory import CATS, _git, counts_from_outcomes, load_outcomes, s4_draws, sequence_claim, summarize
from .population_regate import population_bounds
from .utils.io import atomic_write_json, atomic_write_text, read_json

LABEL = "EXPLORATORY (post hoc, on the already-scored confirmation outcomes; not part of the confirmatory claims)"
ARMS = ("ppl_top1", "cap_top1", "cap_klprobe")
ARM_LABEL = {"ppl_top1": "(a) perplexity", "cap_top1": "(b) gate", "cap_klprobe": "(c) gate + KL"}
TOLS = (0.02, 0.03, 0.04, 0.05)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_models(cfg: dict) -> list[dict]:
    """Every model of ``confirmatory.study.models`` with its runs' records and verified outcomes (read-only)."""
    out = []
    for m in cfg["confirmatory"]["study"]["models"]:
        od = resolve_path(cfg, m["out_dir"])
        reg = read_json(od / "registration.json")
        runs = []
        for b in m["batches"]:
            bdir = od / "batches" / b
            rep = {r["run_id"]: r for r in read_json(bdir / "report.json")["runs"]}
            for rec in read_json(bdir / "confirm.json")["runs"]:
                f = bdir / rec["outcomes_file"]
                if _sha(f) != rec["outcomes_sha256"]:
                    raise SystemExit(f"{f}: SHA-256 does not match the confirmation record")
                runs.append({"run_id": rec["run_id"], "method": rec["method"], "seed": int(rec["seed"]), "batch": b,
                             "confirm": rec, "report": rep[rec["run_id"]], "outcomes_path": str(f),
                             "order": read_json(bdir / f"{rec['run_id']}.order.json"),
                             "journal": od / "runs" / rec["run_id"] / "journal.jsonl", "oc": load_outcomes(f)})
        model_key = read_json(od / "batches" / m["batches"][0] / "report.json")["model_key"]
        out.append({"label": m["label"], "model_key": model_key, "out_dir": str(od), "registration": reg, "runs": runs})
    return out


# ============================================================================ 1. certification frontier
def frontier(models: list[dict], sec: dict, tols=TOLS, alpha: float = 0.05) -> dict:
    sizes = list(sec["s4_sizes"])
    res = {"label": LABEL, "alpha": alpha, "tolerances": list(tols), "sizes": sizes + ["full"],
           "draws": {"per_size": sec["s4_draws"], "seed": sec["s4_seed"],
                     "rule": "the amendment-3 S4 draws: draw d of size n takes n items per category without replacement "
                             "from the qid-sorted items, default_rng([seed, n, d, crc32(category)])"},
           "models": []}
    for M in models:
        oc0 = M["runs"][0]["oc"]
        draws = s4_draws(oc0["qids"], oc0["category"], sizes, sec["s4_draws"], sec["s4_seed"])
        per_run = []
        for r in M["runs"]:
            oc = r["oc"]
            if list(oc["qids"]) != list(oc0["qids"]):
                raise SystemExit(f"{r['run_id']}: item order differs from the model's first run")
            cell = {}
            for t in tols:
                row = {}
                for n in sizes:
                    vals = [sequence_claim(oc, alpha, t, idx)[1] for idx in draws[n]]
                    row[str(n)] = {**summarize(vals), "values": vals}
                full = sequence_claim(oc, alpha, t)
                row["full"] = {**summarize([full[1]]), "values": [full[1]], "position": full[0]}
                cell[f"{t:.2f}"] = row
            per_run.append({"run_id": r["run_id"], "method": r["method"], "seed": r["seed"], "batch": r["batch"],
                            "frozen_checkpoints": int(len(oc["position"])), "by_tolerance": cell})
        agg = {}
        for arm in ARMS:
            xs = [x for x in per_run if x["method"] == arm]
            agg[arm] = {f"{t:.2f}": {str(n): summarize([v for x in xs for v in x["by_tolerance"][f"{t:.2f}"][str(n)]["values"]])
                                     for n in sizes + ["full"]} for t in tols}
        res["models"].append({"label": M["label"], "model_key": M["model_key"],
                              "dense_parameters": int(M["runs"][0]["confirm"]["dense_parameters"]),
                              "per_category_full": {c: int(np.sum(oc0["category"] == c)) for c in CATS},
                              "runs": per_run, "by_arm": agg})
    return res


def check_frontier(F: dict, models: list[dict]) -> list[str]:
    """The 3-point full-n column must reproduce every recorded claim, the 3-point subsample columns S4, and the 5-point
    full column S3 (amendment 3)."""
    errs = []
    for FM, M in zip(F["models"], models):
        rec = {r["run_id"]: r for r in M["runs"]}
        for x in FM["runs"]:
            r = rec[x["run_id"]]
            full3 = x["by_tolerance"]["0.03"]["full"]
            if full3["values"][0] != r["confirm"]["confirmed_parameters_removed"] or full3["position"] != r["confirm"]["claim_position"]:
                errs.append(f"{x['run_id']}: 3-point full claim differs from the recorded claim")
            sec = r["report"]["secondary"]
            if x["by_tolerance"]["0.05"]["full"]["values"][0] != sec["S3"]["parameters_removed"]:
                errs.append(f"{x['run_id']}: 5-point full claim differs from S3")
            for n, v in sec["S4"].items():
                if x["by_tolerance"]["0.03"][n]["values"] != v["values"]:
                    errs.append(f"{x['run_id']}: 3-point n={n} draws differ from S4")
    return errs


# ============================================================================ 2. loss curves
def loss_curves(models: list[dict], alpha: float = 0.05) -> dict:
    """Per run, checkpoint and category: n, lost, gained, the estimate and one-sided (1 - alpha) lower / upper bounds (the
    upper one is the certification bound), plus the run's screen stop and confirmation stop."""
    res = {"label": LABEL, "alpha": alpha, "interval": "one-sided 95 % lower and upper bounds (CP union); the upper one is "
           "the pre-registered certification bound", "models": []}
    for M in models:
        runs = []
        for r in M["runs"]:
            oc, rec = r["oc"], r["confirm"]
            pts = []
            for i in range(len(oc["position"])):
                cnt = counts_from_outcomes(oc["dense"][i], oc["cand"][i], oc["category"])
                cats = {}
                for c, (n, kp, km) in cnt.items():
                    lb, ub = population_bounds(n, kp, km, alpha, alpha)
                    cats[c] = {"n": n, "lost": kp, "gained": km, "estimate": (kp - km) / n, "lower": lb, "upper": ub}
                pts.append({"position": int(oc["position"][i]), "params_removed": int(oc["materialized_parameters_removed"][i]),
                            "categories": cats})
            claim = rec["claim_position"]
            n_tested = len(rec["tested"])
            first_fail = rec["tested"][-1]["position"] if n_tested and not rec["tested"][-1]["pass"] else None
            runs.append({"run_id": r["run_id"], "method": r["method"], "seed": r["seed"], "batch": r["batch"],
                         "stop_reason": r["report"]["stop_reason"], "screen_stop_position": int(oc["position"][-1]),
                         "confirmation_claim_position": claim, "confirmation_first_failure": first_fail,
                         "checkpoints": pts})
        res["models"].append({"label": M["label"], "model_key": M["model_key"],
                              "step_parameters": int(runs[0]["checkpoints"][0]["params_removed"]), "runs": runs})
    return res


# ============================================================================ 3. C1 mechanism
def _gate_decisions(journal: Path) -> list[dict]:
    out = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        g = j.get("gate")
        if not g or not g.get("decision"):
            continue
        out.append({"decision": g["decision"], "order_seed": g.get("order_seed"),
                    "n": {c: int(d["n"]) for c, d in g["domains"].items()},
                    "status": {c: d.get("status") for c, d in g["domains"].items()}})
    return out


def c1_mechanism(models: list[dict]) -> dict:
    """Per model and category: the C1 gap (confirmation - gate pool estimate, tested checkpoints of arms (b) and (c),
    counted per run as in the pre-registered C1), the gate-pool size, and the gate decisions made on that pool."""
    res = {"label": LABEL, "caveat": "2 models x 4 categories = 8 points: far too few for inference; descriptive only.",
           "models": []}
    for M in models:
        pool = M["registration"]["sets"]["gate"]["per_category"]
        gaps = {c: [] for c in CATS}
        gaps_all = {c: [] for c in CATS}
        decisions = []
        per_run = []
        for r in M["runs"]:
            if r["method"] not in ("cap_top1", "cap_klprobe"):
                continue
            rep = r["report"]
            for i, x in enumerate(rep["scored"]):
                if not x.get("pool_loss"):
                    continue
                for c in CATS:
                    g = x["estimate"][c] - x["pool_loss"][c]
                    gaps_all[c].append(g)
                    if i < rep["n_tested"]:
                        gaps[c].append(g)
            ds = _gate_decisions(r["journal"])
            decisions += ds
            per_run.append({"run_id": r["run_id"], "decisions": len(ds),
                            "order_seeds": sorted({str(d["order_seed"]) for d in ds})})
        cats = {}
        for c in CATS:
            ns = [d["n"][c] for d in decisions]
            cats[c] = {"gate_pool": int(pool[c]), "mean_gap_tested": float(np.mean(gaps[c])) if gaps[c] else None,
                       "max_gap_tested": float(np.max(gaps[c])) if gaps[c] else None,
                       "mean_gap_all_scored": float(np.mean(gaps_all[c])) if gaps_all[c] else None,
                       "tested_checkpoints": len(gaps[c]), "gate_decisions": len(ns),
                       "mean_items_per_decision": float(np.mean(ns)) if ns else None,
                       "share_of_decisions_using_whole_pool": float(np.mean([v >= pool[c] for v in ns])) if ns else None,
                       "item_uses_per_pool_item": float(np.sum(ns) / pool[c]) if ns else None,
                       "decisions_failing_here": int(sum(1 for d in decisions if d["status"].get(c) not in (None, "pass")))}
        res["models"].append({"label": M["label"], "model_key": M["model_key"], "categories": cats, "runs": per_run,
                              "gate_decisions": len(decisions)})
    return res


# ============================================================================ the command
def render(F: dict, LC: dict, C1: dict) -> str:
    M_ = lambda v: "–" if v is None else f"{v / 1e6:.1f} M"  # noqa: E731
    pt = lambda v: "–" if v is None else f"{100 * v:+.2f}"  # noqa: E731
    L = ["# Exploratory analyses of the confirmatory outcomes", "", f"**{LABEL}.**", "",
         "## 1. Certification frontier (pre-registered procedure; certified parameters removed, median [IQR] over the 50 "
         "draws of every seed; the full set is one value per seed)", ""]
    for FM in F["models"]:
        L += [f"### {FM['label']}: {FM['model_key']}", "",
              "| arm | tolerance | " + " | ".join(f"n = {n}" for n in F["sizes"]) + " |", "|---|---|" + "---|" * len(F["sizes"])]
        for arm in ARMS:
            for t in F["tolerances"]:
                row = FM["by_arm"][arm][f"{t:.2f}"]
                L.append(f"| {ARM_LABEL[arm]} | {round(100 * t)} points | " + " | ".join(
                    f"{M_(row[str(n)]['median'])} [{M_(row[str(n)]['q25'])}, {M_(row[str(n)]['q75'])}]" for n in F["sizes"]) + " |")
        L.append("")
    L += ["## 2. Loss curves", "", f"Interval: {LC['interval']}. Figures: `docs/paper/workshop/figures/`.", ""]
    L += ["## 3. C1 mechanism", "", f"**{C1['caveat']}**", "",
          "| model | category | gate pool | mean gap, tested (points) | max gap | gate decisions | mean items per decision | "
          "share using the whole pool | item uses per pool item |", "|---|---|---|---|---|---|---|---|---|"]
    for CM in C1["models"]:
        for c, v in CM["categories"].items():
            L.append(f"| {CM['label']} | {c} | {v['gate_pool']:,} | {pt(v['mean_gap_tested'])} | {pt(v['max_gap_tested'])} | "
                     f"{v['gate_decisions']} | {v['mean_items_per_decision']:.0f} | {100 * v['share_of_decisions_using_whole_pool']:.0f} % | "
                     f"{v['item_uses_per_pool_item']:.1f} |")
    return "\n".join(L) + "\n"


def exploratory(cfg: dict) -> Path:
    from .confirmatory import _secondary_cfg

    t0 = time.time()
    sec = _secondary_cfg(cfg)
    models = load_models(cfg)
    F = frontier(models, sec)
    errs = check_frontier(F, models)
    if errs:
        raise SystemExit("frontier does not reproduce the recorded results:\n" + "\n".join(errs))
    LC = loss_curves(models)
    C1 = c1_mechanism(models)
    od = resolve_path(cfg, cfg["confirmatory"].get("exploratory", {}).get("out_dir", "~/cap-prune-runs/exploratory"))
    od.mkdir(parents=True, exist_ok=True)
    meta = {"commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "", "written_at": time.time(),
            "seconds": time.time() - t0, "sources": {M["model_key"]: [r["outcomes_path"] for r in M["runs"]] for M in models}}
    for name, obj in (("frontier", F), ("loss_curves", LC), ("c1_mechanism", C1)):
        atomic_write_json(od / f"{name}.json", {**obj, "meta": meta})
    atomic_write_text(od / "exploratory.md", render(F, LC, C1))
    return od / "exploratory.md"
