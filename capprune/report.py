"""Aggregate finished runs into the metrics the proposal commits to.

Per run (all on the held-out final splits, paired against the dense model on
identical questions):
  achieved sparsity; per-domain MMLU accuracy loss and the *largest* domain loss
  (with a paired bootstrap 95% CI over questions); overall MMLU; MMLU by the 4
  official categories; FreebaseQA exact match; WikiText-2 perplexity;
  prefill / decode latency and peak GPU memory; total GPU time (and time lost
  to failures); probe questions used; controller calls; rollbacks; test early
  stops; search stop reason.

Per (method, target): mean and 95% t-interval across seeds.
Paired comparisons: each method vs. a reference method (default ``paper_agent``)
on the same seed and target - difference in largest-domain loss etc.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .config import runs_dir
from .utils.io import atomic_write_text, read_json

# two-sided 95% Student-t quantiles, df = 1..30
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = float(np.mean(xs))
    if len(xs) < 2:
        return m, float("nan")
    t = _T975[min(len(xs) - 1, 30) - 1]
    return m, t * float(np.std(xs, ddof=1)) / math.sqrt(len(xs))


def _domain_losses(rows, dense_pq, run_pq):
    by: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        q = r["qid"]
        by.setdefault(r["domain"], []).append((int(dense_pq[q]), int(run_pq[q])))
    return {d: np.array(v) for d, v in by.items()}


def worst_domain_bootstrap(arrs: dict[str, np.ndarray], B: int, seed: int = 0) -> tuple[float, float, float, str]:
    losses = {d: float(a[:, 0].mean() - a[:, 1].mean()) for d, a in arrs.items()}
    worst_d = max(losses, key=losses.get)
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        mx = -1.0
        for a in arrs.values():
            idx = rng.integers(0, len(a), len(a))
            s = a[idx]
            mx = max(mx, float(s[:, 0].mean() - s[:, 1].mean()))
        draws[b] = mx
    return losses[worst_d], float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), worst_d


def collect(cfg: dict, bootstrap: int = 1000) -> tuple[list[dict], dict]:
    from .data.prepare import _read_jsonl
    from .config import data_dir

    root = runs_dir(cfg)
    dense_dir = root / "dense" / "final"
    if not (dense_dir / "mmlu.json").exists():
        raise SystemExit("dense run not finished yet (runs/dense/final/mmlu.json missing) - run the 'dense' method first")
    split = cfg.get("final_eval", {}).get("split", "final")
    rows = _read_jsonl(data_dir(cfg) / ("mmlu_dev.jsonl" if split == "dev" else "mmlu_final.jsonl"))
    dense_mmlu = read_json(dense_dir / "mmlu.json")
    dense_fqa = read_json(dense_dir / "freebaseqa.json") if (dense_dir / "freebaseqa.json").exists() else None
    dense_sum = read_json(dense_dir / "summary.json") if (dense_dir / "summary.json").exists() else {}
    out = []
    for d in sorted(root.iterdir()):
        sp = d / "final" / "summary.json"
        if d.name.startswith("_") or not sp.exists():
            continue
        s = read_json(sp)
        mm = read_json(d / "final" / "mmlu.json")
        arrs = _domain_losses(rows, dense_mmlu["per_question"], mm["per_question"])
        worst, lo, hi, wd = worst_domain_bootstrap(arrs, bootstrap)
        rec = {
            "run_id": s["run_id"], "method": s["method"], "target": s["target"], "seed": s["seed"], "mode": s["mode"],
            "achieved_sparsity": s["achieved_sparsity"],
            "worst_domain_loss": worst, "worst_domain_loss_lo": lo, "worst_domain_loss_hi": hi, "worst_domain": wd,
            "mmlu": mm["overall"], "mmlu_loss": dense_mmlu["overall"] - mm["overall"],
            **{f"loss[{k}]": float(a[:, 0].mean() - a[:, 1].mean()) for k, a in sorted(arrs.items())},
            **{f"cat[{k}]": v["acc"] for k, v in mm["by_category"].items()},
        }
        if s.get("freebaseqa"):
            rec["freebaseqa_em"] = s["freebaseqa"]["exact_match"]
            if dense_fqa and dense_fqa["exact_match"] > 0:
                rec["freebaseqa_retention"] = s["freebaseqa"]["exact_match"] / dense_fqa["exact_match"]
        if s.get("wikitext2"):
            rec["wikitext2_ppl"] = s["wikitext2"]["perplexity"]
        b = s.get("benchmark") or {}
        for L, r in (b.get("by_prompt_len") or {}).items():
            rec[f"prefill_ms@{L}"] = r["prefill_ms_median"]
            rec[f"decode_ms_tok@{L}"] = r["decode_ms_per_token_median"]
            if "peak_allocated_bytes" in r:
                rec[f"peak_mem_gib@{L}"] = r["peak_allocated_bytes"] / (1 << 30)
        if b.get("materialize", {}).get("nm_backend"):
            rec["nm_backend"] = b["materialize"]["nm_backend"]
        rec["gpu_hours"] = s["gpu_seconds"] / 3600
        rec["gpu_hours_lost"] = s.get("gpu_seconds_lost_to_failures", 0) / 3600
        rec["attempts"] = s.get("attempts", 1)
        bud = s.get("budget") or {}
        for k in ("probe_questions", "controller_calls", "early_stops", "heuristic_fallbacks"):
            if k in bud:
                rec[k] = bud[k]
        rec["iterations"] = s.get("iterations")
        rec["rollbacks"] = s.get("rollbacks")
        rec["stop_reason"] = s.get("stop_reason")
        out.append(rec)
    return out, {"dense_mmlu": dense_mmlu["overall"], "dense_fqa": dense_fqa["exact_match"] if dense_fqa else None,
                 "dense_summary": dense_sum}


def write_report(cfg: dict, reference: str = "paper_agent", bootstrap: int = 1000) -> Path:
    runs, dense = collect(cfg, bootstrap)
    root = runs_dir(cfg) / "_report"
    root.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in runs for k in r}, key=lambda k: (k not in ("run_id", "method", "target", "seed"), k))
    with open(root / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in runs:
            w.writerow(r)
    lens = sorted({int(k.split("@")[1]) for r in runs for k in r if k.startswith("prefill_ms@")})
    hw = [f"{m}@{L}" for L in lens for m in ("prefill_ms", "decode_ms_tok", "peak_mem_gib")]
    metrics = ["achieved_sparsity", "worst_domain_loss", "mmlu", "mmlu_loss", "freebaseqa_em", "wikitext2_ppl",
               *hw, "gpu_hours", "probe_questions", "controller_calls", "rollbacks", "early_stops"]
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["method"], r["target"]), []).append(r)
    split = cfg.get("final_eval", {}).get("split", "final")
    lines = [f"# CAP-Prune results ({'DEVELOPMENT split - for tuning only' if split == 'dev' else 'held-out final split'})", "",
             f"Dense model: MMLU {dense['dense_mmlu']:.4f}" + (f", FreebaseQA EM {dense['dense_fqa']:.4f}" if dense["dense_fqa"] is not None else ""),
             "", "Mean ± 95% t-interval across seeds (n = number of seeds).", ""]
    hdr = ["method", "target", "n"] + metrics
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    agg_rows = []
    for (m, t), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {"method": m, "target": t, "n": len(rs)}
        cells = [m, f"{t:.2f}", str(len(rs))]
        for k in metrics:
            mu, ci = mean_ci([r.get(k) for r in rs])
            row[k], row[k + "_ci95"] = mu, ci
            cells.append("—" if mu != mu else (f"{mu:.4f}" if ci != ci else f"{mu:.4f} ± {ci:.4f}"))
        agg_rows.append(row)
        lines.append("| " + " | ".join(cells) + " |")
    with open(root / "methods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0]) if agg_rows else ["method"])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    # paired comparisons vs reference
    by_key = {(r["method"], r["target"], r["seed"]): r for r in runs}
    comp = ["", f"## Paired differences vs `{reference}` (same target and seed; negative worst-domain loss = better)", "",
            "| method | target | seeds | Δ worst-domain loss | Δ MMLU | Δ sparsity |", "|---|---|---|---|---|---|"]
    for (m, t) in sorted(groups, key=lambda x: (x[1], x[0])):
        if m == reference:
            continue
        diffs = [(by_key[(m, t, s)], by_key[(reference, t, s)]) for (mm, tt, s) in by_key
                 if mm == m and tt == t and (reference, t, s) in by_key]
        if not diffs:
            continue
        cell = []
        for k in ("worst_domain_loss", "mmlu", "achieved_sparsity"):
            mu, ci = mean_ci([a[k] - b[k] for a, b in diffs])
            cell.append(f"{mu:+.4f}" + ("" if ci != ci else f" ± {ci:.4f}"))
        comp.append(f"| {m} | {t:.2f} | {len(diffs)} | " + " | ".join(cell) + " |")
    lines += comp
    lines += ["", "Per-run details (including per-domain losses, bootstrap CIs, nm kernel status): `runs.csv`."]
    atomic_write_text(root / "report.md", "\n".join(lines) + "\n")
    return root / "report.md"
