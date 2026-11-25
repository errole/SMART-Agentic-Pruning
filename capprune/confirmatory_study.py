"""The confirmatory study across both models (``capprune confirmatory study``; plan section 6 and amendment 3).

Reads only the per-batch outputs of ``confirmatory confirm`` and ``confirmatory report`` (report.json, confirm.json,
the order files and each model's registration) and writes ``study.json`` and ``study.md``:

* the primary endpoint per arm, paired across seeds within each model (the seed is the unit);
* C1, the pool-to-confirmation gap per category (and the section 6 bootstrap criterion);
* C4, arm (a)'s per-category loss at its screened endpoint (descriptive) and the section 6 criterion;
* C6, KL's pick vs gradient rank 2 over all rejections (sign test), TV and S1c vs KL as secondary;
* S1-S4 (added after the pilot), on all seeds and excluding model 1 seed 0;
* wall time and cost per run.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .confirmatory import CATS, _git, _secondary_cfg, summarize
from .utils.io import atomic_write_json, atomic_write_text, read_json

ARMS = ("ppl_top1", "cap_top1", "cap_klprobe")
ARM_LABEL = {"ppl_top1": "(a) perplexity", "cap_top1": "(b) gate", "cap_klprobe": "(c) gate + KL"}


def sign_test(better: int, worse: int) -> float:
    from .matched_signals import sign_test as st

    return st(better, worse)


def bootstrap_mean_ci(vals, resamples: int, seed: int, level: float = 0.95) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval of the mean (resampling the units with replacement)."""
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(int(resamples), len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired(by_seed: dict[int, dict[str, float | None]], a: str, b: str) -> dict:
    """Per-seed differences b - a (seeds where both are defined), with the mean, range and whether the sign is the same
    (and non-zero) in every seed."""
    d = {s: v[b] - v[a] for s, v in sorted(by_seed.items()) if v.get(a) is not None and v.get(b) is not None}
    vals = list(d.values())
    return {"per_seed": d, "mean": float(np.mean(vals)) if vals else None, "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "consistent_sign": (all(x > 0 for x in vals) or all(x < 0 for x in vals)) if vals else False,
            "seeds": len(vals)}


def across_seeds(by_seed: dict[int, dict[str, float | None]]) -> dict:
    """Per arm: per-seed values, mean and range (undefined values are counted, not averaged); paired differences."""
    out = {}
    for arm in ARMS:
        vals = {s: v.get(arm) for s, v in sorted(by_seed.items()) if arm in v}
        defined = [x for x in vals.values() if x is not None]
        out[arm] = {"per_seed": vals, "mean": float(np.mean(defined)) if defined else None,
                    "min": min(defined) if defined else None, "max": max(defined) if defined else None,
                    "undefined": sum(1 for x in vals.values() if x is None)}
    out["b_minus_a"] = paired(by_seed, "ppl_top1", "cap_top1")
    out["c_minus_b"] = paired(by_seed, "cap_top1", "cap_klprobe")
    return out


def _load_model(cfg: dict, m: dict, sec: dict) -> dict:
    od = resolve_path(cfg, m["out_dir"])
    reg = read_json(od / "registration.json")
    runs, passes, batches = [], [], []
    for b in m["batches"]:
        bdir = od / "batches" / b
        conf, rep = read_json(bdir / "confirm.json"), read_json(bdir / "report.json")
        if conf.get("amendment") != 3 or rep.get("amendment") != 3:
            raise SystemExit(f"{bdir}: not confirmed / reported under amendment 3")
        fr = read_json(bdir / "frozen.json")
        batches.append({"batch": b, "frozen_at": fr["frozen_at"], "frozen_commit": fr["commit"], "confirmed_at": conf["finished_at"],
                        "confirm_commit": conf["commit"], "confirm_clean_tree": conf["clean_tree"], "confirm_cost": conf["cost"],
                        "confirm_wall_seconds": conf.get("wall_seconds"), "post_hoc_secondary": b in sec["post_hoc_batches"]})
        passes.append(conf["cost"])
        if (bdir / "superseded" / "confirm.json").exists():  # the pilot's pre-amendment pass (its spilled first pass left no record)
            passes.append(read_json(bdir / "superseded" / "confirm.json")["cost"])
        for r in rep["runs"]:
            runs.append({**r, "batch": b, "post_hoc_secondary": b in sec["post_hoc_batches"]})
    items = sum(p["items_scored"] + p["dense_items_scored"] for p in passes)
    secs = sum(p["seconds"] + p["dense_seconds"] for p in passes)
    return {"label": m.get("label", od.name), "model_key": rep["model_key"], "out_dir": str(od), "runs": runs, "batches": batches,
            "n_confirmation": rep["confirmation"]["items"], "per_category": rep["confirmation"]["per_category"],
            "gate_pool": reg["sets"]["gate"]["per_category"], "registration_sha16": rep["registration_sha256"][:16],
            "confirm_rate": {"items": items, "seconds": secs, "seconds_per_item": secs / items if items else None},
            "admissibility": rep.get("admissibility")}


def _s2(r: dict, n_conf: int, rate: float | None) -> dict:
    """S2 (amendment 3): evaluation cost of a run and per confirmed million parameters."""
    conf_items = n_conf * (1 + int(r["n_tested"]))
    sec = r["cost"]["seconds"]
    items = int(r["cost"]["questions_charged_to_cap"]) + conf_items
    gpu = float(sec.get("gate_s", 0.0)) + float(sec.get("kl_probe_s", 0.0)) + (conf_items * rate if rate else 0.0)
    m = r["confirmed_parameters_removed"] / 1e6
    return {"confirmation_items": conf_items, "items": items, "gpu_seconds": gpu,
            "items_per_confirmed_M": items / m if m else None, "gpu_seconds_per_confirmed_M": gpu / m if m else None,
            "counterfactual_items": r["cost"]["counterfactual_items"], "counterfactual_seconds": float(sec.get("counterfactual_s", 0.0))}


def size_order(n) -> tuple:
    """S4 sizes in numeric order, the full set last."""
    return (str(n) == "full", int(n) if str(n) != "full" else 0)


def _worst(d: dict) -> float:
    return max(float(d[c]) for c in CATS)


def analyse(models: list[dict], sec: dict, exclude: dict) -> dict:
    out = {"models": [], "c6": None}
    c6_rows = []
    for M in models:
        dense_p = next((r["dense_parameters"] for r in M["runs"] if r.get("dense_parameters")), None)
        rate = M["confirm_rate"]["seconds_per_item"]

        def excluded(r):
            return M["model_key"] == exclude.get("model_key") and int(r["seed"]) == int(exclude.get("seed", -1))

        # ---------------------------------------------------------------- primary
        prim = {}
        for r in M["runs"]:
            prim.setdefault(int(r["seed"]), {})[r["method"]] = r["confirmed_parameters_removed"]
        # ---------------------------------------------------------------- secondaries per run
        per_run = []
        for r in M["runs"]:
            s = r["secondary"]
            per_run.append({"run_id": r["run_id"], "method": r["method"], "seed": int(r["seed"]), "batch": r["batch"],
                            "post_hoc": r["post_hoc_secondary"], "excluded_in_sensitivity": excluded(r),
                            "S1": s["S1"]["overshoot_parameters"], "S2": _s2(r, M["n_confirmation"], rate),
                            "S3": s["S3"]["parameters_removed"], "S3_position": s["S3"]["claim_position"],
                            "S4": {n: {k: v[k] for k in ("median", "q25", "q75", "values")} for n, v in s["S4"].items()}})

        def by_seed(key, subset):
            d = {}
            for x in subset:
                d.setdefault(x["seed"], {})[x["method"]] = key(x)
            return d

        sets = {"all seeds": per_run, f"excluding {exclude.get('label', 'the pilot')}": [x for x in per_run if not x["excluded_in_sensitivity"]]}
        secondary = {}
        for name, subset in sets.items():
            s4 = {}
            for arm in ARMS:
                xs = [x for x in subset if x["method"] == arm]
                s4[arm] = {n: summarize([v for x in xs for v in x["S4"][n]["values"]]) for n in (xs[0]["S4"] if xs else {})}
            secondary[name] = {"S1": across_seeds(by_seed(lambda x: x["S1"], subset)),
                               "S2_items_per_M": across_seeds(by_seed(lambda x: x["S2"]["items_per_confirmed_M"], subset)),
                               "S2_gpu_seconds_per_M": across_seeds(by_seed(lambda x: x["S2"]["gpu_seconds_per_confirmed_M"], subset)),
                               "S3": across_seeds(by_seed(lambda x: x["S3"], subset)), "S4": s4}
        # ---------------------------------------------------------------- C1
        c1 = {"tested": [], "all_scored": []}
        for r in M["runs"]:
            if r["method"] not in ("cap_top1", "cap_klprobe"):
                continue
            for i, x in enumerate(r["scored"]):
                if not x.get("pool_loss"):
                    continue
                g = {"run_id": r["run_id"], "method": r["method"], "seed": int(r["seed"]), "position": x["position"],
                     "tested": i < r["n_tested"], "confirmation": {c: x["estimate"][c] for c in CATS},
                     "pool": {c: x["pool_loss"][c] for c in CATS},
                     "gap": {c: x["estimate"][c] - x["pool_loss"][c] for c in CATS},
                     "worst_gap": _worst(x["estimate"]) - _worst(x["pool_loss"])}
                c1["all_scored"].append(g)
                if g["tested"]:
                    c1["tested"].append(g)
        for k in ("tested", "all_scored"):
            gs = c1[k]
            wg = [g["worst_gap"] for g in gs]
            lo, hi = bootstrap_mean_ci(wg, sec["c1_bootstrap"]["resamples"], sec["c1_bootstrap"]["seed"])
            c1[f"{k}_summary"] = {"checkpoints": len(gs), "mean_worst_gap": float(np.mean(wg)) if wg else None,
                                  "ci95": [lo, hi], "per_category_mean_gap": {c: float(np.mean([g["gap"][c] for g in gs])) if gs else None for c in CATS},
                                  "per_category_max_gap": {c: float(np.max([g["gap"][c] for g in gs])) if gs else None for c in CATS}}
        s = c1["tested_summary"]
        c1["model_criterion_met"] = bool(s["mean_worst_gap"] is not None and s["mean_worst_gap"] > 0 and s["ci95"][0] is not None and s["ci95"][0] > 0)
        c1["social_sciences_gate_pool"] = M["gate_pool"]["Social Sciences"]
        # ---------------------------------------------------------------- C4
        c4 = {"per_seed": {}}
        for r in M["runs"]:
            if r["method"] != "ppl_top1":
                continue
            last = r["scored"][-1] if r["scored"] else None
            first_fail = next((x["position"] for x in r["scored"][: r["n_tested"]] if not x["pass"]), None)
            b = prim.get(int(r["seed"]), {}).get("cap_top1")
            c4["per_seed"][int(r["seed"])] = {
                "first_failing_position": first_fail, "frozen_checkpoints": len(r["scored"]),
                "screened_final_estimate": {c: last["estimate"][c] for c in CATS} if last else None,
                "screened_final_exceeds_3pts": bool(last and _worst(last["estimate"]) > 0.03),
                "a_confirms_less_than_b": (b is not None and r["confirmed_parameters_removed"] < b)}
        ps = c4["per_seed"].values()
        c4["a_less_than_b_seeds"] = sum(1 for v in ps if v["a_confirms_less_than_b"])
        c4["final_exceeds_seeds"] = sum(1 for v in ps if v["screened_final_exceeds_3pts"])
        c4["model_criterion_met"] = c4["a_less_than_b_seeds"] >= 2 and c4["final_exceeds_seeds"] >= 2
        # ---------------------------------------------------------------- C6 rows
        for r in M["runs"]:
            if r["method"] != "cap_klprobe":
                continue
            for j, cf in enumerate(r["counterfactual_logs"], 1):
                w = {k: cf["by_rank"][str(rank)]["worst"] for k, rank in cf["picks"].items()}
                c6_rows.append({"model": M["model_key"], "seed": int(r["seed"]), "rejection": j, "picks": cf["picks"],
                                "worst": w, "excluded_in_sensitivity": excluded(r)})
        # ---------------------------------------------------------------- per-run cost and wall time
        runs_tab = []
        for r, x in zip(M["runs"], per_run):
            sc = r["cost"]["seconds"]
            runs_tab.append({"run_id": r["run_id"], "method": r["method"], "seed": int(r["seed"]), "batch": r["batch"],
                             "stop_reason": r["stop_reason"], "accepts": r["accepts"], "attempts": r["attempts"],
                             "wall_minutes": r["wall"]["seconds"] / 60, "worker_attempts": r["wall"]["attempts"],
                             "gate_questions": r["cost"]["screening_gate_questions"], "kl_probe_items": r["cost"]["kl_probe_items"],
                             "charged_to_cap": r["cost"]["questions_charged_to_cap"],
                             "counterfactual_items": r["cost"]["counterfactual_items"],
                             "profile_s": sc.get("profile_s", 0.0), "gate_s": sc.get("gate_s", 0.0), "kl_probe_s": sc.get("kl_probe_s", 0.0),
                             "counterfactual_s": sc.get("counterfactual_s", 0.0), "confirmation_items": x["S2"]["confirmation_items"],
                             "confirmation_s_estimated": x["S2"]["gpu_seconds"] - sc.get("gate_s", 0.0) - sc.get("kl_probe_s", 0.0),
                             "screened_final_parameters": (r["screened_final"] or {}).get("materialized_parameters_removed", 0),
                             "confirmed_parameters": r["confirmed_parameters_removed"], "claim_position": r["claim_position"],
                             "n_tested": r["n_tested"], "frozen_checkpoints": len(r["scored"]), "power_achieved": r["power_achieved"]})
        out["models"].append({"label": M["label"], "model_key": M["model_key"], "dense_parameters": dense_p,
                              "registration_sha16": M["registration_sha16"], "gate_pool": M["gate_pool"],
                              "per_category": M["per_category"], "confirm_rate": M["confirm_rate"], "batches": M["batches"],
                              "admissibility": M["admissibility"],
                              "primary": across_seeds(prim), "secondary": secondary, "c1": c1, "c4": c4, "runs": runs_tab,
                              "per_run_secondary": per_run})
    # -------------------------------------------------------------------- C6, pooled over models and seeds
    def compare(rows, a, b):  # a "beats" b when its hold-out worst-category net flip is lower
        better = sum(1 for x in rows if x["worst"][a] < x["worst"][b])
        worse = sum(1 for x in rows if x["worst"][a] > x["worst"][b])
        ties = len(rows) - better - worse
        return {"rejections": len(rows), "better": better, "worse": worse, "ties": ties, "p_two_sided": sign_test(better, worse)}

    c6 = {"rows": c6_rows}
    for name, rows in (("all", c6_rows), ("excluding_pilot", [x for x in c6_rows if not x["excluded_in_sensitivity"]])):
        c6[name] = {"primary_KL_vs_rank2": compare(rows, "KL", "gradient rank 2"),
                    "secondary_TV_vs_KL": compare(rows, "TV", "KL"), "secondary_S1c_vs_KL": compare(rows, "S1c", "KL")}
    p = c6["all"]["primary_KL_vs_rank2"]
    c6["criterion_met"] = bool(p["rejections"] and p["better"] > p["rejections"] / 2 and p["p_two_sided"] < 0.05)
    out["c6"] = c6
    out["c1_supported"] = all(m["c1"]["model_criterion_met"] for m in out["models"])
    out["c4_supported"] = all(m["c4"]["model_criterion_met"] for m in out["models"])
    return out


def study(cfg: dict) -> Path:
    S = cfg["confirmatory"]["study"]
    sec = _secondary_cfg(cfg)
    models = [_load_model(cfg, m, sec) for m in S["models"]]
    res = analyse(models, sec, S.get("sensitivity_exclude") or {})
    res.update({"secondary_config": sec, "commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "",
                "written_at": time.time()})
    od = resolve_path(cfg, S["out_dir"])
    od.mkdir(parents=True, exist_ok=True)
    atomic_write_json(od / "study.json", res)
    atomic_write_text(od / "study.md", render(res))
    return od / "study.md"


# ============================================================================ rendering
def _M(v) -> str:
    return "–" if v is None else f"{v / 1e6:.1f} M"


def _pt(v) -> str:
    return "–" if v is None else f"{100 * v:+.2f}"


def _num(v, f="{:,.0f}") -> str:
    return "–" if v is None else f.format(v)


def render(R: dict) -> str:
    L = ["# Confirmatory study: results across both models", "",
         "Generated by `capprune confirmatory study` from the per-batch confirmation and report files. Primary endpoint "
         "and C1-C6 as pre-registered (plan sections 5-6); S1-S4 **added after the pilot** (amendment 3), with the pilot "
         "batch's values **post hoc**.", ""]
    for M in R["models"]:
        P = M["primary"]
        L += [f"## {M['label']}: {M['model_key']}", "",
              f"Dense parameters {M['dense_parameters']:,}; confirmation set {sum(M['per_category'].values()):,} items; "
              f"registration `{M['registration_sha16']}`.", "",
              "### Primary endpoint: confirmed parameters removed (seed = unit)", "",
              "| seed | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " | (b) − (a) | (c) − (b) |", "|---|---|---|---|---|---|"]
        for s in sorted(P[ARMS[0]]["per_seed"]):
            L.append(f"| {s} | " + " | ".join(
                f"{_M(P[a]['per_seed'].get(s))} ({100 * P[a]['per_seed'].get(s, 0) / M['dense_parameters']:.2f} %)" for a in ARMS)
                + f" | {_M(P['b_minus_a']['per_seed'].get(s))} | {_M(P['c_minus_b']['per_seed'].get(s))} |")
        L.append("| mean [range] | " + " | ".join(f"{_M(P[a]['mean'])} [{_M(P[a]['min'])}, {_M(P[a]['max'])}]" for a in ARMS)
                 + f" | {_M(P['b_minus_a']['mean'])} [{_M(P['b_minus_a']['min'])}, {_M(P['b_minus_a']['max'])}] | "
                 f"{_M(P['c_minus_b']['mean'])} [{_M(P['c_minus_b']['min'])}, {_M(P['c_minus_b']['max'])}] |")
        L += ["", f"Same sign in every seed: (b) − (a) {P['b_minus_a']['consistent_sign']}; (c) − (b) {P['c_minus_b']['consistent_sign']}.", ""]
        # C1
        c1 = M["c1"]
        t = c1["tested_summary"]
        L += ["### C1: pool-to-confirmation gap (confirmation estimate − gate pool estimate at acceptance; arms (b), (c))", "",
              f"Social Sciences gate pool: **{c1['social_sciences_gate_pool']:,} items** (gate pool: "
              + ", ".join(f"{c} {v:,}" for c, v in M["gate_pool"].items()) + ").", "",
              f"Tested checkpoints (the §6 criterion): {t['checkpoints']}; mean worst-category gap **{_pt(t['mean_worst_gap'])} points**, "
              f"95 % bootstrap interval [{_pt(t['ci95'][0])}, {_pt(t['ci95'][1])}]; criterion met in this model: **{c1['model_criterion_met']}**.", "",
              "| set | checkpoints | " + " | ".join(f"{c}: mean / max gap" for c in CATS) + " | worst-category gap: mean [95 % CI] |",
              "|---|---|" + "---|" * len(CATS) + "---|"]
        for k, lab in (("tested", "tested (primary sequence)"), ("all_scored", "all scored (amendment 3, descriptive)")):
            s = c1[f"{k}_summary"]
            L.append(f"| {lab} | {s['checkpoints']} | " + " | ".join(
                f"{_pt(s['per_category_mean_gap'][c])} / {_pt(s['per_category_max_gap'][c])}" for c in CATS)
                + f" | {_pt(s['mean_worst_gap'])} [{_pt(s['ci95'][0])}, {_pt(s['ci95'][1])}] |")
        L += ["", "Per tested checkpoint (points; confirmation / pool / gap):", "",
              "| run | position | " + " | ".join(CATS) + " | worst gap |", "|---|---|" + "---|" * len(CATS) + "---|"]
        for g in c1["tested"]:
            L.append(f"| {g['run_id']} | {g['position']} | " + " | ".join(
                f"{_pt(g['confirmation'][c])} / {_pt(g['pool'][c])} / {_pt(g['gap'][c])}" for c in CATS) + f" | {_pt(g['worst_gap'])} |")
        # C4
        c4 = M["c4"]
        L += ["", "### C4: arm (a), perplexity screen (descriptive)", "",
              "| seed | first failing position | frozen checkpoints | screened endpoint: " + " / ".join(CATS)
              + " (estimate, points) | > 3 points | (a) confirms less than (b) |", "|---|---|---|---|---|---|"]
        for s, v in sorted(c4["per_seed"].items()):
            L.append(f"| {s} | {v['first_failing_position'] or 'none'} | {v['frozen_checkpoints']} | "
                     + " / ".join(_pt((v['screened_final_estimate'] or {}).get(c)) for c in CATS)
                     + f" | {v['screened_final_exceeds_3pts']} | {v['a_confirms_less_than_b']} |")
        L += ["", f"§6 C4 criterion in this model ((a) < (b) in ≥ 2 of 3 seeds: {c4['a_less_than_b_seeds']}; endpoint > 3 points in "
              f"≥ 2 of 3 seeds: {c4['final_exceeds_seeds']}): **{c4['model_criterion_met']}**.", ""]
        # secondary
        L += ["### S1–S4 (added after the pilot; the pilot batch's values are post hoc)", ""]
        for name, sv in M["secondary"].items():
            L += [f"**{name}**", "", "| endpoint | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " | (b) − (a) | (c) − (b) |",
                  "|---|---|---|---|---|---|"]
            for key, lab, fmt in (("S1", "S1 overshoot (parameters)", _M), ("S3", "S3 confirmed at 5 points", _M),
                                  ("S2_items_per_M", "S2 items per confirmed M", lambda v: _num(v)),
                                  ("S2_gpu_seconds_per_M", "S2 GPU s per confirmed M", lambda v: _num(v, "{:,.1f}"))):
                X = sv[key]
                cells = []
                for a in ARMS:
                    u = f"; {X[a]['undefined']} none confirmed" if X[a]["undefined"] else ""
                    cells.append(f"{fmt(X[a]['mean'])} [{fmt(X[a]['min'])}, {fmt(X[a]['max'])}]{u}")
                L.append(f"| {lab}: mean [range] | " + " | ".join(cells) + f" | {fmt(X['b_minus_a']['mean'])} "
                         f"(same sign: {X['b_minus_a']['consistent_sign']}) | {fmt(X['c_minus_b']['mean'])} (same sign: {X['c_minus_b']['consistent_sign']}) |")
            sizes = sorted(next(iter(sv["S4"].values())).keys(), key=size_order) if sv["S4"] else []
            for n in sizes:
                L.append(f"| S4 n = {n} per category: median [IQR] | " + " | ".join(
                    f"{_M(sv['S4'][a][n]['median'])} [{_M(sv['S4'][a][n]['q25'])}, {_M(sv['S4'][a][n]['q75'])}]" for a in ARMS) + " | | |")
            L.append("")
        L += ["Per run (S1, S2, S3; S4 medians):", "",
              "| run | S1 overshoot | S2 items (per confirmed M) | S2 GPU s (per confirmed M) | counterfactual items, s | S3 (5 pts) | S4 medians by n |",
              "|---|---|---|---|---|---|---|"]
        for x in M["per_run_secondary"]:
            s2 = x["S2"]
            L.append(f"| {x['run_id']}{' (post hoc)' if x['post_hoc'] else ''} | {_M(x['S1'])} | {s2['items']:,} ({_num(s2['items_per_confirmed_M'])}) | "
                     f"{s2['gpu_seconds']:,.0f} ({_num(s2['gpu_seconds_per_confirmed_M'], '{:,.1f}')}) | {s2['counterfactual_items']:,}, {s2['counterfactual_seconds']:.0f} | "
                     f"{_M(x['S3'])} | " + ", ".join(f"{n}: {_M(x['S4'][n]['median'])} [{_M(x['S4'][n]['q25'])}–{_M(x['S4'][n]['q75'])}]"
                                                for n in sorted(x["S4"], key=size_order)) + " |")
        # costs and wall time
        L += ["", "### Wall time and cost per run", "",
              f"Confirmation rate (all of this model's confirmation passes, replays excluded): "
              f"{M['confirm_rate']['items']:,} items in {M['confirm_rate']['seconds']:,.0f} s = "
              f"{_num(M['confirm_rate']['seconds_per_item'] and 1000 * M['confirm_rate']['seconds_per_item'], '{:.1f}')} ms per item.", "",
              "| run | stop | accepts / attempts | wall (min) | worker starts | gate questions | KL-probe items (s) | charged to cap | counterfactual items (s) | profile s | gate / PPL s | confirmation items (est. s) | screened final | confirmed (position) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in M["runs"]:
            L.append(f"| {r['run_id']} | {r['stop_reason']} | {r['accepts']} / {r['attempts']} | {r['wall_minutes']:.1f} | {r['worker_attempts']} | "
                     f"{r['gate_questions']:,} | {r['kl_probe_items']:,} ({r['kl_probe_s']:.0f}) | {r['charged_to_cap']:,} | "
                     f"{r['counterfactual_items']:,} ({r['counterfactual_s']:.0f}) | {r['profile_s']:.0f} | {r['gate_s']:.0f} | "
                     f"{r['confirmation_items']:,} ({r['confirmation_s_estimated']:.0f}) | {_M(r['screened_final_parameters'])} | "
                     f"{_M(r['confirmed_parameters'])} ({r['claim_position'] or 'none'}) |")
        L += ["", "Power achieved (observed disagreement of the claimed, or first tested, checkpoint; joint power at a true loss of 0 / 1.0 / 1.5 points):", ""]
        for r in M["runs"]:
            pw = r["power_achieved"]
            if pw:
                L.append(f"* {r['run_id']}: " + " / ".join(f"{pw[mu]['joint']:.2f}" for mu in ("0.0", "0.01", "0.015"))
                         + "; observed q " + ", ".join(f"{c} {100 * pw['q_observed'][c]:.1f} %" for c in CATS))
        L += ["", "Batches (procedure record, C3):", ""]
        for b in M["batches"]:
            L.append(f"* `{b['batch']}`: frozen at commit `{b['frozen_commit'][:7]}`, {time.strftime('%Y-%m-%d %H:%M', time.localtime(b['frozen_at']))}; "
                     f"confirmed at commit `{b['confirm_commit'][:7]}` (clean tree: {b['confirm_clean_tree']}), finished "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(b['confirmed_at']))}; this pass scored {b['confirm_cost']['items_scored']:,} "
                     f"checkpoint items ({b['confirm_cost']['seconds']:,.0f} s) and replayed {b['confirm_cost']['items_replayed']:,}.")
        L.append("")
    # C6
    c6 = R["c6"]
    L += ["## C6: counterfactual log of arm (c), pooled over models and seeds (the rejection is the unit)", "",
          "Hold-out worst-category net flip loss of each pick (lower is better). Two-sided exact sign test, ties excluded. "
          "TV was favoured post hoc on Qwen3-1.7B, so TV vs KL is the first test of that observation.", "",
          "| model | seed | rejection | KL | gradient rank 2 | TV | S1c | picks (rank) |", "|---|---|---|---|---|---|---|---|"]
    for x in c6["rows"]:
        L.append(f"| {x['model']} | {x['seed']} | {x['rejection']} | {_pt(x['worst']['KL'])} | {_pt(x['worst']['gradient rank 2'])} | "
                 f"{_pt(x['worst']['TV'])} | {_pt(x['worst']['S1c'])} | {x['picks']} |")
    L += ["", "| comparison | rejections | better | worse | ties | p (two-sided) |", "|---|---|---|---|---|---|"]
    for name in ("all", "excluding_pilot"):
        for k, lab in (("primary_KL_vs_rank2", "primary: KL vs gradient rank 2"), ("secondary_TV_vs_KL", "secondary: TV vs KL"),
                       ("secondary_S1c_vs_KL", "secondary: S1c vs KL")):
            v = c6[name][k]
            L.append(f"| {lab} ({name.replace('_', ' ')}) | {v['rejections']} | {v['better']} | {v['worse']} | {v['ties']} | {v['p_two_sided']:.3g} |")
    L += ["", f"§6 C6 criterion (KL beats rank 2 in a majority of rejections and p < 0.05): **{c6['criterion_met']}**.", "",
          f"§6 C1 criterion in both models: **{R['c1_supported']}**. §6 C4 criterion in both models: **{R['c4_supported']}**.", "",
          f"Commit `{R['commit'][:7]}` (clean tree: {R['clean_tree']}).", ""]
    return "\n".join(L)
