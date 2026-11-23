"""Replay experiment: every allocation policy evaluates every cached candidate.

For each (candidate, policy) a fresh :class:`ReplaySource` reveals cached
answers on request; the shared evaluator applies the shared decision rule; the
separate scorer labels the decision against the full-pool reference.

Costs per run
-------------
questions      items revealed (the budget unit)
tokens         prompt tokens of those items (if the cache has token counts)
eval_seconds   measured time to score exactly the batches this run consumed, taken from
               the live cache build (only if the build used this replay's question order
               and look layout; otherwise reported as unavailable)
decision_s     CPU time this replay spent in bounds + decision rule + policy

Search to the first qualifying checkpoint
-----------------------------------------
Candidates are tried in the cache's fixed order; those whose measured latency
misses the target are skipped without evaluation; the search stops at the first
ACCEPT.  Candidate-evaluation time (scoring batches) and total search wall time
(+ building/loading and benchmarking every considered candidate + decision
overhead) are reported separately.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .cache import ReplayCache, ReplaySource
from .evaluator import ACCEPT, Settings, evaluate_candidate, make_plan
from .policies import make_policy
from .scorer import outcome, per_policy, reference


def layout_matches(timing: dict | None, s: Settings) -> bool:
    if not timing or "layout" not in timing:
        return False
    want = {"seed": s.seed, "min_sample": s.min_sample, "batch_size": s.batch_size}
    return json.loads(json.dumps(timing["layout"], sort_keys=True)) == json.loads(json.dumps(want, sort_keys=True))


def run_costs(cache: ReplayCache, cand, run, s: Settings) -> dict:
    ids = [i for v in run.revealed.values() for i in v]
    toks = [cache.tokens.get(i) for i in ids]
    tokens = None if any(t is None for t in toks) else int(sum(toks))
    eval_s = None
    if layout_matches(cand.timing, s):
        bs = cand.timing["batch_seconds"]
        eval_s = sum(sum(bs[d][: j + 1]) for d, j in run.looks_reached.items())
    return {"tokens": tokens, "eval_seconds": eval_s}


def run_replay(cache: ReplayCache, s: Settings, policies: list[str], candidates: list[str] | None = None) -> list[dict]:
    plan = make_plan(s, cache.pool_sizes)
    ref = reference(cache, s)
    names = candidates or [c.name for c in cache.candidates]
    rows = []
    for name in names:
        cand = cache.candidate(name)
        for pol in policies:
            src = ReplaySource(cache, name, s.seed, s.budget)
            run = evaluate_candidate(src, make_policy(pol), plan, name)
            if src.used != run.questions or run.questions > s.budget:
                raise AssertionError(f"accounting mismatch for {name}/{pol}: {src.used} vs {run.questions}")
            rows.append({**run.to_json(), **run_costs(cache, cand, run, s),
                         "outcome": outcome(run.decision, ref[name]["qualifies"]),
                         "reference_qualifies": ref[name]["qualifies"]})
    return rows


# ---------------------------------------------------------------------------- search metric
def latency_ok(model, dense, target: dict | None) -> tuple[bool | None, float | None]:
    """(meets target, latency value).  None when there is no target or no measurement."""
    if not target:
        return True, None
    metric = target.get("metric", "decode_ms_per_token")
    val = (model.latency or {}).get(metric)
    if val is None:
        return None, None
    limit = target.get("max_value")
    if limit is None and target.get("max_ratio_vs_dense") is not None:
        base = (dense.latency or {}).get(metric)
        if base is None:
            return None, val
        limit = float(target["max_ratio_vs_dense"]) * base
    return (True if limit is None else val <= limit), val


def search(rows: list[dict], cache: ReplayCache, ref: dict, policies: list[str], target: dict | None,
           order: list[str]) -> dict[str, dict]:
    by = {(r["policy"], r["candidate"]): r for r in rows}
    out = {}
    for pol in policies:
        res = {"found": None, "considered": 0, "evaluated": 0, "skipped_latency": [], "questions": 0, "tokens": 0,
               "eval_seconds": 0.0, "search_wall_seconds": 0.0, "truly_qualifies": None, "holdout_qualifies": None}
        for name in order:
            c = cache.candidate(name)
            ok, _ = latency_ok(c, cache.dense, target)
            res["considered"] += 1
            t = c.timing or {}
            overhead = None if "setup_seconds" not in t else t["setup_seconds"] + t.get("bench_seconds", 0.0)
            res["search_wall_seconds"] = _add(res["search_wall_seconds"], overhead)
            if not ok:  # missed the target, or no measurement: never evaluated
                res["skipped_latency"].append(name)
                continue
            r = by[(pol, name)]
            res["evaluated"] += 1
            res["questions"] += r["questions"]
            res["tokens"] = _add(res["tokens"], r["tokens"])
            res["eval_seconds"] = _add(res["eval_seconds"], r["eval_seconds"])
            res["search_wall_seconds"] = _add(res["search_wall_seconds"], _add(r["eval_seconds"], r["decision_seconds"]))
            if r["decision"] == ACCEPT:
                res["found"] = name
                res["truly_qualifies"] = ref[name]["qualifies"]
                res["holdout_qualifies"] = ref[name]["holdout_qualifies"]
                break
        out[pol] = res
    return out


def _add(a, b):
    return None if a is None or b is None else a + b


# ---------------------------------------------------------------------------- driver + report
def replay_main(cfg: dict, cache_dir: Path, out_dir: Path) -> Path:
    t0 = time.perf_counter()
    s = Settings.from_dict(cfg["evaluation"])
    policies = list(cfg.get("policies") or ["round_robin", "widest_interval", "threshold_aware"])
    cache = ReplayCache.load(cache_dir)
    srch = cfg.get("search") or {}
    order = srch.get("candidate_order") or [c.name for c in cache.candidates]
    rows = run_replay(cache, s, policies, order)
    ref = reference(cache, s)
    agg = per_policy(rows)
    found = search(rows, cache, ref, policies, srch.get("latency_target"), order)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    cols = ["candidate", "policy", "decision", "outcome", "reference_qualifies", "questions", "tokens", "eval_seconds",
            "steps", "decision_seconds"]
    with open(out_dir / "runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"{d}:{k}" for d in s.domains for k in ("n", "lo", "hi", "status")])
        for r in rows:
            w.writerow([r[c] for c in cols] + [r["final"][d][k] for d in s.domains for k in ("n", "lo", "hi", "status")])
    summary = {"settings": s.to_dict(), "policies": policies, "candidate_order": order, "cache": str(cache_dir),
               "reference": ref, "per_policy": agg, "search": found, "latency_target": srch.get("latency_target"),
               "replay_seconds": time.perf_counter() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    md = render(summary, rows, cache, make_plan(s, cache.pool_sizes))
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    return out_dir / "summary.md"


def _pts(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}"


def _num(x, fmt="{:,}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(S: dict, rows: list[dict], cache: ReplayCache, plan) -> str:
    s = Settings.from_dict(S["settings"])
    pols, ref = S["policies"], S["reference"]
    L = ["# Adaptive candidate evaluation: replay", ""]
    L += [f"Cache: `{S['cache']}`  ", f"Domains (fixed order): {', '.join(s.domains)}  ",
          "Tolerance (max accuracy loss, pts): " + ", ".join(f"{d} {100 * s.tolerance[d]:.1f}" for d in s.domains) + "  ",
          "Minimum sample / batch size: " + ", ".join(f"{d} {s.min_sample[d]}/{s.batch_size[d]}" for d in s.domains) + "  ",
          f"Budget: {s.budget:,} questions per candidate; error spending: {s.spending}"
          f"{f' (rho={s.spending_rho})' if s.spending == 'power' else ''}; "
          f"alpha false-accept {s.alpha_false_accept}, false-reject {s.alpha_false_reject}; question-order seed {s.seed}  ",
          "Planned looks per domain: " + ", ".join(f"{dp.name} {len(dp.looks)} (N={dp.pool_size})" for dp in plan.domains),
          "", "**What the bounds cover.** For each candidate separately, with probability at least "
          f"1 - {s.alpha_false_accept} - {s.alpha_false_reject} over the random question order, every bound computed "
          "at every planned look of every domain contains that domain's *full search-pool* loss "
          "(#broken - #fixed) / N_d against the dense checkpoint. Hence P(false accept) <= "
          f"{s.alpha_false_accept} and P(false reject) <= {s.alpha_false_reject} for every policy. The bounds do not "
          "cover the holdout or the underlying population (holdout losses below are an independent point check), "
          "and there is no correction across candidates.", ""]

    L += ["## Full-pool reference (scorer only; never visible to policies)", ""]
    head = ["candidate", "qualifies"] + [f"{d} loss" for d in s.domains] + ["holdout max loss", "holdout ok", "latency"]
    tr = []
    for c in cache.candidates:
        r = ref[c.name]
        lat = (c.latency or {}).get((S.get("latency_target") or {}).get("metric", "decode_ms_per_token"))
        tr.append([c.name, "yes" if r["qualifies"] else "no"] + [_pts(r["domains"][d]["loss"]) for d in s.domains]
                  + [_pts(r["holdout_max_loss"]), {True: "yes", False: "no", None: "n/a"}[r["holdout_qualifies"]],
                     _num(lat, "{:.2f}")])
    dl = (cache.dense.latency or {}).get((S.get("latency_target") or {}).get("metric", "decode_ms_per_token"))
    L += [_table(head, tr), "", f"Losses in accuracy points (positive = pruning lost accuracy). Dense latency: {_num(dl, '{:.2f}')}.", ""]

    L += ["## Decisions per candidate", ""]
    by = {(r["policy"], r["candidate"]): r for r in rows}
    tr = []
    for name in S["candidate_order"]:
        cells = []
        for p in pols:
            r = by[(p, name)]
            flag = "" if r["outcome"] in ("correct_accept", "correct_reject", "inconclusive") else f" **{r['outcome'].upper()}**"
            cells.append(f"{r['decision']}, {r['questions']:,} q{flag}")
        tr.append([name, "yes" if ref[name]["qualifies"] else "no"] + cells)
    L += [_table(["candidate", "qualifies"] + pols, tr), ""]

    L += ["## Totals per policy", ""]
    rr = S["per_policy"].get("round_robin", {}).get("questions")
    tr = []
    for p in pols:
        a = S["per_policy"][p]
        save = "" if not rr or p == "round_robin" else f" ({100 * (a['questions'] - rr) / rr:+.1f}% vs RR)"
        tr.append([p, f"{a['questions']:,}{save}", _num(a["tokens"]), _num(a["eval_seconds"], "{:.1f}"),
                   a["accept"], a["reject"], a["inconclusive"], a["false_accept"], a["false_reject"],
                   f"{a['decision_seconds']:.2f}"])
    L += [_table(["policy", "questions", "tokens", "eval s (measured)", "accept", "reject", "inconclusive",
                  "false accepts", "false rejects", "decision overhead s"], tr), ""]

    tgt = S.get("latency_target")
    L += ["## Budget to the first qualifying checkpoint", "",
          f"Latency target: {json.dumps(tgt) if tgt else 'none'}. Candidates tried in order "
          f"{', '.join(S['candidate_order'])}; latency misses are skipped unevaluated.", ""]
    tr = []
    for p in pols:
        f = S["search"][p]
        tr.append([p, f["found"] or "none found", f"{f['evaluated']}/{f['considered']}", f"{f['questions']:,}",
                   _num(f["tokens"]), _num(f["eval_seconds"], "{:.1f}"), _num(f["search_wall_seconds"], "{:.1f}"),
                   {True: "yes", False: "NO", None: "n/a"}[f["truly_qualifies"]],
                   {True: "yes", False: "no", None: "n/a"}[f["holdout_qualifies"]]])
    L += [_table(["policy", "first accepted", "evaluated/considered", "questions", "tokens", "candidate-eval s",
                  "search wall s", "truly qualifies", "holdout ok"], tr), "",
          "candidate-eval s = measured scoring time of the consumed batches (live build). search wall s adds building/"
          "loading and benchmarking every considered candidate and the decision overhead; it is reconstructed from "
          "measured components, not timed end to end. n/a = no timing in the cache.", ""]
    return "\n".join(L)
