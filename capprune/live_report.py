"""Arm comparison for the live gate experiment (``capprune live-report``).

Reads, per arm: the journal (one record per attempt), the latest checkpoint (search state, budget and
timings), the final audit (``final/mmlu.json`` etc., held-out final split) and ``_live/bench.json``.
Writes ``<runs_dir>/_live/report.md`` and ``report.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import data_dir, runs_dir
from .utils.io import atomic_write_text, read_json

ROLES = {"llm_ppl": ("A", "LLM", "perplexity"), "llm_cap": ("B", "LLM", "capability, threshold-aware"),
         "greedy_cap": ("C", "greedy sensitivity", "capability, threshold-aware"),
         "llm_cap_rr": ("D", "LLM", "capability, round-robin")}


def _attempts(run_dir: Path) -> list[dict]:
    last: dict[int, dict] = {}
    p = run_dir / "journal.jsonl"
    if not p.exists():
        return []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "iteration":
            last[r["iteration"]] = r  # a replayed iteration after a crash supersedes the earlier record
    return [last[k] for k in sorted(last)]


def _cat_losses(rows: list[dict], dense_pq: dict, run_pq: dict) -> dict[str, float]:
    acc: dict[str, list[int]] = {}
    for r in rows:
        a = acc.setdefault(r["category"], [0, 0, 0])
        a[0] += int(dense_pq[r["qid"]])
        a[1] += int(run_pq[r["qid"]])
        a[2] += 1
    return {c: (v[0] - v[1]) / v[2] for c, v in sorted(acc.items())}


def _validity(att: list[dict]) -> dict:
    """Controller replies per attempt: a proposal is valid when the LLM's reply parsed (no heuristic fallback)
    and at least one known unit survived ``sanitize``."""
    src = [((a.get("controller") or {}).get("source"), a) for a in att]
    if src and all(s_ == "heuristic" for s_, _ in src):
        return {"kind": "heuristic"}

    def has(a, w):
        return any(w in n for n in a.get("notes") or [])

    return {"kind": "llm", "replies": len(att),
            "valid": sum(s_ == "llm" and bool(a.get("decisions")) and not has(a, "unknown unit") for s_, a in src),
            "stop": sum(a.get("outcome") == "stop" for a in att),
            "fallbacks": sum(s_ == "heuristic_fallback" for s_, _ in src),
            "unknown_units": sum(has(a, "unknown unit") for a in att),
            "clamped": sum(has(a, "clamped") for a in att),
            "scaled": sum(has(a, "scaled") for a in att)}


def _collect_root(cfg: dict, methods, rows: list[dict], ref_label):
    from .live_bench import selected_checkpoints
    from .utils.checkpoint import CheckpointStore, NoCheckpoint

    root = runs_dir(cfg)
    tag = f"{ref_label}: " if ref_label else ""
    dense_final = root / "dense" / "final"
    dense = None
    if (dense_final / "summary.json").exists():
        dense = {"mmlu": read_json(dense_final / "mmlu.json"), "summary": read_json(dense_final / "summary.json")}
    # benchmark sessions (bench.json = the first session, bench-<time>.json = later ones); an arm's result
    # comes from the latest session that measured it, against that session's own dense reference
    sessions = []
    for p in sorted((root / "_live").glob("bench*.json"), key=lambda q: q.stat().st_mtime):
        j = read_json(p)
        models = {m["arm"]: m for m in j["models"]}
        sessions.append({"file": tag + p.name, "settings": j["settings"], "dense": models.get("dense"), "models": models})
    bench, bench_session = {}, {}
    for s in sessions:
        for arm, e in s["models"].items():
            if arm != "dense" and "decode_ms_per_token_median" in e:
                bench[arm], bench_session[arm] = e, s["file"]
    target = float(cfg["matrix"]["targets"][0])
    led_p = root / "_matrix" / "ledger.json"
    ledger = read_json(led_p) if led_p.exists() else {}
    max_it = int(cfg["search"]["max_iterations"])
    arms = []
    for sel in selected_checkpoints(cfg, methods):
        arm, rd = sel["arm"], root / sel["run_id"]
        att = _attempts(rd)
        lst = (ledger.get(sel["run_id"]) or {}).get("status")
        if (rd / "DONE").exists():
            status = "complete"
        elif sel["status"] == "not run":
            status = "pending (not started)"
        elif lst == "waiting_controller":
            status = f"paused: LLM controller unavailable ({len(att)}/{max_it} attempts, no final audit yet)"
        else:
            status = f"incomplete ({len(att)}/{max_it} attempts)"
        rec = {"arm": arm + (f" ({ref_label})" if ref_label else ""), "method": arm, "reference": ref_label,
               "runs_dir": str(root), "role": ROLES.get(arm, ("?", "?", "?")), "run_id": sel["run_id"],
               "status": status, "state_id": sel["state_id"], "attempts": att, "controller_validity": _validity(att)}
        counts = {k: 0 for k in ("accept", "reject", "inconclusive", "noop", "stop")}
        for a in att:
            k = a.get("gate_decision") or ("stop" if a.get("outcome") == "stop" else a.get("outcome"))
            counts[k] = counts.get(k, 0) + 1
        rec["counts"] = counts
        rec["questions"] = sum((a.get("gate") or {}).get("questions") or 0 for a in att)
        rec["rollbacks_verified"] = all(a.get("rollback_verified") in (True, None) for a in att)
        rec["parents_verified"] = all(a.get("parent_verified") in (True, None) for a in att)
        try:
            S, _ = CheckpointStore(rd).load_latest()
            srch = S.get("search") or {}
            rec.update(rho=srch.get("rho"), stop_reason=srch.get("stop_reason"), budget=srch.get("budget") or {},
                       target_reached=(srch.get("rho") or 0) >= target - float(cfg["search"].get("target_tolerance", 0.005)),
                       ppl0=srch.get("ppl0"), ppl=srch.get("ppl"))
        except (NoCheckpoint, FileNotFoundError):
            rec.update(rho=None, budget={})
        fin = rd / "final"
        if dense and (fin / "summary.json").exists():  # paired against the dense run of the SAME runs dir
            s = read_json(fin / "summary.json")
            mm = read_json(fin / "mmlu.json")
            cl = _cat_losses(rows, dense["mmlu"]["per_question"], mm["per_question"])
            worst = max(cl, key=cl.get)
            dsum = dense["summary"]
            rec["audit"] = {"category_losses": cl, "worst_category": worst, "worst_loss": cl[worst],
                            "mmlu_loss": dense["mmlu"]["overall"] - mm["overall"], "mmlu": mm["overall"],
                            "fqa_em": (s.get("freebaseqa") or {}).get("exact_match"),
                            "fqa_em_dense": (dsum.get("freebaseqa") or {}).get("exact_match"),
                            "wikitext_ppl": (s.get("wikitext2") or {}).get("perplexity"),
                            "wikitext_ppl_dense": (dsum.get("wikitext2") or {}).get("perplexity"),
                            "tolerance": ((cfg.get("gate") or {}).get("capability") or {}).get("tolerance")}
        rec["bench"] = bench.get(arm)
        rec["bench_session"] = bench_session.get(arm)
        sp = rd / "final" / "summary.json"
        rec["run_wall_seconds"] = read_json(sp).get("wall_seconds") if sp.exists() else None
        arms.append(rec)
    return arms, sessions, dense


def collect(cfg: dict) -> dict:
    from .config import deep_merge
    from .data.prepare import _read_jsonl

    split = (cfg.get("final_eval") or {}).get("split", "final")
    rows = _read_jsonl(data_dir(cfg) / ("mmlu_dev.jsonl" if split == "dev" else "mmlu_final.jsonl"))
    arms, sessions, dense = _collect_root(cfg, None, rows, None)
    # reference arms from another runs dir (read only), e.g. the greedy arm that needs no LLM controller
    for ref in (cfg.get("live_report") or {}).get("reference_arms") or []:
        cfg_r = deep_merge(cfg, {"experiment": {"runs_dir": ref["runs_dir"]}})
        a2, s2, _ = _collect_root(cfg_r, [ref["method"]], rows, ref.get("label", "reference"))
        arms += a2
        sessions += s2
    ctl_p = runs_dir(cfg) / "_live" / "controller_model.json"
    return {"arms": arms, "dense": {"mmlu": dense["mmlu"]["overall"] if dense else None},
            "bench_sessions": sessions, "target": float(cfg["matrix"]["targets"][0]),
            "controller_model": read_json(ctl_p) if ctl_p.exists() else None}


def _f(x, fmt="{:.3f}", na="n/a"):
    return na if x is None else fmt.format(x)


def render(R: dict, cfg: dict) -> str:
    g = (cfg.get("gate") or {}).get("capability") or {}
    tol = g.get("tolerance")
    L = ["# Live gate experiment: arm comparison", "",
         f"Model {cfg['model']['name_or_path']} ({cfg['pruning']['mode']} MLP-channel pruning), target sparsity "
         f"{R['target']:.2f} of MLP weights, proposal budget {cfg['search']['max_iterations']} per arm, seed "
         f"{cfg['matrix']['seeds'][0]}. Capability gate: tolerance {g.get('tolerance')} per category "
         f"({', '.join(g.get('domains', []))}), budget {g.get('budget')} questions per candidate, min sample / batch "
         f"{g.get('min_sample')}/{g.get('batch_size')}, alpha FA/FR {g.get('alpha_false_accept')}/"
         f"{g.get('alpha_false_reject')}, {g.get('spending')} spending.", "",
         "**Scope of the capability bounds: per candidate.** For each evaluated trial, with probability at least "
         "1 - alpha_FA - alpha_FR over that trial's own random question order, every bound covers the trial's loss on "
         "the full pruning-time pool. No error control across the attempts of a search is claimed; the selected "
         "checkpoint's quality is checked on the held-out final split below.", ""]
    cm = R.get("controller_model")
    if cm:
        L += [f"LLM controller: `{cm.get('model')}` served by {cm.get('server')} (quantization {cm.get('quantization')}, "
              f"{cm.get('parameters')} parameters, digest `{str(cm.get('digest'))[:12]}`, context {cm.get('context_length')}).", ""]
    if any(a.get("reference") for a in R["arms"]):
        L += ["Reference arms read from other runs directories (same seed, gate, data, pruning and audit settings; "
              "audits paired with that directory's own dense run): " + ", ".join(
                  f"`{a['arm']}` from `{a['runs_dir']}`" for a in R["arms"] if a.get("reference")) + ".", ""]

    def _valid(a):
        v = a.get("controller_validity") or {}
        if v.get("kind") != "llm":
            return "n/a (heuristic)"
        extra = [f"{v[k]} {k.replace('_', ' ')}" for k in ("stop", "fallbacks", "unknown_units", "clamped", "scaled") if v.get(k)]
        return f"{v['valid']}/{v['replies']}" + (f" ({', '.join(extra)})" if extra else "")

    L += ["## Search outcome", "",
          "| arm | proposer | gate | status | attempts | valid proposals | accepted | rejected | inconclusive | no-op | final MLP sparsity | target reached | gate questions | rollbacks verified |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in R["arms"]:
        c = a["counts"]
        L.append(f"| {a['role'][0]} `{a['arm']}` | {a['role'][1]} | {a['role'][2]} | {a['status']} | {len(a['attempts'])} | {_valid(a)} | "
                 f"{c['accept']} | {c['reject']} | {c['inconclusive']} | {c['noop']} | {_f(a.get('rho'), '{:.4f}')} | "
                 f"{'yes' if a.get('target_reached') else 'no'} | {a['questions']:,} | "
                 f"{('yes' if a['rollbacks_verified'] and a['parents_verified'] else 'NO') if a['attempts'] else ''} |")
    L += ["", f"## Final audit (held-out final split, MMLU 5-shot; losses vs. dense in accuracy points; limit {tol})", "",
          "| arm | worst category | worst-category loss | overall MMLU loss | FreebaseQA EM (dense) | WikiText-2 PPL (dense) |",
          "|---|---|---|---|---|---|"]
    for a in R["arms"]:
        au = a.get("audit")
        if not au:
            L.append(f"| `{a['arm']}` | not audited yet | | | | |")
            continue
        L.append(f"| `{a['arm']}` | {au['worst_category']} | {100 * au['worst_loss']:+.2f} | {100 * au['mmlu_loss']:+.2f} | "
                 f"{_f(au['fqa_em'], '{:.4f}')} ({_f(au['fqa_em_dense'], '{:.4f}')}) | "
                 f"{_f(au['wikitext_ppl'], '{:.3f}')} ({_f(au['wikitext_ppl_dense'], '{:.3f}')}) |")
    L += ["", "Per-category final-split losses (points): " + "; ".join(
        f"`{a['arm']}` " + ", ".join(f"{c} {100 * v:+.1f}" for c, v in a["audit"]["category_losses"].items())
        for a in R["arms"] if a.get("audit")), ""]
    L += ["## Deployment speed (interleaved with the dense model)", ""]
    sessions = [x for x in R.get("bench_sessions") or [] if any(a.get("bench_session") == x["file"] for a in R["arms"])]
    if not sessions:
        L += ["Not measured yet (`capprune live-bench`).", ""]
    max_noise = float((cfg.get("live_bench") or {}).get("max_noise", 0.10))
    for sess in sessions:
        bs, db = sess["settings"], sess["dense"] or {}
        L += [f"Session `{sess['file']}`: {bs['device']}, {bs['dtype']}, torch {bs['torch']}, batch {bs['batch_size']}, "
              f"prompt {bs['prompt_len']}, {bs['gen_tokens']} generated tokens, {bs['warmup']} warm-up + {bs['repeats']} "
              f"interleaved rounds. Dense in this session: decode {_f(db.get('decode_ms_per_token_median'), '{:.2f}')} ms/token "
              f"(IQR {_f(db.get('decode_iqr'), '{:.2f}')}), prefill {_f(db.get('prefill_ms_median'), '{:.1f}')} ms, peak "
              f"{_f((db.get('peak_memory_bytes') or 0) / 2**30 if db.get('peak_memory_bytes') else None, '{:.2f}')} GiB.", ""]
        noise = {k: (db.get(f"{k}_iqr") or 0) / db[f"{k}_ms_per_token_median" if k == "decode" else f"{k}_ms_median"]
                 for k in ("decode", "prefill") if db.get(f"{k}_iqr") is not None}
        noisy = {k for k, v in noise.items() if v > max_noise}
        if noisy:
            L += [f"Timing noise: dense IQR / median = " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in noise.items())
                  + f" (limit {100 * max_noise:.0f}%), so the {' and '.join(sorted(noisy))} speed results are marked "
                  "inconclusive whatever their interval says.", ""]
        L += ["| arm | removed parameters | decode ms/token | decode ratio [95% CI] | prefill ms | prefill ratio [95% CI] | peak GiB | speed result |",
              "|---|---|---|---|---|---|---|---|"]
        for a in R["arms"]:
            b = a.get("bench")
            if a.get("bench_session") != sess["file"]:
                continue
            dr, pr = b["decode_ratio"], b["prefill_ratio"]
            dv = "inconclusive (noisy)" if "decode" in noisy else dr["verdict"]
            pv = "inconclusive (noisy)" if "prefill" in noisy else pr["verdict"]
            verdict = dv if dv == pv else f"decode {dv}, prefill {pv}"
            L.append(f"| `{a['arm']}` | {b['removed_parameters']:,} ({100 * b['removed_parameters'] / (b['parameters'] + b['removed_parameters']):.2f}%) | "
                     f"{b['decode_ms_per_token_median']:.2f} | {dr['median']:.3f} [{dr['ci95'][0]:.3f}, {dr['ci95'][1]:.3f}] | "
                     f"{b['prefill_ms_median']:.1f} | {pr['median']:.3f} [{pr['ci95'][0]:.3f}, {pr['ci95'][1]:.3f}] | "
                     f"{_f(b.get('peak_memory_bytes') / 2**30 if b.get('peak_memory_bytes') else None, '{:.2f}')} | {verdict} |")
        L.append("")
    missing = [a["arm"] for a in R["arms"] if not a.get("bench_session")]
    if missing:
        L += ["Not benchmarked yet: " + ", ".join(f"`{m}`" for m in missing) + ".", ""]
    L += ["## Time (seconds)", "",
          "| arm | profiling | controller | trial build (pruning) | PPL | capability gate | of which scoring | rollback | materialise (bench) | search wall | run wall (end-to-end) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in R["arms"]:
        b = a.get("budget") or {}
        mat = (a.get("bench") or {}).get("materialize_seconds")
        L.append(f"| `{a['arm']}` | {_f(b.get('profile_seconds'), '{:.0f}')} | {_f(b.get('controller_seconds'), '{:.0f}')} | "
                 f"{_f(b.get('build_seconds'), '{:.1f}')} | {_f(b.get('ppl_seconds'), '{:.0f}')} | "
                 f"{_f(b.get('gate_seconds'), '{:.0f}')} | {_f(b.get('gate_scoring_seconds'), '{:.0f}')} | "
                 f"{_f(b.get('rollback_seconds'), '{:.1f}')} | {_f(mat, '{:.1f}')} | {_f(b.get('search_wall_seconds'), '{:.0f}')} | "
                 f"{_f(a.get('run_wall_seconds'), '{:.0f}')} |")
    L += ["", "Run wall = the run's own clock from model load to summary (search + final audit), excluding time "
          "while the run was paused; controller time includes rate-limit back-off.", ""]
    L += ["", "## Attempts", ""]
    for a in R["arms"]:
        L += [f"### {a['role'][0]} `{a['arm']}`", "",
              "| # | parent -> trial | proposal (unit +increment) | gate decision | questions | deciding category: est [lo, hi] (pts) | MLP sparsity after |",
              "|---|---|---|---|---|---|---|"]
        for t in a["attempts"]:
            gd = t.get("gate") or {}
            doms = gd.get("domains") or {}
            if doms:  # the deciding category: the failing one on a reject, else the largest estimate
                fails = [d for d in doms if doms[d]["status"] == "fail"]
                w = max(fails, key=lambda d: doms[d]["lo"]) if fails else max(doms, key=lambda d: doms[d]["est"])
                worst = f"{w} {100 * doms[w]['est']:+.1f} [{100 * doms[w]['lo']:+.1f}, {100 * doms[w]['hi']:+.1f}]"
            elif gd.get("kind", "").startswith("ppl") and gd.get("ppl_trial") is not None:
                worst = f"PPL {gd['ppl_parent']:.3f} -> {gd['ppl_trial']:.3f} ({t.get('ppl_change_pct', 0):+.1f}%)"
            else:
                worst = ""
            dec = ", ".join(f"{u.split('.')[0]}+{d:.3f}" for u, d in t.get("decisions", [])) or "-"
            why = ",".join(t.get("reasons") or [])
            L.append(f"| {t['iteration']} | {str(t.get('parent_id'))[:8]} -> {str(t.get('trial_id') or '-')[:8]} | {dec} | "
                     f"{t.get('gate_decision')}{f' ({why})' if why else ''} | {gd.get('questions') or ''} | {worst} | "
                     f"{t.get('rho_after', '')} |")
        L.append("")
    L += ["## What the controller sees in each arm", "",
          "Arms A, B and D use the same controller model, prompt template, unit table and action space. Differences "
          "(recorded per attempt in `journal.jsonl` as `prompt_variant` and `feedback`):", "",
          "* **Rule line** (system prompt): A is told a step is rolled back if perplexity rises > "
          f"{cfg['search']['ppl_rollback_threshold']:.0%}; B/D are told the capability rule (category limit, question "
          "budget, PPL for information only).",
          "* **Current state** (user prompt): B/D additionally see the accepted model's per-category loss estimate and "
          "bounds; A does not.",
          "* **Feedback on the previous attempt**: all arms get outcome, sparsity gain and PPL change. B/D additionally "
          "get the gate decision, questions used and per-category estimate [lower, upper] vs. the limit, and their "
          "assessment sentence is derived from the capability result (A's from the PPL change).",
          "* B and D see identical text; only the gate's question allocation differs, which can change the bounds "
          "reported after a rejection (and thus later proposals).",
          "* C (greedy) reads no prompt: it uses the unit sensitivities and accept/rollback history (plus PPL change for "
          "its step-size rule).", ""]
    return "\n".join(L) + "\n"


def write_live_report(cfg: dict) -> Path:
    R = collect(cfg)
    out = runs_dir(cfg) / "_live"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    p = out / "report.md"
    atomic_write_text(p, render(R, cfg))
    return p
