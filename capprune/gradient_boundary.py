"""Compression boundary, gate-margin calibration and the live-policy comparison (``capprune gradient-boundary ...``).

boundary-recover  The accepted checkpoints of a finished gradient run (``attempts``) and one fallback candidate
                  (``rank2``: a recorded shortlist action applied to a recorded parent).  Saved masks are reused
                  when present; every other state is REPLAYED from the nearest saved ancestor with the run's own
                  persisted per-attempt profiles, and every state - saved or replayed - is checked against the
                  journal's trial mask hash (and state id) before anything is scored.  The fallback candidate is
                  checked against the stall diagnostic's cache keys.
margin-plan       Records the margin grid, the selection rule, the audit criterion and the fallback for an
                  inconclusive calibration BEFORE the new development states are scored (boundary-audit refuses
                  to run without it).
boundary-audit    Materialises every state and scores it on the already-inspected audit sets with the matched 0-shot
                  protocol (the original audit's ids, order, batches and token budget).  Dense answers come from
                  the earlier materialised audits after a provenance check (same set hash, batch ids and budget;
                  both earlier dense scorings must agree); the two anchors' own results are reused when their mask
                  hashes match.  EXPLORATORY: these sets were inspected before.
calibrate         Replays the threshold-aware gate on the pruning-time pool at every margin (tolerance 3 - m) for
                  every development state with that state's own live question order, and applies the recorded rule.
compare           A (top-1), B (top-2), C (top-2 + locked margin): one table of the finished arms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import load_config, resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("gradient_boundary")


def _gb(cfg: dict) -> dict:
    return cfg["gradient_boundary"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _gb(cfg)["out_dir"])


def _attempts(run_dir: Path) -> dict[int, dict]:
    from .gradient_audit import attempts

    return {a["iteration"]: a for a in attempts(run_dir)}


def _setup_model(rc: dict, dense_copy: bool = False):
    import torch

    from .config import data_dir
    from .data.prepare import PreparedData
    from .gradient_live import load_dense_copy
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pruning.masks import DenseStore

    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    tok = load_tokenizer(rc)
    model = load_model(rc)
    model.eval()
    units = build_units(model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))
    dense = DenseStore(units)
    dm = load_dense_copy(rc, model, units, dense) if dense_copy else None
    torch.manual_seed(0)
    return data, tok, model, units, dense, dm


# ============================================================================ 1. recover the states
def recover(cfg: dict) -> Path:
    from .fixed_state import _geometry
    from .gradient_audit import run_paths
    from .gradient_live import build_trial
    from .pruning.masks import PruneState
    from .pruning.profile import load_profile
    from .ranker import canon, mask_hash

    gb = _gb(cfg)
    rc, _ = run_paths(cfg)
    src = resolve_path(cfg, gb["source_run"])
    att = _attempts(src)
    out = out_dir(cfg)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    data, tok, model, units, dense, _ = _setup_model(rc)
    dev = next(model.parameters()).device
    mode = rc["pruning"]["mode"]
    geo = _geometry(rc, units[0].channels)
    saved = {}  # state id -> mask file (run's retained masks + frozen copies of the anchors)
    for p in (src / "masks").glob("*.safetensors"):
        saved[p.stem] = p
    for a in gb.get("anchors", []):
        fr = read_json(resolve_path(cfg, a["frozen"]))
        if sha256_file(Path(fr["mask_file"])) != fr["mask_sha256"]:
            raise RuntimeError(f"anchor {fr['state_id']}: frozen mask file changed")
        saved[fr["state_id"]] = Path(fr["mask_file"])
    targets = [int(t) for t in gb["attempts"]]
    last = max(targets)
    for t in range(1, last + 1):
        if att[t]["gate_decision"] != "accept":
            raise RuntimeError(f"attempt {t} of the source run was not accepted")
    # replay from the nearest saved ancestor at or before the first target; verify every step against the journal
    start = max((t for t in range(1, targets[0] + 1) if att[t]["cand_id"] in saved), default=0)
    st = PruneState(units, mode)
    if start:
        st.load(saved[att[start]["cand_id"]])
    if mask_hash(st) != (att[start]["trial_mask_hash"] if start else att[1]["parent_mask_hash"]):
        raise RuntimeError(f"saved mask of attempt {start} does not match the journal")
    states, checks = [], []

    def record(name, attempt, sid, stt, how, extra=None):
        p = out / "masks" / f"{sid}.safetensors"
        stt.save(p)
        rec = {"name": name, "attempt": attempt, "state_id": sid, "mask_hash": mask_hash(stt), "mask_file": str(p),
               "mask_sha256": sha256_file(p), "mlp_sparsity": stt.global_sparsity(), "provenance": how, **(extra or {})}
        states.append(rec)
        return rec

    for t in range(start, last + 1):
        if t > start:  # replay attempt t's recorded action from its parent with the run's persisted profile
            a = att[t]
            if mask_hash(st) != a["parent_mask_hash"]:
                raise RuntimeError(f"replay diverged before attempt {t}")
            act, _, _ = load_profile(src / "profiles" / f"iter-{int(a.get('profile_attempt') or t):06d}.safetensors", dev)
            action = {u: int(b) for u, b in a["chosen"]["action"].items()}
            if canon(action) != a["chosen"]["action_key"]:
                raise RuntimeError(f"attempt {t}: decisions do not match the recorded action")
            build_trial(st, dense, action, act, rc, geo)
            ok = mask_hash(st) == a["trial_mask_hash"]
            chk = {"attempt": t, "state_id": a["cand_id"], "replay_matches_journal_mask_hash": ok}
            if a["cand_id"] in saved:  # replayed AND saved: the saved file must hold the same masks
                ref = PruneState(units, mode)
                ref.load(saved[a["cand_id"]])
                chk["saved_mask_equals_replay"] = mask_hash(ref) == mask_hash(st)
                ok &= chk["saved_mask_equals_replay"]
            checks.append(chk)
            if not ok:
                raise RuntimeError(f"attempt {t}: recovered mask does not match the journal / saved mask: {chk}")
        if t in targets:
            a = att[t]
            how = "saved mask (verified against the journal)" if a["cand_id"] in saved and t == start else \
                  ("replayed + equal to the saved mask" if a["cand_id"] in saved else "replayed from the run's profiles")
            record(f"attempt {t}", t, a["cand_id"], st.clone(), how,
                   {"rho_after_journal": a["rho_after"], "order_seed": a["gate"]["order_seed"], "live_decision": "accept",
                    "live_questions": a["gate"]["questions"]})
    # ---- the fallback candidate: a recorded shortlist action applied to its recorded parent
    r2 = gb.get("rank2")
    if r2:
        a = att[int(r2["attempt"])]
        parent = PruneState(units, mode)
        parent.load(saved[a["parent_id"]])
        if mask_hash(parent) != a["parent_mask_hash"]:
            raise RuntimeError("fallback parent mask does not match the journal")
        sl = next(s for s in a["shortlist"] if s["rank"] == int(r2["rank"]))
        if sl["action_key"] != r2["action_key"]:
            raise RuntimeError(f"recorded rank-{r2['rank']} action {sl['action_key']} != configured {r2['action_key']}")
        act, _, _ = load_profile(src / "profiles" / f"iter-{int(a.get('profile_attempt') or a['iteration']):06d}.safetensors", dev)
        build_trial(parent, dense, sl["action"], act, rc, geo)
        h = mask_hash(parent)
        diag = resolve_path(cfg, gb["diagnostic"])
        D = read_json(diag / "diagnostic.json")
        drow = next(c for c in D["candidates"] if c["rank"] == int(r2["rank"]))
        import sqlite3

        con = sqlite3.connect(f"file:{(diag / 'pairs.sqlite').as_posix()}?mode=ro", uri=True)
        n_keys = con.execute("SELECT COUNT(*) FROM kv WHERE k LIKE ?", (f"pair|{h}|%",)).fetchone()[0]
        con.close()
        if n_keys == 0 or drow["action_key"] != r2["action_key"]:
            raise RuntimeError("rebuilt fallback candidate's mask hash is not in the stall diagnostic's cache")
        checks.append({"attempt": a["iteration"], "rank": r2["rank"], "mask_hash_in_diagnostic_cache": True,
                       "diagnostic_batches": n_keys})
        record(f"attempt {a['iteration']} rank {r2['rank']}", a["iteration"], f"A{a['iteration']}r{r2['rank']}-{h[:12]}",
               parent, "rebuilt from the recorded parent + rank-2 action; mask hash found in the stall diagnostic",
               {"parent_state_id": a["parent_id"], "order_seed": D["order_seed"], "live_decision": drow["decision"],
                "live_questions": drow["questions"], "rank": r2["rank"], "action_key": r2["action_key"]})
    res = {"source_run": str(src), "states": states, "checks": checks, "recovered_at": time.time()}
    atomic_write_json(out / "states.json", res)
    return out / "states.json"


# ============================================================================ 2a. the calibration plan (before scoring)
def margin_plan(cfg: dict) -> Path:
    out = out_dir(cfg)
    st = read_json(out / "states.json")
    p = out / "calibration_plan.json"
    gb = _gb(cfg)
    plan = {"margins_pts": [100 * float(m) for m in gb["margins"]], "rule": gb["rule"], "audit_criterion": gb["audit_criterion"],
            "gate_definition": gb["gate_definition"], "if_inconclusive": gb["if_inconclusive"],
            "states": [s["state_id"] for s in st["states"]], "recorded_at": time.time()}
    if (out / "boundary.json").exists() and not p.exists():
        raise SystemExit("development states were already scored; the plan must be recorded first")
    if p.exists():
        old = read_json(p)
        if {k: old[k] for k in plan if k != "recorded_at"} != {k: plan[k] for k in plan if k != "recorded_at"}:
            raise SystemExit(f"{p} already records a different plan; refusing to overwrite")
        return p
    atomic_write_json(p, plan)
    return p


# ============================================================================ 1b. boundary audit (materialised, exploratory)
def _dense_reference(cfg: dict) -> tuple[dict, dict]:
    """(set sha16, category, start, size) -> dense record from the earlier materialised audits; checks agreement."""
    import sqlite3

    ref, info = {}, {"sources": [], "batches": 0, "agree_batches": 0, "items_compared": 0, "items_equal": 0}
    for p in _gb(cfg)["dense_reference"]:
        path = resolve_path(cfg, p)
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        n = 0
        for k, v in con.execute("SELECT k, v FROM kv"):
            role, sha16, tag, cat, i, size = k.split("|")
            rec = json.loads(v)
            key = (sha16, cat, int(i), int(size))
            d = {"pred": rec["dense"]["pred"], "correct": rec["dense"]["correct"], "budget": int(rec["budget"])}
            if key in ref:
                old = ref[key]
                info["batches"] += 1
                info["items_compared"] += len(d["correct"])
                info["items_equal"] += sum(old["correct"][q] == d["correct"][q] and old["pred"][q] == d["pred"][q] for q in d["correct"])
                info["agree_batches"] += old["correct"] == d["correct"] and old["pred"] == d["pred"] and old["budget"] == d["budget"]
            else:
                ref[key] = d
            n += 1
        con.close()
        info["sources"].append({"path": str(path), "batches": n})
    return ref, info


def boundary_audit(cfg: dict) -> Path:
    import torch

    from .adaptive.evaluator import question_order
    from .eval.mmlu import MMLUScorer
    from .gradient_audit import _boot_stats, _clock, _read_jsonl, size_check
    from .gradient_audit import run_paths
    from .pruning.masks import PruneState
    from .pruning.materialize import materialize
    from .utils.cache import KVCache
    from .utils.gpu import free_memory

    gb = _gb(cfg)
    out = out_dir(cfg)
    if not (out / "calibration_plan.json").exists():
        raise SystemExit("record the calibration plan first: capprune gradient-boundary margin-plan --config ...")
    rc, _ = run_paths(cfg)
    S = read_json(out / "states.json")
    inv = read_json(resolve_path(cfg, gb["audit_inventory"]))
    sets = {r: inv["audit_sets"][r] for r in gb["audit_roles"]}
    ref, ref_info = _dense_reference(cfg)
    if ref_info["items_compared"] and ref_info["items_equal"] != ref_info["items_compared"]:
        raise RuntimeError(f"the two earlier dense scorings disagree: {ref_info}")
    anchors = {}
    for a in gb.get("anchors", []):
        am = read_json(resolve_path(cfg, a["audit"]) / "audit_materialized.json")
        anchors[am["provenance"]["frozen"]["state_id"]] = (resolve_path(cfg, a["audit"]), am)
    au = (cfg.get("gradient_live") or {}).get("audit") or {}
    tt = rc["testing"]
    shots, max_len, mbt = int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768))
    bs, seed = int(au.get("batch_size", 100)), int(au.get("order_seed", 2026))
    tol = float(rc["gate"]["capability"]["tolerance"])
    data, tok, model, units, dense, _ = _setup_model(rc)
    dev = next(model.parameters()).device
    cache = KVCache(out / "scores.sqlite")
    results, per_item = [], {}
    for s in S["states"]:
        t0 = time.perf_counter()
        st = PruneState(units, rc["pruning"]["mode"])
        st.load(s["mask_file"])
        st.apply(dense)
        mat, minfo = materialize(model, st, rc)
        size = size_check(rc, model, model, mat, st, units)  # masked model has the dense parameter count
        evals, secs, reused = 0, 0.0, 0
        anc = anchors.get(s["state_id"])
        anc_cache = None
        if anc and anc[1]["provenance"]["frozen"]["mask_sha256"] and sha256_file(Path(anc[1]["provenance"]["frozen"]["mask_file"])) == anc[1]["provenance"]["frozen"]["mask_sha256"]:
            import sqlite3

            anc_cache = sqlite3.connect(f"file:{(anc[0] / 'scores.sqlite').as_posix()}?mode=ro", uri=True)
        sc = MMLUScorer(mat, tok, data.mmlu_fewshot, shots, max_len, {}, "boundary", mbt)
        res = {**s, "size": size, "sets": {}}
        for role, meta in sets.items():
            rows = _read_jsonl(Path(meta["file"]))
            by_cat: dict[str, list[dict]] = {}
            for r in rows:
                by_cat.setdefault(r["category"], []).append(r)
            items = []
            for cat in sorted(by_cat):
                rmap = {r["qid"]: r for r in by_cat[cat]}
                order = question_order(list(rmap), seed, cat)
                for i in range(0, len(order), bs):
                    ids = order[i : i + bs]
                    d = ref[(meta["sha256"][:16], cat, i, len(ids))]
                    if set(d["correct"]) != set(ids):
                        raise RuntimeError("dense reference batch holds different questions")
                    key = f"{s['state_id']}|{meta['sha256'][:16]}|{cat}|{i}|{len(ids)}"
                    m = cache.get(key)
                    if m is None and anc_cache is not None:  # the anchor's own materialised answers, same batch
                        tag = anc[1]["provenance"]["frozen"]["mask_sha256"][:16]
                        for arole in ("fresh", "comparison", "exploratory"):
                            row = anc_cache.execute("SELECT v FROM kv WHERE k=?", (f"{arole}|{meta['sha256'][:16]}|{tag}|{cat}|{i}|{len(ids)}",)).fetchone()
                            if row:
                                v = json.loads(row[0])
                                if int(v["budget"]) == d["budget"] and v["dense"]["correct"] == d["correct"]:
                                    m = {"pred": v["materialized"]["pred"], "correct": v["materialized"]["correct"], "s": 0.0,
                                         "budget": d["budget"], "reused": True}
                                break
                        if m is not None:
                            reused += len(ids)
                    if m is None:
                        sc.budget = d["budget"]
                        tq = _clock(dev)
                        r_ = sc.score([rmap[q] for q in ids])
                        dt = _clock(dev) - tq
                        if sc.budget != d["budget"]:
                            raise RuntimeError("OOM changed the batch layout; the dense reference no longer matches")
                        m = {"pred": {q: int(x["pred"]) for q, x in r_.items()},
                             "correct": {q: int(x["correct"]) for q, x in r_.items()}, "s": dt, "budget": d["budget"]}
                        cache.put(key, m)
                        evals += len(ids)
                        secs += dt
                    items += [{"qid": q, "category": cat, "dense": int(d["correct"][q]), "pruned": int(m["correct"][q]),
                               "pruned_pred": int(m["pred"][q])} for q in ids]
            per = {c: np.array([x["dense"] - x["pruned"] for x in items if x["category"] == c], dtype=np.int8) for c in by_cat}
            stats = _boot_stats(per, int(au.get("bootstrap", 10000)), 0, tol)
            mx = max(by_cat, key=lambda c: stats[c]["ub_simultaneous"])
            res["sets"][role] = {"name": meta["name"], "questions": len(items), "stats": stats, "max_ub_category": mx,
                                 "max_ub": stats[mx]["ub_simultaneous"]}
            per_item[(s["state_id"], role)] = items
        if anc_cache is not None:
            anc_cache.close()
        res["cost"] = {"materialized_evaluations_run": evals, "materialized_scoring_s": secs, "answers_reused": reused,
                       "dense_evaluations_run": 0, "wall_s": time.perf_counter() - t0}
        res["meets_audit"] = all(v["max_ub"] <= tol + 1e-12 for v in res["sets"].values())
        results.append(res)
        log.info("%s: %.3f %% params, worst UB %s, evals %d (reused %d)", s["name"], 100 * size["fraction_of_total_parameters_removed"],
                 {r: round(100 * v["max_ub"], 2) for r, v in res["sets"].items()}, evals, reused)
        del mat
        free_memory()
    cache.close()
    # ---- the fallback candidate vs. its parent (paired on identical items)
    cmp_ = None
    r2 = next((r for r in results if r.get("rank")), None)
    if r2:
        par = next(r for r in results if r["state_id"] == r2["parent_state_id"])
        cmp_ = {"candidate": r2["state_id"], "parent": par["state_id"], "sets": {}}
        for role in sets:
            a, b = per_item[(par["state_id"], role)], per_item[(r2["state_id"], role)]
            cats = sorted({x["category"] for x in a})
            x = {c: np.array([pa["pruned"] - pb["pruned"] for pa, pb in zip(a, b) if pa["category"] == c], dtype=np.int8) for c in cats}
            assert all(pa["qid"] == pb["qid"] for pa, pb in zip(a, b))
            d = _boot_stats(x, int(au.get("bootstrap", 10000)), 1, tol)
            cmp_["sets"][role] = {c: {"delta_loss": d[c]["loss"], "ci95": d[c]["ci95"], "parent_right_only": d[c]["broken"],
                                      "candidate_right_only": d[c]["fixed"]} for c in [*cats, "overall"]}
    B = {"states": results, "dense_reference": ref_info, "rank2_vs_parent": cmp_, "label": "exploratory: audit sets inspected before",
         "protocol": {"shots": shots, "max_seq_len": max_len, "max_batch_tokens": mbt, "batch": bs, "order_seed": seed}}
    with open(out / "per_item.jsonl", "w", encoding="utf-8") as f:
        for (sid, role), items in per_item.items():
            for x in items:
                f.write(json.dumps({"state_id": sid, "role": role, **x}) + "\n")
    atomic_write_json(out / "boundary.json", B)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out / "boundary.json"


# ============================================================================ 2b. calibrate the margin
def calibrate(cfg: dict) -> Path:
    from .adaptive.evaluator import evaluate_candidate
    from .adaptive.live import domain_summary
    from .adaptive.policies import make_policy
    from .gradient_audit import ReadOnlyCache, run_paths
    from .gradient_live import PairedScorer, PairedSource, gate_pool, make_gate
    from .pruning.masks import PruneState
    from .ranker import mask_hash
    from .utils.cache import KVCache

    gb = _gb(cfg)
    out = out_dir(cfg)
    plan = read_json(out / "calibration_plan.json")
    B = read_json(out / "boundary.json")
    rc, _ = run_paths(cfg)
    data, tok, model, units, dense, dm = _setup_model(rc, dense_copy=True)
    tt = rc["testing"]
    scorer = PairedScorer(dm, model, tok, data.mmlu_fewshot, int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)),
                          int(tt.get("max_batch_tokens", 32768)))
    pool, _ = gate_pool(rc, data)
    src_run = resolve_path(cfg, gb["source_run"])
    ro = [ReadOnlyCache(src_run / "caches" / "eval.sqlite"), ReadOnlyCache(resolve_path(cfg, gb["diagnostic"]) / "pairs.sqlite")]
    cache = KVCache(out / "gate_pairs.sqlite")
    margins = [float(m) for m in gb["margins"]]
    rows = []
    for s in B["states"]:
        st = PruneState(units, rc["pruning"]["mode"])
        st.load(s["mask_file"])
        st.apply(dense)
        h = mask_hash(st)
        if h != s["mask_hash"]:
            raise RuntimeError(f"{s['name']}: mask hash changed")
        r = {"state_id": s["state_id"], "name": s["name"], "mlp_sparsity": s["mlp_sparsity"], "meets_audit": s["meets_audit"],
             "order_seed": s["order_seed"], "gate": {}}
        for m in margins:
            gate = make_gate(rc, pool, margin=m)
            e0 = dict(scorer.evals)
            srcq = PairedSource(gate.pool, int(s["order_seed"]), h, scorer, cache, None, ro_caches=ro)
            run = evaluate_candidate(srcq, make_policy(gate.policy_name), gate.plan, f"cal-{s['state_id']}-{m}")
            r["gate"][f"{100 * m:.1f}"] = {"decision": run.decision, "questions": run.questions, "domains": domain_summary(run),
                                         "evaluations_run": (scorer.evals["dense"] - e0["dense"]) + (scorer.evals["trial"] - e0["trial"]),
                                         "pairs_from_cache": srcq.replayed_q}
        r["live_decision_reproduced_at_margin_0"] = (r["gate"]["0.0"]["decision"] == s["live_decision"]
                                                     and r["gate"]["0.0"]["questions"] == s["live_questions"])
        rows.append(r)
        log.info("%s: %s", s["name"], {k: v["decision"] for k, v in r["gate"].items()})
    for c in ro:
        c.close()
    cache.close()
    res = apply_rule(rows, margins)
    res.update(plan=plan, states=rows, label=B["label"],
               cost={"evaluations_run": int(sum(v["evaluations_run"] for r in rows for v in r["gate"].values())),
                     "pairs_from_cache": int(sum(v["pairs_from_cache"] for r in rows for v in r["gate"].values()))})
    atomic_write_json(out / "calibration.json", res)
    atomic_write_text(out / "report.md", render(cfg))
    return out / "report.md"


def apply_rule(rows: list[dict], margins: list[float]) -> dict:
    """Smallest margin m such that every development state passing the gate at 3 - m meets the audit criterion, with
    at least one pruned state passing.  None -> inconclusive."""
    table = []
    for m in margins:
        k = f"{100 * m:.1f}"
        passing = [r for r in rows if r["gate"][k]["decision"] == "accept"]
        bad = [r["name"] for r in passing if not r["meets_audit"]]
        good = [r["name"] for r in passing if r["meets_audit"]]
        blocked_ok = [r["name"] for r in rows if r["gate"][k]["decision"] != "accept" and r["meets_audit"]]
        table.append({"margin_pts": 100 * m, "pass_gate": len(passing), "pass_gate_and_meet_audit": good,
                      "pass_gate_but_fail_audit": bad, "blocked_but_meet_audit": blocked_ok,
                      "qualifies": not bad and len(passing) >= 1})
    chosen = next((t for t in table if t["qualifies"]), None)
    return {"table": table, "chosen_margin_pts": chosen["margin_pts"] if chosen else None,
            "conclusive": chosen is not None,
            "support": None if chosen is None else {"states_passing_and_meeting_audit": len(chosen["pass_gate_and_meet_audit"]),
                                                    "states_blocked": len(rows) - chosen["pass_gate"],
                                                    "states_blocked_that_meet_audit": len(chosen["blocked_but_meet_audit"]),
                                                    "states_total": len(rows)}}


# ============================================================================ 4. compare the live policies
def _arm(cfg_path: Path, label: str) -> dict:
    from .gradient_audit import _cost, analysis_dir, attempts, run_paths

    acfg = load_config(cfg_path)
    rc, rd = run_paths(acfg)
    A = analysis_dir(acfg)
    opt = lambda p: read_json(p) if p.exists() else None  # noqa: E731
    att = attempts(rd)
    fr = opt(A / "frozen" / "frozen.json")
    mat = opt(A / "audit_materialized" / "audit_materialized.json")
    diag = opt(A / "diagnostic" / "diagnostic.json")
    summ = opt(rd / "final" / "summary.json")
    S = None
    try:
        from .utils.checkpoint import CheckpointStore

        S = CheckpointStore(rd).load_latest()[0].get("search")
    except Exception:
        pass
    items = {}
    for p in (A / "audit_materialized").glob("per_item-*.jsonl") if (A / "audit_materialized").exists() else []:
        items[p.stem.split("-", 1)[1]] = {json.loads(l)["qid"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()}
    rs = rc["ranked_search"]
    return {"label": label, "config": str(cfg_path), "run_dir": str(rd), "analysis": str(A), "attempts": att, "frozen": fr,
            "materialized": mat, "diagnostic": diag, "summary": summ, "search": S, "items": items, "cost": _cost(att),
            "settings": {"fallback_k": int(rs.get("fallback_k", 1)), "gate_margin_pts": 100 * float(rs.get("gate_margin", 0) or 0),
                         "max_proposals": rs["max_proposals"], "question_cap": rs.get("question_cap"),
                         "gate_tolerance_pts": 100 * float(rc["gate"]["capability"]["tolerance"])}}


def _paired_delta(a_items: dict, b_items: dict, B: int = 10000, seed: int = 3) -> dict:
    """B minus A (loss, pts) on identical items: x = a_correct - b_correct, stratified by category."""
    from .gradient_audit import _boot_stats

    q = sorted(set(a_items) & set(b_items))
    cats = sorted({a_items[k]["category"] for k in q})
    x = {c: np.array([a_items[k]["materialized_correct"] - b_items[k]["materialized_correct"] for k in q
                      if a_items[k]["category"] == c], dtype=np.int8) for c in cats}
    s = _boot_stats(x, B, seed, 1.0)
    return {c: {"delta": s[c]["loss"], "ci95": s[c]["ci95"]} for c in [*cats, "overall"]}


def compare(cfg: dict) -> Path:
    gc = cfg["gradient_compare"]
    out = resolve_path(cfg, gc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    arms = {a["name"]: _arm(resolve_path(cfg, a["config"]), a["label"]) for a in gc["arms"]}
    deltas = {}
    for x, y in (("A", "B"), ("B", "C")):
        if x in arms and y in arms:
            deltas[f"{y}-{x}"] = {role: _paired_delta(arms[x]["items"][role], arms[y]["items"][role])
                                  for role in arms[x]["items"] if role in arms[y]["items"]}
    cal = read_json(resolve_path(cfg, gc["calibration"])) if gc.get("calibration") and resolve_path(cfg, gc["calibration"]).exists() else None
    bnd = read_json(resolve_path(cfg, gc["boundary"])) if gc.get("boundary") and resolve_path(cfg, gc["boundary"]).exists() else None
    S = {"arms": {k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in arms.items()}, "deltas": deltas,
         "calibration": cal, "boundary": bnd, "fresh_search": gc.get("fresh_search")}
    atomic_write_json(out / "comparison.json", S)
    atomic_write_text(out / "report.md", render_compare(S))
    return out / "report.md"


# ============================================================================ rendering
def _pts(x):
    return "n/a" if x is None else f"{100 * x:+.2f}"


def render(cfg: dict) -> str:
    out = out_dir(cfg)
    B = read_json(out / "boundary.json")
    st = read_json(out / "states.json")
    cal = read_json(out / "calibration.json") if (out / "calibration.json").exists() else None
    plan = read_json(out / "calibration_plan.json")
    L = ["# Compression boundary and gate-margin calibration (EXPLORATORY: audit sets inspected before)", "",
         "## States", ""]
    for c in st["checks"]:
        L.append(f"* check: {c}")
    roles = list(B["states"][0]["sets"])
    L += ["", "Dense reference: " + f"{B['dense_reference']}", "",
          "| state | MLP sparsity | params removed (materialised) | " + " | ".join(
              f"{B['states'][0]['sets'][r]['name']}: worst / largest simultaneous UB" for r in roles)
          + " | meets audit (all UB <= 3) | evaluations run (reused) | GPU s |", "|---|---|---|" + "---|" * len(roles) + "---|---|---|"]
    for s in B["states"]:
        cells = []
        for r in roles:
            v = s["sets"][r]
            c = v["stats"]["conclusion"]
            cells.append(f"{c['worst_category']} {_pts(c['worst_loss'])} / {v['max_ub_category']} {_pts(v['max_ub'])}")
        L.append(f"| {s['name']} (`{s['state_id']}`) | {100 * s['mlp_sparsity']:.2f} % | {s['size']['parameters_removed']:,} "
                 f"({100 * s['size']['fraction_of_total_parameters_removed']:.2f} %) | " + " | ".join(cells)
                 + f" | {s['meets_audit']} | {s['cost']['materialized_evaluations_run']:,} ({s['cost']['answers_reused']:,}) | "
                 f"{s['cost']['materialized_scoring_s']:.1f} |")
    L += ["", "Per-category paired loss (pts) [95 % CI]:", ""]
    for r in roles:
        cats = [k for k in B["states"][0]["sets"][r]["stats"] if k not in ("overall", "conclusion")]
        L += [f"**{B['states'][0]['sets'][r]['name']}**", "", "| state | " + " | ".join(cats) + " | overall |", "|---|" + "---|" * (len(cats) + 1)]
        for s in B["states"]:
            v = s["sets"][r]["stats"]
            L.append(f"| {s['name']} | " + " | ".join(f"{_pts(v[c]['loss'])} [{_pts(v[c]['ci95'][0])}, {_pts(v[c]['ci95'][1])}] "
                                                     f"UB {_pts(v[c]['ub_simultaneous'])}" for c in cats)
                     + f" | {_pts(v['overall']['loss'])} |")
        L.append("")
    if B.get("rank2_vs_parent"):
        cp = B["rank2_vs_parent"]
        L += [f"## Rank-2 candidate `{cp['candidate']}` vs. its parent `{cp['parent']}` (paired, same items)", "",
              "Δ = candidate loss - parent loss (pts) [95 % CI]; items only the parent / only the candidate answers correctly.", ""]
        for r, v in cp["sets"].items():
            L.append(f"* {r}: " + "; ".join(f"{c} {_pts(x['delta_loss'])} [{_pts(x['ci95'][0])}, {_pts(x['ci95'][1])}] "
                                            f"({x['parent_right_only']}/{x['candidate_right_only']})" for c, x in v.items()))
    L += ["", "## Margin calibration", "", f"Plan (recorded {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan['recorded_at']))}, "
          f"before the new states were scored): rule = {plan['rule']}; audit criterion = {plan['audit_criterion']}; gate = "
          f"{plan['gate_definition']}; if inconclusive: {plan['if_inconclusive']}.", ""]
    if cal:
        ms = [f"{m:.1f}" for m in plan["margins_pts"]]
        L += ["| state | meets audit | " + " | ".join(f"gate @ {3 - float(m):.1f} pts" for m in ms) + " |", "|---|---|" + "---|" * len(ms)]
        for r in cal["states"]:
            L.append(f"| {r['name']} | {r['meets_audit']} | " + " | ".join(f"{r['gate'][m]['decision']} ({r['gate'][m]['questions']:,})"
                                                                         for m in ms) + " |")
        L += ["", "| margin (pts) | pass gate | pass gate + meet audit | pass gate but fail audit | qualifies |", "|---|---|---|---|---|"]
        for t in cal["table"]:
            L.append(f"| {t['margin_pts']:.1f} | {t['pass_gate']} | {', '.join(t['pass_gate_and_meet_audit']) or '-'} | "
                     f"{', '.join(t['pass_gate_but_fail_audit']) or '-'} | {t['qualifies']} |")
        L += ["", f"**Chosen margin: {cal['chosen_margin_pts']} pts** (conclusive: {cal['conclusive']}; support: {cal['support']}). "
              f"Live decisions reproduced at margin 0: {all(r['live_decision_reproduced_at_margin_0'] for r in cal['states'])}. "
              f"Calibration cost: {cal['cost']}."]
    return "\n".join(L) + "\n"


def render_compare(S: dict) -> str:
    arms = S["arms"]
    L = ["# Live policies: A gradient top-1 | B gradient top-2 | C gradient top-2 + locked stricter gate", "",
         "All arms: dense start, seed 0, the ranker's action generator, 1,280-channel step, caps of 30 gated proposals and "
         "100,000 paired gate questions (both candidates of a top-2 step count). Audit: materialised checkpoint vs. dense on "
         "the already-inspected sets (EXPLORATORY; no untouched set exists); simultaneous UB = Bonferroni over 4 categories.", ""]
    names = [k for k in ("A", "B", "C") if k in arms]

    def row(title, f):
        L.append(f"| {title} | " + " | ".join(f(arms[k]) for k in names) + " |")

    def set_stat(a, role, what):
        m = a.get("materialized")
        if not m or role not in m["sets"]:
            return "n/a"
        st = m["sets"][role]["stats"]["materialized"]
        c = st["conclusion"]
        if what == "worst":
            w = c["worst_category"]
            return f"{w} {_pts(st[w]['loss'])} [{_pts(st[w]['ci95'][0])}, {_pts(st[w]['ci95'][1])}]"
        cats = [k for k in st if k not in ("overall", "conclusion")]
        mx = max(cats, key=lambda k: st[k]["ub_simultaneous"])
        return f"{mx} {_pts(st[mx]['ub_simultaneous'])} -> {'supported' if c['simultaneous_support'] else 'NOT supported'}"

    L += ["| | " + " | ".join(f"{k}: {arms[k]['label']}" for k in names) + " |", "|---|" + "---|" * len(names)]
    row("settings", lambda a: f"top-{a['settings']['fallback_k']}, gate tolerance {a['settings']['gate_tolerance_pts'] - a['settings']['gate_margin_pts']:.1f} pts")
    row("stop reason", lambda a: f"{(a['frozen'] or {}).get('stop_reason')} after {len(a['attempts'])} proposals")
    row("accepted steps (rank-2 rescues)", lambda a: f"{(a['frozen'] or {}).get('accepts')} ({((a['search'] or {}).get('fallback') or {}).get('rescued', 0)})")
    row("fallback triggers / rescued / failed", lambda a: (lambda f: f"{f.get('triggers', 0)} / {f.get('rescued', 0)} / {f.get('failed', 0)}")(((a['search'] or {}).get('fallback') or {})))
    row("verified params removed (materialised)", lambda a: f"{a['materialized']['size']['parameters_removed']:,} ({100 * a['materialized']['size']['fraction_of_total_parameters_removed']:.2f} %)" if a.get("materialized") else "n/a")
    row("MLP sparsity", lambda a: f"{100 * a['frozen']['mlp_sparsity']:.2f} %" if a.get("frozen") else "n/a")
    roles = sorted({r for a in arms.values() for r in ((a.get("materialized") or {}).get("sets") or {})})
    for role in roles:
        nm = next(a["materialized"]["sets"][role]["name"] for a in arms.values() if a.get("materialized") and role in a["materialized"]["sets"])
        row(f"{nm}: worst-category loss [95 % CI]", lambda a, r=role: set_stat(a, r, "worst"))
        row(f"{nm}: largest simultaneous UB", lambda a, r=role: set_stat(a, r, "ub"))
    row("paired gate questions", lambda a: f"{a['cost']['paired_questions']:,}")
    row("item-model evaluations run (dense + candidate)", lambda a: f"{a['cost']['dense_evaluations_run'] + a['cost']['trial_evaluations_run']:,}")
    row("fallback extra cost (questions / evaluations / s)", lambda a: (lambda f: f"{f.get('extra_questions', 0):,} / {f.get('extra_evaluations', 0):,} / {f.get('extra_seconds', 0.0):.0f}")(((a['search'] or {}).get('fallback') or {})))
    row("end-to-end search s (sum of attempts)", lambda a: f"{a['cost']['search_s']:.0f}")
    row("run wall s (incl. model load)", lambda a: f"{(a['summary'] or {}).get('wall_seconds', 0):.0f}")
    row("first stall: passing ranks of the full shortlist", lambda a: (lambda d: "n/a" if not d or not d.get("candidates") else f"{d['passing_ranks'] or 'none'} of {d['n_candidates']} (rule: {d['recommendation']['rule']})")(a.get("diagnostic")))
    if S["deltas"]:
        L += ["", "## Paired differences on identical audit items (loss of the later arm minus the earlier, pts [95 % CI])", ""]
        for k, v in S["deltas"].items():
            for role, d in v.items():
                L.append(f"* {k}, {role}: " + "; ".join(f"{c} {_pts(x['delta'])} [{_pts(x['ci95'][0])}, {_pts(x['ci95'][1])}]" for c, x in d.items()))
    L += ["", "## Attempt logs", ""]
    for k in names:
        a = arms[k]
        L.append(f"**{k}** ({a['run_dir']}):")
        for t in a["attempts"]:
            fb = " FALLBACK QUEUED" if t.get("fallback_triggered") else (" RESCUED" if t.get("rescued") else "")
            L.append(f"* {t['iteration']}: rank {(t.get('chosen') or {}).get('rank', 1)} {(t.get('chosen') or {}).get('kind')} "
                     f"{(t.get('chosen') or {}).get('layers')} -> {t['gate_decision']} ({t['gate']['questions']:,} q, cum "
                     f"{t['questions_cumulative']:,}; MLP {100 * t['rho_after']:.2f} %){fb}")
        L.append("")
    return "\n".join(L) + "\n"
