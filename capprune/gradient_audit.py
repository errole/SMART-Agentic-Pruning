"""Analysis of a finished gradient-only live run (``capprune gradient-live audit-data|freeze|diagnose|audit|report``).

audit-data  Usage inventory of every prepared MMLU split (which earlier pilot, gate-tuning, ranker or diagnostic
            artifact touched each question id) and the fresh audit set.  The default fresh source is MMLU-CF's
            public validation split (microsoft/MMLU-CF, pinned revision, checksums verified): 4-choice questions
            in 14 fields that map one-to-one onto MMLU's official subcategory -> category table.  Exact duplicates
            of any prepared MMLU question are excluded.
freeze      After the run's ranked_search stage is done: record the final checkpoint (state id, mask SHA-256,
            sparsity, stop reason) and copy its mask.  diagnose and audit verify that copy and that the run's
            latest checkpoint still points at the same state.
diagnose    Offline, from the stalled parent (= the frozen checkpoint, since the failed top-1 trial was rolled
            back): the run's own saved profile rebuilds the identical ordered shortlist; EVERY candidate is gated
            with the same gate and the stalled attempt's question order.  Dense answers are shared across
            candidates batch by batch; the top-1 trial's pairs are read from the run's cache (read-only), so its
            decision must reproduce the live one.  Costs are reported separately from the live run.
audit       The frozen checkpoint vs. the dense model on the fresh set (and, labelled exploratory, on a reused set):
            the live gate's 0-shot prompt and scorer, each category in a seeded order in the gate's batch size,
            dense and pruned scored on identical batches.  Paired losses with stratified bootstrap intervals,
            parameters physically removed (materialised model), MLP sparsity.
report      Markdown report of all of the above (CPU).
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import data_dir, resolve_path, run_config, run_id, runs_dir
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("gradient_audit")

# MMLU's official grouping (hendrycks/test categories.py: subcategory -> category), applied to MMLU-CF's fields.
MMLU_CF_CATEGORY = {
    "Biology": "STEM", "Chemistry": "STEM", "Computer_Science": "STEM", "Engineering": "STEM", "Math": "STEM",
    "Physics": "STEM", "History": "Humanities", "Law": "Humanities", "Philosophy": "Humanities",
    "Economics": "Social Sciences", "Psychology": "Social Sciences", "Business": "Other", "Health": "Other",
    "Other": "Other"}
PROMPT_SUBJECT = {"Other": "miscellaneous"}  # MMLU's own subject name for its "other" subcategory
AUDIT_ROLES = ("fresh", "comparison", "exploratory")  # comparison / exploratory = questions already inspected


def _gl(cfg: dict) -> dict:
    return cfg.get("gradient_live") or {}


def analysis_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _gl(cfg).get("out_dir", "gradient_live_analysis"))


def run_paths(cfg: dict) -> tuple[dict, Path]:
    mx = cfg.get("matrix", {})
    method = _gl(cfg).get("method") or mx["methods"][0]
    target, seed = float(mx.get("targets", [0.0])[0]), int(mx.get("seeds", [0])[0])
    return run_config(cfg, method, target, seed), runs_dir(cfg) / run_id(method, target, seed)


def attempts(run_dir: Path) -> list[dict]:
    """The last journal record of every attempt (a replayed attempt appends a second, identical record)."""
    last = {}
    for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return [last[k] for k in sorted(last)]


class ReadOnlyCache:
    """Read-only view of a run's KVCache (the run's own files are never written by the analysis)."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True) if Path(path).exists() else None

    def get(self, key: str):
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


# ============================================================================ audit data + usage inventory
def _read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _used_ids(cfg: dict, src: dict, fewshot_ids: set[str]) -> set[str]:
    import glob

    kind = src["kind"]
    paths = sorted(Path(p) for p in glob.glob(str(resolve_path(cfg, src["path"]))))
    out: set[str] = set()
    for p in paths:
        if kind == "qid_jsonl":
            out |= {r["qid"] for r in _read_jsonl(p)}
        elif kind == "item_id_jsonl":
            out |= {r["item_id"] for r in _read_jsonl(p)}
        elif kind == "ranker_split":
            s = read_json(p)
            out |= set(s["probe"]) | set(s["label"])
        elif kind == "final_mmlu_json":
            out |= set(read_json(p)["per_question"])
            shots = (read_json(p.parent.parent / "run.json")["config"].get("final_eval") or {}).get("mmlu_shots", 0) \
                if (p.parent.parent / "run.json").exists() else 0
            if int(shots) > 0:  # the 5-shot prompts contain the MMLU-dev exemplars
                out |= fewshot_ids
        elif kind == "fixed_state_answers":
            out |= set(read_json(p))
        else:
            raise ValueError(f"unknown inventory kind {kind!r}")
    return out


def audit_data(cfg: dict) -> Path:
    from .data.prepare import PreparedData
    from .model import tokenizer_id

    out = analysis_dir(cfg) / "audit_data"
    out.mkdir(parents=True, exist_ok=True)
    data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
    splits = {"mmlu_probe": data.mmlu_probe, "mmlu_final": data.mmlu_final, "mmlu_dev": data.mmlu_dev,
              "mmlu_fewshot": data.mmlu_fewshot}
    fewshot_ids = {r["qid"] for r in data.mmlu_fewshot}
    inv = {"sources": [], "splits": {}}
    used_by: dict[str, set[str]] = {}
    for src in _gl(cfg).get("inventory", []):
        ids = _used_ids(cfg, src, fewshot_ids)
        used_by[src["name"]] = ids
        inv["sources"].append({"name": src["name"], "path": src["path"], "kind": src["kind"], "ids": len(ids)})
    for name, rows in splits.items():
        ids = {r["qid"] for r in rows}
        union = set().union(*used_by.values()) if used_by else set()
        inv["splits"][name] = {"questions": len(ids), "used": len(ids & union), "unused": len(ids - union),
                               "by_source": {k: len(ids & v) for k, v in used_by.items() if ids & v}}
    sets = {}
    union_used = set().union(*used_by.values()) if used_by else set()
    for role in AUDIT_ROLES:
        spec = (_gl(cfg).get("audit") or {}).get(role)
        if not spec:
            continue
        rows, prov = _audit_rows(cfg, spec, data)
        internal = {_norm(r["question"]) for rows_ in splits.values() for r in rows_}
        if spec["kind"] != "prepared":
            dup = [r["qid"] for r in rows if _norm(r["question"]) in internal]
            rows = [r for r in rows if _norm(r["question"]) not in internal]
            prov["excluded_exact_duplicates_of_prepared_mmlu"] = dup
        f = out / f"{role}.jsonl"
        atomic_write_text(f, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        cats: dict[str, int] = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        sets[role] = {"name": spec.get("name", role), "kind": spec["kind"], "file": str(f), "sha256": sha256_file(f),
                      "questions": len(rows), "by_category": dict(sorted(cats.items())), "provenance": prov,
                      "fresh": role == "fresh" and spec["kind"] != "prepared",
                      "label": spec.get("label", "fresh" if role == "fresh" else "exploratory (reused questions)")}
        if spec["kind"] == "prepared":
            sets[role]["used_before"] = inv["splits"].get(spec["split"], {}).get("used")
        else:
            sets[role]["used_before"] = len({r["qid"] for r in rows} & union_used)
        if sets[role]["fresh"] and sets[role]["used_before"]:
            raise RuntimeError(f"audit set {role} is labelled fresh but {sets[role]['used_before']} of its questions were used")
    inv["audit_sets"] = sets
    atomic_write_json(out / "inventory.json", inv)
    return out / "inventory.json"


def _audit_rows(cfg: dict, spec: dict, data) -> tuple[list[dict], dict]:
    if spec["kind"] == "prepared":
        rows = [dict(r) for r in getattr(data, spec["split"])]
        return rows, {"source": f"prepared data split {spec['split']}", "data_dir": str(data.dir)}
    if spec["kind"] != "mmlu_cf":
        raise ValueError(f"unknown audit source kind {spec['kind']!r}")
    import pyarrow.parquet as pq

    raw = resolve_path(cfg, spec["raw_dir"])
    (raw / "val").mkdir(parents=True, exist_ok=True)
    rev, repo = spec["revision"], spec.get("repo", "microsoft/MMLU-CF")
    files = {}
    for field, sha in spec["files"].items():
        p = raw / "val" / f"{field}_val.parquet"
        if not p.exists():
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/val/{field}_val.parquet"
            log.info("downloading %s", url)
            tmp = p.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(p)
        got = sha256_file(p)
        if got != sha:
            raise RuntimeError(f"{p}: sha256 {got} != pinned {sha}")
        files[field] = {"sha256": got, "bytes": p.stat().st_size}
    rows = []
    for field in sorted(spec["files"]):
        if field not in MMLU_CF_CATEGORY:
            raise ValueError(f"MMLU-CF field {field!r} has no MMLU category")
        subj = PROMPT_SUBJECT.get(field, field).lower()
        for i, r in enumerate(pq.read_table(raw / "val" / f"{field}_val.parquet").to_pylist()):
            rows.append({"qid": f"mmlucf/{field}/val/{i}", "subject": subj, "field": field,
                         "category": MMLU_CF_CATEGORY[field], "question": str(r["Question"]).strip(),
                         "choices": [str(r[k]).strip() for k in "ABCD"], "answer": "ABCD".index(str(r["Answer"]).strip())})
    return rows, {"source": f"{repo} validation split (val/<field>_val.parquet)", "revision": rev, "files": files,
                  "category_map": "MMLU official subcategory -> category (categories.py)", "prompt_subject": PROMPT_SUBJECT}


# ============================================================================ freeze
def freeze(cfg: dict) -> Path:
    from .utils.checkpoint import CheckpointStore

    rc, rd = run_paths(cfg)
    S, _ = CheckpointStore(rd).load_latest()
    if "ranked_search" not in S.get("stages_done", []):
        raise SystemExit(f"{rd.name}: the live run has not finished its ranked_search stage")
    out = analysis_dir(cfg) / "frozen"
    out.mkdir(parents=True, exist_ok=True)
    sid = S.get("state_id")
    srch = S["search"]
    rec = {"run_id": rd.name, "run_dir": str(rd), "state_id": sid, "mlp_sparsity": srch["rho"],
           "stop_reason": srch["stop_reason"], "attempts": len(attempts(rd)), "accepts": srch["accepts"],
           "questions": srch["questions"], "checkpoint_index": S["ckpt_index"], "science_hash": S.get("science_hash"),
           "journal_sha256": sha256_file(rd / "journal.jsonl")}
    if sid:
        src = rd / "masks" / f"{sid}.safetensors"
        dst = out / f"{sid}.safetensors"
        if not dst.exists():
            shutil.copy2(src, dst)
        rec.update(mask_file=str(dst), mask_sha256=sha256_file(dst), run_mask_sha256=sha256_file(src))
        if rec["mask_sha256"] != rec["run_mask_sha256"]:
            raise RuntimeError("frozen mask copy differs from the run's mask file")
    else:
        rec.update(mask_file=None, mask_sha256="dense", run_mask_sha256="dense")
    p = out / "frozen.json"
    if p.exists():
        old = read_json(p)
        if (old["state_id"], old["mask_sha256"]) != (rec["state_id"], rec["mask_sha256"]):
            raise SystemExit(f"already frozen at {old['state_id']}; the run now ends at {rec['state_id']}")
        return p
    rec["frozen_at"] = time.time()
    atomic_write_json(p, rec)
    return p


def check_frozen(cfg: dict) -> dict:
    from .utils.checkpoint import CheckpointStore

    p = analysis_dir(cfg) / "frozen" / "frozen.json"
    if not p.exists():
        raise SystemExit("freeze the run first: capprune gradient-live freeze --config ...")
    fr = read_json(p)
    _, rd = run_paths(cfg)
    S, _ = CheckpointStore(rd).load_latest()
    if S.get("state_id") != fr["state_id"]:
        raise RuntimeError(f"the run's checkpoint moved ({S.get('state_id')}) since it was frozen at {fr['state_id']}")
    if fr["mask_file"] and sha256_file(Path(fr["mask_file"])) != fr["mask_sha256"]:
        raise RuntimeError("frozen mask file changed since freezing")
    return fr


# ============================================================================ shared model setup
def _setup(rc: dict, fr: dict):
    import torch

    from .data.prepare import PreparedData
    from .gradient_live import load_dense_copy
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pruning.masks import DenseStore, PruneState

    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    tok = load_tokenizer(rc)
    model = load_model(rc)
    model.eval()
    mode = rc["pruning"]["mode"]
    units = build_units(model, mode, rc["pruning"].get("eligible", []))
    dense = DenseStore(units)
    dm = load_dense_copy(rc, model, units, dense)
    state = PruneState(units, mode)
    if fr["mask_file"]:
        state.load(fr["mask_file"])
        state.apply(dense)
    if abs(state.global_sparsity() - fr["mlp_sparsity"]) > 1e-9:
        raise RuntimeError(f"frozen mask sparsity {state.global_sparsity()} != recorded {fr['mlp_sparsity']}")
    torch.manual_seed(0)
    return data, tok, model, dm, units, dense, state


def _clock(dev):
    from .gradient_live import _sync

    _sync(dev)
    return time.perf_counter()


# ============================================================================ stall diagnostic
def diagnose(cfg: dict) -> Path:
    from .adaptive.evaluator import evaluate_candidate
    from .adaptive.live import weights_digest
    from .adaptive.policies import make_policy
    from .fixed_state import _geometry
    from .gradient_live import (PairedScorer, PairedSource, build_trial, gate_pool, legal_shortlist, make_gate,
                                unit_profiles)
    from .pruning.profile import load_profile
    from .ranker import canon, mask_hash
    from .utils.cache import KVCache

    t_all = time.perf_counter()
    rc, rd = run_paths(cfg)
    fr = check_frozen(cfg)
    out = analysis_dir(cfg) / "diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    att = attempts(rd)
    last = att[-1] if att else None
    stalled = last is not None and last["outcome"] != "accept" and last.get("gate_decision") != "none"
    if not stalled and fr["stop_reason"] not in ("proposal_cap", "question_cap"):
        res = {"stall": None, "reason": f"no failed attempt and no cap stop (stop reason {fr['stop_reason']})"}
        atomic_write_json(out / "diagnostic.json", res)
        return out / "diagnostic.json"
    if not stalled and not _gl(cfg).get("probe_on_cap", True):
        res = {"stall": None, "cap_reached": fr["stop_reason"],
               "reason": f"a budget cap ({fr['stop_reason']}) was reached before any top-ranked action failed; the "
                         "failure point is left open (no offline probe, by configuration)"}
        atomic_write_json(out / "diagnostic.json", res)
        return out / "diagnostic.json"
    # mode "stall": the failed live attempt's parent, profile, shortlist and question order.
    # mode "next_attempt": the run ended at a cap with every attempt accepted, so there is no stalled parent; the
    #   frozen checkpoint is probed offline exactly as the live loop's next attempt would have been (same profiling
    #   protocol, candidate stream and the order seed of that attempt number).
    mode = "stall" if stalled else "next_attempt"
    data, tok, model, dm, units, dense, state = _setup(rc, fr)
    dev = next(model.parameters()).device
    geo = _geometry(rc, units[0].channels)
    if mode == "stall":
        if last["parent_id"] != (fr["state_id"] or "dense"):
            raise RuntimeError(f"stalled parent {last['parent_id']} is not the frozen checkpoint {fr['state_id']}")
        if mask_hash(state) != last["parent_mask_hash"]:
            raise RuntimeError("frozen checkpoint mask hash differs from the stalled attempt's parent")
        t, step = last["iteration"], last["candidate_step"]
        pt = int(last.get("profile_attempt") or t)  # a top-k fallback attempt reuses its parent's profile
        act, sens, grad = load_profile(rd / "profiles" / f"iter-{pt:06d}.safetensors", dev)  # the live attempt's profile
    else:
        from .adaptive.live import attempt_order_seed
        from .gradient_live import profile_parent
        from .pruning.profile import save_profile

        t, step = (last["iteration"] + 1) if last else 1, fr["accepts"]
        if last and mask_hash(state) != last["trial_mask_hash"]:
            raise RuntimeError("frozen checkpoint is not the last accepted trial")
        pp = out / f"profile-next-attempt-{t:06d}.safetensors"
        if pp.exists():
            act, sens, grad = load_profile(pp, dev)
        else:
            calib = data.calibration_pool[data.calibration_indices(int(rc["run"]["seed"]),
                                                                   int(rc["data"]["calibration"]["num_samples"]))]
            t0 = _clock(dev)
            act, sens, grad = profile_parent(model, state, calib, rc, {}, None)
            log.info("profiled the frozen checkpoint in %.1f s", _clock(dev) - t0)
            save_profile(pp, act, sens, grad)
    ranked, _ = legal_shortlist(rc, geo, unit_profiles(state, sens, grad), state.global_sparsity(), len(units), step)
    keys = [canon(c["units"]) for c in ranked]
    if mode == "stall" and keys != [s["action_key"] for s in last["shortlist"]]:
        raise RuntimeError("rebuilt shortlist differs from the stalled attempt's logged shortlist")
    pool, _ = gate_pool(rc, data)
    gate = make_gate(rc, pool)
    if mode == "stall":
        seed = int(last["gate"]["order_seed"])
    else:
        seed = attempt_order_seed(gate.base_seed, int(rc["run"]["seed"]), t)
    tt = rc["testing"]
    scorer = PairedScorer(dm, model, tok, data.mmlu_fewshot, int(tt.get("shots", 0)),
                          int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768)))
    cache = KVCache(out / "pairs.sqlite")
    ro = ReadOnlyCache(rd / "caches" / "eval.sqlite")
    parent_digest = weights_digest(state)
    rows = []
    for rank, c in enumerate(ranked, 1):
        parent_masks = dict(state.masks)
        t0 = _clock(dev)
        changed, actual = build_trial(state, dense, c["units"], act, rc, geo)
        build_s = _clock(dev) - t0
        src = PairedSource(gate.pool, seed, mask_hash(state), scorer, cache, None, ro_caches=[ro])
        t0 = _clock(dev)
        run = evaluate_candidate(src, make_policy(gate.policy_name), gate.plan, f"diag-{rank}")
        gate_s = _clock(dev) - t0
        for n in changed:
            state.masks[n] = parent_masks[n]
        if changed:
            state.apply(dense, only=changed)
        if weights_digest(state) != parent_digest:
            raise RuntimeError("parent not restored exactly after a diagnostic trial")
        fin = run.final
        rows.append({"rank": rank, "kind": c["kind"], "action_key": keys[rank - 1],
                     "layers": sorted(int(u.split(".")[0][1:]) for u in c["units"]),
                     "z_g_mean": float(c["features"]["z_g_mean"]), "channels": sum(actual.values()),
                     "decision": run.decision, "questions": run.questions,
                     "domains": {d: {"n": v["n"], "est": v["estimate"], "lo": v["lo"], "hi": v["hi"], "status": v["status"],
                                     "broken": v["k_plus"], "fixed": v["k_minus"]} for d, v in fin.items()},
                     "confirmed_violations": [d for d, v in fin.items() if v["status"] == "fail"],
                     "unresolved": [d for d, v in fin.items() if v["status"] == "open"] if run.decision == "inconclusive" else [],
                     "trial_scoring_s": src.t_trial, "dense_reference_s": src.t_dense, "dense_reference_questions": src.dense_q,
                     "replayed_from_run_cache": src.replayed_q, "build_s": build_s, "gate_wall_s": gate_s})
        log.info("diag rank %d %-22s %s: %s n=%d (replayed %d)", rank, c["kind"], rows[-1]["layers"], run.decision,
                 run.questions, src.replayed_q)
    cache.close()
    ro.close()
    res = summarize_diagnostic(rows, last, int(_gl(cfg).get("near_top_rank", 3)), mode, fr["stop_reason"])
    res.update(mode=mode, stalled_attempt=t, parent=fr["state_id"] or "dense", parent_sparsity=state.global_sparsity(),
               order_seed=seed,
               gate={"policy": gate.policy_name, "budget": gate.budget, "pool": len(pool),
                     "tolerance": gate.settings.tolerance},
               cost={"trial_questions": sum(r["questions"] for r in rows),
                     "trial_questions_scored_here": sum(r["questions"] - r["replayed_from_run_cache"] for r in rows),
                     "dense_reference_questions": sum(r["dense_reference_questions"] for r in rows),
                     "trial_scoring_s": sum(r["trial_scoring_s"] for r in rows),
                     "dense_reference_s": sum(r["dense_reference_s"] for r in rows),
                     "build_s": sum(r["build_s"] for r in rows), "wall_s": time.perf_counter() - t_all},
               candidates=rows)
    atomic_write_json(out / "diagnostic.json", res)
    return out / "diagnostic.json"


def summarize_diagnostic(rows: list[dict], last: dict | None, near_top: int, mode: str = "stall",
                         stop_reason: str | None = None) -> dict:
    """Rank of the first passing candidate, how many pass, what blocked the rest, and the follow-up rule."""
    passing = [r["rank"] for r in rows if r["decision"] == "accept"]
    rejected = [r for r in rows if r["decision"] == "reject"]
    inconcl = [r for r in rows if r["decision"] == "inconclusive"]
    blocked: dict[str, dict[str, int]] = {}
    for r in rows:
        for d in r["confirmed_violations"]:
            blocked.setdefault(d, {"confirmed": 0, "unresolved": 0})["confirmed"] += 1
        for d in r["unresolved"]:
            blocked.setdefault(d, {"confirmed": 0, "unresolved": 0})["unresolved"] += 1
    first = min(passing) if passing else None
    reproduced = None
    if mode == "stall" and last["gate_decision"] != "budget_stop":
        live = rows[int((last.get("chosen") or {}).get("rank") or 1) - 1]  # the live attempt's own (rank, question order)
        reproduced = live["decision"] == last["gate"]["decision"] and live["questions"] == last["gate"]["questions"]
    if mode == "next_attempt" and first == 1:
        rec = {"rule": "no stall",
               "text": f"No stall: every live attempt was accepted, the run ended at its {stop_reason}, and the top-ranked "
                       "action still passes at the frozen checkpoint. None of the three stall rules applies; the selector "
                       "has not been pushed to its first failure. Raise the caps (a new run) to reach the first stall, "
                       "then rerun this diagnostic."}
    elif first is not None and first <= near_top:
        k = first
        rec = {"rule": "top-k fallback",
               "text": f"A passing action ranks near the top (rank {first} of {len(rows)}): test a budgeted top-{k} "
                       f"fallback (gate the ranked candidates in order, at most {k} per parent, within the question cap).",
               "fallback_k": k, "questions_to_first_pass": sum(r["questions"] for r in rows[:first])}
    elif first is not None:
        rec = {"rule": "ranking signals",
               "text": f"Passing actions exist but rank low (first at rank {first} of {len(rows)}; near the top = rank "
                       f"<= {near_top}): investigate ranking signals that can distinguish them.",
               "questions_to_first_pass": sum(r["questions"] for r in rows[:first])}
    elif rows and not inconcl and len(rejected) == len(rows):
        rec = {"rule": "smaller step",
               "text": "Every candidate fails with a confirmed violation: test a smaller removal step on the same parent."}
    else:
        rec = {"rule": "undetermined",
               "text": f"No candidate passes but {len(inconcl)} ended inconclusive (budget ran out): the evidence does "
                       "not show that every candidate clearly fails."}
    return {"n_candidates": len(rows), "n_pass": len(passing), "passing_ranks": passing, "first_pass_rank": first,
            "n_reject": len(rejected), "n_inconclusive": len(inconcl), "blocking_domains": blocked,
            "near_top_rank": near_top, "live_decision_reproduced": reproduced, "recommendation": rec}


# ============================================================================ audit
def _boot_stats(per: dict[str, np.ndarray], B: int, seed: int, tol: float, alpha: float = 0.05) -> dict:
    """Paired, category-stratified bootstrap of dense-minus-pruned accuracy.

    Per item x = dense_correct - pruned_correct in {-1, 0, 1}; resampling n items of one category with replacement
    gives multinomial counts of (+1, -1, 0), so the draws are exact item-bootstrap draws without materialising
    item indices.  Overall = item-weighted mean of the category draws (strata kept fixed)."""
    rng = np.random.default_rng(seed)
    cats = sorted(per)
    draws, out = {}, {}
    k = len(cats)
    for c in cats:
        x = per[c]
        n = len(x)
        kp, km = int((x == 1).sum()), int((x == -1).sum())
        m = rng.multinomial(n, np.array([kp, km, n - kp - km], dtype=np.float64) / n, size=B)
        draws[c] = (m[:, 0] - m[:, 1]) / n
        est = (kp - km) / n
        out[c] = {"n": n, "loss": est, "broken": kp, "fixed": km, "se": float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                  "ci95": [float(np.quantile(draws[c], 0.025)), float(np.quantile(draws[c], 0.975))],
                  "ub_simultaneous": float(np.quantile(draws[c], 1 - alpha / k))}
    N = sum(len(per[c]) for c in cats)
    ov = sum(draws[c] * len(per[c]) for c in cats) / N
    allx = np.concatenate([per[c] for c in cats])
    out["overall"] = {"n": N, "loss": float(allx.mean()), "broken": int((allx == 1).sum()), "fixed": int((allx == -1).sum()),
                      "se": float(allx.std(ddof=1) / np.sqrt(N)),
                      "ci95": [float(np.quantile(ov, 0.025)), float(np.quantile(ov, 0.975))],
                      "ub95": float(np.quantile(ov, 1 - alpha))}
    worst = max(cats, key=lambda c: out[c]["loss"])
    point = all(out[c]["loss"] <= tol + 1e-12 for c in cats)
    per_ci = all(out[c]["ci95"][1] <= tol + 1e-12 for c in cats)
    simul = all(out[c]["ub_simultaneous"] <= tol + 1e-12 for c in cats)
    if point and simul:
        verdict = "meets the limit; the simultaneous upper bounds support it"
    elif point and per_ci:
        verdict = "meets the limit; each category's 95 % interval is below it, but not all simultaneous bounds"
    elif point:
        verdict = "point estimates meet the limit, but the intervals do not exclude a loss above it"
    else:
        verdict = "does NOT meet the limit (a category's point estimate is above it)"
    out["conclusion"] = {"tolerance": tol, "worst_category": worst, "worst_loss": out[worst]["loss"],
                         "point_estimates_meet": point, "per_category_ci_support": per_ci,
                         "simultaneous_support": simul, "verdict": verdict,
                         "bootstrap": {"B": B, "seed": seed, "simultaneous": f"Bonferroni over {k} categories, one-sided "
                                                                         f"{100 * (1 - alpha / k):.2f} % per category"}}
    return out


def _count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())  # parameters() de-duplicates tied weights


def audit(cfg: dict) -> Path:
    import torch

    from .adaptive.evaluator import question_order
    from .eval.mmlu import MMLUScorer, build_prompt
    from .gradient_live import PairedScorer
    from .pruning.materialize import materialize
    from .utils.cache import KVCache
    from .utils.gpu import free_memory

    t_all = time.perf_counter()
    rc, rd = run_paths(cfg)
    fr = check_frozen(cfg)
    inv = read_json(analysis_dir(cfg) / "audit_data" / "inventory.json")
    out = analysis_dir(cfg) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    au = _gl(cfg).get("audit") or {}
    data, tok, model, dm, units, dense, state = _setup(rc, fr)
    dev = next(model.parameters()).device
    tt = rc["testing"]
    shots, max_len, mbt = int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768))
    bs = int(au.get("batch_size", rc["gate"]["capability"]["batch_size"]))
    seed = int(au.get("order_seed", 2026))
    tol = float(rc["gate"]["capability"]["tolerance"])
    scorer = PairedScorer(dm, model, tok, data.mmlu_fewshot, shots, max_len, mbt)
    cache = KVCache(out / "pairs.sqlite")
    tag = (fr["mask_sha256"] or "dense")[:16]
    res = {"frozen": fr, "protocol": {"shots": shots, "max_seq_len": max_len, "max_batch_tokens": mbt, "batch": bs,
                                      "order_seed": seed, "representation": "masked weights (dense * mask), " + str(rc["model"].get("dtype")),
                                      "layout": "per category, seeded order, batches of the gate's batch size; dense and "
                                                "pruned scored on identical batches"},
           "sets": {}}
    for role, meta in inv["audit_sets"].items():
        rows = _read_jsonl(Path(meta["file"]))
        if sha256_file(Path(meta["file"])) != meta["sha256"]:
            raise RuntimeError(f"audit set {role} changed since audit-data")
        long = sum(len(tok(build_prompt(r, scorer.t.fewshot, shots), add_special_tokens=False)["input_ids"]) > max_len
                   for r in rows)
        by_cat: dict[str, list[dict]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        items, sd, st, scored = [], 0.0, 0.0, 0
        for cat in sorted(by_cat):
            rmap = {r["qid"]: r for r in by_cat[cat]}
            order = question_order(list(rmap), seed, cat)
            for i in range(0, len(order), bs):
                ids = order[i : i + bs]
                key = f"audit|{role}|{meta['sha256'][:16]}|{tag}|{cat}|{i}|{len(ids)}"
                rec = cache.get(key)
                if rec is None:
                    r_ = scorer.score([rmap[q] for q in ids])
                    rec = {"d": r_["d"], "t": r_["t"], "sd": r_["sd"], "st": r_["st"], "budget": r_["budget"]}
                    cache.put(key, rec)
                    scored += len(ids)
                sd += rec["sd"]
                st += rec["st"]
                items += [{"qid": q, "category": cat, "dense": int(rec["d"][q]), "pruned": int(rec["t"][q])} for q in ids]
            log.info("audit %s %s: %d items", role, cat, len(by_cat[cat]))
        with open(out / f"per_item-{role}.jsonl", "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in items))
        per = {c: np.array([x["dense"] - x["pruned"] for x in items if x["category"] == c], dtype=np.int8) for c in by_cat}
        stats = _boot_stats(per, int(au.get("bootstrap", 10000)), 0, tol)
        accs = {c: {"dense": float(np.mean([x["dense"] for x in items if x["category"] == c])),
                    "pruned": float(np.mean([x["pruned"] for x in items if x["category"] == c]))} for c in by_cat}
        accs["overall"] = {"dense": float(np.mean([x["dense"] for x in items])), "pruned": float(np.mean([x["pruned"] for x in items]))}
        res["sets"][role] = {**{k: meta[k] for k in ("name", "kind", "questions", "by_category", "fresh", "label", "sha256")},
                             "used_before": meta.get("used_before"), "prompts_over_max_len": int(long),
                             "accuracy": accs, "stats": stats, "dense_scoring_s": sd, "pruned_scoring_s": st,
                             "questions_scored_now": scored}
    cache.close()
    # ---- physical size: materialise the pruned model (MLP channels sliced out)
    dense_params = _count_params(dm)
    mat, info = materialize(model, state, rc)
    pruned_params = _count_params(mat)
    first = inv["audit_sets"].get("fresh") or next(iter(inv["audit_sets"].values()))
    shared = _read_jsonl(Path(first["file"]))[:256]
    a = MMLUScorer(model, tok, data.mmlu_fewshot, shots, max_len, {}, "x", mbt).score(shared)
    b = MMLUScorer(mat, tok, data.mmlu_fewshot, shots, max_len, {}, "y", mbt).score(shared)
    del mat
    free_memory()
    mlp_total = sum(u.channels for u in units)
    removed_ch = int(sum((~state.masks[u.name]).sum().item() for u in units))
    res["size"] = {"dense_parameters": dense_params, "pruned_parameters": pruned_params,
                   "parameters_removed": dense_params - pruned_params,
                   "fraction_of_total_parameters_removed": (dense_params - pruned_params) / dense_params,
                   "mlp_sparsity": state.global_sparsity(), "mlp_channels_removed": removed_ch, "mlp_channels_total": mlp_total,
                   "materialize": info, "per_layer_channels_removed": {u.name: int((~state.masks[u.name]).sum().item())
                                                                       for u in units if (~state.masks[u.name]).any()},
                   "masked_vs_materialized_0shot": {"items": len(shared),
                                                    "agree": sum(a[q]["pred"] == b[q]["pred"] for q in a)}}
    res["wall_s"] = time.perf_counter() - t_all
    del model, dm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    atomic_write_json(out / "audit.json", res)
    return out / "audit.json"


# ============================================================================ materialized audit / verification
class MultiScorer:
    """Scores the same rows on several models with one identical batch layout (rows, order, token budget);
    if a CUDA OOM shrinks any layout, every model is rescored with the smaller budget."""

    def __init__(self, models: dict, tok, fewshot, shots: int, max_len: int, max_batch_tokens: int):
        from .eval.mmlu import MMLUScorer

        self.sc = {k: MMLUScorer(m, tok, fewshot, shots, max_len, {}, f"multi_{k}", max_batch_tokens)
                   for k, m in models.items()}
        self.budget = int(max_batch_tokens)
        self.dev = next(next(iter(models.values())).parameters()).device
        self.evals = {k: 0 for k in models}

    def score(self, rows: list[dict]) -> tuple[dict, int]:
        while True:
            b, out, ok = self.budget, {}, True
            for k, sc in self.sc.items():
                sc.budget = b
                self.evals[k] += len(rows)
                t0 = _clock(self.dev)
                res = sc.score(rows)
                s = _clock(self.dev) - t0
                out[k] = {"pred": {q: int(r["pred"]) for q, r in res.items()},
                          "correct": {q: int(r["correct"]) for q, r in res.items()}, "s": s}
                if sc.budget != b:
                    ok, self.budget = False, min(self.budget, sc.budget)
            if ok:
                return out, b
            log.warning("OOM changed the batch layout; rescoring every model with token budget %d", self.budget)


def size_check(rc: dict, dm, model, mat, state, units) -> dict:
    """Parameter count of the materialised model vs. the dense model and vs. the masks, plus an exact check that
    every kept gate/up row and down column equals the masked model's."""
    import torch

    from .model import build_units

    dense_p, mat_p = _count_params(dm), _count_params(mat)
    munits = {u.name: u for u in build_units(mat, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))}
    per_ch = {u.name: sum(lin.weight.shape[1] if p != "down" else lin.weight.shape[0] for p, lin in u.linears.items())
              for u in units}
    expect_removed, layers, shape_ok, weights_ok = 0, {}, True, True
    with torch.no_grad():
        for u in units:
            keep = state.masks[u.name]
            pruned = int((~keep).sum().item())
            expect_removed += pruned * per_ch[u.name]
            mu = munits[u.name]
            kept = mu.linears["gate"].weight.shape[0]
            ok = (kept == u.channels - pruned and mu.linears["up"].weight.shape[0] == kept
                  and mu.linears["down"].weight.shape[1] == kept)
            shape_ok &= ok
            idx = keep.nonzero().flatten()
            weights_ok &= bool(torch.equal(mu.linears["gate"].weight, u.linears["gate"].weight[idx])
                               and torch.equal(mu.linears["up"].weight, u.linears["up"].weight[idx])
                               and torch.equal(mu.linears["down"].weight, u.linears["down"].weight[:, idx]))
            if pruned:
                layers[u.name] = {"removed": pruned, "kept": kept}
    return {"dense_parameters": dense_p, "materialized_parameters": mat_p, "parameters_removed": dense_p - mat_p,
            "expected_from_masks": expect_removed, "count_matches_masks": dense_p - mat_p == expect_removed,
            "fraction_of_total_parameters_removed": (dense_p - mat_p) / dense_p, "mlp_sparsity": state.global_sparsity(),
            "mlp_channels_removed": sum(v["removed"] for v in layers.values()),
            "mlp_channels_total": sum(u.channels for u in units), "intermediate_sizes_match": shape_ok,
            "kept_weights_identical_to_masked": weights_ok, "per_layer": layers}


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    files = ["eval/mmlu.py", "pruning/materialize.py", "pruning/masks.py", "pruning/wanda.py", "pruning/profile.py",
             "model.py", "gradient_live.py", "gradient_audit.py", "ranker.py", "adaptive/evaluator.py",
             "adaptive/bounds.py", "adaptive/policies.py", "adaptive/live.py"]
    return {f: sha256_file(root / f)[:16] for f in files if (root / f).exists()}


def audit_materialized(cfg: dict) -> Path:
    """Audit the frozen checkpoint in its deployable form: dense, masked and materialised scored on identical batches
    (the original audit's question ids, order, batch size, prompt, dtype and token budget)."""
    import torch

    from .adaptive.evaluator import question_order
    from .pruning.materialize import materialize
    from .utils.cache import KVCache
    from .utils.gpu import free_memory

    t_all = time.perf_counter()
    rc, rd = run_paths(cfg)
    fr = check_frozen(cfg)
    am = _gl(cfg).get("audit_materialized") or {}
    exp = am.get("expect") or {}
    for k in ("state_id", "mask_sha256"):
        if exp.get(k) and fr[k] != exp[k]:
            raise RuntimeError(f"frozen {k} {fr[k]} != expected {exp[k]}")
    A = analysis_dir(cfg)
    inv = read_json(A / "audit_data" / "inventory.json")
    roles = am.get("sets") or list(inv["audit_sets"])
    out = A / "audit_materialized"
    out.mkdir(parents=True, exist_ok=True)
    au = _gl(cfg).get("audit") or {}
    data, tok, model, dm, units, dense, state = _setup(rc, fr)
    dev = next(model.parameters()).device
    mat, minfo = materialize(model, state, rc)
    for p in mat.parameters():
        p.requires_grad_(False)
    size = size_check(rc, dm, model, mat, state, units)
    size["materialize"] = minfo
    if not (size["count_matches_masks"] and size["intermediate_sizes_match"] and size["kept_weights_identical_to_masked"]):
        raise RuntimeError(f"materialised model does not match the checkpoint: {size}")
    tt = rc["testing"]
    shots, max_len, mbt = int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768))
    bs = int(au.get("batch_size", rc["gate"]["capability"]["batch_size"]))
    seed = int(au.get("order_seed", 2026))
    tol = float(rc["gate"]["capability"]["tolerance"])
    protocol = {"shots": shots, "max_seq_len": max_len, "max_batch_tokens": mbt, "batch": bs, "order_seed": seed,
                "dtype": str(rc["model"].get("dtype")), "layout": "per category (sorted), seeded order, consecutive batches; "
                "dense, masked and materialised scored on identical batches"}
    orig = read_json(A / "audit" / "audit.json") if (A / "audit" / "audit.json").exists() else None
    prov = {"frozen": fr, "code_sha256_16": _code_hashes(), "protocol": protocol,
            "original_audit_protocol": orig["protocol"] if orig else None,
            "protocol_matches_original": None if orig is None else all(
                orig["protocol"][k] == protocol[k2] for k, k2 in (("shots", "shots"), ("max_seq_len", "max_seq_len"),
                                                                    ("max_batch_tokens", "max_batch_tokens"), ("batch", "batch"),
                                                                    ("order_seed", "order_seed")))
            and str(rc["model"].get("dtype")) in orig["protocol"]["representation"]}
    ro = ReadOnlyCache(A / "audit" / "pairs.sqlite")
    ms = MultiScorer({"dense": dm, "masked": model, "materialized": mat}, tok, data.mmlu_fewshot, shots, max_len, mbt)
    cache = KVCache(out / "scores.sqlite")
    tag = (fr["mask_sha256"] or "dense")[:16]
    res = {"provenance": prov, "size": size, "sets": {}}
    for role in roles:
        meta = inv["audit_sets"][role]
        if sha256_file(Path(meta["file"])) != meta["sha256"]:
            raise RuntimeError(f"audit set {role} changed since audit-data")
        rows = _read_jsonl(Path(meta["file"]))
        by_cat: dict[str, list[dict]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        items, secs, cached_batches = [], {"dense": 0.0, "masked": 0.0, "materialized": 0.0}, 0
        match = {"batches": 0, "batches_same_ids": 0, "dense_items": 0, "dense_equal": 0, "masked_items": 0, "masked_equal": 0}
        for cat in sorted(by_cat):
            rmap = {r["qid"]: r for r in by_cat[cat]}
            order = question_order(list(rmap), seed, cat)
            for i in range(0, len(order), bs):
                ids = order[i : i + bs]
                key = f"{role}|{meta['sha256'][:16]}|{tag}|{cat}|{i}|{len(ids)}"
                rec = cache.get(key)
                if rec is None:
                    sc, b = ms.score([rmap[q] for q in ids])
                    rec = {**sc, "budget": b}
                    cache.put(key, rec)
                else:
                    cached_batches += 1
                for k in secs:
                    secs[k] += rec[k]["s"]
                old = ro.get(f"audit|{role}|{meta['sha256'][:16]}|{tag}|{cat}|{i}|{len(ids)}")
                if old is not None:  # the original (masked) audit's cached pair for this exact batch
                    match["batches"] += 1
                    match["batches_same_ids"] += set(old["d"]) == set(ids) and int(old["budget"]) == int(rec["budget"])
                    match["dense_items"] += len(ids)
                    match["dense_equal"] += sum(int(old["d"][q]) == rec["dense"]["correct"][q] for q in ids)
                    match["masked_items"] += len(ids)
                    match["masked_equal"] += sum(int(old["t"][q]) == rec["masked"]["correct"][q] for q in ids)
                items += [{"qid": q, "category": cat, **{f"{k}_pred": rec[k]["pred"][q] for k in secs},
                           **{f"{k}_correct": rec[k]["correct"][q] for k in secs}} for q in ids]
            log.info("materialised audit %s %s: %d items", role, cat, len(by_cat[cat]))
        with open(out / f"per_item-{role}.jsonl", "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in items))
        cats = sorted(by_cat)
        stats = {}
        for rep in ("masked", "materialized"):
            per = {c: np.array([x["dense_correct"] - x[f"{rep}_correct"] for x in items if x["category"] == c], dtype=np.int8)
                   for c in cats}
            stats[rep] = _boot_stats(per, int(au.get("bootstrap", 10000)), 0, tol)
        agree = {c: {"n": sum(x["category"] == c for x in items),
                     "pred_agree": sum(x["category"] == c and x["masked_pred"] == x["materialized_pred"] for x in items),
                     "correct_agree": sum(x["category"] == c and x["masked_correct"] == x["materialized_correct"] for x in items),
                     "mat_minus_masked_loss": float(np.mean([x["masked_correct"] - x["materialized_correct"]
                                                             for x in items if x["category"] == c]))}
                 for c in cats}
        agree["overall"] = {"n": len(items), "pred_agree": sum(x["masked_pred"] == x["materialized_pred"] for x in items),
                            "correct_agree": sum(x["masked_correct"] == x["materialized_correct"] for x in items),
                            "mat_minus_masked_loss": float(np.mean([x["masked_correct"] - x["materialized_correct"] for x in items]))}
        accs = {rep: {c: float(np.mean([x[f"{rep}_correct"] for x in items if x["category"] == c])) for c in cats}
                | {"overall": float(np.mean([x[f"{rep}_correct"] for x in items]))} for rep in ("dense", "masked", "materialized")}
        res["sets"][role] = {**{k: meta[k] for k in ("name", "kind", "questions", "by_category", "fresh", "label", "sha256")},
                             "used_before": meta.get("used_before"), "accuracy": accs, "stats": stats, "agreement": agree,
                             "original_audit_match": match if match["batches"] else None, "scoring_s": secs,
                             "batches_from_cache": cached_batches}
    cache.close()
    ro.close()
    res["item_model_evaluations_run"] = dict(ms.evals)
    res["wall_s"] = time.perf_counter() - t_all
    del mat, model, dm
    free_memory()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    atomic_write_json(out / "audit_materialized.json", res)
    atomic_write_text(out / "audit_materialized.md", render_materialized(res))
    return out / "audit_materialized.md"


def render_materialized(res: dict) -> str:
    pts = lambda x: "n/a" if x is None else f"{100 * x:+.2f}"  # noqa: E731
    ci = lambda c: f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"  # noqa: E731
    fr, sz, pr = res["provenance"]["frozen"], res["size"], res["provenance"]
    L = [f"# Materialised-checkpoint audit: `{fr['state_id']}` (mask sha256 `{fr['mask_sha256'][:16]}...`)", "",
         f"* Frozen checkpoint and mask verified before scoring (run `{fr['run_id']}`, stop {fr['stop_reason']}).",
         f"* Materialised with `pruning/materialize.py`: {sz['dense_parameters']:,} -> {sz['materialized_parameters']:,} "
         f"parameters, **{sz['parameters_removed']:,} removed = {100 * sz['fraction_of_total_parameters_removed']:.3f} %** "
         f"(masks predict {sz['expected_from_masks']:,}: {'match' if sz['count_matches_masks'] else 'MISMATCH'}); "
         f"MLP sparsity {100 * sz['mlp_sparsity']:.3f} % ({sz['mlp_channels_removed']:,} / {sz['mlp_channels_total']:,} "
         f"channels); intermediate sizes match the masks: {sz['intermediate_sizes_match']}; kept weights identical to the "
         f"masked model: {sz['kept_weights_identical_to_masked']}.",
         f"* Protocol: {pr['protocol']}; matches the original audit: {pr['protocol_matches_original']}.",
         f"* Item-model evaluations run: {res['item_model_evaluations_run']}.", ""]
    for role, s in res["sets"].items():
        a = s["agreement"]["overall"]
        L += [f"## {role}: {s['name']} ({s['questions']:,} questions; {s['label']})", ""]
        m = s.get("original_audit_match")
        if m:
            L.append(f"Original audit cache, same batches: {m['batches_same_ids']}/{m['batches']} batches with identical ids "
                     f"and token budget; dense correctness reproduced on {m['dense_equal']:,}/{m['dense_items']:,} items, "
                     f"masked on {m['masked_equal']:,}/{m['masked_items']:,}.")
        L += [f"Materialised vs. masked: predictions agree on {a['pred_agree']:,}/{a['n']:,} items "
              f"({100 * a['pred_agree'] / a['n']:.2f} %), correctness on {a['correct_agree']:,}; loss difference "
              f"(materialised - masked) {pts(a['mat_minus_masked_loss'])} pts.", "",
              "| category | n | dense acc | masked loss [95 % CI] | materialised loss [95 % CI] | materialised simultaneous one-sided 95 % UB | prediction agreement |",
              "|---|---|---|---|---|---|---|"]
        for c in [*sorted(s["by_category"]), "overall"]:
            mk, mt = s["stats"]["masked"][c], s["stats"]["materialized"][c]
            ub = mt.get("ub_simultaneous", mt.get("ub95"))
            ag = s["agreement"][c]
            L.append(f"| {c} | {mt['n']:,} | {100 * s['accuracy']['dense'][c]:.2f} | {pts(mk['loss'])} {ci(mk['ci95'])} | "
                     f"{pts(mt['loss'])} {ci(mt['ci95'])} | {pts(ub)}{' (overall, one-sided 95 %)' if c == 'overall' else ''} | "
                     f"{ag['pred_agree']:,}/{ag['n']:,} |")
        c = s["stats"]["materialized"]["conclusion"]
        L += ["", f"**Materialised model: {c['verdict']}.** Worst category {c['worst_category']} {pts(c['worst_loss'])} pts; "
              f"simultaneous bounds below {100 * c['tolerance']:.0f} pts: {c['simultaneous_support']} "
              f"({c['bootstrap']['simultaneous']}).", "",
              "Scoring GPU s: " + ", ".join(f"{k} {v:.1f}" for k, v in s["scoring_s"].items()) + ".", ""]
    return "\n".join(L) + "\n"


# ============================================================================ pre-launch registration
def register(cfg: dict) -> Path:
    """Record the run's caps, replay prefix and audit-set decision BEFORE it is launched."""
    from .utils.io import short_hash

    rc, rd = run_paths(cfg)
    A = analysis_dir(cfg)
    A.mkdir(parents=True, exist_ok=True)
    p = A / "registration.json"
    started = (rd / "journal.jsonl").exists()
    rs = rc["ranked_search"]
    vp = rs.get("verify_prefix") or {}
    rec = {"run_id": rd.name, "run_dir": str(rd), "config": rc.get("_config_path"),
           "config_sha256": sha256_file(Path(rc["_config_path"])), "science_hash_inputs": short_hash(
               {k: v for k, v in rc.items() if k in ("ranked_search", "gate", "ranker", "search", "pruning", "profiling")}),
           "caps": {"max_proposals": rs["max_proposals"], "question_cap": rs.get("question_cap"),
                    "question_unit": "one paired gate question = one item scored on the dense model AND the trial (same "
                                     "batch); the cap counts every paired question of the run, including a replayed prefix"},
           "fixed": {"selector": rs.get("selector", "gradient"), "tie_break": "generator index",
                     "step": f"{rc['ranker']['shortlist']['units_per_action']} units x {rc['ranker']['shortlist']['blocks_per_unit']} "
                             f"blocks x {rc['pruning']['structured']['channel_multiple']} channels",
                     "gate": {k: rc["gate"]["capability"].get(k) for k in ("policy", "tolerance", "budget", "min_sample",
                                                                          "batch_size", "alpha_false_accept", "alpha_false_reject")}},
           "prefix": None, "audit_decision": _gl(cfg).get("fresh_search"), "code_sha256_16": _code_hashes(),
           "launched_before_registration": started}
    from .gradient_live import gate_margin

    m = gate_margin(rc)
    tol = float(rc["gate"]["capability"]["tolerance"])
    rec["fixed"].update(fallback_k=int(rs.get("fallback_k", 1)), gate_margin_pts=100 * m,
                        pruning_time_tolerance_pts=100 * (tol - m), audit_target_pts=100 * tol)
    ml = _gl(cfg).get("margin_lock")
    if ml:  # the margin must be the one the recorded calibration rule chose (or its predeclared fallback)
        cal = read_json(resolve_path(cfg, ml["calibration"]))
        locked = cal["chosen_margin_pts"] if cal["conclusive"] else float(ml["if_inconclusive_pts"])
        if abs(100 * m - locked) > 1e-9:
            raise SystemExit(f"gate_margin {100 * m:.2f} pts != locked calibration margin {locked} pts")
        rec["margin_lock"] = {"calibration": str(resolve_path(cfg, ml["calibration"])), "locked_margin_pts": locked,
                              "conclusive": cal["conclusive"], "support": cal.get("support"),
                              "label": "confirmatory margin" if cal["conclusive"] else "EXPLORATORY margin (calibration inconclusive)"}
    if vp:
        jp = resolve_path(rc, vp["journal"])
        rec["prefix"] = {**vp, "journal_sha256": sha256_file(jp)}
    if p.exists():
        old = read_json(p)
        keep = ("caps", "prefix", "config_sha256")
        if any(old.get(k) != rec.get(k) for k in keep):
            raise SystemExit(f"{p} already records different caps/prefix/config; refusing to overwrite")
        return p
    if started:
        raise SystemExit(f"{rd.name} has already started; registration must precede the launch")
    rec["registered_at"] = time.time()
    atomic_write_json(p, rec)
    return p


# ============================================================================ report
def _ranker_crosscheck(cfg: dict, att: list[dict]) -> dict | None:
    p = resolve_path(cfg, _gl(cfg).get("ranker_dir")) if _gl(cfg).get("ranker_dir") else None
    if p is None or not (p / "decisions.json").exists():
        return None
    D = read_json(p / "decisions.json")
    by_parent: dict[str, list[dict]] = {}
    for d in D["decisions"]:
        by_parent.setdefault(d["parent_mask"], []).append(d)
    out = []
    for a in att:
        ds = by_parent.get(a["parent_mask_hash"])
        if not ds:
            out.append({"attempt": a["iteration"], "ranker_parent": None})
            continue
        d = ds[0]
        cand = {c["action_key"]: c for c in d["candidates"]}
        live = {s["action_key"]: s for s in a["shortlist"]}
        common = [k for k in live if k in cand]
        dz = max((abs(live[k]["z_g_mean"] - cand[k]["features"]["z_g_mean"]) for k in common), default=None)
        ch = cand.get(a["chosen"]["action_key"])
        off_order = sorted(cand, key=lambda k: (round(cand[k]["features"]["z_g_mean"], 12), cand[k]["idx"]))
        out.append({"attempt": a["iteration"], "ranker_parent": f"{d['group']} {d['trajectory']}/{d['step']}",
                    "shared_actions": len(common), "of_live": len(live), "max_abs_dz_grad": dz,
                    "offline_top1": off_order[0], "live_top1": a["chosen"]["action_key"],
                    "same_top1": off_order[0] == a["chosen"]["action_key"],
                    "offline_chosen": None if ch is None else {
                        "label_worst": ch["worst"], "worst_domain": ch["worst_domain"], "label_loss": ch["label"],
                        "gate_replay": ch["gate"], "rank_by_true_loss": 1 + sum(c["worst"] < ch["worst"] - 1e-12 for c in d["candidates"]),
                        "n": len(d["candidates"]), "best_label_worst": min(c["worst"] for c in d["candidates"])}})
    return {"dir": str(p), "eval_key": D["eval_key"], "attempts": out}


def report(cfg: dict) -> Path:
    rc, rd = run_paths(cfg)
    A = analysis_dir(cfg)
    att = attempts(rd)
    fr = read_json(A / "frozen" / "frozen.json") if (A / "frozen" / "frozen.json").exists() else None
    diag = read_json(A / "diagnostic" / "diagnostic.json") if (A / "diagnostic" / "diagnostic.json").exists() else None
    aud = read_json(A / "audit" / "audit.json") if (A / "audit" / "audit.json").exists() else None
    inv = read_json(A / "audit_data" / "inventory.json") if (A / "audit_data" / "inventory.json").exists() else None
    summ = read_json(rd / "final" / "summary.json") if (rd / "final" / "summary.json").exists() else None
    xc = _ranker_crosscheck(cfg, att)
    S = {"run_id": rd.name, "attempts": att, "frozen": fr, "diagnostic": diag, "audit": aud, "inventory": inv,
         "summary": summ, "ranker_crosscheck": xc, "caps": rc["ranked_search"], "gate": rc["gate"]["capability"]}
    if _gl(cfg).get("compare_with"):  # an extension of an earlier run: its own report, the earlier one untouched
        B = resolve_path(cfg, _gl(cfg)["compare_with"])
        opt = lambda p: read_json(p) if p.exists() else None  # noqa: E731
        S.update(registration=opt(A / "registration.json"), materialized=opt(A / "audit_materialized" / "audit_materialized.json"),
                 search_state=_search_state(rd), starts=_starts(rd),
                 original={"dir": str(B), "frozen": opt(B / "frozen" / "frozen.json"),
                           "materialized": opt(B / "audit_materialized" / "audit_materialized.json")})
        if S["original"]["frozen"]:
            S["original"]["attempts"] = attempts(Path(S["original"]["frozen"]["run_dir"]))
        atomic_write_json(A / "report.json", S)
        atomic_write_text(A / "report.md", render_extended(S, rc))
        return A / "report.md"
    atomic_write_json(A / "report.json", S)
    atomic_write_text(A / "report.md", render(S, rc))
    return A / "report.md"


def _search_state(rd: Path) -> dict | None:
    from .utils.checkpoint import CheckpointStore, NoCheckpoint

    try:
        return CheckpointStore(rd).load_latest()[0].get("search")
    except NoCheckpoint:
        return None


def _starts(rd: Path) -> list[float]:
    p = rd / "journal.jsonl"
    if not p.exists():
        return []
    return [json.loads(l)["ts"] for l in p.read_text(encoding="utf-8").splitlines() if '"type": "start"' in l]


def _cost(att: list[dict]) -> dict:
    tm = lambda k: sum(a.get("timings", {}).get(k, 0.0) for a in att)  # noqa: E731

    def ev(k):  # journals written before the evaluation counters existed: one dense + one trial run per paired question
        tot = 0
        for a in att:
            e = a.get("evaluations")
            q = (a.get("gate") or {}).get("questions", 0)
            tot += e.get(k, 0) if e else (q if k in ("dense_run", "trial_run") else 0)
        return tot
    return {"attempts": len(att), "accepted": sum(a["gate_decision"] == "accept" for a in att),
            "paired_questions": sum(a["gate"]["questions"] for a in att if a.get("gate")),
            "dense_evaluations_run": ev("dense_run"), "trial_evaluations_run": ev("trial_run"),
            "pairs_replayed": ev("pairs_replayed_from_cache"), "profile_s": tm("profile_s"), "pruning_s": tm("build_s"),
            "dense_scoring_s": tm("gate_dense_reference_s"), "candidate_scoring_s": tm("gate_trial_scoring_s"),
            "gate_wall_s": tm("gate_s"), "rollback_s": tm("rollback_s"), "search_s": tm("attempt_s")}


def render_extended(S: dict, rc: dict) -> str:
    from .gradient_live import STOP_KINDS

    pts = lambda x: "n/a" if x is None else f"{100 * x:+.2f}"  # noqa: E731
    att, fr, diag, mat, reg = S["attempts"], S["frozen"], S["diagnostic"], S.get("materialized"), S.get("registration")
    O = S["original"]
    n_pre = ((reg or {}).get("prefix") or {}).get("attempts") or 0
    sst = S.get("search_state") or {}
    L = ["# Extended gradient-only search: where does the top-ranked action first fail?", ""]
    # ---------------- lead
    if mat and fr:
        sz = mat["size"]
        L.append(f"**Verified (materialised) reduction: {sz['parameters_removed']:,} of {sz['dense_parameters']:,} parameters "
                 f"= {100 * sz['fraction_of_total_parameters_removed']:.2f} %** ({100 * sz['mlp_sparsity']:.2f} % MLP sparsity; "
                 f"count equals the masks' prediction: {sz['count_matches_masks']}).")
        for role, s in mat["sets"].items():
            st = s["stats"]["materialized"]
            c = st["conclusion"]
            mx = max(s["by_category"], key=lambda k: st[k]["ub_simultaneous"])
            L.append(f"* {s['name']} ({'fresh' if s.get('fresh') else 'exploratory, already inspected'}): worst category "
                     f"{c['worst_category']} {pts(c['worst_loss'])} pts; largest simultaneous one-sided 95 % upper bound "
                     f"{mx} {pts(st[mx]['ub_simultaneous'])} pts -> {c['verdict']}.")
    if fr:
        L.append(f"**Stop:** `{fr['stop_reason']}` ({STOP_KINDS.get(fr['stop_reason'], '?')}) after {fr['attempts']} attempts, "
                 f"{fr['accepts']} accepted, {fr['questions']:,} paired gate questions; frozen checkpoint `{fr['state_id']}`.")
    if diag and diag.get("recommendation"):
        L.append(f"**First stall:** {diag['n_pass']}/{diag['n_candidates']} candidates pass at the stalled parent; first pass "
                 f"at rank {diag['first_pass_rank']}. Recommendation: **{diag['recommendation']['rule']}** - "
                 f"{diag['recommendation']['text']}")
    elif diag:
        L.append(f"**Stall diagnostic:** {diag.get('reason')}")
    # ---------------- registration and prefix
    L += ["", "## Registration (before launch)", ""]
    if reg:
        first = min(S.get("starts") or [float("inf")])
        L += [f"* Recorded {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reg['registered_at']))}; run started "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first)) if first != float('inf') else 'n/a'} "
              f"(registered first: {reg['registered_at'] < first}).",
              f"* Caps: {reg['caps']['max_proposals']} proposals, {reg['caps']['question_cap']:,} paired gate questions "
              f"in total. {reg['caps']['question_unit']}.",
              f"* Unchanged: {reg['fixed']}.",
              f"* Prefix: replay of `{(reg.get('prefix') or {}).get('journal')}` ({n_pre} attempts, final state "
              f"`{(reg.get('prefix') or {}).get('state_id')}`, mask sha256 `{str((reg.get('prefix') or {}).get('mask_sha256'))[:16]}...`).",
              f"* Audit-set decision: {(reg.get('audit_decision') or {}).get('decision', 'n/a')}"]
        for cnd in (reg.get("audit_decision") or {}).get("candidates", []):
            L.append(f"  * {cnd['name']}: {cnd['verdict']}")
    else:
        L.append("Not registered.")
    pf = sst.get("prefix")
    L += ["", "## Replay of the original pilot (prefix check)", "",
          (f"Verified through attempt {pf['verified_through']} of {pf['n']} (parent, ordered shortlist, chosen action, trial "
           f"mask hash, order seed, gate decision, questions and per-category counts at every attempt): ok={pf['ok']}; final "
           f"mask content identical to the frozen checkpoint (packed-bit hash; the frozen file's recorded sha256 re-checked): "
           f"{pf.get('final_mask_verified')}; mask file bytes identical: {pf.get('final_mask_file_bytes_equal')} (safetensors "
           f"writes metadata keys in an unspecified order, so equal masks need not give equal file hashes); differences: "
           f"{pf['diffs'] or 'none'}." if pf else "No prefix check recorded.")]
    # ---------------- attempts
    L += ["", "## Attempts", "",
          "| # | parent MLP sparsity | rank-1 action | gate | paired questions (cum.) | blocking (reject) / largest estimate: est [lo, hi] | "
          "params removed after | s: profile / prune / dense / candidate / attempt | prefix |", "|---|---|---|---|---|---|---|---|---|"]
    for a in att:
        doms = a["gate"]["domains"]
        failed = [d for d in doms if doms[d]["status"] == "fail"]
        w = failed[0] if failed else max(doms, key=lambda d: doms[d]["est"] if doms[d]["est"] is not None else -9)
        t = a["timings"]
        L.append(f"| {a['iteration']} | {100 * a['parent_sparsity']:.3f} % | {a['chosen']['kind']} {a['chosen']['layers']} "
                 f"({a['chosen']['z_g_mean']:+.3f}) | **{a['gate_decision']}** | {a['gate']['questions']:,} "
                 f"({a['questions_cumulative']:,}) | {w} {pts(doms[w]['est'])} [{pts(doms[w]['lo'])}, {pts(doms[w]['hi'])}] | "
                 f"{a.get('params_removed', 0):,} | {t.get('profile_s', 0):.1f} / {t.get('build_s', 0):.2f} / "
                 f"{t.get('gate_dense_reference_s', 0):.1f} / {t.get('gate_trial_scoring_s', 0):.1f} / {t.get('attempt_s', 0):.1f} | "
                 f"{'ok' if (a.get('prefix_check') or {}).get('ok') else ('MISMATCH' if a.get('prefix_check') else '-')} |")
    # ---------------- cost
    pre, inc = [a for a in att if a["iteration"] <= n_pre], [a for a in att if a["iteration"] > n_pre]
    ref_name = Path(O["frozen"]["run_dir"]).parent.name if O.get("frozen") else "reference run"
    rows = [(f"reference run ({ref_name})", _cost(O.get("attempts") or [])), ("this run: replayed prefix", _cost(pre)),
            ("this run: incremental (beyond the pilot)", _cost(inc)), ("this run: cumulative", _cost(att))]
    L += ["", "## Cost", "",
          "| scope | attempts (accepted) | paired gate questions | item-model evaluations run (dense + candidate) | pairs from cache | "
          "profiling s | pruning s | dense scoring s | candidate scoring s | end-to-end search s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in rows:
        L.append(f"| {name} | {c['attempts']} ({c['accepted']}) | {c['paired_questions']:,} | "
                 f"{c['dense_evaluations_run'] + c['trial_evaluations_run']:,} ({c['dense_evaluations_run']:,} + "
                 f"{c['trial_evaluations_run']:,}) | {c['pairs_replayed']:,} | {c['profile_s']:.1f} | {c['pruning_s']:.2f} | "
                 f"{c['dense_scoring_s']:.1f} | {c['candidate_scoring_s']:.1f} | {c['search_s']:.1f} |")
    L.append("\nThe original pilot's journal predates the per-attempt evaluation counters: its evaluations are derived as one "
             "dense + one candidate evaluation per paired question (one worker start, so no cache replays). Timings are "
             "synchronised GPU wall seconds per stage; model loading and the dense-copy check are outside the per-attempt "
             "timings.")
    # ---------------- diagnostic
    L += ["", "## First stall", ""]
    if diag and diag.get("candidates"):
        L += [f"Parent `{diag['parent']}` ({100 * diag['parent_sparsity']:.3f} % MLP sparsity), attempt {diag['stalled_attempt']}; "
              f"every shortlist action gated offline with the same gate and question order (seed {diag['order_seed']}). "
              f"Live top-1 decision reproduced: {diag['live_decision_reproduced']}.", "",
              "| rank | action | z_grad | decision | questions | blocked by |", "|---|---|---|---|---|---|"]
        for r in diag["candidates"]:
            blk = ", ".join([f"{d} (confirmed)" for d in r["confirmed_violations"]] + [f"{d} (unresolved)" for d in r["unresolved"]]) or "-"
            L.append(f"| {r['rank']} | {r['kind']} {r['layers']} | {r['z_g_mean']:+.3f} | {r['decision']} | {r['questions']:,} | {blk} |")
        c = diag["cost"]
        L += ["", f"* Passing {diag['n_pass']}/{diag['n_candidates']} (ranks {diag['passing_ranks'] or 'none'}); rejected "
              f"{diag['n_reject']}; inconclusive {diag['n_inconclusive']}; blocking categories {diag['blocking_domains'] or 'none'}.",
              f"* Recommendation ({diag['recommendation']['rule']}): {diag['recommendation']['text']}",
              f"* Diagnostic compute (separate): {c['trial_questions']:,} paired questions ({c['trial_questions_scored_here']:,} "
              f"scored, rest from the run's cache), {c['dense_reference_questions']:,} dense evaluations; GPU s candidate "
              f"{c['trial_scoring_s']:.1f}, dense {c['dense_reference_s']:.1f}; wall {c['wall_s']:.0f} s."]
    else:
        L.append((diag or {}).get("reason", "Not run."))
    # ---------------- audit
    L += ["", "## Audit of the frozen materialised checkpoint", ""]
    if mat:
        sz = mat["size"]
        L += [f"Materialised with `pruning/materialize.py`: {sz['parameters_removed']:,} parameters removed (masks predict "
              f"{sz['expected_from_masks']:,}; intermediate sizes and kept weights verified: {sz['intermediate_sizes_match']}, "
              f"{sz['kept_weights_identical_to_masked']}). Per layer channels removed: "
              + ", ".join(f"{k.split('.')[0]} {v['removed']}" for k, v in sz["per_layer"].items()) + ".", ""]
        for role, s in mat["sets"].items():
            st = s["stats"]["materialized"]
            a = s["agreement"]["overall"]
            L += [f"### {s['name']} ({s['label']}; {s['questions']:,} questions, {s.get('used_before') or 0:,} used before)", "",
                  "| category | dense acc | materialised loss [95 % CI] | simultaneous one-sided 95 % UB |", "|---|---|---|---|"]
            for k in [*sorted(s["by_category"]), "overall"]:
                v = st[k]
                L.append(f"| {k} | {100 * s['accuracy']['dense'][k]:.2f} | {pts(v['loss'])} [{pts(v['ci95'][0])}, "
                         f"{pts(v['ci95'][1])}] | {pts(v.get('ub_simultaneous', v.get('ub95')))} |")
            L += ["", f"{st['conclusion']['verdict'].capitalize()} (worst {st['conclusion']['worst_category']} "
                  f"{pts(st['conclusion']['worst_loss'])}). Materialised vs. masked predictions agree on {a['pred_agree']:,}/{a['n']:,}.", ""]
    else:
        L.append("Not run.")
    # ---------------- comparison
    L += ["## Original vs. extended checkpoint (materialised; MMLU-CF and MMLU held-out are both already inspected)", ""]
    om, of = O.get("materialized"), O.get("frozen")

    def col(m, f):
        if not (m and f):
            return None
        out = {"state": f["state_id"], "attempts": f"{f['attempts']} ({f['accepts']} accepted)", "questions": f"{f['questions']:,}",
               "stop": f["stop_reason"], "mlp": f"{100 * m['size']['mlp_sparsity']:.2f} %",
               "params": f"{m['size']['parameters_removed']:,} ({100 * m['size']['fraction_of_total_parameters_removed']:.2f} %)"}
        for name, s in m["sets"].items():
            st = s["stats"]["materialized"]
            c = st["conclusion"]
            key = "MMLU-CF" if "MMLU-CF" in s["name"] else "MMLU held-out"
            out[key] = (f"worst {c['worst_category']} {pts(c['worst_loss'])}, overall {pts(st['overall']['loss'])}, max UB "
                        f"{pts(max(st[k]['ub_simultaneous'] for k in s['by_category']))}")
        return out

    a_, b_ = col(om, of), col(mat, fr)
    keys = ["state", "attempts", "questions", "stop", "mlp", "params", "MMLU-CF", "MMLU held-out"]
    names = {"state": "checkpoint", "attempts": "attempts", "questions": "paired gate questions", "stop": "stop reason",
             "mlp": "MLP sparsity", "params": "parameters removed (materialised)",
             "MMLU-CF": "MMLU-CF val: loss (pts), simultaneous UB", "MMLU held-out": "MMLU held-out: loss (pts), simultaneous UB"}
    L += ["| | original pilot | extended run |", "|---|---|---|"]
    for k in keys:
        L.append(f"| {names[k]} | {(a_ or {}).get(k, 'n/a')} | {(b_ or {}).get(k, 'n/a')} |")
    return "\n".join(L) + "\n"


def render(S: dict, rc: dict) -> str:
    from .gradient_live import STOP_KINDS

    pts = lambda x: "n/a" if x is None else f"{100 * x:+.2f}"  # noqa: E731
    ci = lambda c: f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"  # noqa: E731
    att, fr, diag, aud = S["attempts"], S["frozen"], S["diagnostic"], S["audit"]
    caps = S["caps"]
    g = S["gate"]
    L = ["# Gradient-only selector in the live pruning loop (Qwen3-1.7B, structured MLP channels)", ""]
    # ---------------- headline
    if fr:
        L.append(f"Run `{S['run_id']}`: {fr['accepts']} accepted step(s), {fr['attempts']} attempt(s), "
                 f"{fr['questions']:,} gate questions; stop: **{fr['stop_reason']}** "
                 f"({STOP_KINDS.get(fr['stop_reason'], '?')}). Frozen checkpoint `{fr['state_id'] or 'dense'}` at "
                 f"{100 * fr['mlp_sparsity']:.3f} % MLP sparsity.")
    if aud:
        sz = aud["size"]
        L.append(f"Physically removed: {sz['parameters_removed']:,} of {sz['dense_parameters']:,} parameters "
                 f"(**{100 * sz['fraction_of_total_parameters_removed']:.2f} %** of the model); MLP sparsity "
                 f"{100 * sz['mlp_sparsity']:.3f} % ({sz['mlp_channels_removed']:,} of {sz['mlp_channels_total']:,} channels).")
        for role, s in aud["sets"].items():
            c = s["stats"]["conclusion"]
            L.append(f"* {s['label']} audit ({s['name']}, {s['questions']:,} questions): worst category {c['worst_category']} "
                     f"{pts(c['worst_loss'])} pts, overall {pts(s['stats']['overall']['loss'])} pts -> {c['verdict']}.")
    if diag and diag.get("recommendation"):
        where = ("the stalled parent" if diag.get("mode", "stall") == "stall"
                 else "the frozen checkpoint (no live stall; offline next-attempt probe)")
        L.append(f"* Stall diagnostic: {diag['n_pass']}/{diag['n_candidates']} candidates pass at {where}"
                 + (f", first at rank {diag['first_pass_rank']}" if diag["first_pass_rank"] else "")
                 + f". Recommendation: **{diag['recommendation']['rule']}** - {diag['recommendation']['text']}")
    L += ["", "## Setup", "",
          f"* Selector: gradient-only (lowest mean z_grad of the action's units; ties by generator index), top-1 per parent; "
          f"candidate set = the ranker experiment's generator ({rc['ranker']['shortlist']['units_per_action']} units x "
          f"{rc['ranker']['shortlist']['blocks_per_unit']} blocks x {rc['pruning']['structured']['channel_multiple']} channels "
          f"= matched step; {rc['ranker'].get('random_actions', 2)} seeded random sets), profiling = the ranker's fixed "
          f"calibration subsets (profile seed {rc['ranker'].get('profile_seed', 0)}), run seed {rc['run']['seed']}.",
          f"* Gate: {g.get('policy', 'threshold_aware')}, tolerance {100 * float(g['tolerance']):.1f} pts per category vs. the "
          f"dense model (cumulative), alpha {g['alpha_false_accept']}/{g['alpha_false_reject']}, min sample / batch "
          f"{g['min_sample']}/{g['batch_size']}, {g.get('spending', 'uniform')} spending, pool = "
          f"{caps.get('gate_pool', 'label')} set ({att[0]['gate']['pool']['size'] if att else '?'} questions; budget = pool), "
          "fresh question order per attempt. Dense and trial scored on identical batches (separate dense copy).",
          f"* Caps: {caps['max_proposals']} proposals, {caps.get('question_cap')} gate questions (cumulative; dense-reference "
          "rescoring of the same batches is reported separately and not counted).", ""]
    # ---------------- attempts
    L += ["## Live attempts", "",
          "| # | parent MLP sparsity | chosen (rank 1) | z_grad | channels proposed / actual | gate | questions (cum.) | "
          "per-category loss, est [lo, hi] (pts) | GPU s (profile / gate trial / dense ref / total) | verified |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in att:
        gd = a.get("gate", {})
        doms = gd.get("domains") or {}
        dl = "; ".join(f"{d} {pts(v['est'])} [{pts(v['lo'])}, {pts(v['hi'])}] {v['status']}" for d, v in doms.items())
        tm = a.get("timings", {})
        L.append(f"| {a['iteration']} | {100 * a['parent_sparsity']:.3f} % | {a['chosen']['kind']} {a['chosen']['layers']} | "
                 f"{a['chosen']['z_g_mean']:+.3f} | {a['proposed_channels']} / {a['actual_channels']} | "
                 f"**{a['gate_decision']}** | {gd.get('questions', 0):,} ({a['questions_cumulative']:,}) | {dl} | "
                 f"{tm.get('profile_s', 0):.1f} / {tm.get('gate_trial_scoring_s', 0):.1f} / {tm.get('gate_dense_reference_s', 0):.1f}"
                 f" / {a.get('gpu_seconds', 0):.1f} | parent {a.get('parent_verified')}, rollback {a.get('rollback_verified')} |")
    if att:
        L += ["", "Ordered shortlist at each parent (rank: kind [layers] mean z_grad):", ""]
        for a in att:
            L.append(f"* attempt {a['iteration']}: " + "; ".join(f"{s['rank']}: {s['kind']} {s['layers']} {s['z_g_mean']:+.3f}"
                                                               for s in a["shortlist"])
                     + (f" (illegal, dropped: {len(a['illegal'])})" if a.get("illegal") else ""))
    if fr:
        bs = (S.get("summary") or {}).get("budget") or {}
        L += ["", f"**Why the search stopped:** `{fr['stop_reason']}` = {STOP_KINDS.get(fr['stop_reason'], '?')}."]
        if att and att[-1]["gate_decision"] in ("reject", "inconclusive"):
            a = att[-1]
            bad = [d for d, v in a["gate"]["domains"].items() if v["status"] == "fail"]
            L.append(f"Attempt {a['iteration']} ({a['chosen']['kind']} {a['chosen']['layers']}) was "
                     f"{a['gate_decision']}ed" + (f": confirmed violation in {', '.join(bad)}" if bad else "")
                     + "; the parent was restored and kept as the final checkpoint.")
        if bs:
            L.append(f"Run totals: {bs.get('probe_questions', 0):,} gate questions; GPU s profile "
                     f"{bs.get('profile_seconds', 0):.1f}, gate trial scoring {bs.get('gate_scoring_seconds', 0):.1f}, dense "
                     f"reference {bs.get('gate_dense_reference_seconds', 0):.1f}, gate wall {bs.get('gate_seconds', 0):.1f}; "
                     f"search wall {bs.get('search_wall_seconds', 0):.1f} s.")
    xc = S.get("ranker_crosscheck")
    if xc:
        L += ["", "### Cross-check against the offline ranker experiment (cached label-set scores, read only)", ""]
        for r in xc["attempts"]:
            if not r["ranker_parent"]:
                L.append(f"* attempt {r['attempt']}: parent not in the ranker's data.")
                continue
            oc = r["offline_chosen"]
            L.append(f"* attempt {r['attempt']}: parent = ranker {r['ranker_parent']}; {r['shared_actions']}/{r['of_live']} live "
                     f"actions also offline, max |dz_grad| {r['max_abs_dz_grad']:.2g}; same gradient top-1: {r['same_top1']}"
                     + ("" if oc is None else
                        f"; offline exact label-set worst loss of the chosen action {pts(oc['label_worst'])} ({oc['worst_domain']}), "
                        f"rank {oc['rank_by_true_loss']}/{oc['n']} by true loss (best {pts(oc['best_label_worst'])}), offline gate "
                        f"replay {oc['gate_replay']['decision']} after {oc['gate_replay']['questions']:,}") + ".")
    # ---------------- diagnostic
    L += ["", "## Stall diagnostic (offline, after freezing; not part of the live search)", ""]
    if not diag:
        L.append("Not run.")
    elif diag.get("stall") is None and "candidates" not in diag:
        L.append(diag.get("reason", "No stall."))
    else:
        if diag.get("mode", "stall") == "stall":
            head = (f"Parent: `{diag['parent']}` ({100 * diag['parent_sparsity']:.3f} % MLP sparsity), stalled attempt "
                    f"{diag['stalled_attempt']}; every candidate gated with the same gate (budget {diag['gate']['budget']:,} = "
                    f"pool) and the stalled attempt's question order (seed {diag['order_seed']}). Live top-1 decision "
                    f"reproduced: {diag['live_decision_reproduced']}.")
        else:
            head = (f"**No live stall**: every attempt was accepted and the run ended at a cap. Offline probe instead: the "
                    f"frozen checkpoint `{diag['parent']}` ({100 * diag['parent_sparsity']:.3f} % MLP sparsity) profiled with "
                    f"the live protocol and its full shortlist gated exactly as live attempt {diag['stalled_attempt']} would "
                    f"have been (same gate, budget {diag['gate']['budget']:,} = pool, that attempt's question order, seed "
                    f"{diag['order_seed']}).")
        L += [head, "",
              "| rank | candidate | z_grad | decision | questions | per-category est [lo, hi] (pts) | blocked by |",
              "|---|---|---|---|---|---|---|"]
        for r in diag["candidates"]:
            dl = "; ".join(f"{d} {pts(v['est'])} [{pts(v['lo'])}, {pts(v['hi'])}]" for d, v in r["domains"].items())
            blk = ", ".join([f"{d} (confirmed)" for d in r["confirmed_violations"]]
                            + [f"{d} (unresolved)" for d in r["unresolved"]]) or "-"
            L.append(f"| {r['rank']} | {r['kind']} {r['layers']} | {r['z_g_mean']:+.3f} | {r['decision']} | {r['questions']:,} | "
                     f"{dl} | {blk} |")
        bl = "; ".join(f"{d}: {v['confirmed']} confirmed, {v['unresolved']} unresolved" for d, v in diag["blocking_domains"].items())
        c = diag["cost"]
        L += ["", f"* Passing: {diag['n_pass']}/{diag['n_candidates']} (ranks {diag['passing_ranks'] or 'none'}); first passing "
              f"rank: {diag['first_pass_rank']}. Rejected (confirmed violations): {diag['n_reject']}; inconclusive: "
              f"{diag['n_inconclusive']}.",
              f"* Blocking categories: {bl or 'none'}.",
              f"* Recommendation ({diag['recommendation']['rule']}): {diag['recommendation']['text']}",
              f"* Diagnostic cost (separate from the live run): {c['trial_questions']:,} gate questions over all candidates "
              f"({c['trial_questions_scored_here']:,} scored here, the rest read from the live run's cache), "
              f"{c['dense_reference_questions']:,} dense-reference questions; GPU s trial {c['trial_scoring_s']:.1f}, dense "
              f"{c['dense_reference_s']:.1f}; wall {c['wall_s']:.0f} s."]
    # ---------------- audit
    L += ["", "## Audit of the frozen checkpoint", ""]
    inv = S.get("inventory")
    if inv:
        L += ["Question usage before this audit (prepared MMLU splits):", "",
              "| split | questions | used before | unused | by source |", "|---|---|---|---|---|"]
        for n, v in inv["splits"].items():
            L.append(f"| {n} | {v['questions']:,} | {v['used']:,} | {v['unused']:,} | "
                     + "; ".join(f"{k}: {x:,}" for k, x in v["by_source"].items()) + " |")
        L.append("")
        for role, s in inv["audit_sets"].items():
            p = s["provenance"]
            L.append(f"* **{role}** = {s['name']} ({s['label']}): {s['questions']:,} questions {s['by_category']}; "
                     f"source {p.get('source')}" + (f" @ {p['revision'][:12]}" if p.get("revision") else "")
                     + (f"; {len(p['excluded_exact_duplicates_of_prepared_mmlu'])} exact duplicates of prepared MMLU excluded"
                        if "excluded_exact_duplicates_of_prepared_mmlu" in p else "") + ".")
    if not aud:
        L.append("\nNot run.")
    else:
        sz, pr = aud["size"], aud["protocol"]
        L += ["", f"Checkpoint `{aud['frozen']['state_id'] or 'dense'}` (mask sha256 `{str(aud['frozen']['mask_sha256'])[:16]}`), fixed "
              f"throughout. Protocol = the live gate's: {pr['shots']}-shot, max_seq_len {pr['max_seq_len']}, token budget "
              f"{pr['max_batch_tokens']:,}, {pr['representation']}; {pr['layout']} ({pr['batch']} per batch, order seed "
              f"{pr['order_seed']}).", "",
              f"* Parameters: dense {sz['dense_parameters']:,}, materialised pruned {sz['pruned_parameters']:,} -> removed "
              f"{sz['parameters_removed']:,} = **{100 * sz['fraction_of_total_parameters_removed']:.3f} %** of all parameters.",
              f"* MLP sparsity {100 * sz['mlp_sparsity']:.3f} % ({sz['mlp_channels_removed']:,} / {sz['mlp_channels_total']:,} "
              f"channels); per layer: " + ", ".join(f"{k.split('.')[0]} {v}" for k, v in sz["per_layer_channels_removed"].items()) + ".",
              f"* Masked vs. materialised predictions on {sz['masked_vs_materialized_0shot']['items']} fresh items: "
              f"{sz['masked_vs_materialized_0shot']['agree']} agree.", ""]
        for role, s in aud["sets"].items():
            st = s["stats"]
            c = st["conclusion"]
            L += [f"### {s['label'].capitalize()}: {s['name']} ({s['questions']:,} questions)", "",
                  f"Prompts longer than max_seq_len (left-truncated by the scorer): {s['prompts_over_max_len']}. "
                  + (f"Used before this audit: {s['used_before']:,} of these questions." if s.get("used_before") else ""), "",
                  "| category | n | dense acc | pruned acc | paired loss (pts) | 95 % CI | simultaneous 95 % upper bound | broken / fixed |",
                  "|---|---|---|---|---|---|---|---|"]
            for k in [*sorted(s["by_category"]), "overall"]:
                v = st[k]
                ub = v.get("ub_simultaneous", v.get("ub95"))
                L.append(f"| {k} | {v['n']:,} | {100 * s['accuracy'][k]['dense']:.2f} | {100 * s['accuracy'][k]['pruned']:.2f} | "
                         f"{pts(v['loss'])} | {ci(v['ci95'])} | {pts(ub)}{' (one-sided 95 %)' if k == 'overall' else ''} | "
                         f"{v['broken']} / {v['fixed']} |")
            L += ["", f"**{c['verdict'].capitalize()}.** Worst category {c['worst_category']} {pts(c['worst_loss'])} pts vs. the "
                  f"{100 * c['tolerance']:.0f}-point limit; point estimates meet it: {c['point_estimates_meet']}; every per-category "
                  f"95 % interval below it: {c['per_category_ci_support']}; simultaneous bounds below it: {c['simultaneous_support']} "
                  f"({c['bootstrap']['simultaneous']}, {c['bootstrap']['B']:,} paired stratified bootstrap draws).", "",
                  f"Audit GPU s: dense {s['dense_scoring_s']:.1f}, pruned {s['pruned_scoring_s']:.1f}.", ""]
    return "\n".join(L) + "\n"
