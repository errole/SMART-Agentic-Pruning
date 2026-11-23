"""2x2 gap diagnostic for one live arm's selected checkpoint (``capprune gap-diagnostic``).

Question split  {pruning-time pool (MMLU probe), held-out (MMLU final)}  x  prompting {0-shot, 5-shot}.

Every cell is a *paired* loss: dense accuracy minus pruned accuracy on the same items under the same
prompt.  All cells use one scoring procedure - the final audit's: ``eval/mmlu.py::MMLUScorer``, masked
weights in the run's dtype, ``final_eval.max_seq_len``, ``final_eval.max_batch_tokens`` and chunks of
``final_eval.chunk_size`` items in data order; only the number of shots differs.  The 5-shot cells use
the first five ``mmlu_fewshot`` (MMLU dev) examples of each subject - the same fixed exemplars on both
splits, never scored.

Reuse: the held-out 5-shot cell is the run's final audit when (a) the audit's cache keys carry the
checkpoint's state id for every held-out item, (b) its settings match, (c) the scoring code is unchanged
and (d) a recomputed chunk agrees item by item; otherwise it is recomputed.  The other cells are scored
here; the gate's cached 0-shot answers and the cached dense pool answers are compared as checks.

Intervals: bootstrap over items, stratified by category, items kept paired (dense/pruned and 0/5-shot
use the same resampled items within a split; the two splits are resampled independently).
Outputs go to ``out_dir`` only; the run's own artifacts are read, never written.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import runs_dir
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("gap")
CELLS = [("pool", 0), ("pool", 5), ("holdout", 0), ("holdout", 5)]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception as e:  # not a git checkout
        return f"unavailable: {e}"


def _loss(d: np.ndarray, p: np.ndarray) -> float:
    return float(d.mean() - p.mean())


def bootstrap(items: dict, cats: list[str], B: int, seed: int) -> dict:
    """items[(split, shots)] -> (dense, pruned, category) arrays, same item order for both shot counts."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("pool", "holdout"):
        cat = items[(split, 0)][2]
        x = {s: items[(split, s)][0].astype(np.int8) - items[(split, s)][1].astype(np.int8) for s in (0, 5)}
        per: dict[int, dict] = {0: {}, 5: {}}
        sizes = {}
        for c in cats:  # stratified by category; one resample per category shared by both shot counts
            ix = np.flatnonzero(cat == c).astype(np.int32)
            sizes[c] = len(ix)
            draw = ix[rng.integers(0, len(ix), size=(B, len(ix)), dtype=np.int32)]
            for s in (0, 5):
                per[s][c] = x[s][draw].mean(axis=1)
            del draw
        n = sum(sizes.values())
        for s in (0, 5):
            out[(split, s)] = {"overall": sum(per[s][c] * sizes[c] for c in cats) / n, **per[s]}
    return out


def _ci(a: np.ndarray) -> list[float]:
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def run(cfg: dict, arm: str, out_dir: Path, bootstrap_B: int = 10000) -> Path:
    import torch

    from .config import data_dir, run_id
    from .data.prepare import PreparedData
    from .eval.mmlu import MMLUScorer, build_prompt
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pruning.masks import DenseStore, PruneState
    from .pruning.materialize import materialize
    from .utils.cache import KVCache
    from .utils.checkpoint import CheckpointStore
    from .utils.io import short_hash

    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = runs_dir(cfg)
    rid = run_id(arm, float(cfg["matrix"]["targets"][0]), int(cfg["matrix"]["seeds"][0]))
    rd, dd = root / rid, root / "dense"
    store = CheckpointStore(rd)
    S, _ = store.load_latest()
    if "final_mmlu" not in S.get("stages_done", []):
        raise SystemExit(f"{rid}: no final audit yet - run the arm to completion first")
    sid = S["state_id"]
    mask = store.mask_path(sid)
    fe = cfg["final_eval"]
    shots5, max_len = int(fe.get("mmlu_shots", 5)), int(fe.get("max_seq_len", 4096))
    budget, chunk = int(fe.get("max_batch_tokens", 32768)), int(fe.get("chunk_size", 512))
    prov: dict = {"arm": arm, "run_id": rid, "state_id": sid, "mask_file": str(mask), "mask_sha256": sha256_file(mask),
                  "search_rho": (S.get("search") or {}).get("rho"),
                  "procedure": {"representation": "masked weights (dense * mask)", "dtype": cfg["model"].get("dtype"),
                                "max_seq_len": max_len, "max_batch_tokens": budget, "chunk_size": chunk,
                                "shots": [0, shots5], "fewshot_source": "mmlu_fewshot (MMLU dev), first 5 per subject"},
                  "code": {"git_head": _git("rev-parse", "HEAD"),
                           "uncommitted_changes_to_scoring_code": _git("status", "--porcelain", "--", "capprune/eval",
                                                                       "capprune/pruning", "capprune/model.py", "capprune/data") or "none",
                           "eval_mmlu_blob": _git("hash-object", "capprune/eval/mmlu.py")}}

    data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
    tok = load_tokenizer(cfg)
    splits = {"pool": data.mmlu_probe, "holdout": data.mmlu_final}
    cats = sorted({r["category"] for r in data.mmlu_final})

    # ---- few-shot exemplars: fixed, five per subject, disjoint from every scored item
    fs: dict[str, list[dict]] = {}
    for e in data.mmlu_fewshot:
        fs.setdefault(e["subject"], []).append(e)
    subjects = {r["subject"] for rows in splits.values() for r in rows}
    scored_text = {r["question"].strip() for rows in splits.values() for r in rows}
    scored_ids = {r["qid"] for rows in splits.values() for r in rows}
    used = [e for s in sorted(subjects) for e in fs.get(s, [])[:shots5]]
    prov["fewshot"] = {"subjects": len(subjects), "subjects_with_5": sum(len(fs.get(s, [])) >= shots5 for s in subjects),
                       "exemplar_ids_sha256": hashlib.sha256("\n".join(e["qid"] for e in used).encode()).hexdigest(),
                       "exemplar_id_overlap_with_scored": len({e["qid"] for e in used} & scored_ids),
                       "exemplar_text_overlap_with_scored": sum(e["question"].strip() in scored_text for e in used),
                       "duplicate_items_excluded": duplicate_items(splits, used)}

    # ---- prompt lengths: 0-shot prompts must fit the gate's 2048; count 5-shot prompts that drop exemplars
    def shots_used(r) -> int:
        for k in range(shots5, -1, -1):
            if len(tok(build_prompt(r, fs, k), add_special_tokens=False)["input_ids"]) <= max_len:
                return k
        return 0

    prov["prompts"] = {}
    for split, rows in splits.items():
        l0 = [len(tok(build_prompt(r, fs, 0), add_special_tokens=False)["input_ids"]) for r in rows]
        ks = [shots_used(r) for r in rows]
        prov["prompts"][split] = {"max_0shot_tokens": max(l0), "zero_shot_over_2048": sum(x > 2048 for x in l0),
                                  "five_shot_items_with_fewer_than_5_exemplars": sum(k < shots5 for k in ks),
                                  "short_items": [r["qid"] for r, k in zip(rows, ks) if k < shots5][:50]}

    # ---- model, dense weights, checkpoint
    model = load_model(cfg)
    dev = next(model.parameters()).device
    units = build_units(model, cfg["pruning"]["mode"], cfg["pruning"].get("eligible", []))
    dense_store = DenseStore(units)
    state = PruneState(units, cfg["pruning"]["mode"])
    state.load(mask)
    prov["checkpoint_sparsity"] = state.global_sparsity()
    if abs(prov["checkpoint_sparsity"] - (prov["search_rho"] or 0)) > 1e-9:
        raise RuntimeError(f"mask sparsity {prov['checkpoint_sparsity']} != search state {prov['search_rho']}")

    def score(rows, shots, m=None):
        sc = MMLUScorer(m or model, tok, data.mmlu_fewshot, shots, max_len, {}, "gap", budget)
        res = {}
        for i in range(0, len(rows), chunk):
            res.update({q: int(r["correct"]) for q, r in sc.score(rows[i : i + chunk]).items()})
        return res

    # ---- held-out 5-shot: provenance of the final audits
    audit = {}
    ok = True
    for who, d_ in (("dense", dd), ("pruned", rd)):
        mm = read_json(d_ / "final" / "mmlu.json")["per_question"]
        key_sid = "dense" if who == "dense" else sid
        cache = KVCache(d_ / "caches" / "eval.sqlite")
        keys = {q: f"final_mmlu|{key_sid}|{q}" for q in mm}
        got = cache.get_many(keys.values())
        cache.close()
        rj = read_json(d_ / "run.json")["config"]["final_eval"]
        checks = {"items": len(mm), "ids_equal_holdout": set(mm) == {r["qid"] for r in data.mmlu_final},
                  "cache_keys_with_state_id": sum(keys[q] in got for q in mm),
                  "cache_values_equal": sum(keys[q] in got and bool(got[keys[q]]) == bool(mm[q]) for q in mm),
                  "settings_match": all(rj.get(k) == fe.get(k) for k in ("mmlu_shots", "max_seq_len", "max_batch_tokens", "chunk_size"))}
        audit[who] = {k: int(v) for k, v in mm.items()}
        prov.setdefault("audit_provenance", {})[who] = checks
        ok &= all([checks["ids_equal_holdout"], checks["cache_keys_with_state_id"] == len(mm),
                   checks["cache_values_equal"] == len(mm), checks["settings_match"]])

    preds: dict = {}  # (who, split, shots) -> {qid: 0/1}
    t0 = time.perf_counter()
    PruneState(units, cfg["pruning"]["mode"]).apply(dense_store)  # the original dense weights
    check_rows = data.mmlu_final[:chunk]
    for who in ("dense", "pruned"):
        if who == "pruned":
            state.apply(dense_store)
        re5 = score(check_rows, shots5)
        agree = sum(re5[r["qid"]] == audit[who][r["qid"]] for r in check_rows)
        prov.setdefault("audit_recompute_check", {})[who] = {"items": len(check_rows), "agree": agree}
        ok &= agree == len(check_rows)
        for split, shots in (("pool", 0), ("pool", shots5), ("holdout", 0)):
            log.info("scoring %s %s %d-shot (%d items)", who, split, shots, len(splits[split]))
            preds[(who, split, shots)] = score(splits[split], shots)
        if who == "pruned":
            # masked vs materialised predictions on a shared batch (0-shot and 5-shot)
            mat, _ = materialize(model, state, cfg)
            shared = data.mmlu_final[:256]
            agr = {}
            for shots in (0, shots5):
                a, b = score(shared, shots), score(shared, shots, mat)
                agr[f"{shots}-shot"] = {"items": len(shared), "agree": sum(a[q] == b[q] for q in a)}
            prov["masked_vs_materialized"] = agr
            del mat
    prov["audit_reused"] = ok
    for who in ("dense", "pruned"):
        if ok:
            preds[(who, "holdout", shots5)] = audit[who]
        else:
            log.warning("audit provenance not verified: recomputing held-out %d-shot for %s", shots5, who)
            if who == "dense":
                PruneState(units, cfg["pruning"]["mode"]).apply(dense_store)
            else:
                state.apply(dense_store)
            preds[(who, "holdout", shots5)] = score(splits["holdout"], shots5)
    PruneState(units, cfg["pruning"]["mode"]).apply(dense_store)
    prov["gpu_scoring_seconds"] = time.perf_counter() - t0

    # ---- checks against cached 0-shot answers on the pool (gate / dense reference)
    mkey = short_hash(cfg["model"]["name_or_path"], cfg["model"].get("revision"), cfg["model"].get("dtype"))
    man = data.manifest["files"]
    qkey = short_hash(man["mmlu_probe.jsonl"]["sha256"], man["mmlu_fewshot.jsonl"]["sha256"],
                      cfg["testing"].get("shots", 0), cfg["model"].get("max_seq_len", 2048))
    dp = root / "_shared" / mkey / f"dense_probe-{qkey}.json"
    if dp.exists():
        cached = read_json(dp)
        prov["dense_pool_0shot_vs_cached_reference"] = {
            "items": len(cached), "agree": sum(int(cached[q]) == preds[("dense", "pool", 0)][q] for q in cached)}
    gate = KVCache(rd / "caches" / "eval.sqlite")
    gk = {q: f"probe|{sid}|{q}" for q in preds[("pruned", "pool", 0)]}
    gv = gate.get_many(gk.values())
    gate.close()
    prov["pruned_pool_0shot_vs_gate_cache"] = {
        "items_scored_by_gate": len(gv),
        "agree": sum(int(bool(gv[gk[q]])) == v for q, v in preds[("pruned", "pool", 0)].items() if gk[q] in gv)}

    # ---- per-item answers (every scored item) -> analysis
    per_item = []
    for split, rows in splits.items():
        for r in rows:
            q = r["qid"]
            per_item.append({"qid": q, "split": split, "category": r["category"],
                             "dense_0": preds[("dense", split, 0)][q], "pruned_0": preds[("pruned", split, 0)][q],
                             "dense_5": preds[("dense", split, shots5)][q], "pruned_5": preds[("pruned", split, shots5)][q]})
    prov["wall_seconds"] = time.perf_counter() - t_start
    with open(out_dir / "per_item.jsonl", "w", encoding="utf-8") as f:
        for r in per_item:
            f.write(json.dumps(r) + "\n")
    atomic_write_json(out_dir / "provenance.json", prov)
    return analyze(out_dir, bootstrap_B)


def duplicate_items(splits: dict, exemplars: list[dict]) -> list[str]:
    """Scored items whose question AND choices equal a few-shot exemplar's (MMLU repeats some questions
    across subjects/splits).  They are excluded from every cell so exemplars and scored items are disjoint."""
    key = lambda r: (r["question"].strip(), tuple(c.strip() for c in r["choices"]))  # noqa: E731
    ex = {key(e) for e in exemplars}
    return sorted(r["qid"] for rows in splits.values() for r in rows if key(r) in ex)


def analyze(out_dir: Path, bootstrap_B: int = 10000) -> Path:
    """Tables and paired-bootstrap intervals from ``per_item.jsonl`` (no GPU)."""
    prov = read_json(out_dir / "provenance.json")
    with open(out_dir / "per_item.jsonl", encoding="utf-8") as f:
        per_item = [json.loads(l) for l in f if l.strip()]
    excl = set(prov["fewshot"].get("duplicate_items_excluded", []))
    rows = [r for r in per_item if r["qid"] not in excl]
    cats = sorted({r["category"] for r in rows})
    shots5 = int(prov["procedure"]["shots"][1])
    items = {}
    for split in ("pool", "holdout"):
        rs = [r for r in rows if r["split"] == split]
        cat = np.array([r["category"] for r in rs])
        for s_ in (0, 5):
            items[(split, s_)] = (np.array([r[f"dense_{s_}"] for r in rs]), np.array([r[f"pruned_{s_}"] for r in rs]), cat)
    boot = bootstrap(items, cats, bootstrap_B, seed=0)
    table = {}
    for (split, shots), (d, p, cat) in items.items():
        cell = {"n": int(len(d)), "dense_acc": float(d.mean()), "pruned_acc": float(p.mean()),
                "overall": _loss(d, p), "overall_ci": _ci(boot[(split, shots)]["overall"])}
        for c in cats:
            m = cat == c
            cell[c] = _loss(d[m], p[m])
            cell[f"{c}_ci"] = _ci(boot[(split, shots)][c])
            cell[f"{c}_n"] = int(m.sum())
        cell["worst"] = max(cats, key=lambda c: cell[c])
        table[f"{split}|{shots}"] = cell
    b = boot
    stem = "STEM"
    k = f"{shots5}-shot"
    contrasts = {
        f"pool: {k} minus 0-shot": b[("pool", 5)][stem] - b[("pool", 0)][stem],
        f"holdout: {k} minus 0-shot": b[("holdout", 5)][stem] - b[("holdout", 0)][stem],
        "0-shot: holdout minus pool": b[("holdout", 0)][stem] - b[("pool", 0)][stem],
        f"{k}: holdout minus pool": b[("holdout", 5)][stem] - b[("pool", 5)][stem],
        f"split gap change ({k} minus 0-shot)": (b[("holdout", 5)][stem] - b[("pool", 5)][stem])
                                                  - (b[("holdout", 0)][stem] - b[("pool", 0)][stem]),
    }
    point = {
        f"pool: {k} minus 0-shot": table["pool|5"][stem] - table["pool|0"][stem],
        f"holdout: {k} minus 0-shot": table["holdout|5"][stem] - table["holdout|0"][stem],
        "0-shot: holdout minus pool": table["holdout|0"][stem] - table["pool|0"][stem],
        f"{k}: holdout minus pool": table["holdout|5"][stem] - table["pool|5"][stem],
    }
    point[f"split gap change ({k} minus 0-shot)"] = point[f"{k}: holdout minus pool"] - point["0-shot: holdout minus pool"]
    stem_contrasts = {k_: {"estimate": point[k_], "ci95": _ci(v)} for k_, v in contrasts.items()}
    prov["bootstrap"] = {"B": bootstrap_B, "seed": 0, "scheme": "items resampled within category; same items for both "
                         "shot counts and for dense/pruned within a split; splits resampled independently"}
    summary = {"table": table, "stem_contrasts": stem_contrasts, "categories": cats, "provenance": prov, "shots": shots5,
               "excluded_items": sorted(excl)}
    atomic_write_json(out_dir / "summary.json", summary)
    atomic_write_text(out_dir / "summary.md", render(summary))
    return out_dir / "summary.md"


def reanalyze(cfg: dict, out_dir: Path, bootstrap_B: int = 10000) -> Path:
    """Re-run the analysis of an existing diagnostic; fills in the duplicate check for older outputs."""
    from .config import data_dir
    from .data.prepare import PreparedData
    from .model import tokenizer_id

    prov = read_json(out_dir / "provenance.json")
    if "duplicate_items_excluded" not in prov["fewshot"]:
        data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
        fs: dict[str, list[dict]] = {}
        for e in data.mmlu_fewshot:
            fs.setdefault(e["subject"], []).append(e)
        splits = {"pool": data.mmlu_probe, "holdout": data.mmlu_final}
        shots5 = int(prov["procedure"]["shots"][1])
        subjects = {r["subject"] for rows in splits.values() for r in rows}
        used = [e for s_ in sorted(subjects) for e in fs.get(s_, [])[:shots5]]
        prov["fewshot"]["duplicate_items_excluded"] = duplicate_items(splits, used)
        atomic_write_json(out_dir / "provenance.json", prov)
    return analyze(out_dir, bootstrap_B)


def render(S: dict) -> str:
    t, cats, prov = S["table"], S["categories"], S["provenance"]
    pts = lambda x: f"{100 * x:+.2f}"  # noqa: E731
    ci = lambda c: f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"  # noqa: E731
    L = [f"# Gap diagnostic: `{prov['arm']}` checkpoint {prov['state_id']} ({100 * prov['checkpoint_sparsity']:.2f} % MLP sparsity)",
         "", "Paired loss = dense accuracy - pruned accuracy on the same items and prompt (accuracy points); 95 % "
         "intervals from a paired, category-stratified item bootstrap.", "",
         "## STEM loss", "", f"| split | 0-shot | {S['shots']}-shot |", "|---|---:|---:|"]
    for split, name in (("pool", "Pruning-time pool"), ("holdout", "Held-out audit")):
        L.append(f"| {name} (n={t[f'{split}|0']['STEM_n']}) | {pts(t[f'{split}|0']['STEM'])} {ci(t[f'{split}|0']['STEM_ci'])} | "
                 f"{pts(t[f'{split}|5']['STEM'])} {ci(t[f'{split}|5']['STEM_ci'])} |")
    L += ["", "## All categories", "",
          "| cell | n | dense acc | pruned acc | overall loss | " + " | ".join(cats) + " | worst |",
          "|---|---|---|---|---|" + "---|" * len(cats) + "---|"]
    for key in ("pool|0", "pool|5", "holdout|0", "holdout|5"):
        c = t[key]
        split, shots = key.split("|")
        shots = S["shots"] if shots == "5" else 0
        L.append(f"| {'pruning-time pool' if split == 'pool' else 'held-out'}, {shots}-shot | {c['n']} | {100 * c['dense_acc']:.2f} | "
                 f"{100 * c['pruned_acc']:.2f} | {pts(c['overall'])} {ci(c['overall_ci'])} | "
                 + " | ".join(f"{pts(c[k])} {ci(c[f'{k}_ci'])}" for k in cats) + f" | {c['worst']} |")
    L += ["", "## STEM contrasts", "", "| contrast | estimate | 95 % interval |", "|---|---:|---|"]
    for k, v in S["stem_contrasts"].items():
        L.append(f"| {k} | {pts(v['estimate'])} | {ci(v['ci95'])} |")
    a = prov["audit_provenance"]
    L += ["", "## Provenance and checks", "",
          f"* Checkpoint: state `{prov['state_id']}`, mask sha256 `{prov['mask_sha256'][:16]}...`, sparsity "
          f"{prov['checkpoint_sparsity']:.6f} (matches the search state).",
          f"* Procedure (every cell): {prov['procedure']}.",
          f"* Scoring code: git HEAD `{prov['code']['git_head'][:12]}`, uncommitted changes to scoring/pruning/data code: "
          f"{prov['code']['uncommitted_changes_to_scoring_code']}.",
          f"* Few-shot: {prov['fewshot']['subjects_with_5']}/{prov['fewshot']['subjects']} subjects have 5 exemplars; "
          f"exemplar overlap with scored items: {prov['fewshot']['exemplar_id_overlap_with_scored']} ids, "
          f"{prov['fewshot']['exemplar_text_overlap_with_scored']} question texts; scored items identical to an exemplar "
          f"(question and choices) excluded from every cell: {S.get('excluded_items') or 'none'}.",
          "* Prompt lengths: " + "; ".join(
              f"{s}: longest 0-shot prompt {v['max_0shot_tokens']} tokens ({v['zero_shot_over_2048']} over 2048), "
              f"{v['five_shot_items_with_fewer_than_5_exemplars']} 5-shot prompts dropped exemplars to fit"
              for s, v in prov["prompts"].items()) + ".",
          f"* Held-out 5-shot = the run's final audit, reused: {prov['audit_reused']} "
          f"(dense: {a['dense']}; pruned: {a['pruned']}; recomputed first chunk agrees: {prov['audit_recompute_check']}).",
          f"* Masked vs materialised predictions on a shared batch: {prov['masked_vs_materialized']}.",
          f"* Pool 0-shot vs cached answers: dense reference {prov.get('dense_pool_0shot_vs_cached_reference')}; "
          f"pruned vs the gate's cache {prov['pruned_pool_0shot_vs_gate_cache']}.",
          f"* Bootstrap: {prov['bootstrap']}.",
          f"* GPU scoring {prov['gpu_scoring_seconds']:.0f} s, wall {prov['wall_seconds']:.0f} s."]
    return "\n".join(L) + "\n"
