"""The confirmatory study (docs/confirmatory_study_plan.md): ``capprune confirmatory <phase> --config <model config>``.

build-data     BEFORE the model is scored on anything: pool MMLU (test + validation) and MMLU-CF validation, drop
               duplicate questions and few-shot exemplars, draw the sealed confirmation set (seeded, stratified by
               category and source), write the rest as the model's screening pool, derive the gate / probe / hold-out
               sets exactly as the live loop will, and register every set by SHA-256 together with the achievable
               power. The confirmation rows go to ``sealed/``; only their ids are visible to screening (overlap check).
admissibility  the pre-registered first scoring of the model: dense 0-shot accuracy on the gate pool.
freeze         after a batch of runs ends: each run's accepted checkpoints, ordered by parameters removed, with mask
               hashes; written and hashed before any confirmation item is read.
confirm        amendment 1 exactly: per run, fixed-sequence testing over the frozen order (materialised checkpoint vs.
               dense in identical batches, Clopper-Pearson union bound, per-category IUT at alpha), stopping at the first
               failure; the run's final checkpoint is also scored as description if it was not reached. Refuses unless
               every run of the batch is frozen and the plan is committed.
report         per-arm confirmed compression, per-category bounds, power achieved, costs and wall time.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("confirmatory")
REPO = Path(__file__).resolve().parents[1]
CATS = ("STEM", "Humanities", "Social Sciences", "Other")


def _c(cfg: dict) -> dict:
    return cfg["confirmatory"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _c(cfg)["out_dir"])


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").split("\n") if line.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def set_sha(qids) -> str:
    """SHA-256 of a set of item ids (sorted, newline-joined): the same as ``gradient_live.qid_set_sha``."""
    return hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest()


# ============================================================================ sources and the split (pure, CPU-tested)
def normalize(r: dict) -> str:
    return re.sub(r"\s+", " ", (str(r["question"]) + " || " + " | ".join(map(str, r["choices"]))).lower()).strip()


def dedupe(rows: list[dict], fewshot: list[dict]) -> tuple[list[dict], dict]:
    """Keep the first occurrence of every question text (rows are in source priority order) and drop questions that
    are few-shot exemplars."""
    shots = {normalize(r) for r in fewshot}
    seen, out, dup, ex = set(), [], 0, 0
    for r in rows:
        k = normalize(r)
        if k in shots:
            ex += 1
            continue
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, {"input": len(rows), "kept": len(out), "duplicates_dropped": dup, "fewshot_exemplars_dropped": ex}


def seal_split(rows: list[dict], conf_per_cat: dict[str, int], reserve_per_cat: dict[str, int], seed: int):
    """The sealed confirmation set: per category, ``min(target, available - reserve)`` items, allocated to the sources
    in proportion to their size (largest remainder) and drawn without replacement by a seeded generator per
    (category, source) from the qid-sorted rows. Everything else is the screening pool."""
    conf, alloc = [], {}
    for cat in CATS:
        rc = [r for r in rows if r["category"] == cat]
        avail = len(rc)
        n = max(0, min(int(conf_per_cat[cat]), avail - int(reserve_per_cat[cat])))
        srcs = sorted({r["source"] for r in rc})
        size = {s: sum(1 for r in rc if r["source"] == s) for s in srcs}
        exact = {s: n * size[s] / avail for s in srcs} if avail else {}
        take = {s: int(np.floor(exact[s])) for s in srcs}
        for s in sorted(srcs, key=lambda s: (-(exact[s] - take[s]), s))[: n - sum(take.values())]:
            take[s] += 1
        for s in srcs:
            pool = sorted((r for r in rc if r["source"] == s), key=lambda r: r["qid"])
            rng = np.random.default_rng([int(seed), zlib.crc32(cat.encode()), zlib.crc32(s.encode())])
            conf += [pool[i] for i in sorted(rng.choice(len(pool), size=take[s], replace=False).tolist())]
        alloc[cat] = {"available": avail, "target": int(conf_per_cat[cat]), "confirmation": n,
                      "shortfall": max(0, int(conf_per_cat[cat]) - n), "by_source": take, "source_sizes": size}
    cq = {r["qid"] for r in conf}
    return conf, [r for r in rows if r["qid"] not in cq], alloc


def power_at(n: int, q: float, mu: float, alpha: float = 0.05, tol: float = 0.03, draws: int = 2000, seed: int = 0) -> float:
    """Probability that one category certifies (one-sided bound at ``alpha`` <= ``tol``) with ``n`` fresh items, paired
    disagreement rate ``q`` and true loss ``mu``: p+ = (q + mu)/2, p- = (q - mu)/2 (seeded simulation)."""
    from .screen_confirm import certifies

    pp, pm = (q + mu) / 2, (q - mu) / 2
    if pm < 0:
        pp, pm = mu, 0.0
    if n <= 0:
        return 0.0
    k = np.random.default_rng([int(seed), int(n)]).multinomial(n, [pp, pm, 1 - pp - pm], size=draws)
    return float(np.mean([certifies(n, int(a), int(b), alpha, tol) for a, b, _ in k]))


def power_table(n_by_cat: dict[str, int], q_by_cat: dict[str, dict[str, float]], losses, alpha: float, tol: float) -> dict:
    """Per-category power and the joint power (every category certifies; categories independent) of the IUT."""
    out = {}
    for which in ("final", "median", "max"):
        for mu in losses:
            per = {c: power_at(n_by_cat[c], q_by_cat[c][which], float(mu), alpha, tol) for c in CATS}
            out[f"{which}|{mu}"] = {"per_category": per, "joint": float(np.prod(list(per.values())))}
    return out


# ============================================================================ build-data
def source_rows(cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """(pooled rows in source priority order, few-shot exemplars, provenance): MMLU test and validation from the
    prepared files (identical to ``cais/mmlu`` at the pinned revision), then MMLU-CF validation (pinned files)."""
    from .gradient_audit import _audit_rows

    src = _c(cfg)["sources"]
    d = resolve_path(cfg, src["mmlu_prepared_dir"])
    files = {f: sha256_file(d / f) for f in ("mmlu_probe.jsonl", "mmlu_final.jsonl", "mmlu_dev.jsonl", "mmlu_fewshot.jsonl")}
    for f, want in (src.get("mmlu_file_sha256") or {}).items():
        if files[f] != want:
            raise RuntimeError(f"{d / f}: sha256 {files[f]} != pinned {want}")
    test = [{**r, "source": "mmlu/test"} for f in ("mmlu_probe.jsonl", "mmlu_final.jsonl") for r in _read_jsonl(d / f)]
    val = [{**r, "source": "mmlu/validation"} for r in _read_jsonl(d / "mmlu_dev.jsonl")]
    fewshot = _read_jsonl(d / "mmlu_fewshot.jsonl")
    cf, cf_info = _audit_rows(cfg, {"kind": "mmlu_cf", **src["mmlu_cf"]}, None)
    cf = [{**r, "source": "mmlu-cf/validation", "domain": r["category"]} for r in cf]
    rows = sorted(test, key=lambda r: r["qid"]) + sorted(val, key=lambda r: r["qid"]) + sorted(cf, key=lambda r: r["qid"])
    return rows, fewshot, {"mmlu": {"dataset": src.get("mmlu_dataset"), "prepared_dir": str(d), "file_sha256": files},
                           "mmlu_cf": cf_info}


def _regate_q(cfg: dict) -> dict[str, dict[str, float]]:
    """Planning disagreement rates (Qwen3-1.7B run A, as in the amendment-1 sample-size table)."""
    sc = _c(cfg)["planning_rates"]
    R = read_json(resolve_path(cfg, sc["regate"]))
    A = R["runs"][sc["run"]]["attempts"]
    final = next(a for a in A if a["attempt"] == int(sc["final_attempt"]))
    out = {}
    for c in CATS:
        qs = [a["full_pool_projection"][c]["q_disagree"] for a in A]
        out[c] = {"final": final["full_pool_projection"][c]["q_disagree"], "median": float(np.median(qs)), "max": float(max(qs))}
    return out


def _run_rc(cfg: dict, method: str | None = None, seed: int = 0) -> dict:
    from .config import run_config

    return run_config(cfg, method or cfg["matrix"]["methods"][0], 0.0, seed)


def build_data(cfg: dict, rows_override=None) -> Path:
    """Seal the model's data (see the module docstring). Refuses to run twice with a different result."""
    from .config import data_dir
    from .data.prepare import PreparedData
    from .gradient_live import screening_split
    from .model import tokenizer_id

    C = _c(cfg)
    out = out_dir(cfg)
    reg_p = out / "registration.json"
    if reg_p.exists():
        raise SystemExit(f"{reg_p} exists: the data of {C['model_key']} are sealed already (use a new out_dir)")
    rows, fewshot, prov = rows_override if rows_override is not None else source_rows(cfg)
    rows, dd = dedupe(rows, fewshot)
    reserve = {c: int(C["probe_per_category"]) + int(C["holdout_per_category"]) + int(C["min_gate_per_category"]) for c in CATS}
    conf, screen, alloc = seal_split(rows, C["confirmation_per_category"], reserve, int(C["split_seed"]))
    # ---- sealed: the confirmation rows (read only by `confirm`), and their ids (read by screening for the overlap check)
    sealed = out / "sealed"
    _write_jsonl(sealed / "confirmation.jsonl", conf)
    atomic_write_json(sealed / "confirmation_qids.json", sorted(r["qid"] for r in conf))
    # ---- the model's data dir: the screening pool replaces the prepared MMLU splits; no confirmation item remains
    rc = _run_rc(cfg)
    dd_dir = data_dir(rc)
    man_p = dd_dir / "manifest.json"
    man = read_json(man_p)
    _write_jsonl(dd_dir / "mmlu_probe.jsonl", sorted(screen, key=lambda r: r["qid"]))
    _write_jsonl(dd_dir / "mmlu_final.jsonl", [])
    _write_jsonl(dd_dir / "mmlu_dev.jsonl", [])
    for f in ("mmlu_probe.jsonl", "mmlu_final.jsonl", "mmlu_dev.jsonl"):
        man["files"][f] = {"sha256": sha256_file(dd_dir / f), "bytes": (dd_dir / f).stat().st_size}
    man.setdefault("counts", {}).update({"mmlu_probe": len(screen), "mmlu_final": 0, "mmlu_dev": 0})
    man.setdefault("info", {})["confirmatory"] = {"screening_pool": "mmlu_probe.jsonl (gate + probe + hold-out)",
                                                  "mmlu_final": "empty (sealed confirmation set is elsewhere)",
                                                  "mmlu_dev": "empty"}
    atomic_write_json(man_p, man)
    # ---- the gate / probe / hold-out sets exactly as the live loop derives them (no registration check yet)
    rs = dict(rc.get("ranked_search") or {})
    rs.pop("split_registration", None)
    split = screening_split({**rc, "ranked_search": rs}, PreparedData(dd_dir, tokenizer_id(rc)))
    field = split["field"]
    sets = {"confirmation": conf, "gate": split["gate"], "probe": split["probe"], "holdout": split["holdout"]}
    all_ids = [r["qid"] for v in sets.values() for r in v]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(rows):
        raise RuntimeError("the four sets are not a disjoint partition of the pooled items")
    texts = {k: {normalize(r) for r in v} for k, v in sets.items()}
    if any(texts["confirmation"] & texts[k] for k in ("gate", "probe", "holdout")):
        raise RuntimeError("a confirmation question text also appears in a screening set")
    n_conf = {c: sum(1 for r in conf if r[field] == c) for c in CATS}
    alpha, tol = float(C["alpha"]), float(C["tolerance"])
    reg = {"model_key": C["model_key"], "model": {k: rc["model"].get(k) for k in ("name_or_path", "revision", "dtype")},
           "model_files": C.get("model_files"), "split_seed": int(C["split_seed"]), "sources": prov, "dedupe": dd,
           "allocation": alloc, "reserve_per_category": reserve,
           "sets": {k: {"sha256": set_sha(r["qid"] for r in v), "n": len(v),
                        "per_category": {c: sum(1 for r in v if r[field] == c) for c in CATS},
                        "per_source": {s: sum(1 for r in v if r["source"] == s) for s in sorted({r["source"] for r in v})}}
                    for k, v in sets.items()},
           "files": {"confirmation_rows": {"path": str(sealed / "confirmation.jsonl"), "sha256": sha256_file(sealed / "confirmation.jsonl")},
                     "confirmation_qids": {"path": str(sealed / "confirmation_qids.json"), "sha256": sha256_file(sealed / "confirmation_qids.json")},
                     "screening_pool": {"path": str(dd_dir / "mmlu_probe.jsonl"), "sha256": man["files"]["mmlu_probe.jsonl"]["sha256"]},
                     "data_manifest": {"path": str(man_p), "sha256": sha256_file(man_p)}},
           "screening_split_settings": {"probe_per_domain": rc["ranker"]["probe_per_domain"], "probe_seed": rc["ranker"]["probe_seed"],
                                        "screening_holdout": rs.get("screening_holdout")},
           "power": {"alpha": alpha, "tolerance": tol, "planning_rates": _regate_q(cfg) if C.get("planning_rates") else None},
           "commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "", "sealed_at": time.time()}
    if reg["power"]["planning_rates"]:
        reg["power"]["table"] = power_table(n_conf, reg["power"]["planning_rates"], C.get("true_losses", [0.0, 0.01, 0.015]),
                                            alpha, tol)
    reg["shortfalls"] = {c: a["shortfall"] for c, a in alloc.items() if a["shortfall"]}
    atomic_write_json(reg_p, reg)
    atomic_write_text(out / "registration.md", render_registration(reg))
    return reg_p


def render_registration(reg: dict) -> str:
    L = [f"# Sealed data: {reg['model_key']} ({reg['model']['name_or_path']} @ {reg['model']['revision']})", "",
         f"Split seed {reg['split_seed']}; dedupe {reg['dedupe']}.", "",
         "| set | n | " + " | ".join(CATS) + " | sha256 |", "|---|---|" + "---|" * len(CATS) + "---|"]
    for k, v in reg["sets"].items():
        L.append(f"| {k} | {v['n']:,} | " + " | ".join(f"{v['per_category'][c]:,}" for c in CATS) + f" | `{v['sha256'][:16]}` |")
    L += ["", "Confirmation allocation (target -> drawn; shortfall):", ""]
    for c, a in reg["allocation"].items():
        L.append(f"* {c}: {a['target']:,} -> {a['confirmation']:,}" + (f" (SHORT by {a['shortfall']:,})" if a["shortfall"] else "")
                 + f"; by source {a['by_source']} of {a['source_sizes']}")
    P = reg["power"]
    if P.get("table"):
        L += ["", f"Power of the per-category one-sided test at alpha {P['alpha']} (3 points), with the planning disagreement "
              "rates of the development model (Qwen3-1.7B run A); joint = every category certifies, categories independent.", "",
              "| rates | true loss | " + " | ".join(CATS) + " | joint |", "|---|---|" + "---|" * len(CATS) + "---|"]
        for k, v in P["table"].items():
            which, mu = k.split("|")
            L.append(f"| {which} | {100 * float(mu):.1f} pts | " + " | ".join(f"{v['per_category'][c]:.2f}" for c in CATS)
                     + f" | {v['joint']:.2f} |")
    return "\n".join(L) + "\n"


# ============================================================================ admissibility (the first scoring)
def admissibility(cfg: dict) -> Path:
    """Dense 0-shot accuracy on the gate pool, per category; the plan's threshold decides whether the model stays."""
    from .config import data_dir
    from .data.prepare import PreparedData
    from .eval.mmlu import MMLUScorer
    from .gradient_live import screening_split
    from .model import load_model, load_tokenizer, tokenizer_id

    C = _c(cfg)
    out = out_dir(cfg)
    p = out / "admissibility.json"
    if p.exists():
        return p
    _require_plan_committed(cfg)
    rc = _run_rc(cfg)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    split = screening_split(rc, data)  # verifies the sealed registration
    t = rc["testing"]
    sc = MMLUScorer(load_model(rc), load_tokenizer(rc), data.mmlu_fewshot, int(t.get("shots", 0)),
                    int(rc["model"].get("max_seq_len", 2048)), {}, "admissibility", int(t.get("max_batch_tokens", 32768)))
    t0 = time.time()
    res = sc.score(split["gate"])
    acc = {c: float(np.mean([res[r["qid"]]["correct"] for r in split["gate"] if r[split["field"]] == c])) for c in CATS}
    overall = float(np.mean([res[r["qid"]]["correct"] for r in split["gate"]]))
    th = C["admissibility"]
    ok = overall >= th["overall_min"] and all(v >= th["per_category_min"] for v in acc.values())
    rec = {"model_key": C["model_key"], "items": len(split["gate"]), "accuracy": {"overall": overall, **acc}, "threshold": th,
           "admissible": ok, "seconds": time.time() - t0, "commit": _git("rev-parse", "HEAD")}
    atomic_write_json(p, rec)
    return p


# ============================================================================ freeze
def _mask_file_hash(path: Path) -> str:
    """``ranker.mask_hash`` from a saved mask file (packed bits per unit, names sorted) without building the model."""
    from safetensors import safe_open

    h = hashlib.sha256()
    with safe_open(str(path), framework="np") as f:
        for n in sorted(f.keys()):
            h.update(n.encode())
            h.update(f.get_tensor(n).tobytes())
    return h.hexdigest()[:20]


def _batch_runs(cfg: dict, batch: str) -> list[tuple[str, int, Path]]:
    from .config import run_id, runs_dir

    rc = _run_rc(cfg)
    return [(m, int(s), runs_dir(rc) / run_id(m, 0.0, int(s))) for m, s in _c(cfg)["batches"][batch]]


def _journal(rd: Path) -> dict[int, dict]:
    last = {}
    for line in (rd / "journal.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "iteration":
                last[r["iteration"]] = r
    return last


def freeze(cfg: dict, batch: str) -> Path:
    from .utils.checkpoint import CheckpointStore

    bdir = out_dir(cfg) / "batches" / batch
    fp = bdir / "frozen.json"
    if fp.exists():
        return fp
    runs = []
    for method, seed, rd in _batch_runs(cfg, batch):
        S, _ = CheckpointStore(rd).load_latest()
        if "ranked_search" not in S.get("stages_done", []):
            raise SystemExit(f"{rd.name}: the run has not finished its ranked_search stage")
        J = _journal(rd)
        acc = [a for a in J.values() if a["outcome"] == "accept"]
        order = []
        for a in sorted(acc, key=lambda a: (int(a["params_removed"]), int(a["iteration"]))):
            mf = rd / "masks" / f"{a['cand_id']}.safetensors"
            h = _mask_file_hash(mf)
            if h != a["trial_mask_hash"]:
                raise RuntimeError(f"{rd.name} attempt {a['iteration']}: mask file hash {h} != journal {a['trial_mask_hash']}")
            doms = (a.get("gate") or {}).get("domains") or {}
            order.append({"position": len(order) + 1, "attempt": int(a["iteration"]), "cand_id": a["cand_id"],
                          "params_removed": int(a["params_removed"]), "mlp_sparsity": a["rho_after"], "mask_hash": h,
                          "mask_file": str(mf), "mask_sha256": sha256_file(mf), "rank": a.get("rank", 1),
                          "pool_loss": {d: v.get("est") for d, v in doms.items()} if doms else None,
                          "screen": (a.get("gate") or {}).get("kind")})
        srch = S["search"]
        rec = {"method": method, "seed": seed, "run_id": rd.name, "run_dir": str(rd), "journal_sha256": sha256_file(rd / "journal.jsonl"),
               "stop_reason": srch["stop_reason"], "accepts": srch["accepts"], "attempts": len(J), "order": order,
               "cost": {"gate_questions": sum((a.get("gate") or {}).get("questions") or 0 for a in J.values()),
                        "questions_charged": srch.get("questions", 0), "kl_probe": srch.get("kl_probe"),
                        "counterfactual": srch.get("counterfactual"), "budget": srch.get("budget"),
                        "evaluations": srch.get("evaluations")}}
        atomic_write_json(bdir / f"{rd.name}.order.json", rec)
        runs.append({"run_id": rd.name, "order_sha256": sha256_file(bdir / f"{rd.name}.order.json"), "checkpoints": len(order)})
    atomic_write_json(fp, {"batch": batch, "runs": runs, "frozen_at": time.time(), "commit": _git("rev-parse", "HEAD")})
    return fp


# ============================================================================ confirm
def _require_plan_committed(cfg: dict) -> None:
    """Scoring of any kind needs the plan committed, unchanged, and recording the sealed registration's hash.
    (``confirmatory.require_committed_plan: false`` exists for the CPU smoke test only.)"""
    if _c(cfg).get("require_committed_plan", True) is False:
        return
    pf = resolve_path(cfg, _c(cfg)["plan_file"])
    try:
        rel = pf.resolve().relative_to(REPO).as_posix()
    except ValueError:
        raise SystemExit(f"the plan {pf} is not inside the repository") from None
    if not re.fullmatch(r"[0-9a-f]{40}", _git("log", "-1", "--format=%H", "--", rel)) or _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"commit {rel} (unchanged) before any scoring")
    reg = out_dir(cfg) / "registration.json"
    if reg.exists() and sha256_file(reg)[:16] not in pf.read_text(encoding="utf-8"):
        raise SystemExit(f"the plan does not record the sealed registration's hash ({sha256_file(reg)[:16]})")


def _check_frozen(cfg: dict, batch: str) -> list[dict]:
    bdir = out_dir(cfg) / "batches" / batch
    if not (bdir / "frozen.json").exists():
        raise SystemExit(f"batch {batch} is not frozen: the confirmation items stay sealed")
    fr = read_json(bdir / "frozen.json")
    runs = []
    want = {rd.name for _, _, rd in _batch_runs(cfg, batch)}
    if {r["run_id"] for r in fr["runs"]} != want:
        raise SystemExit(f"batch {batch}: frozen runs differ from the batch definition")
    for r in fr["runs"]:
        p = bdir / f"{r['run_id']}.order.json"
        if sha256_file(p) != r["order_sha256"]:
            raise SystemExit(f"{p} changed after freezing")
        o = read_json(p)
        if sha256_file(Path(o["run_dir"]) / "journal.jsonl") != o["journal_sha256"]:
            raise SystemExit(f"{o['run_id']}: journal changed after freezing")
        for ck in o["order"]:
            if sha256_file(Path(ck["mask_file"])) != ck["mask_sha256"]:
                raise SystemExit(f"{ck['mask_file']} changed after freezing")
        runs.append(o)
    return runs


def _paired_counts(dense: dict, cand: dict, rows: list[dict], field: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for c in CATS:
        rr = [r for r in rows if r[field] == c]
        lost = sum(1 for r in rr if dense[r["qid"]] and not cand[r["qid"]])
        gained = sum(1 for r in rr if cand[r["qid"]] and not dense[r["qid"]])
        out[c] = (len(rr), lost, gained)
    return out


def confirm(cfg: dict, batch: str) -> Path:
    import torch

    from .config import data_dir
    from .data.prepare import PreparedData
    from .gradient_audit import size_check
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pool_signals import LogprobScorer
    from .pruning.masks import PruneState
    from .pruning.materialize import materialize
    from .screen_confirm import fixed_sequence, iut_test
    from .utils.cache import KVCache

    C = _c(cfg)
    _require_plan_committed(cfg)
    runs = _check_frozen(cfg, batch)  # refuses before anything sealed is read
    out = out_dir(cfg)
    reg = read_json(out / "registration.json")
    sealed = out / "sealed" / "confirmation.jsonl"
    if sha256_file(sealed) != reg["files"]["confirmation_rows"]["sha256"]:
        raise SystemExit("the sealed confirmation file changed after sealing")
    rows = _read_jsonl(sealed)
    field = "category"
    conf_sha = reg["sets"]["confirmation"]["sha256"][:16]
    alpha, tol = float(C["alpha"]), float(C["tolerance"])
    rc = _run_rc(cfg)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    # ONE dense model, never modified: every checkpoint is materialised from it (kept channels keep their dense weights,
    # exactly as in the masked model), so only the dense model and one materialised copy are on the GPU at a time
    tok, dm = load_tokenizer(rc), load_model(rc)
    dm.eval()
    for prm in dm.parameters():
        prm.requires_grad_(False)
    mode = rc["pruning"]["mode"]
    units = build_units(dm, mode, rc["pruning"].get("eligible", []))
    dev = next(dm.parameters()).device
    t = rc["testing"]
    shots, max_len, mbt = int(t.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(t.get("max_batch_tokens", 32768))
    from .adaptive.evaluator import question_order

    batches = []
    for c in CATS:
        rmap = {r["qid"]: r for r in rows if r[field] == c}
        order = question_order(sorted(rmap), int(C.get("confirm_order_seed", 2027)), c)
        batches += [(c, i, [rmap[q] for q in order[i : i + 100]]) for i in range(0, len(order), 100)]
    cache = KVCache(out / "confirm" / "scores.sqlite")
    cost = {"items_scored": 0, "dense_items_scored": 0, "items_replayed": 0, "seconds": 0.0, "dense_seconds": 0.0}

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def score_pair(key: str, cand_model) -> tuple[dict, dict]:
        """Correctness of the dense copy and the candidate on every confirmation item, identical batch layouts;
        every batch persisted before it is used."""
        dsc = LogprobScorer(dm, tok, data.mmlu_fewshot, shots, max_len, mbt)
        csc = LogprobScorer(cand_model, tok, data.mmlu_fewshot, shots, max_len, mbt)
        D, T = {}, {}
        for c, i, br in batches:
            pk, dk = f"ckpt|{key}|{conf_sha}|{c}|{i}|{len(br)}", f"dense|{conf_sha}|{c}|{i}|{len(br)}"
            rec = cache.get(pk)
            if rec is None:
                drec = cache.get(dk)
                b = int(drec["budget"]) if drec else mbt
                while True:  # an OOM on either side shrinks the layout; both sides end on the same one
                    if drec is None or int(drec["budget"]) != b:
                        sync()
                        t0 = time.perf_counter()
                        dres, bd = dsc.score(br, b)
                        sync()
                        cost["dense_seconds"] += time.perf_counter() - t0
                        cost["dense_items_scored"] += len(br)
                        drec = {"items": dres, "budget": bd}
                        cache.put(dk, drec)
                    sync()
                    t0 = time.perf_counter()
                    cres, bc = csc.score(br, int(drec["budget"]))
                    sync()
                    cost["seconds"] += time.perf_counter() - t0
                    cost["items_scored"] += len(br)
                    if bc == int(drec["budget"]):
                        break
                    b = bc
                rec = {"d": {q: v["correct"] for q, v in drec["items"].items()}, "t": {q: v["correct"] for q, v in cres.items()},
                       "d_logp": {q: v["logp"] for q, v in drec["items"].items()}, "t_logp": {q: v["logp"] for q, v in cres.items()},
                       "budget": int(drec["budget"])}
                cache.put(pk, rec)
            else:
                cost["items_replayed"] += len(br)
            D.update(rec["d"])
            T.update(rec["t"])
        return D, T

    results = []
    for o in runs:
        log.info("confirming %s: %d frozen checkpoints", o["run_id"], len(o["order"]))
        tested, final_desc = [], None
        seq = []
        for ck in o["order"]:
            st = PruneState(units, mode)
            st.load(ck["mask_file"])  # masks only; the dense model's weights are not touched
            mat, _ = materialize(dm, st, rc)
            mat.eval()
            sz = size_check(rc, dm, dm, mat, st, units)  # kept rows / columns == the dense weights at those indices
            D, T = score_pair(ck["mask_hash"], mat)
            counts = _paired_counts(D, T, rows, field)
            r = iut_test(counts, alpha, tol)
            tested.append({**{k: ck[k] for k in ("position", "attempt", "cand_id", "params_removed", "mlp_sparsity", "mask_hash", "pool_loss")},
                           "materialized_parameters_removed": sz["parameters_removed"],
                           "fraction_of_total_parameters_removed": sz["fraction_of_total_parameters_removed"],
                           "count_matches_masks": sz["count_matches_masks"], "counts": counts, **r})
            seq.append({"id": ck["position"], "params_removed": ck["params_removed"], "counts": counts})
            del mat
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if not r["pass"]:
                break
        fs = fixed_sequence(seq, alpha, tol)  # the same decisions, through the protocol's function
        claim = next((x for x in tested if x["position"] == fs["claim"]), None)
        last = o["order"][-1] if o["order"] else None
        if last is not None and last["position"] not in {x["position"] for x in tested}:
            st = PruneState(units, mode)
            st.load(last["mask_file"])
            mat, _ = materialize(dm, st, rc)
            mat.eval()
            sz = size_check(rc, dm, dm, mat, st, units)
            D, T = score_pair(last["mask_hash"], mat)
            counts = _paired_counts(D, T, rows, field)
            final_desc = {**{k: last[k] for k in ("position", "attempt", "params_removed", "mlp_sparsity", "pool_loss")},
                          "materialized_parameters_removed": sz["parameters_removed"], "counts": counts,
                          **iut_test(counts, alpha, tol), "label": "DESCRIPTIVE ONLY (after the first failure; not part of any claim)"}
            del mat
        rec = {"run_id": o["run_id"], "method": o["method"], "seed": o["seed"], "frozen_checkpoints": len(o["order"]),
               "tested": tested, "claim_position": fs["claim"],
               "confirmed_parameters_removed": claim["materialized_parameters_removed"] if claim else 0,
               "confirmed_fraction_removed": claim["fraction_of_total_parameters_removed"] if claim else 0.0,
               "screened_final": {k: last[k] for k in ("position", "params_removed", "mlp_sparsity")} if last else None,
               "final_descriptive": final_desc}
        atomic_write_json(out / "batches" / batch / f"{o['run_id']}.confirm.json", rec)
        results.append(rec)
    cache.close()
    S = {"batch": batch, "runs": results, "confirmation_items": len(rows), "per_category": {c: sum(1 for r in rows if r[field] == c) for c in CATS},
         "cost": cost, "alpha": alpha, "tolerance": tol, "commit": _git("rev-parse", "HEAD"),
         "clean_tree": _git("status", "--porcelain") == "", "finished_at": time.time()}
    atomic_write_json(out / "batches" / batch / "confirm.json", S)
    return out / "batches" / batch / "confirm.json"


# ============================================================================ report
def report(cfg: dict, batch: str) -> Path:
    C = _c(cfg)
    out = out_dir(cfg)
    bdir = out / "batches" / batch
    reg = read_json(out / "registration.json")
    conf = read_json(bdir / "confirm.json")
    adm = read_json(out / "admissibility.json") if (out / "admissibility.json").exists() else None
    q_plan = reg["power"]["planning_rates"]
    rows = []
    for r in conf["runs"]:
        o = read_json(bdir / f"{r['run_id']}.order.json")
        rd = Path(o["run_dir"])
        J = _journal(rd)
        tim = {k: sum((a.get("timings") or {}).get(k, 0.0) for a in J.values()) for k in
               ("profile_s", "build_s", "gate_s", "gate_trial_scoring_s", "gate_dense_reference_s", "kl_probe_s", "counterfactual_s",
                "rollback_s", "attempt_s")}
        # power achieved: the claimed (or first tested) checkpoint's observed disagreement, at the confirmation n
        ref = next((x for x in r["tested"] if x["position"] == r["claim_position"]), r["tested"][0] if r["tested"] else None)
        pw = None
        if ref:
            q_obs = {c: (ref["counts"][c][1] + ref["counts"][c][2]) / max(1, ref["counts"][c][0]) for c in CATS}
            pw = {f"{mu}": {"per_category": {c: power_at(ref["counts"][c][0], q_obs[c], mu, conf["alpha"], conf["tolerance"]) for c in CATS}}
                  for mu in (0.0, 0.01, 0.015)}
            for v in pw.values():
                v["joint"] = float(np.prod(list(v["per_category"].values())))
            pw["q_observed"] = q_obs
        cf_logs = [a["counterfactual"] for a in J.values() if a.get("counterfactual")]
        rows.append({"run_id": r["run_id"], "method": r["method"], "seed": r["seed"], "stop_reason": o["stop_reason"],
                     "accepts": o["accepts"], "attempts": o["attempts"], "screened_final": r["screened_final"],
                     "claim_position": r["claim_position"], "confirmed_parameters_removed": r["confirmed_parameters_removed"],
                     "confirmed_fraction_removed": r["confirmed_fraction_removed"],
                     "tested": [{k: x[k] for k in ("position", "params_removed", "pass", "upper", "upper_bonferroni_descriptive",
                                                   "estimate", "pool_loss", "counts")} for x in r["tested"]],
                     "final_descriptive": r["final_descriptive"], "power_achieved": pw,
                     "cost": {"screening_gate_questions": o["cost"]["gate_questions"],
                              "questions_charged_to_cap": o["cost"]["questions_charged"],
                              "kl_probe_items": (o["cost"].get("kl_probe") or {}).get("paired_items_charged", 0),
                              "counterfactual_items": (o["cost"].get("counterfactual") or {}).get("paired_items", 0),
                              "seconds": {k: round(v, 1) for k, v in tim.items()}},
                     "counterfactual_logs": cf_logs})
    S = {"batch": batch, "model_key": C["model_key"], "admissibility": adm, "registration_sha256": sha256_file(out / "registration.json"),
         "confirmation": {"items": conf["confirmation_items"], "per_category": conf["per_category"], "cost": conf["cost"]},
         "planning_rates": q_plan, "runs": rows, "commit": _git("rev-parse", "HEAD")}
    atomic_write_json(bdir / "report.json", S)
    atomic_write_text(bdir / "report.md", render_report(S))
    return bdir / "report.md"


def render_report(S: dict) -> str:
    pt = lambda v: "n/a" if v is None else f"{100 * v:+.2f}"  # noqa: E731
    L = [f"# Confirmatory study, batch `{S['batch']}`: {S['model_key']}", ""]
    if S["admissibility"]:
        a = S["admissibility"]
        L += [f"Admissibility (dense 0-shot accuracy on the gate pool, {a['items']:,} items): overall {100 * a['accuracy']['overall']:.1f} %; "
              + ", ".join(f"{c} {100 * a['accuracy'][c]:.1f} %" for c in CATS) + f" -> admissible: {a['admissible']}", ""]
    c = S["confirmation"]
    L += [f"Confirmation set: {c['items']:,} items (" + ", ".join(f"{k} {v:,}" for k, v in c["per_category"].items())
          + f"); scored {c['cost']['items_scored']:,} checkpoint items in {c['cost']['seconds']:.0f} s and "
          f"{c['cost']['dense_items_scored']:,} dense items in {c['cost']['dense_seconds']:.0f} s.", "",
          "| run | arm | stop | accepts / attempts | screened final (params removed) | confirmed checkpoint | confirmed params removed |",
          "|---|---|---|---|---|---|---|"]
    for r in S["runs"]:
        sf = r["screened_final"]
        L.append(f"| {r['run_id']} | {r['method']} | {r['stop_reason']} | {r['accepts']} / {r['attempts']} | "
                 f"{(sf or {}).get('params_removed', 0):,} | {r['claim_position'] or 'none'} | "
                 f"{r['confirmed_parameters_removed']:,} ({100 * r['confirmed_fraction_removed']:.2f} %) |")
    for r in S["runs"]:
        L += ["", f"## {r['run_id']} ({r['method']})", "",
              "| position | params removed | pass | " + " | ".join(f"{c}: est / UB / Bonf. UB / pool" for c in CATS) + " |",
              "|---|---|---|" + "---|" * len(CATS)]
        for x in r["tested"]:
            L.append(f"| {x['position']} | {x['params_removed']:,} | {x['pass']} | " + " | ".join(
                f"{pt(x['estimate'][c])} / {pt(x['upper'][c])} / {pt(x['upper_bonferroni_descriptive'][c])} / "
                f"{pt((x['pool_loss'] or {}).get(c))}" for c in CATS) + " |")
        if r["final_descriptive"]:
            f = r["final_descriptive"]
            L.append(f"| final {f['position']} (descriptive) | {f['params_removed']:,} | ({f['pass']}) | " + " | ".join(
                f"{pt(f['estimate'][c])} / {pt(f['upper'][c])} / {pt(f['upper_bonferroni_descriptive'][c])} / "
                f"{pt((f['pool_loss'] or {}).get(c))}" for c in CATS) + " |")
        pw = r["power_achieved"]
        if pw:
            L.append("")
            L.append("Power achieved (observed disagreement of the confirmed / first tested checkpoint, at the confirmation n): "
                     + "; ".join(f"true loss {100 * float(mu):.1f} pts: joint {v['joint']:.2f} (" + ", ".join(
                         f"{c} {v['per_category'][c]:.2f}" for c in CATS) + ")" for mu, v in pw.items() if mu != "q_observed")
                     + ". Observed q: " + ", ".join(f"{c} {100 * pw['q_observed'][c]:.1f} %" for c in CATS) + ".")
        co = r["cost"]
        L.append("")
        L.append(f"Cost: gate {co['screening_gate_questions']:,} questions; KL probe {co['kl_probe_items']:,} paired items; "
                 f"counterfactual {co['counterfactual_items']:,} paired items; charged to the cap {co['questions_charged_to_cap']:,}. "
                 "Seconds: " + ", ".join(f"{k} {v}" for k, v in co["seconds"].items() if v) + ".")
        for i, cf in enumerate(r["counterfactual_logs"], 1):
            L.append(f"Counterfactual log {i}: picks {cf['picks']}; hold-out worst-category net flip by rank "
                     + ", ".join(f"r{k} {pt(v['worst'])}" for k, v in cf["by_rank"].items()) + ".")
    return "\n".join(L) + "\n"
