"""Matched-compression signal test (``capprune matched-signals register|score|analyze``), plan docs/matched_signal_plan.md.

register  Writes the stratified two-half split of the gate pool and records the SHA-256 of the plan and the split.
score     GPU: rebuilds and verifies every stall parent's 11 equal-size candidates, scores the dense model and every
          candidate on the whole pool in the pool-signals layout (identical batches), reusing pool-signals answers
          whose provenance matches; saves the four letters' log-probabilities.
analyze   CPU: subsample signals on half 1 (S0 net flip, S2 KL, S1c correct-answer log-prob change on dense-correct
          items), half-2 net flip targets, Spearman correlations, top-1 choice regret (with gradient-rank-1 and random
          references), pass/fail call accuracy at 3 points, the lead criterion.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("matched_signals")
REPO = Path(__file__).resolve().parents[1]


def _ms(cfg: dict) -> dict:
    return cfg["matched_signals"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _ms(cfg)["out_dir"])


def plan(cfg: dict) -> tuple[dict, str]:
    p = resolve_path(cfg, _ms(cfg)["plan_file"])
    text = p.read_bytes().replace(b"\r\n", b"\n")
    return json.loads(re.search(rb"```json\s*(\{.*?\})\s*```", text, re.S).group(1)), hashlib.sha256(text).hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


# ============================================================================ pure building blocks (CPU-tested)
def make_split(qids_by_cat: dict[str, list[str]], seed: int) -> dict[str, dict[str, list[str]]]:
    """Per category: sorted qids, seeded permutation, the first ceil(n/2) -> half 1, the rest -> half 2."""
    out = {"half1": {}, "half2": {}}
    for c in sorted(qids_by_cat):
        q = sorted(qids_by_cat[c])
        perm = np.random.default_rng([int(seed), zlib.crc32(c.encode())]).permutation(len(q))
        k = (len(q) + 1) // 2
        out["half1"][c] = sorted(q[i] for i in perm[:k])
        out["half2"][c] = sorted(q[i] for i in perm[k:])
    return out


def subsample(half1: dict[str, list[str]], n: int, draw: int, seed: int) -> dict[str, list[str]]:
    """``n`` items per category of half 1, without replacement, seeded by (seed, n, draw, category)."""
    out = {}
    for c in sorted(half1):
        rng = np.random.default_rng([int(seed), int(n), int(draw), zlib.crc32(c.encode())])
        out[c] = [half1[c][i] for i in rng.choice(len(half1[c]), size=min(n, len(half1[c])), replace=False)]
    return out


def _renorm(lp: np.ndarray) -> np.ndarray:
    z = lp.astype(np.float64) - lp.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def item_terms(lp_dense: np.ndarray, lp_cand: np.ndarray, answer: np.ndarray, c_dense: np.ndarray, c_cand: np.ndarray) -> dict:
    """Per-item terms whose means are the signals: flip x in {-1,0,1}, KL over A-D, and the renormalised correct-answer
    log-prob change (kept with a dense-correct mask for S1c)."""
    a, b = _renorm(lp_dense), _renorm(lp_cand)
    idx = np.arange(len(answer))
    return {"flip": c_dense.astype(int) - c_cand.astype(int), "kl": (np.exp(a) * (a - b)).sum(1),
            "dlogp": a[idx, answer] - b[idx, answer], "dense_correct": c_dense.astype(bool)}


def signal_means(t: dict, sel: np.ndarray) -> dict:
    """S0, S2 and S1c over the selected items (S1c only over the dense-correct ones)."""
    dc = sel & t["dense_correct"]
    return {"S0": float(t["flip"][sel].mean()), "S2": float(t["kl"][sel].mean()),
            "S1c": float(t["dlogp"][dc].mean()) if dc.any() else float("nan")}


def choose(scores: list[float], ranks: list[int]) -> int:
    """Index of the action with the lowest score; ties -> lower gradient rank."""
    return min(range(len(scores)), key=lambda i: (round(scores[i], 12), ranks[i]))


def spearman(x, y) -> float:
    """Spearman correlation with average ranks for ties; NaN if either side does not vary (or n < 3)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    for v, r in ((x, rx), (y, ry)):  # average ranks for ties
        for u in np.unique(v):
            m = v == u
            r[m] = r[m].mean()
    if len(x) < 3 or rx.min() == rx.max() or ry.min() == ry.max():
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


SIGNALS = ("S0", "S2", "S1c")


def decide(ranks: list[int], vals: list[dict], target: list[dict], tol: float, calib: dict | None = None) -> dict:
    """One parent at one (n, draw). ``vals[j][category][signal]`` are action j's subsample signals and
    ``target[j][category]`` its half-2 net flip loss. Returns, per signal, the regret of its top-1 choice (choice =
    lowest worst-category value, ties -> lower gradient rank), the per-category Spearman correlation averaged over the
    categories where both sides vary, and the accuracy of its pass/fail call (S2/S1c mapped to net flip by ``calib``
    {signal: (intercept, slope)}); plus the gradient-rank-1 and random-choice reference regrets."""
    cats = sorted(target[0])
    worst = [max(t.values()) for t in target]
    best = min(worst)
    truth = [w <= tol + 1e-12 for w in worst]
    out = {"regret": {}, "spearman": {}, "call_acc": {}}
    for s in SIGNALS:
        agg = [float(np.nanmax([v[k][s] for k in cats])) for v in vals]
        out["regret"][s] = worst[choose(agg, ranks)] - best
        sp = [spearman([v[k][s] for v in vals], [t[k] for t in target]) for k in cats]
        out["spearman"][s] = float(np.nanmean(sp)) if not np.all(np.isnan(sp)) else float("nan")
        a, b = (0.0, 1.0) if s == "S0" else calib[s]
        calls = [max(a + b * v[k][s] for k in cats) <= tol + 1e-12 for v in vals]
        out["call_acc"][s] = float(np.mean([x == y for x, y in zip(calls, truth)]))
    out["regret"]["gradient rank 1"] = worst[ranks.index(1)] - best
    out["regret"]["random (expected)"] = float(np.mean(worst)) - best
    return out


def lead_n(sizes: list[int], kl: dict, s0: dict) -> int | None:
    """Smallest n whose mean KL regret is below S0's at n and at every larger n in the grid."""
    for n in sorted(sizes):
        if all(kl[m] < s0[m] - 1e-12 for m in sizes if m >= n):
            return n
    return None


# ============================================================================ register
def register(cfg: dict) -> Path:
    from .config import data_dir
    from .data.prepare import PreparedData
    from .gradient_audit import run_paths
    from .gradient_live import gate_pool
    from .model import tokenizer_id

    P, plan_sha = plan(cfg)
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    reg = out / "registration.json"
    rc, _ = run_paths(cfg)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    pool, _ = gate_pool(rc, data)
    field = rc["gate"]["capability"].get("domain_field", "category")
    by = {}
    for r in pool:
        by.setdefault(r[field], []).append(r["qid"])
    split = make_split(by, int(P["split_seed"]))
    sp = out / "split.json"
    body = {"split_seed": P["split_seed"], "pool_items": len(pool), **split}
    split_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if reg.exists():
        old = read_json(reg)
        if old["plan_sha256_lf"] != plan_sha or old["split_sha256"] != split_sha:
            raise SystemExit("a different plan or split was registered here; use a new out_dir")
        return reg
    if (out / "logprobs.npz").exists() or (out / "scores.sqlite").exists():
        raise SystemExit("scores already exist; registration must precede scoring")
    pf = resolve_path(cfg, _ms(cfg)["plan_file"])
    rel = pf.resolve().relative_to(REPO).as_posix()
    committed = _git("log", "-1", "--format=%H", "--", rel)
    if not re.fullmatch(r"[0-9a-f]{40}", committed) or _git("status", "--porcelain", "--", rel) != "":
        raise SystemExit(f"commit {rel} (unchanged) before registering")
    atomic_write_json(sp, body)
    rec = {"plan_file": rel, "plan_sha256_lf": plan_sha, "plan_committed_in": committed,
           "split_file": str(sp), "split_sha256": split_sha, "split_file_sha256": sha256_file(sp),
           "split_sizes": {h: {c: len(v) for c, v in split[h].items()} for h in ("half1", "half2")},
           "protocol_commit": _ms(cfg).get("protocol_commit"), "registered_at": time.time()}
    atomic_write_json(reg, rec)
    return reg


def check(cfg: dict) -> tuple[dict, dict, dict]:
    P, plan_sha = plan(cfg)
    out = out_dir(cfg)
    reg = read_json(out / "registration.json")
    split = read_json(out / "split.json")
    if reg["plan_sha256_lf"] != plan_sha or hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest() != reg["split_sha256"]:
        raise SystemExit("the plan or the split changed after registration")
    return P, reg, split


# ============================================================================ score (GPU)
SCORING_CODE = ("capprune/pool_signals.py", "capprune/eval/mmlu.py", "capprune/gradient_live.py", "capprune/model.py",
                "capprune/pruning/masks.py", "capprune/pruning/wanda.py", "capprune/adaptive/evaluator.py")


def _attempts(run_dir: Path) -> dict[int, dict]:
    last = {}
    for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return last


def _batches(pool: list[dict], field: str, order_seed: int, batch: int) -> list[tuple[str, int, list[dict]]]:
    """The pool-signals layout: per category (sorted), seeded question order, fixed-size batches."""
    from .adaptive.evaluator import question_order

    by_cat: dict[str, list[dict]] = {}
    for r in pool:
        by_cat.setdefault(r[field], []).append(r)
    out = []
    for cat in sorted(by_cat):
        rmap = {r["qid"]: r for r in by_cat[cat]}
        order = question_order(list(rmap), order_seed, cat)
        for i in range(0, len(order), batch):
            out.append((cat, i, [rmap[q] for q in order[i : i + batch]]))
    return out


def _reuse_source(src: Path, pool_sha: str, protocol: dict) -> dict:
    """Provenance of the pool-signals cache: same pool, protocol, clean committed scoring code, unchanged since."""
    info = {"source": str(src), "used": False, "eligible": False}
    if not (src / "manifest.json").exists():
        info["why_not"] = "no pool-signals manifest"
        return info
    pm = read_json(src / "manifest.json")
    diff = subprocess.run(["git", "diff", "--quiet", pm["commit"], "HEAD", "--", *SCORING_CODE], cwd=REPO).returncode
    info.update(cached_commit=pm["commit"], cached_clean_tree=pm["clean_tree"], pool_match=pm["pool_sha16"] == pool_sha,
                protocol_match=all(pm["protocol"].get(k) == v for k, v in protocol.items()),
                scoring_code_unchanged=diff == 0, models={"dense": "dense", **{s["mask_hash"]: s["state_id"] for s in pm["states"]}})
    info["eligible"] = bool(pm["clean_tree"] and info["pool_match"] and info["protocol_match"] and info["scoring_code_unchanged"])
    return info


def score(cfg: dict) -> Path:
    import torch

    from .adaptive.live import weights_digest
    from .config import data_dir
    from .data.prepare import PreparedData
    from .fixed_state import _geometry
    from .gradient_audit import _clock, run_paths
    from .gradient_live import build_trial, gate_pool, load_dense_copy
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pool_signals import LogprobScorer
    from .pruning.masks import DenseStore, PruneState
    from .pruning.profile import load_profile
    from .ranker import canon, mask_hash
    from .utils.cache import KVCache

    t_all = time.time()
    P, reg, split = check(cfg)
    root = resolve_path(cfg, _ms(cfg)["runs_root"])
    out = out_dir(cfg)
    rc, _ = run_paths(cfg)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    tok = load_tokenizer(rc)
    model = load_model(rc)
    model.eval()
    mode = rc["pruning"]["mode"]
    units = build_units(model, mode, rc["pruning"].get("eligible", []))
    dense = DenseStore(units)
    dm = load_dense_copy(rc, model, units, dense)
    dev = next(model.parameters()).device
    geo = _geometry(rc, units[0].channels)
    tt = rc["testing"]
    shots, max_len, mbt = int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768))
    pool, _ = gate_pool(rc, data)
    field = rc["gate"]["capability"].get("domain_field", "category")
    pool_sha = hashlib.sha256("\n".join(r["qid"] for r in pool).encode()).hexdigest()[:16]
    if {r["qid"] for r in pool} != {q for h in ("half1", "half2") for c in split[h] for q in split[h][c]}:
        raise RuntimeError("the gate pool is not the registered split's pool")
    batches = _batches(pool, field, int(P["order_seed"]), int(P["batch_size"]))
    protocol = {"shots": shots, "max_seq_len": max_len, "max_batch_tokens": mbt, "batch": int(P["batch_size"]),
                "order_seed": int(P["order_seed"]), "weights": "masked bf16"}
    reuse = _reuse_source(root / P["reuse_from"], pool_sha, protocol)
    ro = None
    if reuse["eligible"]:
        ro = sqlite3.connect(f"file:{(root / P['reuse_from'] / 'scores.sqlite').as_posix()}?mode=ro", uri=True)
    cache = KVCache(out / "scores.sqlite")
    budgets: dict[tuple, int] = {}
    res, evals, reused, secs = {}, {}, {}, {}

    def cached(key: str, cat: str, i: int, rows: list[dict]):
        k = f"{key}|{pool_sha}|{cat}|{i}|{len(rows)}"
        rec = cache.get(k)
        if rec is None and reuse["used"] and key in reuse["models"]:
            row = ro.execute("SELECT v FROM kv WHERE k=?", (f"{reuse['models'][key]}|{pool_sha}|{cat}|{i}|{len(rows)}",)).fetchone()
            if row is not None:
                rec = json.loads(row[0])
                cache.put(k, rec)
                reused[key] += len(rows)
        return rec

    def run_model(key: str, sc) -> None:
        res[key], evals[key], reused[key] = {}, 0, 0
        t0 = _clock(dev)
        for cat, i, rows in batches:
            rec = cached(key, cat, i, rows)
            if rec is None:
                r_, b = sc.score(rows, budgets.get((cat, i), mbt))
                if (cat, i) in budgets and b != budgets[(cat, i)]:
                    raise RuntimeError(f"batch layout changed for {key} {cat} {i} (OOM); rerun with a smaller token budget")
                rec = {"items": r_, "budget": b}
                cache.put(f"{key}|{pool_sha}|{cat}|{i}|{len(rows)}", rec)
                evals[key] += len(rows)
            budgets.setdefault((cat, i), int(rec["budget"]))
            if int(rec["budget"]) != budgets[(cat, i)]:
                raise RuntimeError(f"{key} {cat} {i}: answers from a different batch layout")
            res[key].update(rec["items"])
        secs[key] = _clock(dev) - t0

    # ---- candidates: every stall parent's full shortlist, rebuilt exactly as the stall diagnostic built it. Pass 1
    #      verifies all 33 before anything is scored; pass 2 rebuilds each again (must give the verified hash) and scores.
    def walk(score_it: bool):
        for par in P["parents"]:
            rd = root / par["run"]
            diag_dir = root / par["diagnostic"]
            att = _attempts(rd)
            a = att[int(par["attempt"])]
            fr = read_json(diag_dir.parent / "frozen" / "frozen.json")
            dg = read_json(diag_dir / "diagnostic.json")
            if sha256_file(Path(fr["mask_file"])) != fr["mask_sha256"] or fr["state_id"] != a["parent_id"]:
                raise RuntimeError(f"{par['name']}: frozen parent file changed or is not the stalled attempt's parent")
            if dg["mode"] != "stall" or int(dg["stalled_attempt"]) != int(par["attempt"]) or dg["parent"] != a["parent_id"]:
                raise RuntimeError(f"{par['name']}: the diagnostic is not of this stalled attempt")
            if [c["action_key"] for c in dg["candidates"]] != [s["action_key"] for s in a["shortlist"]]:
                raise RuntimeError(f"{par['name']}: the diagnostic's shortlist differs from the journal's")
            state = PruneState(units, mode)
            state.load(fr["mask_file"])
            state.apply(dense)
            if mask_hash(state) != a["parent_mask_hash"]:
                raise RuntimeError(f"{par['name']}: parent mask hash differs from the journal")
            live = {int(a.get("rank", 1)): (int(a["iteration"]), a["trial_mask_hash"])}
            if a.get("fallback_of"):  # the same parent's rank-1 attempt that preceded the stalled fallback
                f = att[int(a["fallback_of"])]
                if f["parent_mask_hash"] != a["parent_mask_hash"]:
                    raise RuntimeError(f"{par['name']}: the fallback's rank-1 attempt has a different parent")
                live[int(f.get("rank", 1))] = (int(f["iteration"]), f["trial_mask_hash"])
            pt = int(a.get("profile_attempt") or a["iteration"])
            act, _, _ = load_profile(rd / "profiles" / f"iter-{pt:06d}.safetensors", dev)
            digest = weights_digest(state)
            dcon = sqlite3.connect(f"file:{(diag_dir / 'pairs.sqlite').as_posix()}?mode=ro", uri=True)
            rcon = sqlite3.connect(f"file:{(rd / 'caches' / 'eval.sqlite').as_posix()}?mode=ro", uri=True)
            for s, drow in zip(a["shortlist"], dg["candidates"]):
                parent_masks = dict(state.masks)
                changed, actual = build_trial(state, dense, s["action"], act, rc, geo)
                h = mask_hash(state)
                if not score_it:
                    # the diagnostic's answers for this mask at its order seed: in its own pairs cache, or - for the
                    # rank it replayed - in the run's answer cache, which it read through (``replayed_from_run_cache``)
                    pre = (f"pair|{h}|{int(dg['order_seed'])}|%",)
                    n_diag = dcon.execute("SELECT COUNT(*) FROM kv WHERE k LIKE ?", pre).fetchone()[0]
                    n_run = rcon.execute("SELECT COUNT(*) FROM kv WHERE k LIKE ?", pre).fetchone()[0]
                    lv = live.get(int(s["rank"]))
                    chk = {"parent": par["name"], "rank": int(s["rank"]), "mask_hash": h,
                           "diagnostic_pairs_keys": n_diag, "run_cache_keys": n_run,
                           "diagnostic_replayed_questions": int(drow["replayed_from_run_cache"]),
                           "channels": sum(actual.values()), "action_key_canonical": canon(s["action"]) == s["action_key"],
                           "live_attempt": lv[0] if lv else None, "matches_live_trial": (h == lv[1]) if lv else None}
                    checks.append(chk)
                    found = n_run > 0 if chk["diagnostic_replayed_questions"] else n_diag > 0
                    if not found or not chk["action_key_canonical"] or chk["matches_live_trial"] is False:
                        raise RuntimeError(f"{par['name']} rank {s['rank']}: rebuilt mask fails verification {chk}")
                    cands.append({"parent": par["name"], "parent_id": a["parent_id"], "attempt": int(par["attempt"]),
                                  "rank": int(s["rank"]), "kind": s["kind"], "layers": s["layers"],
                                  "action_key": s["action_key"], "z_g_mean": s["z_g_mean"], "mask_hash": h,
                                  "channels": sum(actual.values()), "mlp_sparsity": state.global_sparsity(),
                                  "diagnostic_decision": drow["decision"]})
                else:
                    if h != next(c["mask_hash"] for c in cands if (c["parent"], c["rank"]) == (par["name"], int(s["rank"]))):
                        raise RuntimeError(f"{par['name']} rank {s['rank']}: rebuild is not deterministic")
                    if h not in res:
                        run_model(h, LogprobScorer(model, tok, data.mmlu_fewshot, shots, max_len, mbt))
                    log.info("%s rank %d %s: %d scored, %d reused, %.1f s", par["name"], s["rank"], h, evals[h], reused[h], secs[h])
                for n in changed:
                    state.masks[n] = parent_masks[n]
                if changed:
                    state.apply(dense, only=changed)
                if weights_digest(state) != digest:
                    raise RuntimeError("parent not restored exactly after a candidate")
            dcon.close()
            rcon.close()

    cands, checks = [], []
    walk(False)
    log.info("verified %d candidates of %d parents against the journals and the stall diagnostics", len(cands), len(P["parents"]))
    # ---- reuse gate: re-score the first batch of every category with the dense model; the cache must reproduce it exactly
    dsc = LogprobScorer(dm, tok, data.mmlu_fewshot, shots, max_len, mbt)
    if reuse["eligible"]:
        spot = {}
        for cat, i, rows in batches:
            if i != 0:
                continue
            row = ro.execute("SELECT v FROM kv WHERE k=?", (f"dense|{pool_sha}|{cat}|{i}|{len(rows)}",)).fetchone()
            if row is None:
                spot[cat] = None
                continue
            old = json.loads(row[0])
            new, _ = dsc.score(rows, int(old["budget"]))
            spot[cat] = max(abs(x - y) for q in new for x, y in zip(new[q]["logp"], old["items"][q]["logp"]))
        reuse["spot_check_max_abs_diff"] = spot
        reuse["used"] = all(v == 0.0 for v in spot.values())
    log.info("pool-signals reuse: %s", {k: v for k, v in reuse.items() if k != "models"})
    run_model("dense", dsc)
    log.info("dense: %d scored, %d reused, %.1f s", evals["dense"], reused["dense"], secs["dense"])
    walk(True)
    cache.close()
    if ro is not None:
        ro.close()
    qids = [r["qid"] for _, _, rows in batches for r in rows]
    arrays = {"qid": np.array(qids), "category": np.array([r[field] for _, _, rows in batches for r in rows]),
              "answer": np.array([int(r["answer"]) for _, _, rows in batches for r in rows], dtype=np.int8)}
    for key in res:
        arrays[f"logp__{key}"] = np.array([res[key][q]["logp"] for q in qids], dtype=np.float32)
        arrays[f"correct__{key}"] = np.array([res[key][q]["correct"] for q in qids], dtype=np.int8)
    np.savez_compressed(out / "logprobs.npz", **arrays)
    man = {"registration": reg, "commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "",
           "protocol": {**protocol, "logprob": "full-vocabulary log-softmax at the last position, letters A-D"},
           "pool_items": len(qids), "pool_sha16": pool_sha, "reuse": {k: v for k, v in reuse.items() if k != "models"},
           "verification": checks, "candidates": cands, "item_model_evaluations_run": evals, "answers_reused": reused,
           "gpu_seconds": secs, "wall_s": time.time() - t_all, "npz_sha256": sha256_file(out / "logprobs.npz")}
    atomic_write_json(out / "manifest.json", man)
    del model, dm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out / "manifest.json"


# ============================================================================ analyze (CPU)
def analyze(cfg: dict) -> Path:
    P, reg, split = check(cfg)
    out = out_dir(cfg)
    man = read_json(out / "manifest.json")
    if man["registration"]["split_sha256"] != reg["split_sha256"] or man["registration"]["plan_sha256_lf"] != reg["plan_sha256_lf"]:
        raise SystemExit("scores were produced under a different registration")
    if sha256_file(out / "logprobs.npz") != man["npz_sha256"]:
        raise SystemExit("logprobs.npz changed after scoring")
    Z = np.load(out / "logprobs.npz")
    pos = {q: i for i, q in enumerate(Z["qid"].tolist())}
    N, cats, tol = len(pos), sorted(split["half1"]), float(P["tolerance"])

    def sel_of(qids: list[str]) -> np.ndarray:
        m = np.zeros(N, bool)
        m[[pos[q] for q in qids]] = True
        return m

    h1 = {k: sel_of(split["half1"][k]) for k in cats}
    h2 = {k: sel_of(split["half2"][k]) for k in cats}
    ans = Z["answer"].astype(int)
    cands = man["candidates"]
    terms = {c["mask_hash"]: item_terms(Z["logp__dense"], Z[f"logp__{c['mask_hash']}"], ans, Z["correct__dense"],
                                        Z[f"correct__{c['mask_hash']}"]) for c in cands}
    by_parent: dict[str, list[dict]] = {}
    for c in cands:
        by_parent.setdefault(c["parent"], []).append(c)
    parents = list(by_parent)
    target = {c["mask_hash"]: {k: float(terms[c["mask_hash"]]["flip"][h2[k]].mean()) for k in cats} for c in cands}
    full_h1 = {c["mask_hash"]: {k: signal_means(terms[c["mask_hash"]], h1[k]) for k in cats} for c in cands}
    # leave-one-parent-out straight-line maps signal -> net flip loss (full half 1 of the other parents' action x category)
    calib = {}
    for p in parents:
        pairs = [full_h1[c["mask_hash"]][k] for q in parents if q != p for c in by_parent[q] for k in cats]
        calib[p] = {}
        for s in ("S2", "S1c"):
            slope, icpt = np.polyfit([x[s] for x in pairs], [x["S0"] for x in pairs], 1)
            calib[p][s] = (float(icpt), float(slope))
    rows = []
    grid = [(int(n), d) for n in P["sizes"] for d in range(int(P["draws"]))]
    for n, d in grid + [("full half 1", 0)]:
        if n == "full half 1":  # supplementary (not in the plan's grid): the whole of half 1
            sel = h1
        else:
            sub = subsample(split["half1"], n, d, int(P["subsample_seed"]))
            sel = {k: sel_of(sub[k]) for k in cats}
        for p, cs in by_parent.items():
            vals = [{k: signal_means(terms[c["mask_hash"]], sel[k]) for k in cats} for c in cs]
            r = decide([c["rank"] for c in cs], vals, [target[c["mask_hash"]] for c in cs], tol, calib[p])
            rows.append({"parent": p, "n": n, "draw": d, **r})

    def agg(key: str, s: str, n, scope: str) -> dict:
        v = [r[key][s] for r in rows if r["n"] == n and scope in ("pooled", r["parent"])]
        v = [x for x in v if x == x]
        return {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v)) if v else None, "k": len(v)}

    refs = ("gradient rank 1", "random (expected)")
    summary = {"regret": {}, "spearman": {}, "call_acc": {}, "kl_minus_s0_regret": {}}
    for n in [*map(int, P["sizes"]), "full half 1"]:
        for scope in ["pooled", *parents]:
            tag = f"{n}|{scope}"
            summary["regret"][tag] = {s: agg("regret", s, n, scope) for s in (*SIGNALS, *refs)}
            summary["spearman"][tag] = {s: agg("spearman", s, n, scope) for s in SIGNALS}
            summary["call_acc"][tag] = {s: agg("call_acc", s, n, scope) for s in SIGNALS}
            diffs = [r["regret"]["S2"] - r["regret"]["S0"] for r in rows if r["n"] == n and scope in ("pooled", r["parent"])]
            summary["kl_minus_s0_regret"][tag] = {"mean": float(np.mean(diffs)), "share_kl_better": float(np.mean([x < -1e-12 for x in diffs])),
                                                  "share_tie": float(np.mean([abs(x) <= 1e-12 for x in diffs])),
                                                  "share_s0_better": float(np.mean([x > 1e-12 for x in diffs]))}
    sizes = [int(n) for n in P["sizes"]]
    lead = lead_n(sizes, {n: summary["regret"][f"{n}|pooled"]["S2"]["mean"] for n in sizes},
                  {n: summary["regret"][f"{n}|pooled"]["S0"]["mean"] for n in sizes})
    truth = {p: [max(target[c["mask_hash"]].values()) <= tol + 1e-12 for c in cs] for p, cs in by_parent.items()}
    truth["pooled"] = [x for p in parents for x in truth[p]]
    base = {p: {"pass_rate": float(np.mean(v)), "majority_class_accuracy": float(max(np.mean(v), 1 - np.mean(v)))}
            for p, v in truth.items()}
    land = {p: [{"rank": c["rank"], "kind": c["kind"], "layers": c["layers"], "z_g_mean": c["z_g_mean"],
                 "diagnostic_decision": c["diagnostic_decision"], "half2_worst": max(target[c["mask_hash"]].values()),
                 "half2": target[c["mask_hash"]], "half1_full": full_h1[c["mask_hash"]]} for c in cs]
            for p, cs in by_parent.items()}
    S = {"label": "EXPLORATORY: 3 stall parents x 11 equal-size actions; the held-out target is half-2 net flip loss "
                  "on the gate pool; no audit sets",
         "registration": reg, "plan": P,
         "scoring": {k: man[k] for k in ("commit", "clean_tree", "reuse", "item_model_evaluations_run", "answers_reused",
                                         "gpu_seconds", "wall_s", "npz_sha256")},
         "analysis_commit": _git("rev-parse", "HEAD"), "analysis_clean_tree": _git("status", "--porcelain") == "",
         "lead_n": lead, "summary": summary, "truth_base_rates": base, "calibration": calib, "landscape": land,
         "rows": rows}
    atomic_write_json(out / "analysis.json", S)
    atomic_write_text(out / "report.md", render(S))
    return out / "report.md"


def render(S: dict) -> str:
    P, sc, reg = S["plan"], S["scoring"], S["registration"]
    parents = [p for p in S["landscape"]]
    ns = [*map(int, P["sizes"]), "full half 1"]
    pt = lambda v: "n/a" if v is None else f"{100 * v:+.2f}"  # noqa: E731
    ru = sc["reuse"]
    L = ["# Matched-compression signal test (EXPLORATORY; no audit sets)", "",
         f"Plan `{reg['plan_file']}` (sha256 LF `{reg['plan_sha256_lf'][:16]}`, committed in `{reg['plan_committed_in'][:12]}`); "
         f"split sha256 `{reg['split_sha256'][:16]}`; both registered before scoring. Scoring commit `{sc['commit'][:12]}` "
         f"(clean tree: {sc['clean_tree']}); analysis commit `{S['analysis_commit'][:12]}` (clean: {S['analysis_clean_tree']}).",
         f"{sum(sc['item_model_evaluations_run'].values()):,} item-model evaluations run, "
         f"{sum(sc['answers_reused'].values()):,} answers reused from pool-signals (reuse used: {ru.get('used')}, spot-check "
         f"max |Δ log p| {ru.get('spot_check_max_abs_diff')}); {sum(sc['gpu_seconds'].values()):.0f} GPU s.", "",
         f"**Lead criterion** (smallest n per category from which KL's pooled mean regret stays below the net-flip "
         f"statistic's): **{S['lead_n'] if S['lead_n'] else 'none in the grid'}**.", "",
         "Regret = the chosen action's worst-category half-2 net flip loss minus the best of the parent's 11 actions "
         "(points; oracle = 0). Means over 20 subsample draws (pooled: over 3 parents x 20 draws). The `full half 1` rows "
         "use all of half 1 (one draw) and are supplementary.", "",
         "## Regret of the top-1 choice", "",
         "| n per category | scope | S0 net flip | S2 KL | S1c Δlogp (dense-correct) | gradient rank 1 | random (expected) | KL − S0 | KL better / tie / S0 better |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in ns:
        for scope in ["pooled", *parents]:
            r = S["summary"]["regret"][f"{n}|{scope}"]
            d = S["summary"]["kl_minus_s0_regret"][f"{n}|{scope}"]
            L.append(f"| {n} | {scope} | {pt(r['S0']['mean'])} | {pt(r['S2']['mean'])} | {pt(r['S1c']['mean'])} | "
                     f"{pt(r['gradient rank 1']['mean'])} | {pt(r['random (expected)']['mean'])} | {pt(d['mean'])} | "
                     f"{100 * d['share_kl_better']:.0f} / {100 * d['share_tie']:.0f} / {100 * d['share_s0_better']:.0f} % |")
    L += ["", "## Spearman correlation with the half-2 net flip loss", "",
          "Per category across the 11 actions, averaged over the categories where both sides vary, then over draws.", "",
          "| n | scope | S0 | S2 KL | S1c |", "|---|---|---|---|---|"]
    for n in ns:
        for scope in ["pooled", *parents]:
            r = S["summary"]["spearman"][f"{n}|{scope}"]
            L.append(f"| {n} | {scope} | " + " | ".join("n/a" if r[s]["mean"] is None else f"{r[s]['mean']:+.2f}" for s in SIGNALS) + " |")
    b = S["truth_base_rates"]
    L += ["", "## Pass/fail call on half 2 at 3 points", "",
          "Truth: every category's half-2 net flip loss ≤ 3 points. Pass rate / majority-class accuracy: "
          + "; ".join(f"{p} {100 * b[p]['pass_rate']:.0f} % / {100 * b[p]['majority_class_accuracy']:.0f} %" for p in ["pooled", *parents])
          + ". S2 and S1c are mapped to net flip loss by a straight line fitted on the other two parents' full half 1.", "",
          "| n | scope | S0 | S2 KL (mapped) | S1c (mapped) |", "|---|---|---|---|---|"]
    for n in ns:
        for scope in ["pooled", *parents]:
            r = S["summary"]["call_acc"][f"{n}|{scope}"]
            L.append(f"| {n} | {scope} | " + " | ".join(f"{100 * r[s]['mean']:.0f} %" for s in SIGNALS) + " |")
    L += ["", "## Landscape: half-2 worst-category net flip loss per action (points)", "",
          "| parent | rank | kind | layers | diagnostic gate | half-2 worst | full-half-1 worst S0 | full-half-1 worst KL |",
          "|---|---|---|---|---|---|---|---|"]
    for p, cs in S["landscape"].items():
        for c in sorted(cs, key=lambda c: c["rank"]):
            L.append(f"| {p} | {c['rank']} | {c['kind']} | {c['layers']} | {c['diagnostic_decision']} | {pt(c['half2_worst'])} | "
                     f"{pt(max(v['S0'] for v in c['half1_full'].values()))} | {max(v['S2'] for v in c['half1_full'].values()):.4f} |")
    return "\n".join(L) + "\n"
