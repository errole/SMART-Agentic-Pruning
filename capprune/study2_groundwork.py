"""Study 2 groundwork: EXPLORATORY engineering checks (docs/study2_groundwork.md). Nothing here scores a model on a
benchmark item, seals a model, or registers anything.

``capprune groundwork <phase>``:

* ``wanda-speed --model K``  (GPU) P1 and P2. One resident copy of the model: the prunable linears are restored from
                             the pinned weight files between levels (bit checksum verified). Levels are pruned with the
                             original sequential Wanda (control) and with cached block inputs; each mask hash is compared
                             with the feasibility run's recorded hash. Peak GPU memory per phase: load, prune, C4
                             validation perplexity, and a forward pass shaped like a scoring batch on C4 text.
* ``memory``                 (CPU) P2. The expected peak for Qwen3-4B and Qwen3-14B in bf16, from the pinned configs
                             (meta device) and the measured 1.7B/8B components.
* ``exact-test``             (CPU) P4. The exact score test with the refined nuisance grid (``bounds`` grid v2): its
                             critical values and its exact size on and off the grid.
* ``letters``                (CPU) P5. Qwen3-1.7B, stored outputs only: answer letters, dense predictions, and what the
                             stored correctness says about the pruned models' letter preference.
* ``candidates``             (network, Hub metadata only) P6, and the pins of the Qwen3 size ladder.
* ``power``                  (CPU) planning for the draft plan: the feasibility per-item outcomes resampled to the
                             confirmation-set sizes, paper A's rule applied (and the exact test v2).
* ``accounting``             (CPU) planning: the zeroed share of all parameters at each study-2 grid level, per model.

Data: only ``calibration_pool.safetensors`` (C4 train) and ``ppl_gate.safetensors`` (C4 validation) are read from a
model's prepared data folder, after their hashes are checked against its manifest. No MMLU file is opened, except the
Qwen3-1.7B feasibility evaluation set (items seen) in ``letters``.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

from . import unstructured_feasibility as uf
from .config import resolve_path
from .confirmatory import CATS
from .utils.io import atomic_write_json, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("groundwork")
LABEL = "EXPLORATORY (groundwork; no benchmark item is scored)"
LETTERS = "ABCD"


def _G(cfg: dict) -> dict:
    return cfg["groundwork"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _G(cfg)["out_dir"])


def _meta() -> dict:
    return uf._meta()


# ============================================================================ C4 token files only
def c4_tokens(cfg: dict, key: str):
    """(calibration pool, C4 validation windows) of a model's prepared data folder, hash-checked against its manifest."""
    import torch
    from safetensors.torch import load_file

    d = resolve_path(cfg, uf._M(cfg, key)["data_dir"])
    man = read_json(d / "manifest.json")
    out = []
    for f in ("calibration_pool.safetensors", "ppl_gate.safetensors"):
        want = man["files"][f]["sha256"]
        if sha256_file(d / f) != want:
            raise SystemExit(f"{d / f} differs from its manifest")
        out.append(load_file(str(d / f))["tokens"].long())
    assert isinstance(out[0], torch.Tensor)
    return out[0], out[1], {"data_dir": str(d), "calibration_pool_sha256": man["files"]["calibration_pool.safetensors"]["sha256"],
                            "ppl_gate_sha256": man["files"]["ppl_gate.safetensors"]["sha256"]}


def calibration_indices(pool_size: int, seed: int, n: int) -> list[int]:
    """= ``PreparedData.calibration_indices``."""
    rng = np.random.default_rng([int(seed), 7_919])
    return sorted(rng.choice(pool_size, size=n, replace=False).tolist())


# ============================================================================ P1 / P2 on the GPU
def score_proxy_batch(windows, batch: int, length: int, pad_id: int, seed: int = 0):
    """A batch shaped like a scoring batch of ``batch x length`` tokens, cut from C4 validation text, left-padded to
    random lengths (so the 4-D attention mask is materialised, as in ``LogprobScorer``)."""
    import torch

    flat = windows.reshape(-1)
    if flat.numel() < batch * length:
        raise ValueError("not enough C4 validation tokens for this shape")
    ids = flat[: batch * length].view(batch, length).clone()
    rng = np.random.default_rng([seed, batch, length])
    att = torch.ones_like(ids)
    for j, k in enumerate(rng.integers(0, length // 2, size=batch).tolist()):
        if j == 0:
            continue  # the longest sequence fills the batch, as in the scorer's sorted batches
        ids[j, :k] = pad_id
        att[j, :k] = 0
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    return ids, att, pos


def run_score_proxy(model, windows, shapes, pad_id: int) -> list[dict]:
    """Peak memory of LogprobScorer's forward call (``logits_to_keep=1``, the last position's log-softmax) on C4 text."""
    import torch

    dev = next(model.parameters()).device
    out = []
    for b, length in shapes:
        ids, att, pos = score_proxy_batch(windows, int(b), int(length), pad_id)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model(input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev), use_cache=False,
                      logits_to_keep=1)
            lp = o.logits[:, -1, :].float().log_softmax(-1)
            del o, lp
        torch.cuda.synchronize(dev)
        out.append({"batch": int(b), "length": int(length), "tokens": int(b) * int(length),
                    "seconds": time.perf_counter() - t0, "peak_bytes": int(torch.cuda.max_memory_allocated(dev)),
                    "peak_over_resident_bytes": int(torch.cuda.max_memory_allocated(dev) - base),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(dev))})
    return out


def wanda_speed(cfg: dict, key: str) -> Path:
    """P1 and P2 on one model (see the module docstring). Writes ``<out>/<model>/wanda_speed.json`` after every check."""
    import torch

    from .eval.perplexity import perplexity
    from .model import build_units, load_model
    from .pruning.masks import LiveDenseStore, PruneState
    from .pruning.reload import fingerprint, restore_dense
    from .pruning.wanda import sequential_wanda

    G, U = _G(cfg), cfg["unstructured"]
    M = uf._M(cfg, key)
    rc = uf.model_cfg(cfg, key)
    od = out_dir(cfg) / key
    od.mkdir(parents=True, exist_ok=True)
    feas = uf.model_dir(cfg, key)  # the feasibility run of this model: recorded hashes, times, perplexities
    rec_dense = read_json(feas / "dense.json")
    files = uf.verify_model_files(M)
    snap = uf.snapshot_dir(M)
    pool, ppl_windows, data_info = c4_tokens(cfg, key)
    cal = U["calibration"]
    idx = calibration_indices(pool.shape[0], int(cal["seed"]), int(cal["windows"]))
    if idx != rec_dense["calibration"]["indices"] or data_info["calibration_pool_sha256"] != rec_dense["calibration"]["pool_sha256"]:
        raise SystemExit("the calibration windows differ from the feasibility run's")
    calib = pool[idx]
    dev = torch.device("cuda")
    gpu_busy = [ln for ln in _nvidia_compute_apps() if "python" in ln.lower() and not ln.startswith(f"{os.getpid()},")]
    res = {"label": LABEL, "model": key, "revision": M["model"]["revision"], "weights": files, "data": data_info,
           "calibration": {"seed": int(cal["seed"]), "windows": int(cal["windows"]), "batch_size": int(cal.get("batch_size", 4)),
                           "indices_match_feasibility": True},
           "other_python_gpu_processes_at_start": gpu_busy, "checks": [], "meta": _meta()}
    if gpu_busy:
        raise SystemExit(f"another python process is on the GPU: {gpu_busy}")
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    model = load_model(rc)
    torch.cuda.synchronize(dev)
    elig = rc["pruning"]["eligible"]
    units = build_units(model, "unstructured", elig)
    total = int(sum(p.numel() for p in model.parameters()))
    res["load"] = {"seconds": time.perf_counter() - t0, "resident_bytes": int(torch.cuda.memory_allocated(dev)),
                   "peak_bytes": int(torch.cuda.max_memory_allocated(dev)), "parameters": total,
                   "parameter_bytes": int(sum(p.numel() * p.element_size() for p in model.parameters())),
                   "prunable_parameters": int(sum(u.weight.numel() for u in units))}
    t1 = time.perf_counter()
    fp0 = fingerprint(units)
    res["load"]["fingerprint_seconds"] = time.perf_counter() - t1
    live = LiveDenseStore(units)
    tok_pad = _pad_id(rc)
    out_p = od / "wanda_speed.json"
    for i, ch in enumerate(G["checks"][key]):
        s, method = float(ch["level"]), str(ch["method"])
        name = uf.level_name(s)
        cached = method == "cached"
        entry: dict = {"level": s, "name": name, "method": method}
        if i > 0:  # every level starts from the dense weights, restored from the pinned files
            r = restore_dense(model, units, snap)
            t2 = time.perf_counter()
            fp = fingerprint(units)
            r["fingerprint_seconds"] = time.perf_counter() - t2
            r["fingerprint_match"] = fp == fp0
            entry["restore"] = r
            if fp != fp0:
                raise SystemExit("the restored dense weights differ from the loaded ones")
            live.reset()
        st = PruneState(units, "unstructured", mask_device="cpu" if cached else None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t3 = time.perf_counter()
        sequential_wanda(model, st, live, calib, s, rc, batch_size=int(cal.get("batch_size", 4)), cache_inputs=cached,
                         offload_masks=cached)
        torch.cuda.synchronize(dev)
        entry["prune_seconds"] = time.perf_counter() - t3
        entry["prune_peak_bytes"] = int(torch.cuda.max_memory_allocated(dev))
        entry["prune_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
        mh = uf.mask_hash(st.masks)
        stats = uf.mask_stats(st, units, s, total)
        recm = read_json(feas / "levels" / name / "mask.json")
        recr = read_json(feas / "levels" / name / "result.json")
        entry.update({"mask_hash": mh, "recorded_mask_hash": recm["mask_hash"], "hash_matches_feasibility": mh == recm["mask_hash"],
                      "per_row_exact": stats["per_row_exact"], "masked_weights_not_zero": stats["masked_weights_not_zero"],
                      "zeroed": stats["zeroed"], "recorded_prune_seconds": recm["prune_seconds"],
                      "recorded_prune_peak_bytes": recm["prune_peak_bytes"],
                      "recorded_level_peak_bytes": recr["peak_gpu_bytes"]["level"],
                      "recorded_seconds": recr["seconds"]})
        log.info("%s %s %s: %.1f s (recorded %.1f s), hash %s %s", key, name, method, entry["prune_seconds"],
                 recm["prune_seconds"], mh[:16], "matches" if mh == recm["mask_hash"] else "DIFFERS")
        del st
        gc.collect()
        torch.cuda.empty_cache()
        if ch.get("ppl"):
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            t4 = time.perf_counter()
            ppl = perplexity(model, ppl_windows, {}, "ppl", initial_bs=4)
            torch.cuda.synchronize(dev)
            entry["ppl"] = {"value": ppl, "recorded": recr["ppl"], "equal": ppl == recr["ppl"], "seconds": time.perf_counter() - t4,
                            "peak_bytes": int(torch.cuda.max_memory_allocated(dev)),
                            "reserved_bytes": int(torch.cuda.max_memory_reserved(dev))}
        if ch.get("score_proxy"):
            entry["score_proxy"] = run_score_proxy(model, ppl_windows, G["score_proxy_shapes"], tok_pad)
        res["checks"].append(entry)
        res["meta"] = _meta()
        atomic_write_json(out_p, res)
    res["all_hashes_match"] = all(c["hash_matches_feasibility"] for c in res["checks"])
    res["meta"] = _meta()
    atomic_write_json(out_p, res)
    return out_p


def _pad_id(rc: dict) -> int:
    from .model import load_tokenizer

    tok = load_tokenizer(rc)
    return int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)


def _nvidia_compute_apps() -> list[str]:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# ============================================================================ P5: letters (CPU, stored outputs only)
def chi2_sf_even(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution with an even number of degrees of freedom (closed form)."""
    import math

    if df % 2:
        raise ValueError("even df only")
    k, h = df // 2, x / 2.0
    term, acc = 1.0, 1.0
    for i in range(1, k):
        term *= h / i
        acc += term
    return float(math.exp(-h) * acc)


def change_table(answer: np.ndarray, dense: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Rows: answer letter A-D; columns: lost (dense right, pruned wrong), gained, unchanged."""
    t = np.zeros((4, 3), dtype=np.int64)
    for L in range(4):
        m = answer == L
        lost = int(np.count_nonzero(m & dense & ~cand))
        gained = int(np.count_nonzero(m & ~dense & cand))
        t[L] = [lost, gained, int(m.sum()) - lost - gained]
    return t


def homogeneity(t: np.ndarray) -> dict:
    """Pearson chi-square test that the lost / gained / unchanged split is the same for every answer letter."""
    t = t.astype(float)
    exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    ok = exp > 0
    x2 = float((((t - exp) ** 2)[ok] / exp[ok]).sum())
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return {"chi2": x2, "df": df, "p": chi2_sf_even(x2, df)}


def letters(cfg: dict) -> Path:
    """P5 on Qwen3-1.7B from the feasibility run's stored files: answer letters, the dense model's predicted letters
    (stored in the score cache's dense records), and per level the change split by answer letter. The pruned models'
    predicted letters were not stored (the cache keeps only their correctness); nothing is rescored."""
    import sqlite3

    key = "qwen3-1.7b"
    md = uf.model_dir(cfg, key)
    U = cfg["unstructured"]
    es = read_json(md / "eval" / "eval_set.json")
    if sha256_file(md / "eval" / "items.jsonl") != es["items_file_sha256"]:
        raise SystemExit("the evaluation items changed after they were fixed")
    rows = {r["qid"]: r for r in uf._read_jsonl(md / "eval" / "items.jsonl")}
    con = sqlite3.connect(f"file:{(md / 'eval' / 'scores.sqlite').as_posix()}?mode=ro", uri=True)
    dense_pred, kinds, ckpt_fields = {}, {}, set()
    for k, v in con.execute("SELECT k, v FROM kv"):
        kind = k.split("|", 1)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
        rec = json.loads(v)
        if kind == "dense":
            for q, it in rec["items"].items():
                dense_pred[q] = (int(it["pred"]), int(it["correct"]), [float(x) for x in it["logp"]])
        elif kind == "ckpt":
            ckpt_fields |= {type(x).__name__ for x in rec["t"].values()}
    con.close()
    with np.load(md / "dense.npz", allow_pickle=False) as z:
        qids, cats, dense = list(z["qids"]), z["category"], z["dense"].astype(bool)
    answer = np.array([int(rows[q]["answer"]) for q in qids])
    dpred = np.array([dense_pred[q][0] for q in qids])
    dlogp = np.array([dense_pred[q][2] for q in qids])
    if not np.array_equal(np.array([bool(dense_pred[q][1]) for q in qids]), dense) or not np.array_equal(dpred == answer, dense):
        raise SystemExit("the stored dense predictions disagree with the dense record")
    res = {"label": LABEL, "model": key, "items": len(qids),
           "stored": {"score_cache_records": kinds,
                      "dense_records": "per item: 4 letter log-probabilities, predicted letter, correct",
                      "pruned_records": f"per item: correctness only ({sorted(ckpt_fields)}); predicted letters NOT stored",
                      "rescored": False},
           "answer_letters": {}, "dense_predicted_letters": {}, "levels": []}
    for c in list(CATS) + ["all"]:
        m = np.ones(len(qids), bool) if c == "all" else cats == c
        res["answer_letters"][c] = {LETTERS[L]: int(np.count_nonzero(m & (answer == L))) for L in range(4)}
        res["dense_predicted_letters"][c] = {LETTERS[L]: int(np.count_nonzero(m & (dpred == L))) for L in range(4)}
    for s in sorted(float(x) for x in U["levels"]):
        ld = md / "levels" / uf.level_name(s)
        r = read_json(ld / "result.json")
        if sha256_file(ld / "outcomes.npz") != r["outcomes_sha256"]:
            raise SystemExit(f"{ld / 'outcomes.npz'} changed after it was written")
        with np.load(ld / "outcomes.npz", allow_pickle=False) as z:
            if list(z["qids"]) != qids or not np.array_equal(z["dense"], dense):
                raise SystemExit(f"{ld}: items or dense answers differ")
            cand = z["cand"].astype(bool)
        lv = {"level": s, "name": uf.level_name(s), "worst": r["worst"], "worst_category": r["worst_category"],
              "categories": {}}
        for c in list(CATS) + ["all"]:
            m = np.ones(len(qids), bool) if c == "all" else cats == c
            t = change_table(answer[m], dense[m], cand[m])
            n_l = t.sum(1)
            right = np.array([int(np.count_nonzero(m & cand & (answer == L))) for L in range(4)])
            wrong_not = np.array([int(np.count_nonzero(m & ~cand & (answer != L))) for L in range(4)])
            lv["categories"][c] = {
                "by_answer_letter": {LETTERS[L]: {"n": int(n_l[L]), "lost": int(t[L, 0]), "gained": int(t[L, 1]),
                                                  "loss": float((t[L, 0] - t[L, 1]) / n_l[L]) if n_l[L] else None,
                                                  "dense_acc": float(dense[m & (answer == L)].mean()) if n_l[L] else None,
                                                  "pruned_acc": float(cand[m & (answer == L)].mean()) if n_l[L] else None}
                                     for L in range(4)},
                "pruned_predicted_letter_bounds": {LETTERS[L]: [int(right[L]), int(right[L] + wrong_not[L])]
                                                   for L in range(4)},
                "loss": float((t[:, 0].sum() - t[:, 1].sum()) / t.sum()),
                "homogeneity": homogeneity(t)}
        lv["bias_shift"] = bias_shift(dlogp, answer, dense, cand)
        res["levels"].append(lv)
    res["meta"] = _meta()
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    atomic_write_json(od / "letters.json", res)
    return od / "letters.json"


def bias_shift(dlogp: np.ndarray, answer: np.ndarray, dense: np.ndarray, cand: np.ndarray) -> dict:
    """The dense model with a constant letter bias b (added to its four stored letter log-probabilities; b_D = 0) that
    best reproduces the pruned model's accuracy on the items of each answer letter (least squares, weighted by n;
    grid 0.25 on [-3, 3], refined to 0.05). How much of the paired loss such a pure change of letter preference
    produces, and whether the items it loses are the items the pruned model lost."""
    n_l = np.array([np.count_nonzero(answer == L) for L in range(4)])
    target = np.array([cand[answer == L].mean() for L in range(4)])

    def acc_by_letter(pred):
        ok = pred == answer
        return np.array([ok[answer == L].mean() for L in range(4)])

    def err(b):
        return float((n_l * (acc_by_letter((dlogp + b).argmax(1)) - target) ** 2).sum())

    best, best_e = np.zeros(4), err(np.zeros(4))
    for step, half in ((0.25, 3.0), (0.05, 0.25)):
        c0 = best.copy()
        ax = np.arange(-half, half + 1e-9, step)
        for a in ax:
            for b_ in ax:
                for c in ax:
                    b = c0 + np.array([a, b_, c, 0.0])
                    e = err(b)
                    if e < best_e - 1e-12:
                        best, best_e = b, e
    pred = (dlogp + best).argmax(1)
    shifted = pred == answer
    lost_b = dense & ~shifted
    lost_p = dense & ~cand
    n = len(answer)
    return {"bias": [float(x) for x in best], "fit_sq_error": best_e,
            "acc_by_letter_model": [float(x) for x in acc_by_letter(pred)], "acc_by_letter_pruned": [float(x) for x in target],
            "loss_model": float((np.count_nonzero(lost_b) - np.count_nonzero(~dense & shifted)) / n),
            "loss_pruned": float((np.count_nonzero(lost_p) - np.count_nonzero(~dense & cand)) / n),
            "items_lost_by_model": int(np.count_nonzero(lost_b)),
            "pruned_lost_given_model_lost": float(lost_p[lost_b].mean()) if lost_b.any() else None,
            "pruned_lost_given_model_kept_right": float(lost_p[dense & shifted].mean()) if (dense & shifted).any() else None,
            "predicted_letters_model": {LETTERS[L]: int(np.count_nonzero(pred == L)) for L in range(4)}}


# ============================================================================ P4: the exact test's registered fix (CPU)
def _exact_one(args) -> dict:
    from . import bounds as B

    n, tol, alpha, seed = args
    t0 = time.perf_counter()
    r = B.exact_score_critical_v2(int(n), float(tol), float(alpha))
    t_cert = time.perf_counter() - t0
    z2, z1 = r["z"], r["z_v1"]
    top = (1 - tol) / 2 * (1 - 1e-9)
    rng = np.random.default_rng([int(seed), int(n), int(round(1000 * tol))])
    data = (np.arange(0.040, 0.15001, 0.0005) - tol) / 2          # q = 4-15 %, the range the data occupy
    check = np.concatenate([rng.uniform(0.0, top, 2000), rng.uniform(0.9 * top, top, 1000), data])
    s2 = np.array([B.size_point(int(n), float(tol), z2, g) for g in check])
    s1 = np.array([B.size_point(int(n), float(tol), z1, g) for g in check])
    return {**r, "seconds_certificate": t_cert,
            "offgrid_check": {"points": int(len(check)), "seed": [int(seed), int(n), int(round(1000 * tol))],
                              "max_size_v2": float(s2.max()), "argmax_q_v2": float(2 * check[int(s2.argmax())] + tol),
                              "max_size_v1_same_points": float(s1.max()),
                              "argmax_q_v1": float(2 * check[int(s1.argmax())] + tol),
                              "max_size_v2_data_range": float(s2[-len(data):].max()),
                              "v1_points_over_alpha": int((s1 > alpha).sum())}}


def exact_test(cfg: dict) -> Path:
    from concurrent.futures import ProcessPoolExecutor

    from . import bounds as B

    E = _G(cfg)["exact_test"]
    jobs = [(int(n), float(t), float(E["alpha"]), int(E["check_seed"])) for t in E["tolerances"] for n in E["n"]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(jobs), int(E.get("workers", 8)))) as ex:
        out = list(ex.map(_exact_one, jobs))
    res = {"label": "method development (data-free; no real data is used)",
           "definition": {
               "grid": f"v1 grid (400 quadratic points + the boundary points of q in {list(B.V2_EXTRA_Q)}) plus a uniform "
                       f"grid of spacing 1/({B.V2_PER_UNIT} n) in p_gained over [0, (1 - tol)/2)",
               "critical_value": f"largest z (bisection to 1e-9) with the exact size <= alpha - {B.V2_MARGIN:g} at every grid point",
               "certificate": "branch and bound over every grid interval with a Lipschitz bound from the Fisher information "
                              "of the boundary curve (on S and on 2 arcsin sqrt S); certified: the size is <= alpha at every "
                              "nuisance value in [0, (1 - tol)/2)",
               "independent_check": "3,000 seeded random nuisance values (1,000 in the top 10 % of the range) and the data "
                                    "range q = 4-15 % in steps of 0.05 points, with the v2 and the v1 critical values"},
           "results": out,
           "all_certified": all(r["certified"] and r["certified_sup_bound"] <= r["alpha"] for r in out),
           "all_offgrid_at_most_alpha": all(r["offgrid_check"]["max_size_v2"] <= r["alpha"] for r in out),
           "wall_seconds": time.time() - t0, "meta": _meta()}
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    atomic_write_json(od / "exact_test.json", res)
    return od / "exact_test.json"


# ============================================================================ P6: Hub metadata only (no weights)
def candidates(cfg: dict) -> Path:
    """Hub metadata of the second-family candidates and the Qwen3 size ladder: the current commit (to pin), licence,
    gating, parameter count, architecture; config.json (a few kB) where it can be read; the model card's MMLU lines.
    No weight file is downloaded."""
    import re

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    C = _G(cfg)["candidates"]
    out = {"label": LABEL, "checked_at": time.time(), "models": {}}
    for group in ("ladder", "second_family"):
        for repo in C[group]:
            rec: dict = {"group": group}
            try:
                info = api.model_info(repo, files_metadata=False)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                out["models"][repo] = rec
                continue
            card = getattr(info, "card_data", None)
            cg = (lambda k: card.get(k)) if card is not None and hasattr(card, "get") else (lambda k: None)
            st = getattr(info, "safetensors", None)
            rec.update({"sha": info.sha, "last_modified": str(getattr(info, "last_modified", "")),
                        "gated": getattr(info, "gated", None), "licence": cg("license"), "licence_name": cg("license_name"),
                        "base_model": cg("base_model"), "pipeline_tag": getattr(info, "pipeline_tag", None),
                        "safetensors_total": int(st.total) if st is not None and getattr(st, "total", None) else None,
                        "safetensors_dtypes": dict(st.parameters) if st is not None and getattr(st, "parameters", None) else None,
                        "files": sorted(s.rfilename for s in (info.siblings or []))[:80]})
            for f, field in (("config.json", "config"), ("README.md", "card")):
                try:
                    p = hf_hub_download(repo, f, revision=info.sha)
                    txt = Path(p).read_text(encoding="utf-8", errors="replace")
                    if field == "config":
                        cj = json.loads(txt)
                        rec["config"] = {k: cj.get(k) for k in (
                            "architectures", "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings",
                            "torch_dtype", "dtype", "max_position_embeddings", "layer_types", "sliding_window",
                            "use_sliding_window")}
                        rec["config_sha256"] = sha256_file(Path(p))
                        rec["config_path"] = str(p)
                    else:
                        lines = txt.splitlines()
                        rec["card_mmlu_lines"] = [ln.strip()[:400] for ln in lines if re.search(r"MMLU", ln)][:60]
                        rec["card_head"] = [ln.strip()[:200] for ln in lines[:30]]
                except Exception as e:  # noqa: BLE001
                    rec[f"{field}_error"] = f"{type(e).__name__}: {e}"[:300]
            out["models"][repo] = rec
    out["meta"] = _meta()
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    atomic_write_json(od / "candidates.json", out)
    return od / "candidates.json"


# ============================================================================ P2: expected peak for the size ladder (CPU)
def _dims(M: dict) -> dict:
    shapes, total, info = uf.meta_shapes(M)
    big = max(r * c for r, c in shapes.values())
    return {"total_parameters": total, "parameter_bytes_bf16": 2 * total, "prunable_parameters": sum(r * c for r, c in shapes.values()),
            "hidden": info["hidden_size"], "intermediate": info["intermediate_size"], "layers": info["layers"],
            "largest_linear": big, "tie_word_embeddings": info["tie_word_embeddings"]}


def _extras(ws: dict) -> dict:
    """Peak bytes above the resident model, per phase, from a ``wanda_speed.json``."""
    R = ws["load"]["resident_bytes"]
    cached = [c for c in ws["checks"] if c["method"] == "cached"]
    ppl = [c["ppl"]["peak_bytes"] for c in cached if "ppl" in c]
    sp = [s["peak_bytes"] for c in cached for s in c.get("score_proxy", [])]
    return {"resident": R, "prune": max(c["prune_peak_bytes"] for c in cached) - R,
            "ppl": (max(ppl) - R) if ppl else None, "score": (max(sp) - R) if sp else None}


def _scale(src: dict, dst: dict) -> float:
    """Conservative growth of the non-weight memory from one model to another: the largest of the ratios of hidden
    size, intermediate size and largest linear (never below 1 when the target is the smaller model)."""
    return max(1.0, dst["hidden"] / src["hidden"], dst["intermediate"] / src["intermediate"],
               dst["largest_linear"] / src["largest_linear"])


def memory(cfg: dict) -> Path:
    od = out_dir(cfg)
    ws = {k: read_json(od / k / "wanda_speed.json") for k in ("qwen3-1.7b", "qwen3-8b")}
    cand = read_json(od / "candidates.json")["models"]
    models = {"qwen3-1.7b": uf._M(cfg, "qwen3-1.7b"), "qwen3-8b": uf._M(cfg, "qwen3-8b")}
    for key, repo in (("qwen3-4b", "Qwen/Qwen3-4B"), ("qwen3-14b", "Qwen/Qwen3-14B")):
        models[key] = {"model": {"name_or_path": repo, "revision": cand[repo]["sha"]}}
    dims = {k: _dims(M) for k, M in models.items()}
    ex = {k: _extras(v) for k, v in ws.items()}
    card = 97887 * 2**20  # nvidia-smi: memory.total of the RTX PRO 6000 (MiB)
    phases = ("prune", "ppl", "score")
    check = {}  # the rule's prediction of 8B from 1.7B, against the 8B measurement
    f = _scale(dims["qwen3-1.7b"], dims["qwen3-8b"])
    for ph in phases:
        check[ph] = {"predicted": ex["qwen3-1.7b"][ph] * f, "measured": ex["qwen3-8b"][ph],
                     "conservative": ex["qwen3-1.7b"][ph] * f >= ex["qwen3-8b"][ph]}
    proj = {}
    for k in ("qwen3-4b", "qwen3-14b"):
        f8 = _scale(dims["qwen3-8b"], dims[k])
        per = {ph: ex["qwen3-8b"][ph] * f8 for ph in phases}
        R = dims[k]["parameter_bytes_bf16"]
        peak = R + max(per.values())
        proj[k] = {"scale_from_8b": f8, "resident_weights": R, "extra_by_phase": per, "expected_peak": peak,
                   "two_copy_design_weights_only": 2 * R, "share_of_card": peak / card,
                   "bool_masks_on_gpu_if_not_offloaded": dims[k]["prunable_parameters"]}
    res = {"label": LABEL, "card_bytes": card, "measured": {k: {"dims": dims[k], "extras": ex[k],
                                                                 "level_peak_single_copy": ex[k]["resident"] + max(ex[k][p] for p in phases),
                                                                 "recorded_feasibility_level_peak": max(c["recorded_level_peak_bytes"] for c in ws[k]["checks"])}
                                                             for k in ws},
           "rule_check_1.7b_to_8b": {"scale": f, **check}, "projection": proj, "dims": dims,
           "note": "Peak = torch.cuda.max_memory_allocated (tensors only). The CUDA context, the caching allocator's reserve "
                   "and the desktop's own use of the card (about 2.7 GB here) come on top.", "meta": _meta()}
    atomic_write_json(od / "memory.json", res)
    return od / "memory.json"


# ============================================================================ planning: power from the feasibility outcomes (CPU)
def _level_outcomes(cfg: dict, key: str) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """The feasibility run's per-item paired outcomes: (levels, categories, lost[item, level], gained[item, level])."""
    md = uf.model_dir(cfg, key)
    levels, lost, gained, cats, qids = [], [], [], None, None
    for s in sorted(float(x) for x in cfg["unstructured"]["levels"]):
        ld = md / "levels" / uf.level_name(s)
        r = read_json(ld / "result.json")
        if sha256_file(ld / "outcomes.npz") != r["outcomes_sha256"]:
            raise SystemExit(f"{ld / 'outcomes.npz'} changed after it was written")
        with np.load(ld / "outcomes.npz", allow_pickle=False) as z:
            if qids is None:
                qids, cats = list(z["qids"]), z["category"]
            elif list(z["qids"]) != qids:
                raise SystemExit("items differ between levels")
            d, c = z["dense"].astype(bool), z["cand"].astype(bool)
        levels.append(s)
        lost.append(d & ~c)
        gained.append(~d & c)
    return levels, cats, np.stack(lost, 1), np.stack(gained, 1)


def bootstrap_claims(cats, lost, gained, n_conf: dict, reps: int, seed: int, alpha: float, tol: float,
                     z_exact: dict | None = None) -> dict:
    """Confirmation sets of n_conf[c] items per category drawn with replacement from the feasibility items (each item
    keeps its outcomes at every level). Per replicate: the per-category CP union decision at every level, the IUT, and
    the fixed sequence in ascending order. Returns, per level, P(certified within the fixed sequence), and the claim
    distribution; with ``z_exact`` (critical z per n), the same for the exact score test (v2)."""
    from . import bounds as B

    rng = np.random.default_rng(seed)
    nl = lost.shape[1]
    cert_cp = np.ones((reps, nl), bool)
    cert_ex = np.ones((reps, nl), bool) if z_exact else None
    for c in CATS:
        idx = np.flatnonzero(cats == c)
        n = int(n_conf[c])
        cnt = rng.multinomial(n, np.full(len(idx), 1 / len(idx)), size=reps)  # (reps, items)
        kp = cnt @ lost[idx].astype(np.int64)                                   # (reps, levels)
        km = cnt @ gained[idx].astype(np.int64)
        cert_cp &= B.cp_union_region(n, kp.ravel(), km.ravel(), alpha, tol).reshape(reps, nl)
        if z_exact:
            cert_ex &= (B.score_z(n, kp, km, tol) <= z_exact[n])
    out = {}
    for name, cert in (("cp_union", cert_cp), ("exact_v2", cert_ex)):
        if cert is None:
            continue
        seq = np.cumprod(cert, axis=1).astype(bool)        # certified AND every lower level certified
        claim = seq.sum(1)                                  # number of levels claimed
        out[name] = {"p_through_level": [float(x) for x in seq.mean(0)],
                     "claim_counts": [int(np.count_nonzero(claim == k)) for k in range(nl + 1)]}
    return out


def power(cfg: dict) -> Path:
    P = _G(cfg)["power"]
    od = out_dir(cfg)
    alpha, tol = float(P["alpha"]), float(P["tolerance"])
    zx = None
    if (od / "exact_test.json").exists():
        ex = read_json(od / "exact_test.json")
        zx = {int(r["n"]): float(r["z"]) for r in ex["results"] if abs(r["tol"] - tol) < 1e-12}
    res = {"label": "planning (EXPLORATORY feasibility outcomes, resampled; not a prediction of a sealed set)",
           "alpha": alpha, "tolerance": tol, "reps": int(P["reps"]), "models": {}}
    for key in P["models"]:
        levels, cats, lost, gained = _level_outcomes(cfg, key)
        avail = {c: int(np.count_nonzero(cats == c)) for c in CATS}
        mres = {"levels": levels, "feasibility_n": avail, "by_size": {}}
        for mult in P["size_multipliers"]:
            n_conf = {c: int(round(float(mult) * int(P["confirmation_n"][c]))) for c in CATS}
            z_use = zx if (zx and float(mult) == 1.0 and all(n in zx for n in n_conf.values())) else None
            mres["by_size"][str(mult)] = {"n": n_conf, **bootstrap_claims(cats, lost, gained, n_conf, int(P["reps"]),
                                                                          int(P["seed"]), alpha, tol, z_use)}
        res["models"][key] = mres
    res["meta"] = _meta()
    atomic_write_json(od / "power.json", res)
    return od / "power.json"


# ============================================================================ planning: share of all parameters per grid level (CPU)
def accounting_grid(cfg: dict) -> Path:
    """Zeroed share of ALL parameters at each study-2 grid level, from pinned configs on the meta device (no weights):
    the Qwen3 ladder and the second-family candidates whose config.json is readable. A gated candidate whose config
    cannot be read is computed from its published architecture, labelled as such."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from .model import build_units

    A = _G(cfg)["accounting"]
    levels = [float(x) for x in A["levels"]]
    cand = read_json(out_dir(cfg) / "candidates.json")["models"]
    res = {"label": "planning (parameter accounting only)", "levels": levels, "models": {}}
    elig = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for repo in A["models"]:
        c = cand.get(repo, {})
        src = None
        if c.get("config_path"):
            conf = AutoConfig.from_pretrained(str(Path(c["config_path"]).parent))
            src = f"config.json at {c['sha']}"
        elif repo in A.get("published_architectures", {}):
            from transformers import LlamaConfig

            conf = LlamaConfig(**A["published_architectures"][repo])
            src = "published architecture (config.json is gated for this account)"
        else:
            res["models"][repo] = {"error": "no readable config"}
            continue
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(conf)
        units = build_units(m, "unstructured", elig)
        shapes = {u.name: tuple(u.weight.shape) for u in units}
        total = int(sum(p.numel() for p in m.parameters()))
        acc = uf.accounting(shapes, total, levels)
        first10 = next((v["level"] for v in acc["levels"].values() if v["share_of_all"] >= 0.10 - 1e-12), None)
        res["models"][repo] = {"source": src, "total_parameters": total, "prunable_parameters": acc["prunable_parameters"],
                               "prunable_share": acc["prunable_share"],
                               "share_by_level": {f"{100 * v['level']:g}": v["share_of_all"] for v in acc["levels"].values()},
                               "first_level_with_10pct_share": first10}
        del m
    res["meta"] = _meta()
    atomic_write_json(out_dir(cfg) / "accounting.json", res)
    return out_dir(cfg) / "accounting.json"
