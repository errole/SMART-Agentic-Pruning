"""Unstructured feasibility: EXPLORATORY (docs/unstructured_feasibility_plan.md).

Uniform one-shot unstructured Wanda, as published (``pruning/wanda.py::sequential_wanda`` with
``pruning.unstructured_count: floor``), at fixed layer-sparsity levels, on Qwen3-1.7B (the development model; the
compensation experiment's evaluation set) and Qwen3-8B (sealed with the confirmatory procedure; its evaluation set is
drawn from the screening pool only). ``capprune unstructured <phase>``:

* ``attainability``           the parameter accounting of both models and the best reach the level grid allows (CPU,
                              from the pinned config.json; no weights are loaded);
* ``models --model K``        the pinned weight files are hashed; the model is loaded in bf16 on the GPU (no scoring);
* ``build-eval --model K``    the evaluation set (Qwen3-1.7B: the compensation set, hash-checked; Qwen3-8B: drawn from
                              the screening pool, only after the admissibility test passed);
* ``run --model K``           dense, then every level one-shot from the dense model, in ascending order. A level's mask
                              hash is persisted before it is scored and its per-item outcomes before the next level; a
                              level is regenerated (after a crash) only if its regenerated mask hash verifies;
* ``analyze``                 reach, the verdict on Qwen3-8B, the descriptive tables and the figure.

Never opens a sealed confirmation set (the Qwen3-8B registration and admissibility records are read; ``sealed/`` is
not) and refuses any model outside ``unstructured.allowed_models``. Every output records the commit, the tree state and
the GPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

from .config import deep_merge, resolve_path
from .confirmatory import CATS, _git, _read_jsonl, _write_jsonl, set_sha
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("unstructured")
LABEL = "EXPLORATORY"
LABEL_43 = "uniform Wanda at the same overall sparsity, not their method"
NOT_CERT = "not a certificate"
ZERO_NOTE = ("Zeroed weights are still stored and multiplied: unstructured sparsity gives no memory or speed gain on "
             "standard hardware.")


def _U(cfg: dict) -> dict:
    return cfg["unstructured"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _U(cfg)["out_dir"])


def model_dir(cfg: dict, key: str) -> Path:
    return out_dir(cfg) / key


def _M(cfg: dict, key: str) -> dict:
    models = _U(cfg)["models"]
    if key not in models:
        raise SystemExit(f"--model must be one of {sorted(models)}")
    M = models[key]
    if M["model"]["name_or_path"] not in _U(cfg)["allowed_models"]:
        raise SystemExit(f"{M['model']['name_or_path']} is not an allowed model for this experiment")
    return M


def model_cfg(cfg: dict, key: str) -> dict:
    """The experiment config with this model's pinned id/revision and its prepared data folder."""
    M = _M(cfg, key)
    return deep_merge({k: v for k, v in cfg.items()}, {"model": M["model"], "experiment": {"data_dir": M["data_dir"]}})


def level_name(s: float) -> str:
    return f"s{int(round(100 * s)):02d}"


def _meta() -> dict:
    import torch

    return {"commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "written_at": time.time()}


def snapshot_dir(M: dict) -> Path:
    """The local snapshot folder of the pinned revision in the Hugging Face cache (its weight files are then hashed)."""
    from huggingface_hub import constants

    repo, rev = M["model"]["name_or_path"], M["model"]["revision"]
    d = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not (d / "config.json").exists():
        raise SystemExit(f"{repo} @ {rev} is not in the local cache ({d})")
    return d


def verify_model_files(M: dict) -> dict:
    """SHA-256 of every pinned weight file in the local snapshot of the pinned revision; refuses on any mismatch."""
    d = snapshot_dir(M)
    got = {f: sha256_file(d / f) for f in M["model_files"]}
    bad = {f: (got[f][:16], w[:16]) for f, w in M["model_files"].items() if got[f] != w}
    if bad:
        raise SystemExit(f"{M['model']['name_or_path']}: weight files differ from the pinned hashes: {bad}")
    return {"snapshot": str(d), "files": got, "verified": True}


# ============================================================================ parameter accounting (pure)
def pruned_per_row(cols: int, s: float) -> int:
    from .pruning.wanda import unstructured_count

    return unstructured_count(cols, s, "floor")


def accounting(shapes: dict[str, tuple[int, int]], total_params: int, levels) -> dict:
    """Per level: weights zeroed by uniform per-row Wanda (rows x int(cols x s) per linear), the achieved layer sparsity
    (zeroed / prunable weights) and the zeroed share of ALL model parameters."""
    prunable = sum(r * c for r, c in shapes.values())
    out = {"total_parameters": int(total_params), "prunable_parameters": int(prunable),
           "prunable_share": prunable / total_params, "levels": {}}
    for s in levels:
        z = sum(r * pruned_per_row(c, float(s)) for r, c in shapes.values())
        out["levels"][level_name(s)] = {"level": float(s), "zeroed": int(z), "layer_sparsity": z / prunable,
                                        "share_of_all": z / total_params}
    return out


def meta_shapes(M: dict) -> tuple[dict[str, tuple[int, int]], int, dict]:
    """Shapes of every prunable linear and the total parameter count, from the pinned config.json on the meta device."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from .model import build_units

    conf = AutoConfig.from_pretrained(str(snapshot_dir(M)))
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(conf)
    units = build_units(m, "unstructured", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    shapes = {u.name: tuple(u.weight.shape) for u in units}
    total = sum(p.numel() for p in m.parameters())
    emb = m.get_input_embeddings().weight.numel()
    tied = bool(getattr(conf, "tie_word_embeddings", False))
    info = {"layers": conf.num_hidden_layers, "hidden_size": conf.hidden_size, "intermediate_size": conf.intermediate_size,
            "tie_word_embeddings": tied, "embedding_parameters": int(emb),
            "lm_head_parameters": 0 if tied else int(m.get_output_embeddings().weight.numel()),
            "other_parameters": int(total - sum(r * c for r, c in shapes.values()) - emb - (0 if tied else emb))}
    return shapes, int(total), info


def attainability(cfg: dict) -> Path:
    U = _U(cfg)
    levels = [float(s) for s in U["levels"]]
    rule = U["rule"]
    res = {"label": LABEL, "levels": levels, "rule": rule, "models": {}}
    for key in U["models"]:
        M = _M(cfg, key)
        shapes, total, info = meta_shapes(M)
        acc = accounting(shapes, total, levels)
        best = acc["levels"][level_name(max(levels))]["share_of_all"]
        verdict_if = {k: verdict(v["share_of_all"], rule) for k, v in acc["levels"].items()}
        res["models"][key] = {**acc, **info, "best_attainable_share": best,
                              "best_attainable_formula": f"prunable share {acc['prunable_share']:.4f} x zeroed fraction at "
                                                         f"{max(levels):.0%} = {best:.4f}",
                              "verdict_if_reach_at_level": verdict_if if key == rule["model"] else None}
    r8 = res["models"][rule["model"]]
    res["go_attainable"] = r8["best_attainable_share"] >= float(rule["go_min_share"])
    res["meta"] = _meta()
    p = out_dir(cfg) / "attainability.json"
    atomic_write_json(p, res)
    return p


# ============================================================================ models: hashes and the bf16 load
def models(cfg: dict, key: str) -> Path:
    import torch

    from .model import load_model

    M = _M(cfg, key)
    rc = model_cfg(cfg, key)
    files = verify_model_files(M)
    d = snapshot_dir(M)
    lic = (d / "LICENSE").read_text(encoding="utf-8", errors="replace").splitlines()[:3] if (d / "LICENSE").exists() else None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = load_model(rc)
    torch.cuda.synchronize()
    dtypes = sorted({str(p.dtype) for p in m.parameters()})
    rec = {"label": LABEL, "model": M["model"], "licence_declared": M.get("licence"), "licence_file_head": lic,
           "licence_file_sha256": sha256_file(d / "LICENSE") if (d / "LICENSE").exists() else None, "weights": files,
           "load": {"dtype_requested": rc["model"].get("dtype"), "parameter_dtypes": dtypes, "device": str(next(m.parameters()).device),
                    "parameters": int(sum(p.numel() for p in m.parameters())), "seconds": time.time() - t0,
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated()), "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
                    "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory)},
           "note": "load only: no item is scored here", "meta": _meta()}
    p = model_dir(cfg, key) / "model.json"
    atomic_write_json(p, rec)
    return p


# ============================================================================ the evaluation sets
def draw_eval(rows: list[dict], per_category: int, seed: int) -> tuple[list[dict], dict]:
    """Per category: the items sorted by qid; ``per_category`` of them chosen without replacement with
    ``default_rng([seed, crc32(category)])``, or every item when the category has no more than that."""
    chosen, per = [], {}
    for c in CATS:
        items = sorted((r for r in rows if r["category"] == c), key=lambda r: r["qid"])
        k = min(int(per_category), len(items))
        if k == len(items):
            sel = items
        else:
            rng = np.random.default_rng([int(seed), zlib.crc32(c.encode())])
            sel = [items[i] for i in sorted(rng.choice(len(items), size=k, replace=False).tolist())]
        chosen += sel
        per[c] = {"pool": len(items), "chosen": k, "all_items_used": k == len(items), "short_of_target": k < int(per_category),
                  "qid_sha256": set_sha(r["qid"] for r in sel),
                  "by_source": {s: sum(1 for r in sel if r.get("source") == s) for s in sorted({r.get("source") for r in sel})}}
    return sorted(chosen, key=lambda r: (CATS.index(r["category"]), r["qid"])), per


def build_eval(cfg: dict, key: str) -> Path:
    M = _M(cfg, key)
    E = M["eval"]
    od = model_dir(cfg, key) / "eval"
    if (od / "eval_set.json").exists():
        raise SystemExit(f"{od / 'eval_set.json'} exists: the evaluation set is fixed")
    if E["kind"] == "fixed":
        src = resolve_path(cfg, E["dir"])
        es = read_json(src / "eval_set.json")
        if sha256_file(src / "items.jsonl") != E["items_file_sha256"] or es["items_file_sha256"] != E["items_file_sha256"]:
            raise SystemExit(f"{src / 'items.jsonl'} differs from the pinned file")
        rows = _read_jsonl(src / "items.jsonl")
        if set_sha(r["qid"] for r in rows) != E["qid_sha256"] or es["qid_sha256"] != E["qid_sha256"]:
            raise SystemExit("the reused evaluation set's qid hash differs from the pinned one")
        _write_jsonl(od / "items.jsonl", rows)
        per = {c: {"chosen": sum(1 for r in rows if r["category"] == c),
                   "qid_sha256": set_sha(r["qid"] for r in rows if r["category"] == c)} for c in CATS}
        res = {"kind": "fixed", "source": str(src), "source_eval_set": es.get("rule"), "source_seed": es.get("seed")}
    elif E["kind"] == "screening_pool":
        sd = resolve_path(cfg, E["sealed_dir"])
        reg_p = sd / "registration.json"
        if sha256_file(reg_p) != E["registration_sha256"]:
            raise SystemExit(f"{reg_p} differs from the pinned registration")
        adm_p = sd / "admissibility.json"
        if not adm_p.exists():
            raise SystemExit("run the admissibility test first (confirmatory admissibility)")
        adm = read_json(adm_p)
        if not adm["admissible"]:
            raise SystemExit(f"{key} failed the admissibility test: stop")
        reg = read_json(reg_p)
        pool_p = Path(reg["files"]["screening_pool"]["path"])
        if sha256_file(pool_p) != reg["files"]["screening_pool"]["sha256"]:
            raise SystemExit(f"{pool_p} changed after the registration")
        rows = _read_jsonl(pool_p)  # the screening pool: gate + probe + hold-out (no confirmation item)
        want = {c: sum(reg["sets"][s]["per_category"][c] for s in ("gate", "probe", "holdout")) for c in CATS}
        got = {c: sum(1 for r in rows if r["category"] == c) for c in CATS}
        if want != got or len(rows) != sum(reg["sets"][s]["n"] for s in ("gate", "probe", "holdout")):
            raise SystemExit(f"the screening pool does not match the registration: {got} != {want}")
        rows, per = draw_eval(rows, int(E["per_category"]), int(E["seed"]))
        _write_jsonl(od / "items.jsonl", rows)
        res = {"kind": "screening_pool", "pool_file": str(pool_p), "pool_sha256": reg["files"]["screening_pool"]["sha256"],
               "registration_sha256": E["registration_sha256"], "admissibility": adm, "per_category_target": int(E["per_category"]),
               "seed": int(E["seed"]), "rule": "per category: the screening pool's items sorted by qid; 3,000 chosen without "
                                              "replacement with default_rng([seed, crc32(category)]), or every item when "
                                              "the category has no more"}
    else:
        raise SystemExit(f"unknown evaluation set kind {E['kind']}")
    res.update({"label": LABEL, "model": key, "per_category": per, "total": len(rows), "qid_sha256": set_sha(r["qid"] for r in rows),
                "items_file_sha256": sha256_file(od / "items.jsonl"), "meta": _meta()})
    atomic_write_json(od / "eval_set.json", res)
    return od / "eval_set.json"


# ============================================================================ masks: hash and checks
def mask_hash(masks: dict) -> str:
    """SHA-256 over the unit names (sorted) and each unit's packed keep-mask bits (row-major): ``ranker.mask_hash``,
    with the full digest."""
    h = hashlib.sha256()
    for n in sorted(masks):
        h.update(n.encode())
        h.update(np.packbits(masks[n].detach().to("cpu").numpy().reshape(-1)).tobytes())
    return h.hexdigest()


def mask_stats(state, units, s: float, total_params: int) -> dict:
    """Exact per-row sparsity of every linear, the zeroed counts and the check that every masked weight is zero."""
    import torch

    zeroed = prunable = 0
    exact, nonzero_masked = True, 0
    per_kind: dict[str, list[int]] = {}
    for u in units:
        m = state.masks[u.name]
        rows, cols = m.shape
        k = pruned_per_row(cols, s)
        per_row = (~m).sum(dim=1)
        exact &= bool(torch.all(per_row == k).item())
        z = int((~m).sum().item())
        zeroed += z
        prunable += m.numel()
        nonzero_masked += int((u.weight.detach()[~m] != 0).sum().item())
        pk = per_kind.setdefault(u.kind, [0, 0])
        pk[0] += z
        pk[1] += m.numel()
    return {"per_row_exact": exact, "zeroed": int(zeroed), "prunable": int(prunable),
            "layer_sparsity": zeroed / prunable, "share_of_all": zeroed / total_params,
            "masked_weights_not_zero": int(nonzero_masked),
            "per_kind_sparsity": {k: v[0] / v[1] for k, v in sorted(per_kind.items())}}


# ============================================================================ the level loop (CPU-testable)
class LevelRunner:
    """Levels in ascending order, each pruned one-shot from the dense weights with sequential Wanda.

    Per level: prune -> mask hash -> ``mask.json`` (persisted before scoring; on a rerun the regenerated hash must equal
    it, or the level is refused) -> ``score_fn(hash, model)`` -> ``ppl_fn(hash, model)`` -> ``outcomes.npz`` and
    ``result.json`` (persisted before the next level). A level with a verified ``result.json`` is not pruned again."""

    def __init__(self, out: Path, model, units, dense, calib, levels, cfg: dict, qids, cats, score_fn, ppl_fn,
                 total_params: int, batch_size: int = 4, crash_hook=None, regeneration_check=(), device=None):
        self.out, self.model, self.units, self.dense, self.calib = Path(out), model, units, dense, calib
        self.levels = sorted(float(s) for s in levels)
        self.cfg, self.qids, self.cats = cfg, list(qids), np.asarray(cats)
        self.score_fn, self.ppl_fn, self.total = score_fn, ppl_fn, int(total_params)
        self.bs, self.crash_hook = int(batch_size), crash_hook
        self.regen = {level_name(float(s)) for s in regeneration_check}
        self.device = device

    def _sync(self):
        import torch

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _peak(self) -> int | None:
        import torch

        return int(torch.cuda.max_memory_allocated(self.device)) if self.device is not None and self.device.type == "cuda" else None

    def _reset_peak(self):
        import torch

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def prune(self, s: float):
        from .pruning.masks import PruneState
        from .pruning.wanda import sequential_wanda

        st = PruneState(self.units, "unstructured")
        st.apply(self.dense)  # every level starts from the dense weights
        return sequential_wanda(self.model, st, self.dense, self.calib, float(s), self.cfg, batch_size=self.bs)

    def _hook(self, where: str, s: float):
        if self.crash_hook:
            self.crash_hook(where, s)

    def run(self) -> list[dict]:
        results = []
        for s in self.levels:
            name = level_name(s)
            ld = self.out / "levels" / name
            rp, mp, op = ld / "result.json", ld / "mask.json", ld / "outcomes.npz"
            if rp.exists():
                r = read_json(rp)
                if sha256_file(op) != r["outcomes_sha256"]:
                    raise SystemExit(f"{op} changed after it was written")
                results.append(r)
                continue
            prior = read_json(mp) if mp.exists() else None
            self._reset_peak()
            self._sync()
            t0 = time.perf_counter()
            st = self.prune(s)
            self._sync()
            prune_s = time.perf_counter() - t0
            prune_peak = self._peak()
            t1 = time.perf_counter()
            mh = mask_hash(st.masks)
            hash_s = time.perf_counter() - t1
            stats = mask_stats(st, self.units, s, self.total)
            if not stats["per_row_exact"] or stats["masked_weights_not_zero"]:
                raise RuntimeError(f"level {name}: the mask is not the requested per-row sparsity or is not applied")
            regen_check = None
            if prior is None and name in self.regen:  # generate the masks a second time; the hashes must agree
                t2 = time.perf_counter()
                st2 = self.prune(s)  # the model is left in this (identical) second-generation state
                h2 = mask_hash(st2.masks)
                regen_check = {"second_hash": h2, "match": h2 == mh, "seconds": time.perf_counter() - t2}
                if h2 != mh:
                    raise SystemExit(f"level {name}: a second generation gave {h2[:16]} != {mh[:16]}: levels cannot be "
                                     "regenerated after a crash; stop and ask")
            if prior is not None:
                if prior["mask_hash"] != mh:
                    raise SystemExit(f"level {name}: the regenerated mask hash {mh[:16]} differs from the persisted "
                                     f"{prior['mask_hash'][:16]}; the level may not be regenerated")
                regenerated = {"prune_seconds": prune_s, "prune_peak_bytes": prune_peak}
            else:
                regenerated = None
                atomic_write_json(mp, {"label": LABEL, "level": s, "name": name, "mask_hash": mh, **stats,
                                       "prune_seconds": prune_s, "hash_seconds": hash_s, "prune_peak_bytes": prune_peak,
                                       "regeneration_check": regen_check, "calibration_windows": int(self.calib.shape[0]),
                                       "meta": _meta()})
            self._hook("mask_persisted", s)
            self._sync()
            t3 = time.perf_counter()
            D, T, cost = self.score_fn(mh, self.model)
            self._sync()
            score_s = time.perf_counter() - t3
            self._hook("scored", s)
            t4 = time.perf_counter()
            ppl = self.ppl_fn(mh, self.model)
            self._sync()
            ppl_s = time.perf_counter() - t4
            dense = np.array([bool(D[q]) for q in self.qids])
            cand = np.array([bool(T[q]) for q in self.qids])
            tmp = ld / "outcomes.tmp.npz"
            np.savez_compressed(tmp, qids=np.array(self.qids), category=self.cats, dense=dense, cand=cand)
            tmp.replace(op)
            counts = counts_of(dense, cand, self.cats)
            w, wc = worst_of(counts)
            mrec = read_json(mp)
            r = {"label": LABEL, "level": s, "name": name, "mask_hash": mh, "zeroed": stats["zeroed"],
                 "layer_sparsity": stats["layer_sparsity"], "share_of_all": stats["share_of_all"], "counts": counts,
                 "worst": w, "worst_category": wc, "ppl": ppl, "cost": cost,
                 "seconds": {"prune": mrec["prune_seconds"], "hash": mrec["hash_seconds"], "score": score_s, "ppl": ppl_s,
                             "regeneration_check": (mrec.get("regeneration_check") or {}).get("seconds")},
                 "peak_gpu_bytes": {"prune": mrec["prune_peak_bytes"], "level": self._peak()},
                 "regenerated_after_crash": regenerated, "outcomes_sha256": sha256_file(op), "meta": _meta()}
            atomic_write_json(rp, r)
            log.info("level %s: worst %+.2f points (%s), ppl %.3f, %.0f s prune, %.0f s score", name, 100 * w, wc, ppl,
                     mrec["prune_seconds"], score_s)
            self._hook("result_persisted", s)
            results.append(r)
        return results


def counts_of(dense: np.ndarray, cand: np.ndarray, cats: np.ndarray) -> dict[str, list[int]]:
    from .confirmatory import counts_from_outcomes

    return {c: list(v) for c, v in counts_from_outcomes(dense, cand, cats).items()}


def worst_of(counts: dict) -> tuple[float, str]:
    """The largest estimated paired loss over the categories, and its category (ties: the first in CATS)."""
    est = {c: (counts[c][1] - counts[c][2]) / counts[c][0] for c in CATS}
    wc = max(CATS, key=lambda c: (est[c], -CATS.index(c)))
    return est[wc], wc


def reach(worst: list[float], limit: float) -> int:
    """How many levels, in ascending order, have the worst-category loss <= limit at that level and every lower one."""
    r = 0
    for w in worst:
        if w > limit + 1e-12:
            break
        r += 1
    return r


def verdict(share: float, rule: dict) -> str:
    if share >= float(rule["go_min_share"]) - 1e-12:
        return "GO"
    if share < float(rule["nogo_below_share"]) - 1e-12:
        return "NO-GO"
    return "INCONCLUSIVE"


# ============================================================================ the GPU context
class _Ctx:
    def __init__(self, cfg: dict, key: str):
        import torch

        from .adaptive.evaluator import question_order
        from .data.prepare import PreparedData
        from .model import build_units, load_model, load_tokenizer
        from .pruning.masks import DenseStore
        from .utils.cache import KVCache

        self.cfg, self.key, self.U, self.M = cfg, key, _U(cfg), _M(cfg, key)
        self.rc = model_cfg(cfg, key)
        self.md = model_dir(cfg, key)
        self.files = verify_model_files(self.M)
        es = read_json(self.md / "eval" / "eval_set.json")
        if sha256_file(self.md / "eval" / "items.jsonl") != es["items_file_sha256"]:
            raise SystemExit("the evaluation items changed after they were fixed")
        self.rows = _read_jsonl(self.md / "eval" / "items.jsonl")
        self.qids = [r["qid"] for r in self.rows]
        self.cats = np.array([r["category"] for r in self.rows])
        self.set_sha = es["qid_sha256"][:16]
        self.data = PreparedData(resolve_path(cfg, self.M["data_dir"]), self.M["tokenizer_id"])
        self.tok, self.dm = load_tokenizer(self.rc), load_model(self.rc)  # the dense model: never modified
        self.work = copy.deepcopy(self.dm)                                   # pruned in place, level by level
        self.dev = next(self.dm.parameters()).device
        elig = self.rc["pruning"]["eligible"]
        self.units = build_units(self.work, "unstructured", elig)
        dunits = {u.name: u for u in build_units(self.dm, "unstructured", elig)}
        self.dense = DenseStore.__new__(DenseStore)                         # the dense weights ARE the dense model's
        self.dense.w = {(u.name, "w"): dunits[u.name].weight.detach() for u in self.units}
        self.total_params = int(sum(p.numel() for p in self.dm.parameters()))
        t = self.rc["testing"]
        self.shots, self.max_len, self.mbt = int(t.get("shots", 0)), int(self.rc["model"].get("max_seq_len", 2048)), int(t.get("max_batch_tokens", 32768))
        self.cache = KVCache(self.md / "eval" / "scores.sqlite")
        bsz, oseed = int(self.U["eval"]["batch"]), int(self.U["eval"]["order_seed"])
        self.batches = []
        for c in CATS:
            rmap = {r["qid"]: r for r in self.rows if r["category"] == c}
            order = question_order(sorted(rmap), oseed, c)
            self.batches += [(c, i, [rmap[q] for q in order[i:i + bsz]]) for i in range(0, len(order), bsz)]
        cal = self.U["calibration"]
        self.cal_idx = self.data.calibration_indices(int(cal["seed"]), int(cal["windows"]))
        self.calib = self.data.calibration_pool[self.cal_idx]
        self.torch = torch
        self.ppl_store: dict = {}

    def _sync(self):
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)

    def score_dense(self) -> tuple[dict, dict]:
        from .pool_signals import LogprobScorer

        dsc = LogprobScorer(self.dm, self.tok, self.data.mmlu_fewshot, self.shots, self.max_len, self.mbt)
        D, cost = {}, {"items_scored": 0, "items_replayed": 0, "seconds": 0.0}
        for c, i, br in self.batches:
            dk = f"dense|{self.set_sha}|{c}|{i}|{len(br)}"
            drec = self.cache.get(dk)
            if drec is None:
                self._sync()
                t0 = time.perf_counter()
                dres, bd = dsc.score(br, self.mbt)
                self._sync()
                cost["seconds"] += time.perf_counter() - t0
                cost["items_scored"] += len(br)
                drec = {"items": dres, "budget": bd}
                self.cache.put(dk, drec)
            else:
                cost["items_replayed"] += len(br)
            D.update({q: v["correct"] for q, v in drec["items"].items()})
        return D, cost

    def score_pair(self, key: str, cand) -> tuple[dict, dict, dict]:
        """Correctness of the dense model and the candidate on every item, identical batch layouts (an OOM on either
        side shrinks both to the same layout), every batch persisted before it is used."""
        from .pool_signals import LogprobScorer

        dsc = LogprobScorer(self.dm, self.tok, self.data.mmlu_fewshot, self.shots, self.max_len, self.mbt)
        csc = LogprobScorer(cand, self.tok, self.data.mmlu_fewshot, self.shots, self.max_len, self.mbt)
        D, T = {}, {}
        cost = {"items_scored": 0, "items_replayed": 0, "seconds": 0.0, "dense_items_rescored": 0}
        for c, i, br in self.batches:
            pk, dk = f"ckpt|{key}|{self.set_sha}|{c}|{i}|{len(br)}", f"dense|{self.set_sha}|{c}|{i}|{len(br)}"
            rec = self.cache.get(pk)
            if rec is None:
                drec = self.cache.get(dk)
                b = int(drec["budget"]) if drec else self.mbt
                while True:
                    if drec is None or int(drec["budget"]) != b:
                        dres, bd = dsc.score(br, b)
                        drec = {"items": dres, "budget": bd}
                        self.cache.put(dk, drec)
                        cost["dense_items_rescored"] += len(br)
                    self._sync()
                    t0 = time.perf_counter()
                    cres, bc = csc.score(br, int(drec["budget"]))
                    self._sync()
                    cost["seconds"] += time.perf_counter() - t0
                    cost["items_scored"] += len(br)
                    if bc == int(drec["budget"]):
                        break
                    b = bc
                rec = {"d": {q: v["correct"] for q, v in drec["items"].items()}, "t": {q: v["correct"] for q, v in cres.items()},
                       "budget": int(drec["budget"])}
                self.cache.put(pk, rec)
            else:
                cost["items_replayed"] += len(br)
            D.update(rec["d"])
            T.update(rec["t"])
        return D, T, cost

    def perplexity(self, key: str, model) -> float:
        from .eval.perplexity import perplexity

        ck = f"ppl|{key}"
        rec = self.cache.get(ck)
        if rec is None:
            rec = {"ppl": perplexity(model, self.data.ppl_gate, self.ppl_store, "ppl", initial_bs=4)}
            self.cache.put(ck, rec)
        return float(rec["ppl"])


def run(cfg: dict, key: str) -> Path:
    import torch

    U = _U(cfg)
    md = model_dir(cfg, key)
    if not (md / "eval" / "eval_set.json").exists():
        raise SystemExit(f"run `unstructured build-eval --model {key}` first")
    if _M(cfg, key)["eval"]["kind"] == "screening_pool":
        adm = read_json(resolve_path(cfg, _M(cfg, key)["eval"]["sealed_dir"]) / "admissibility.json")
        if not adm["admissible"]:
            raise SystemExit(f"{key} failed the admissibility test: stop")
    t_start = time.time()
    ctx = _Ctx(cfg, key)
    # ---- dense: outcomes, accuracy, perplexity (once)
    dp = md / "dense.json"
    if not dp.exists():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        D, cost = ctx.score_dense()
        ppl = ctx.perplexity("dense", ctx.dm)
        dense = np.array([bool(D[q]) for q in ctx.qids])
        np.savez_compressed(md / "dense.tmp.npz", qids=np.array(ctx.qids), category=ctx.cats, dense=dense)
        (md / "dense.tmp.npz").replace(md / "dense.npz")
        cross = None
        xc = _M(cfg, key)["eval"].get("dense_crosscheck")
        if xc:  # the compensation experiment's dense answers on the same items, batches and budget
            with np.load(resolve_path(cfg, xc), allow_pickle=False) as z:
                if list(z["qids"]) == ctx.qids:
                    cross = {"file": str(resolve_path(cfg, xc)), "items": len(ctx.qids),
                             "identical_items": int(np.count_nonzero(z["dense"] == dense))}
        zero_dense = int(sum(int((u.weight.detach() == 0).sum().item()) for u in ctx.units))
        atomic_write_json(dp, {"label": LABEL, "model": key, "revision": ctx.M["model"]["revision"], "weights": ctx.files,
                               "total_parameters": ctx.total_params,
                               "prunable_parameters": int(sum(u.weight.numel() for u in ctx.units)),
                               "dense_zero_weights_in_prunable_layers": zero_dense,
                               "accuracy": {c: float(dense[ctx.cats == c].mean()) for c in CATS} | {"all": float(dense.mean())},
                               "n": {c: int((ctx.cats == c).sum()) for c in CATS}, "ppl": ppl, "cost": cost,
                               "seconds": time.perf_counter() - t0, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                               "dense_crosscheck": cross, "outcomes_sha256": sha256_file(md / "dense.npz"),
                               "calibration": {"seed": int(U["calibration"]["seed"]), "indices": ctx.cal_idx,
                                               "pool_sha256": ctx.data.manifest["files"]["calibration_pool.safetensors"]["sha256"]},
                               "meta": _meta()})
    runner = LevelRunner(md, ctx.work, ctx.units, ctx.dense, ctx.calib, U["levels"], ctx.rc, ctx.qids, ctx.cats,
                         ctx.score_pair, ctx.perplexity, ctx.total_params, int(U["calibration"].get("batch_size", 4)),
                         regeneration_check=U.get("regeneration_check", ()), device=ctx.dev)
    res = runner.run()
    rec = {"label": LABEL, "model": key, "levels": [r["name"] for r in res], "invocation_seconds": time.time() - t_start,
           "meta": _meta()}
    atomic_write_json(md / "run.json", rec)
    return md / "run.json"


# ============================================================================ analysis (CPU)
def _model_results(cfg: dict, key: str) -> dict | None:
    from .population_regate import population_bounds

    U = _U(cfg)
    md = model_dir(cfg, key)
    if not (md / "dense.json").exists():
        return None
    dense = read_json(md / "dense.json")
    if sha256_file(md / "dense.npz") != dense["outcomes_sha256"]:
        raise SystemExit(f"{md / 'dense.npz'} changed after it was written")
    alpha = float(U["confirmation_rule"]["alpha"])
    rows = []
    for s in sorted(float(x) for x in U["levels"]):
        ld = md / "levels" / level_name(s)
        if not (ld / "result.json").exists():
            break
        r = read_json(ld / "result.json")
        m = read_json(ld / "mask.json")
        if sha256_file(ld / "outcomes.npz") != r["outcomes_sha256"] or m["mask_hash"] != r["mask_hash"]:
            raise SystemExit(f"{ld}: outcomes or mask record changed after they were written")
        with np.load(ld / "outcomes.npz", allow_pickle=False) as z:
            oc = {k: z[k] for k in z.files}
        with np.load(md / "dense.npz", allow_pickle=False) as z:
            if not np.array_equal(z["dense"], oc["dense"]):
                raise SystemExit(f"{ld}: the dense answers differ from the dense record")
        cats = {}
        for c in CATS:
            n, kp, km = r["counts"][c]
            lb, ub = population_bounds(n, kp, km, alpha, alpha)
            cats[c] = {"n": n, "lost": kp, "gained": km, "estimate": (kp - km) / n, "lower": lb, "upper": ub,
                       "dense_acc": float(oc["dense"][oc["category"] == c].mean()),
                       "pruned_acc": float(oc["cand"][oc["category"] == c].mean())}
        lost_all = int(np.count_nonzero(oc["dense"] & ~oc["cand"]))
        gained_all = int(np.count_nonzero(oc["cand"] & ~oc["dense"]))
        rows.append({"level": s, "name": r["name"], "mask_hash": r["mask_hash"], "zeroed": r["zeroed"],
                     "layer_sparsity": r["layer_sparsity"], "share_of_all": r["share_of_all"], "categories": cats,
                     "worst": r["worst"], "worst_category": r["worst_category"],
                     "pooled": {"n": int(len(oc["dense"])), "estimate": (lost_all - gained_all) / len(oc["dense"]),
                                "dense_acc": float(oc["dense"].mean()), "pruned_acc": float(oc["cand"].mean())},
                     "ppl": r["ppl"], "ppl_ratio": r["ppl"] / dense["ppl"], "seconds": r["seconds"],
                     "peak_gpu_bytes": r["peak_gpu_bytes"], "regenerated_after_crash": r["regenerated_after_crash"],
                     "regeneration_check": m.get("regeneration_check"), "per_row_exact": m["per_row_exact"],
                     "per_kind_sparsity": m["per_kind_sparsity"],
                     "source": str(ld / "result.json")})
    lim = float(U["reach_limit"])
    R = reach([x["worst"] for x in rows], lim)
    reach_share = rows[R - 1]["share_of_all"] if R else 0.0
    from .screen_confirm import fixed_sequence

    tol = float(U["confirmation_rule"]["tolerance"])
    fs = fixed_sequence([{"id": x["name"], "params_removed": x["zeroed"],
                          "counts": {c: (v["n"], v["lost"], v["gained"]) for c, v in x["categories"].items()}} for x in rows],
                        alpha, tol)
    claim = next((x for x in rows if x["name"] == fs["claim"]), None)
    return {"model": key, "label": _U(cfg)["models"][key]["label"], "dense": dense, "levels": rows,
            "complete": len(rows) == len(U["levels"]), "reach_levels": R,
            "reach_level": rows[R - 1]["level"] if R else None, "reach_share": reach_share,
            "reach_layer_sparsity": rows[R - 1]["layer_sparsity"] if R else 0.0,
            "confirmation_rule": {"label": NOT_CERT, "claim": fs["claim"], "tested": len(fs["tested"]),
                                  "claim_share": claim["share_of_all"] if claim else 0.0,
                                  "claim_level": claim["level"] if claim else None,
                                  "upper": {t["id"]: t["upper"] for t in fs["tested"]}}}


def _channel_reference(cfg: dict, uns: dict | None) -> dict | None:
    """Qwen3-1.7B: the compensation experiment's channel-removal trajectories, matched on the share of parameters."""
    p = resolve_path(cfg, _U(cfg)["channel_reference"])
    if not p.exists() or uns is None:
        return None
    A = read_json(p)
    dp = int(A["dense_parameters"])
    steps = [{"step": r["step"], "share": r["params_removed"] / dp, "params_removed": r["params_removed"],
              "mask_worst": r["ops"]["mask"]["worst"], "comp_worst": r["ops"]["comp"]["worst"],
              "mask_worst_category": max(CATS, key=lambda c: (r["ops"]["mask"]["categories"][c]["estimate"], -CATS.index(c))),
              "mask_ppl_ratio": r["ops"]["mask"]["ppl_ratio"]} for r in A["per_step"]]
    top = steps[-1]["share"]
    matched = []
    for x in uns["levels"]:
        if x["share_of_all"] > top + 1e-9:
            matched.append({"level": x["name"], "share": x["share_of_all"], "channel": None})
            continue
        ch = min(steps, key=lambda t: (abs(t["share"] - x["share_of_all"]), t["step"]))
        matched.append({"level": x["name"], "share": x["share_of_all"], "worst": x["worst"],
                        "worst_category": x["worst_category"], "ppl_ratio": x["ppl_ratio"], "channel": ch})
    return {"source": str(p), "dense_parameters": dp, "R_mask": A["R_mask"], "R_comp": A["R_comp"],
            "R_mask_share": steps[A["R_mask"] - 1]["share"] if A["R_mask"] else 0.0, "max_share": top,
            "steps": steps, "matched": matched}


def analyze(cfg: dict) -> Path:
    U = _U(cfg)
    rule = U["rule"]
    res = {"label": LABEL, "zero_note": ZERO_NOTE, "rule": rule, "reach_limit": float(U["reach_limit"]),
           "attainability": read_json(out_dir(cfg) / "attainability.json"), "models": {}}
    for key in U["models"]:
        res["models"][key] = _model_results(cfg, key)
    m8 = res["models"].get(rule["model"])
    if m8 and m8["complete"]:
        res["verdict"] = verdict(m8["reach_share"], rule)
    else:
        res["verdict"] = "NOT REACHED (the decision model's levels are incomplete)"
    res["channel_comparison"] = _channel_reference(cfg, res["models"].get("qwen3-1.7b"))
    bp = U["base_paper"]
    at = level_name(float(U["base_paper_level"]))
    x43 = next((x for x in (m8 or {}).get("levels", []) if x["name"] == at), None)
    res["base_paper_comparison"] = {"label": LABEL_43, "base_paper": bp, "base_paper_drop_points": bp["mmlu_dense"] - bp["mmlu_pruned"],
                                    "ours": x43}
    for key, m in res["models"].items():
        if m:
            m["admissibility"] = None
            adm_dir = (_U(cfg)["models"][key]["eval"].get("sealed_dir"))
            if adm_dir and (resolve_path(cfg, adm_dir) / "admissibility.json").exists():
                m["admissibility"] = read_json(resolve_path(cfg, adm_dir) / "admissibility.json")
            m["eval_set"] = {k: v for k, v in read_json(model_dir(cfg, key) / "eval" / "eval_set.json").items() if k not in ("admissibility",)}
    res["meta"] = _meta()
    ad = out_dir(cfg) / "analysis"
    ad.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ad / "analysis.json", res)
    _figure(res, ad)
    atomic_write_text(ad / "analysis.md", render(res))
    return ad / "analysis.md"


def _figure(res: dict, ad: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#191919", "#666666", "#e1e0d9"
    series = [("qwen3-1.7b", "#257de0", "o"), ("qwen3-8b", "#f47230", "s")]  # reference palette slots 1-2 (validated)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lim, style, txt in ((0.02, (0, (4, 3)), "2-point reach limit"), (0.03, (0, (1, 2)), "3-point confirmation tolerance")):
        ax.axhline(100 * lim, color=ink2, lw=0.9, ls=style, zorder=1)
    ax.axhline(0, color="#c3c2b7", lw=0.6, zorder=1)
    cc = res.get("channel_comparison")
    if cc:  # context: the 1.7B channel-removal trajectory (mask-only) of the compensation experiment
        ax.plot([100 * t["share"] for t in cc["steps"]], [100 * t["mask_worst"] for t in cc["steps"]], color="#00b894",
                marker="^", ms=3.5, lw=1.2, zorder=2,
                label=f"Qwen3-1.7B, MLP-channel removal (compensation experiment; reach {100 * cc['R_mask_share']:.1f} %)")
    for key, col, mk in series:
        m = res["models"].get(key)
        if not m or not m["levels"]:
            continue
        x = [0.0] + [100 * r["share_of_all"] for r in m["levels"]]
        y = [0.0] + [100 * r["worst"] for r in m["levels"]]
        ax.plot(x, y, color=col, marker=mk, ms=5, lw=2, zorder=3,
                label=f"{m['label'].split(' (')[0]}, unstructured Wanda (reach {100 * m['reach_share']:.1f} %)")
    ymax = max([3.5] + [100 * r["worst"] for m in res["models"].values() if m for r in m["levels"]])
    ax.set_ylim(-1.5, min(ymax + 1.0, 40))
    ax.text(0.3, 2.08, "2-point reach limit", fontsize=7, color=ink2, va="bottom")
    ax.text(0.3, 3.08, "3-point confirmation tolerance", fontsize=7, color=ink2, va="bottom")
    ax.set_xlabel("weights zeroed or removed, share of ALL model parameters (%)", color=ink)
    ax.set_ylabel("worst-category paired loss vs dense (points)", color=ink)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.grid(color=grid, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", labelcolor=ink)
    ax.set_title(f"EXPLORATORY: worst-category loss against compression; verdict on Qwen3-8B: {res['verdict']}",
                 fontsize=8.5, color=ink)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ad / f"worst_loss.{ext}", dpi=220, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else {})
    plt.close(fig)


def render(R: dict) -> str:
    pt = lambda v: "n/a" if v is None else f"{100 * v:+.2f}"  # noqa: E731
    pc = lambda v: f"{100 * v:.2f} %"  # noqa: E731
    L = ["# Unstructured feasibility: analysis", "", f"**{R['label']}.** {R['zero_note']}", "",
         f"Reach limit {100 * R['reach_limit']:.1f} points (worst category, point estimate, this level and every lower one). "
         f"Rule on {R['rule']['model']}: GO if reach >= {pc(R['rule']['go_min_share'])} of all parameters, NO-GO if "
         f"< {pc(R['rule']['nogo_below_share'])}, otherwise INCONCLUSIVE.", "", f"**Verdict: {R['verdict']}**", ""]
    for key, m in R["models"].items():
        if not m:
            L += [f"## {key}: not run", ""]
            continue
        d = m["dense"]
        L += [f"## {m['label']}", "",
              f"Dense: {d['total_parameters']:,} parameters ({d['prunable_parameters']:,} in prunable linears); 0-shot accuracy "
              + ", ".join(f"{c} {100 * d['accuracy'][c]:.2f} % (n {d['n'][c]:,})" for c in CATS)
              + f"; C4 validation perplexity {d['ppl']:.3f}.", "",
              f"**Reach: {m['reach_levels']} level(s)"
              + (f", up to {100 * m['reach_level']:.0f} % layer sparsity = {pc(m['reach_share'])} of all parameters**" if m["reach_levels"] else ", none**"),
              "", "| level | layer sparsity (achieved) | zeroed, share of all parameters | worst category | worst loss | ppl / dense | prune s | score s | peak GPU (GB) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in m["levels"]:
            L.append(f"| {100 * x['level']:.0f} % | {pc(x['layer_sparsity'])} | {pc(x['share_of_all'])} ({x['zeroed']:,}) | "
                     f"{x['worst_category']} | {pt(x['worst'])} | {x['ppl_ratio']:.3f} | {x['seconds']['prune']:.0f} | "
                     f"{x['seconds']['score']:.0f} | {(x['peak_gpu_bytes']['level'] or 0) / 1e9:.1f} |")
        L += ["", "Per category: paired loss against dense, estimate [one-sided 95 % lower, upper] (points):", "",
              "| level | " + " | ".join(CATS) + " | pooled |", "|---|" + "---|" * (len(CATS) + 1)]
        for x in m["levels"]:
            L.append(f"| {100 * x['level']:.0f} % | " + " | ".join(
                f"{pt(v['estimate'])} [{pt(v['lower'])}, {pt(v['upper'])}]" for v in (x["categories"][c] for c in CATS))
                + f" | {pt(x['pooled']['estimate'])} |")
        cr = m["confirmation_rule"]
        L += ["", f"Confirmation rule on this set ({cr['label']}): per-category IUT, Clopper-Pearson union bound, 3 points, "
              f"alpha 0.05, fixed sequence in ascending sparsity: claim "
              + (f"{100 * cr['claim_level']:.0f} % ({pc(cr['claim_share'])} of all parameters)" if cr["claim"] else "none")
              + f"; {cr['tested']} level(s) tested.", ""]
    cc = R.get("channel_comparison")
    if cc:
        L += ["## Qwen3-1.7B: unstructured against MLP-channel removal, matched on share of parameters", "",
              f"Channel removal (compensation experiment, mask-only): reach {cc['R_mask']} steps = {pc(cc['R_mask_share'])}; "
              f"trajectory to {pc(cc['max_share'])}.", "",
              "| level | share | unstructured worst (category) | channel step | channel share | channel worst, mask-only "
              "(category) | channel worst, compensated |",
              "|---|---|---|---|---|---|---|"]
        for x in cc["matched"]:
            ch = x["channel"]
            if ch is None:
                L.append(f"| {x['level']} | {pc(x['share'])} | beyond the channel trajectory | | | | |")
            else:
                L.append(f"| {x['level']} | {pc(x['share'])} | {pt(x['worst'])} ({x['worst_category']}) | {ch['step']} | "
                         f"{pc(ch['share'])} | {pt(ch['mask_worst'])} ({ch['mask_worst_category']}) | {pt(ch['comp_worst'])} |")
        L.append("")
    bp = R["base_paper_comparison"]
    if bp["ours"]:
        x = bp["ours"]
        L += [f"## 43 % on Qwen3-8B ({bp['label']})", "",
              f"Base paper ({bp['base_paper']['source']}): {bp['base_paper']['model']} at {100 * bp['base_paper']['sparsity']:.2f} % "
              f"sparsity, {bp['base_paper']['shots']}-shot MMLU {bp['base_paper']['mmlu_dense']} -> {bp['base_paper']['mmlu_pruned']} "
              f"({bp['base_paper_drop_points']:.2f} points).", "",
              "| category | paired loss [one-sided 95 %] | dense acc | pruned acc |", "|---|---|---|---|"]
        for c in CATS:
            v = x["categories"][c]
            L.append(f"| {c} | {pt(v['estimate'])} [{pt(v['lower'])}, {pt(v['upper'])}] | {100 * v['dense_acc']:.2f} % | {100 * v['pruned_acc']:.2f} % |")
        L.append(f"| pooled (this set) | {pt(x['pooled']['estimate'])} | {100 * x['pooled']['dense_acc']:.2f} % | {100 * x['pooled']['pruned_acc']:.2f} % |")
    return "\n".join(L) + "\n"
