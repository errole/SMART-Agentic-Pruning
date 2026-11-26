"""Compensation feasibility: EXPLORATORY, on the development model Qwen3-1.7B (docs/compensation_feasibility_plan.md).

``capprune compensation <phase>``:
* ``reference``: freeze the reference trajectory's masks (hash-checked) and compare them with gradient-live-ext's;
* ``build-eval``: the evaluation set (3,000 items per MMLU category from the Qwen items);
* ``score --operator mask``: dense + mask-only steps; the reach R_mask; the extension rule (plan section 5);
* ``compensate``: the refit trajectory over the same masks (``capprune.compensate``; persisted, resumable);
* ``score --operator comp``: the compensated steps;
* ``analyze``: the verdict, the descriptive tables and the figure.

Never reads a sealed confirmation set and never loads a confirmatory model. Every output records the commit, the tree
state and the GPU.
"""
from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

from .config import resolve_path, run_config, run_id, runs_dir
from .confirmatory import CATS, _git, _mask_file_hash, _read_jsonl, _write_jsonl, set_sha
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("compensation")
LABEL = "EXPLORATORY: development model Qwen3-1.7B; every evaluation item has been seen on Qwen before"
OPS = ("mask", "comp")
OP_LABEL = {"mask": "mask-only", "comp": "compensated"}


def _C(cfg: dict) -> dict:
    return cfg["compensation"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _C(cfg)["out_dir"])


def _meta() -> dict:
    import torch

    return {"commit": _git("rev-parse", "HEAD"), "clean_tree": _git("status", "--porcelain") == "",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "written_at": time.time()}


def _rc(cfg: dict) -> dict:
    C = _C(cfg)
    return run_config(cfg, C["method"], 0.0, int(C["seed"]))


def _run_dir(cfg: dict) -> Path:
    C = _C(cfg)
    return runs_dir(cfg) / run_id(C["method"], 0.0, int(C["seed"]))


def _iterations(journal: Path) -> list[dict]:
    return [j for j in (json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines() if x.strip())
            if j.get("type") == "iteration"]


# ============================================================================ the reference trajectory
def reference(cfg: dict) -> Path:
    C = _C(cfg)
    rd = _run_dir(cfg)
    if not (rd / "DONE").exists():
        raise SystemExit(f"{rd}: the reference run has not finished")
    its = _iterations(rd / "journal.jsonl")
    acc = [j for j in its if j["outcome"] == "accept"]
    if len(acc) != len(its):
        raise SystemExit("the reference trajectory has a rejected proposal: it was meant to have no gate")
    order = []
    for j in sorted(acc, key=lambda j: int(j["iteration"])):
        mf = rd / "masks" / f"{j['cand_id']}.safetensors"
        h = _mask_file_hash(mf)
        if h != j["trial_mask_hash"]:
            raise SystemExit(f"step {j['iteration']}: mask file hash {h} != journal {j['trial_mask_hash']}")
        g = j.get("gate") or {}
        order.append({"step": len(order) + 1, "iteration": int(j["iteration"]), "cand_id": j["cand_id"], "mask_hash": h,
                      "mask_file": str(mf), "mask_sha256": sha256_file(mf), "params_removed": int(j["params_removed"]),
                      "mlp_sparsity": j.get("rho_after"), "action": j["chosen"].get("action_key"),
                      "ppl_trial": g.get("ppl_trial"), "ppl0": g.get("ppl0")})
    # the check against the earlier gated run: its trials, in order, up to and including its first rejection
    other = _iterations(resolve_path(cfg, C["check_against"]))
    cmp_rows = [{"iteration": int(e["iteration"]), "earlier": e["trial_mask_hash"], "earlier_outcome": e["outcome"],
                 "reference": order[i]["mask_hash"] if i < len(order) else None,
                 "match": i < len(order) and order[i]["mask_hash"] == e["trial_mask_hash"]} for i, e in enumerate(other)]
    first_diff = next((r["iteration"] for r in cmp_rows if not r["match"]), None)
    res = {"label": LABEL, "run_dir": str(rd), "journal_sha256": sha256_file(rd / "journal.jsonl"),
           "run_json": read_json(rd / "run.json").get("git"), "steps": len(order), "order": order,
           "check_against": {"journal": str(resolve_path(cfg, C["check_against"])), "compared": len(cmp_rows),
                             "all_match": first_diff is None, "first_difference_at_iteration": first_diff, "rows": cmp_rows},
           "meta": _meta()}
    p = out_dir(cfg) / "reference_order.json"
    atomic_write_json(p, res)
    return p


def _order(cfg: dict) -> list[dict]:
    return read_json(out_dir(cfg) / "reference_order.json")["order"]


# ============================================================================ the evaluation set
def build_eval(cfg: dict) -> Path:
    from .confirmatory import dedupe, source_rows

    C = _C(cfg)
    od = out_dir(cfg) / "eval"
    if (od / "eval_set.json").exists():
        raise SystemExit(f"{od / 'eval_set.json'} exists: the evaluation set is fixed")
    rows, fewshot, prov = source_rows({**cfg, "confirmatory": {"sources": C["sources"]}})
    pool, dd = dedupe(rows, fewshot)
    n, seed = int(C["eval"]["per_category"]), int(C["eval"]["seed"])
    chosen, per = [], {}
    for c in CATS:
        items = sorted((r for r in pool if r["category"] == c), key=lambda r: r["qid"])
        k = min(n, len(items))
        rng = np.random.default_rng([seed, zlib.crc32(c.encode())])
        idx = sorted(rng.choice(len(items), size=k, replace=False).tolist())
        sel = [items[i] for i in idx]
        chosen += sel
        per[c] = {"pool": len(items), "chosen": k, "short_of_target": k < n, "qid_sha256": set_sha(r["qid"] for r in sel),
                  "by_source": {s: sum(1 for r in sel if r.get("source") == s) for s in sorted({r.get("source") for r in sel})}}
    _write_jsonl(od / "items.jsonl", sorted(chosen, key=lambda r: (CATS.index(r["category"]), r["qid"])))
    res = {"label": LABEL, "per_category": per, "total": len(chosen), "qid_sha256": set_sha(r["qid"] for r in chosen),
           "items_file_sha256": sha256_file(od / "items.jsonl"), "dedupe": dd, "provenance": prov, "seed": seed,
           "rule": "per category: the pool's items sorted by qid, chosen without replacement with "
                   "default_rng([seed, crc32(category)])", "meta": _meta()}
    atomic_write_json(od / "eval_set.json", res)
    return od / "eval_set.json"


# ============================================================================ GPU context
class _Ctx:
    """One dense model (never modified), the evaluation items, the scorer and the caches."""

    def __init__(self, cfg: dict):
        import torch

        from .config import data_dir
        from .data.prepare import PreparedData
        from .model import build_units, load_model, load_tokenizer, tokenizer_id
        from .utils.cache import KVCache

        self.cfg, self.C = cfg, _C(cfg)
        self.rc = _rc(cfg)
        self.out = out_dir(cfg)
        es = read_json(self.out / "eval" / "eval_set.json")
        if sha256_file(self.out / "eval" / "items.jsonl") != es["items_file_sha256"]:
            raise SystemExit("the evaluation items changed after they were fixed")
        self.rows = _read_jsonl(self.out / "eval" / "items.jsonl")
        self.qids = [r["qid"] for r in self.rows]
        self.cats = np.array([r["category"] for r in self.rows])
        self.set_sha = es["qid_sha256"][:16]
        self.data = PreparedData(data_dir(self.rc), tokenizer_id(self.rc))
        self.tok, self.dm = load_tokenizer(self.rc), load_model(self.rc)
        self.dm.eval()
        for p in self.dm.parameters():
            p.requires_grad_(False)
        self.units = build_units(self.dm, "structured", self.rc["pruning"].get("eligible", []))
        self.dev = next(self.dm.parameters()).device
        t = self.rc["testing"]
        self.shots, self.max_len, self.mbt = int(t.get("shots", 0)), int(self.rc["model"].get("max_seq_len", 2048)), int(t.get("max_batch_tokens", 32768))
        self.cache = KVCache(self.out / "eval" / "scores.sqlite")
        from .adaptive.evaluator import question_order

        self.batches = []
        bsz, oseed = int(self.C["eval"]["batch"]), int(self.C["eval"]["order_seed"])
        for c in CATS:
            rmap = {r["qid"]: r for r in self.rows if r["category"] == c}
            order = question_order(sorted(rmap), oseed, c)
            self.batches += [(c, i, [rmap[q] for q in order[i:i + bsz]]) for i in range(0, len(order), bsz)]
        self.torch = torch
        self.ppl_store: dict = {}

    def score_pair(self, key: str, cand) -> tuple[dict, dict, dict]:
        """Correctness of the dense model and the candidate on every item, identical batch layouts, every batch persisted
        before use (as in ``confirmatory.confirm``). Returns (dense, cand, cost)."""
        from .pool_signals import LogprobScorer

        torch = self.torch
        dsc = LogprobScorer(self.dm, self.tok, self.data.mmlu_fewshot, self.shots, self.max_len, self.mbt)
        csc = LogprobScorer(cand, self.tok, self.data.mmlu_fewshot, self.shots, self.max_len, self.mbt)
        D, T, cost = {}, {}, {"items_scored": 0, "items_replayed": 0, "seconds": 0.0}
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
                    t0 = time.perf_counter()
                    cres, bc = csc.score(br, int(drec["budget"]))
                    if self.dev.type == "cuda":
                        torch.cuda.synchronize()
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


def _counts(dense: np.ndarray, cand: np.ndarray, cats: np.ndarray) -> dict[str, tuple[int, int, int]]:
    from .confirmatory import counts_from_outcomes

    return counts_from_outcomes(dense, cand, cats)


def worst_loss(counts: dict[str, tuple[int, int, int]]) -> float:
    return max((kp - km) / n for n, kp, km in counts.values())


def reach(worst: list[float], limit: float) -> int:
    """The largest t with worst[s] <= limit for every s = 1..t (worst[0] is step 1); 0 if step 1 exceeds it."""
    r = 0
    for w in worst:
        if w > limit + 1e-12:
            break
        r += 1
    return r


def extension(r_mask: int, L: int, max_steps: int) -> dict:
    """The plan's order-of-work rule: extend to min(60, ceil(2.5 R)) if 2R + 2 > L; stop if 2R + 2 > 60 after that."""
    if 2 * r_mask + 2 <= L:
        return {"extend": False, "L": L, "proceed": True}
    newL = min(max_steps, math.ceil(2.5 * r_mask))
    return {"extend": True, "L": max(newL, L), "proceed": 2 * r_mask + 2 <= max_steps}


# ============================================================================ scoring
def _score_steps(ctx: _Ctx, op: str, steps: list[int], prev: dict | None) -> dict:
    """Score the given steps of one operator; returns the (merged) per-step record."""
    from .pruning.masks import PruneState
    from .pruning.materialize import materialize
    from .gradient_audit import size_check

    torch = ctx.torch
    order = _order(ctx.cfg)
    res = prev or {"op": op, "steps": {}, "dense": None}
    if op == "comp":
        traj, work = _comp_trajectory(ctx)
    t_all = time.time()
    for t in steps:
        o = order[t - 1]
        st = PruneState(ctx.units, "structured")
        st.load(o["mask_file"])
        if op == "mask":
            key, src_model = o["mask_hash"], ctx.dm
        else:
            traj.restore(t)
            key, src_model = traj.checkpoint_hash_at(t), work
        mat, _ = materialize(src_model, st, ctx.rc)
        mat.eval()
        sz = size_check(ctx.rc, ctx.dm, ctx.dm, mat, st, ctx.units) if op == "mask" else None
        t0 = time.time()
        D, T, cost = ctx.score_pair(key, mat)
        ppl = ctx.perplexity(f"{op}|{key}", mat)
        dense = np.array([bool(D[q]) for q in ctx.qids])
        cand = np.array([bool(T[q]) for q in ctx.qids])
        counts = _counts(dense, cand, ctx.cats)
        n_params = sum(p.numel() for p in mat.parameters())
        res["steps"][str(t)] = {"step": t, "key": key, "mask_hash": o["mask_hash"], "params_removed": o["params_removed"],
                                "materialized_parameters": int(n_params), "counts": counts, "worst": worst_loss(counts),
                                "ppl": ppl, "cost": cost, "seconds": time.time() - t0,
                                "size_check": None if sz is None else {k: sz[k] for k in ("parameters_removed", "count_matches_masks")}}
        res.setdefault("_cand", {})[str(t)] = cand
        res["_dense"] = dense
        log.info("%s step %d: worst %+.2f points, ppl %.3f (%.0f s since start)", op, t, 100 * res["steps"][str(t)]["worst"],
                 ppl, time.time() - t_all)
        del mat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


def _save_scores(ctx: _Ctx, op: str, res: dict) -> None:
    od = ctx.out / "eval"
    steps = sorted(int(s) for s in res["steps"])
    cand = np.stack([res["_cand"][str(s)] for s in steps]) if steps else np.zeros((0, len(ctx.qids)), bool)
    tmp = od / f"{op}.outcomes.tmp.npz"
    np.savez_compressed(tmp, qids=np.array(ctx.qids), category=ctx.cats, dense=res["_dense"], cand=cand,
                        step=np.array(steps), params_removed=np.array([res["steps"][str(s)]["params_removed"] for s in steps]))
    tmp.replace(od / f"{op}.outcomes.npz")
    rec = {k: v for k, v in res.items() if not k.startswith("_")}
    rec.update({"label": LABEL, "outcomes_sha256": sha256_file(od / f"{op}.outcomes.npz"), "meta": _meta()})
    atomic_write_json(od / f"{op}.json", rec)


def _load_scores(ctx: _Ctx, op: str) -> dict | None:
    od = ctx.out / "eval"
    if not (od / f"{op}.json").exists():
        return None
    rec = read_json(od / f"{op}.json")
    if sha256_file(od / f"{op}.outcomes.npz") != rec["outcomes_sha256"]:
        raise SystemExit(f"{op}.outcomes.npz changed after it was written")
    with np.load(od / f"{op}.outcomes.npz", allow_pickle=False) as z:
        rec["_dense"] = z["dense"]
        rec["_cand"] = {str(int(s)): z["cand"][i] for i, s in enumerate(z["step"])}
    return rec


def score(cfg: dict, op: str) -> Path:
    if op not in OPS:
        raise SystemExit(f"--operator must be one of {OPS}")
    C = _C(cfg)
    out = out_dir(cfg)
    order = _order(cfg)
    limit, L0, Lmax = float(C["reach_limit"]), int(C["trajectory_steps"]), int(C["max_steps"])
    if op == "comp":
        dec = read_json(out / "eval" / "mask_decision.json")
        if not dec["proceed"]:
            raise SystemExit("the extension rule stopped the experiment: ask before running the compensated trajectory")
        L = int(dec["L"])
        cm = read_json(out / "compensated" / "manifest.json")
        if len(cm["steps"]) < L:
            raise SystemExit(f"the compensated trajectory has {len(cm['steps'])} of {L} steps: run `compensation compensate`")
    ctx = _Ctx(cfg)
    prev = _load_scores(ctx, op)
    t0 = time.time()
    if op == "mask":
        if (out / "eval" / "mask_decision.json").exists():
            raise SystemExit("the mask-only trajectory is scored and its decision recorded already")
        res = _score_steps(ctx, op, list(range(1, L0 + 1)), prev)
        worst = [res["steps"][str(t)]["worst"] for t in range(1, L0 + 1)]
        r = reach(worst, limit)
        ext = extension(r, L0, Lmax)
        dec = {"label": LABEL, "L_initial": L0, "R_mask_initial": r, **ext, "history": []}
        if ext["extend"]:
            if ext["L"] > len(order):
                raise SystemExit(f"the reference has {len(order)} steps; the rule asks for {ext['L']}")
            res = _score_steps(ctx, op, list(range(L0 + 1, ext["L"] + 1)), res)
            worst = [res["steps"][str(t)]["worst"] for t in range(1, ext["L"] + 1)]
            r2 = reach(worst, limit)
            dec["history"].append({"after_extension_to": ext["L"], "R_mask": r2})
            dec.update({"R_mask": r2, "proceed": 2 * r2 + 2 <= Lmax})
        else:
            dec["R_mask"] = r
        dec["dense_ppl"] = ctx.perplexity("dense", ctx.dm)
        dec["meta"] = _meta()
        dec["seconds"] = time.time() - t0
        _save_scores(ctx, op, {**res, "dense_ppl": dec["dense_ppl"]})
        atomic_write_json(out / "eval" / "mask_decision.json", dec)  # recorded BEFORE any compensated model is scored
        if not dec["proceed"]:
            raise SystemExit(f"2 x R_mask + 2 > {Lmax} (R_mask = {dec['R_mask']}): stop and ask")
        return out / "eval" / "mask_decision.json"
    res = _score_steps(ctx, op, list(range(1, L + 1)), prev)
    res["dense_ppl"] = ctx.perplexity("dense", ctx.dm)
    _save_scores(ctx, op, res)
    return out / "eval" / f"{op}.json"


# ============================================================================ the compensated trajectory
def _comp_trajectory(ctx: _Ctx, crash_hook=None):
    """The CompensatedTrajectory on a working copy of the dense model, with the calibration windows of the plan."""
    import copy

    from .compensate import CompensatedTrajectory
    from .model import build_units, decoder_layers
    from .pruning.masks import PruneState

    torch = ctx.torch
    C = ctx.C
    if not hasattr(ctx, "_traj"):
        work = copy.deepcopy(ctx.dm)
        work.eval()
        wunits = build_units(work, "structured", ctx.rc["pruning"].get("eligible", []))
        units = {u.name: (u.linears["gate"], u.linears["up"], u.linears["down"]) for u in wunits}
        dense = {u.name: tuple(u.linears[p].weight.detach().clone() for p in ("gate", "up", "down")) for u in wunits}
        order = _order(ctx.cfg)
        masks, hashes = [], []
        for o in order:
            st = PruneState(wunits, "structured")
            st.load(o["mask_file"])
            masks.append({n: m.detach().to("cpu").clone() for n, m in st.masks.items()})
            hashes.append(o["mask_hash"])
        cal = C["calibration"]
        idx = ctx.data.calibration_indices(int(cal["seed"]), int(cal["windows"]))
        windows = ctx.data.calibration_pool[idx]
        layers = decoder_layers(work)
        act = layers[0].mlp.act_fn

        class _Stop(Exception):
            pass

        def inputs(name):
            i = int(name.split(".")[0][1:])
            mlp = layers[i].mlp
            got = []

            def hook(mod, args):
                got.append(args[0].detach())
                raise _Stop

            hd = mlp.register_forward_pre_hook(hook)
            try:
                for s in range(0, windows.shape[0], 8):
                    ids = windows[s:s + 8].to(ctx.dev)
                    try:
                        with torch.no_grad():
                            work.model(input_ids=ids, use_cache=False)
                    except _Stop:
                        pass
                    x = got.pop()
                    yield x.reshape(-1, x.shape[-1])
            finally:
                hd.remove()

        traj = CompensatedTrajectory(ctx.out / "compensated", units, dense, inputs, act, masks, hashes,
                                     ridge_fraction=float(C["ridge_fraction"]), crash_hook=crash_hook, log=log)
        ctx._traj = (traj, work, idx)
    return ctx._traj[0], ctx._traj[1]


def compensate(cfg: dict) -> Path:
    out = out_dir(cfg)
    dec = read_json(out / "eval" / "mask_decision.json")
    if not dec["proceed"]:
        raise SystemExit("the extension rule stopped the experiment: ask before running the compensated trajectory")
    L = int(dec["L"])
    ctx = _Ctx(cfg)
    traj, work = _comp_trajectory(ctx)
    idx = ctx._traj[2]
    prof_idx = ctx.data.calibration_indices(int(_C(cfg)["seed"]), int(ctx.rc["data"]["calibration"]["num_samples"]))
    t0 = time.time()
    hashes = traj.run(L)
    rec = {"label": LABEL, "steps": L, "checkpoint_hashes": hashes, "ridge_fraction": float(_C(cfg)["ridge_fraction"]),
           "calibration": {"seed": int(_C(cfg)["calibration"]["seed"]), "windows": len(idx), "indices": idx,
                           "profiling_seed": int(_C(cfg)["seed"]), "profiling_indices": prof_idx,
                           "shared_with_profiling": len(set(idx) & set(prof_idx))},
           "seconds_this_invocation": time.time() - t0, "meta": _meta()}
    atomic_write_json(out / "compensated" / "trajectory.json", rec)
    return out / "compensated" / "trajectory.json"


# ============================================================================ analysis (CPU)
def verdict(r_mask: int, r_comp: int) -> str:
    """The plan's section 5 rule."""
    if r_comp >= 2 * r_mask and r_comp - r_mask >= 5:
        return "GO"
    if r_comp < 1.5 * r_mask:
        return "NO-GO"
    return "INCONCLUSIVE"


def _load_outcomes(out: Path, op: str) -> tuple[dict, dict]:
    rec = read_json(out / "eval" / f"{op}.json")
    if sha256_file(out / "eval" / f"{op}.outcomes.npz") != rec["outcomes_sha256"]:
        raise SystemExit(f"{op}.outcomes.npz changed after it was written")
    with np.load(out / "eval" / f"{op}.outcomes.npz", allow_pickle=False) as z:
        oc = {k: z[k] for k in z.files}
    return rec, oc


def analyze(cfg: dict) -> Path:
    from .population_regate import population_bounds
    from .screen_confirm import fixed_sequence

    C = _C(cfg)
    out = out_dir(cfg)
    dec = read_json(out / "eval" / "mask_decision.json")
    L = int(dec["L"])
    limit = float(C["reach_limit"])
    alpha, tol = float(C["confirmation_rule"]["alpha"]), float(C["confirmation_rule"]["tolerance"])
    recs, ocs = {}, {}
    for op in OPS:
        recs[op], ocs[op] = _load_outcomes(out, op)
    qids = ocs["mask"]["qids"]
    if list(qids) != list(ocs["comp"]["qids"]) or not np.array_equal(ocs["mask"]["dense"], ocs["comp"]["dense"]):
        raise SystemExit("the two operators were not scored on the same items with the same dense answers")
    cats, dense = ocs["mask"]["category"], ocs["mask"]["dense"]
    order = _order(cfg)
    dense_params = int(recs["mask"]["steps"]["1"]["materialized_parameters"]) + int(order[0]["params_removed"])
    per_step, worst = [], {op: [] for op in OPS}
    for t in range(1, L + 1):
        row = {"step": t, "params_removed": int(order[t - 1]["params_removed"]), "ops": {}, "difference": {}}
        cand = {}
        for op in OPS:
            i = int(np.flatnonzero(ocs[op]["step"] == t)[0])
            cand[op] = ocs[op]["cand"][i]
            cnt = {}
            for c in CATS:
                m = cats == c
                d, k = dense[m], cand[op][m]
                n, kp, km = int(m.sum()), int(np.count_nonzero(d & ~k)), int(np.count_nonzero(k & ~d))
                lb, ub = population_bounds(n, kp, km, alpha, alpha)
                cnt[c] = {"n": n, "lost": kp, "gained": km, "estimate": (kp - km) / n, "lower": lb, "upper": ub}
            w = max(v["estimate"] for v in cnt.values())
            worst[op].append(w)
            s = recs[op]["steps"][str(t)]
            row["ops"][op] = {"categories": cnt, "worst": w, "ppl": s["ppl"], "ppl_ratio": s["ppl"] / recs[op]["dense_ppl"],
                              "key": s["key"]}
        for c in CATS:  # mask-only minus compensated, on the same items: P(comp right, mask wrong) - P(mask right, comp wrong)
            m = cats == c
            a = int(np.count_nonzero(cand["comp"][m] & ~cand["mask"][m]))
            b = int(np.count_nonzero(cand["mask"][m] & ~cand["comp"][m]))
            n = int(m.sum())
            lo, hi = population_bounds(n, a, b, 0.025, 0.025)
            row["difference"][c] = {"n": n, "comp_right_mask_wrong": a, "mask_right_comp_wrong": b, "estimate": (a - b) / n,
                                    "lower95": lo, "upper95": hi}
        per_step.append(row)
    r_mask, r_comp = reach(worst["mask"], limit), reach(worst["comp"], limit)
    if r_mask != int(dec["R_mask"]):
        raise SystemExit(f"R_mask recomputed ({r_mask}) differs from the recorded decision ({dec['R_mask']})")
    v = verdict(r_mask, r_comp)
    claims = {}
    for op in OPS:
        seq = [{"id": r["step"], "params_removed": r["params_removed"],
                "counts": {c: (x["n"], x["lost"], x["gained"]) for c, x in r["ops"][op]["categories"].items()}} for r in per_step]
        fs = fixed_sequence(seq, alpha, tol)
        claims[op] = {"claim_step": fs["claim"], "tested": len(fs["tested"]),
                      "parameters_removed": 0 if fs["claim"] is None else per_step[fs["claim"] - 1]["params_removed"],
                      "label": "on seen items, not a certificate"}
    # the compensation records
    comp_steps = []
    for t in range(1, L + 1):
        r = read_json(out / "compensated" / "steps" / f"step_{t:03d}.json")
        comp_steps.append({"step": t, "seconds": r["seconds"], "peak_gpu_bytes": r["peak_gpu_bytes"],
                           "checkpoint_hash": r["checkpoint_hash"], "units": r["diagnostics"]})
    traj = read_json(out / "compensated" / "trajectory.json")
    res = {"label": LABEL, "L": L, "decision": dec, "R_mask": r_mask, "R_comp": r_comp, "verdict": v,
           "rule": {"GO": "R_comp >= 2 R_mask and R_comp - R_mask >= 5", "NO-GO": "R_comp < 1.5 R_mask",
                    "INCONCLUSIVE": "otherwise"},
           "reach_limit": limit, "dense_parameters": dense_params, "dense_ppl": recs["mask"]["dense_ppl"],
           "worst": worst, "per_step": per_step, "confirmation_rule": claims, "compensation": comp_steps,
           "calibration": traj["calibration"],
           "sources": {"mask": str(out / "eval" / "mask.json"), "comp": str(out / "eval" / "comp.json"),
                       "decision": str(out / "eval" / "mask_decision.json"), "steps": str(out / "compensated" / "steps"),
                       "eval_set": str(out / "eval" / "eval_set.json"), "reference": str(out / "reference_order.json")},
           "meta": _meta()}
    ad = out / "analysis"
    ad.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ad / "analysis.json", res)
    _figure(res, ad)
    atomic_write_text(ad / "analysis.md", render(res))
    return ad / "analysis.md"


def _figure(res: dict, ad: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"mask": "#2a78d6", "comp": "#eb6834"}  # reference palette, slots 1-2 (validated)
    mark = {"mask": "o", "comp": "s"}
    x = [r["params_removed"] / 1e6 for r in res["per_step"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for lim, lab in ((0.02, "2-point reach limit"), (0.03, "3-point confirmation limit")):
        ax.axhline(100 * lim, color="#52514e", lw=0.8, ls=(0, (4, 3)) if lim == 0.02 else (0, (1, 2)))
        ax.annotate(lab, (x[-1], 100 * lim), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=7, color="#52514e")
    for op in OPS:
        y = [100 * w for w in res["worst"][op]]
        R = res["R_mask"] if op == "mask" else res["R_comp"]
        ax.plot(x, y, color=color[op], marker=mark[op], ms=3.5, lw=1.5, label=f"{OP_LABEL[op]} (reach {R} steps)")
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.set_xlabel("parameters removed (M)")
    ax.set_ylabel("worst-category paired loss (points)")
    ax.grid(color="#e1e0d9", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"EXPLORATORY, Qwen3-1.7B, seen items: verdict {res['verdict']}", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ad / f"worst_loss.{ext}", dpi=220, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else {})
    plt.close(fig)


def render(R: dict) -> str:
    pt = lambda v: f"{100 * v:+.2f}"  # noqa: E731
    L = ["# Compensation feasibility: analysis", "", f"**{R['label']}.**", "",
         f"Trajectory length L = {R['L']} (initial {R['decision']['L_initial']}; extended: {R['decision']['extend']}). "
         f"Reach limit {100 * R['reach_limit']:.1f} points, worst category, point estimate.", "",
         f"**R_mask = {R['R_mask']}, R_comp = {R['R_comp']}: {R['verdict']}** (GO: R_comp >= 2 R_mask and R_comp - R_mask >= 5; "
         "NO-GO: R_comp < 1.5 R_mask).", "",
         f"Dense: {R['dense_parameters']:,} parameters; C4 validation perplexity {R['dense_ppl']:.3f}.", "",
         "## Worst-category loss and perplexity per step", "",
         "| step | parameters removed | mask-only worst | compensated worst | mask-only ppl / dense | compensated ppl / dense |",
         "|---|---|---|---|---|---|"]
    for r in R["per_step"]:
        m, c = r["ops"]["mask"], r["ops"]["comp"]
        L.append(f"| {r['step']} | {r['params_removed'] / 1e6:.1f} M | {pt(m['worst'])} | {pt(c['worst'])} | {m['ppl_ratio']:.3f} | {c['ppl_ratio']:.3f} |")
    L += ["", "## Per category: paired loss against dense (estimate [one-sided 95 % lower, upper])", ""]
    for op in OPS:
        L += [f"### {OP_LABEL[op]}", "", "| step | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
        for r in R["per_step"]:
            L.append(f"| {r['step']} | " + " | ".join(
                f"{pt(x['estimate'])} [{pt(x['lower'])}, {pt(x['upper'])}]" for x in (r["ops"][op]["categories"][c] for c in CATS)) + " |")
        L.append("")
    L += ["## Mask-only minus compensated, same items (estimate [two-sided 95 % interval]; positive = compensation better)", "",
          "| step | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for r in R["per_step"]:
        L.append(f"| {r['step']} | " + " | ".join(
            f"{pt(x['estimate'])} [{pt(x['lower95'])}, {pt(x['upper95'])}]" for x in (r["difference"][c] for c in CATS)) + " |")
    L += ["", "## The pre-registered confirmation rule on this set (on seen items, not a certificate)", "",
          "| operator | claim (step) | parameters removed | steps tested |", "|---|---|---|---|"]
    for op in OPS:
        x = R["confirmation_rule"][op]
        L.append(f"| {OP_LABEL[op]} | {x['claim_step'] or 'none'} | {x['parameters_removed'] / 1e6:.1f} M | {x['tested']} |")
    L += ["", "## Compensation steps", "",
          "| step | seconds | peak GPU (GB) | unit | kept | lambda | cond C[K,K] | cond with ridge | rel. error before | after |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in R["compensation"]:
        for i, (n, d) in enumerate(s["units"].items()):
            head = f"| {s['step']} | {s['seconds']:.1f} | {s['peak_gpu_bytes'] / 1e9:.1f} |" if i == 0 else "| | | |"
            L.append(f"{head} {n} | {d['kept']} | {d['lambda']:.3g} | {d['cond']:.3g} | {d['cond_ridge']:.3g} | "
                     f"{d['rel_error_before']:.4f} | {d['rel_error_after']:.4f} |")
    cal = R["calibration"]
    L += ["", f"Calibration: {cal['windows']} windows (seed {cal['seed']}); {cal['shared_with_profiling']} shared with the "
          f"profiling selection (seed {cal['profiling_seed']}).", ""]
    return "\n".join(L) + "\n"
