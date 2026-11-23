"""Continuous paired signals on the gate pool (``capprune pool-signals score|analyze``), plan docs/pool_signals_plan.md.

score    GPU: the dense model and every state of the compression-boundary map on the gate pool, with identical batches
         (per category, seeded order, 100 per batch, same token budget); saves, per item and model, the full-vocabulary
         log-softmax of the four answer letters, the prediction and correctness (``logprobs.npz``).
analyze  CPU: per state and category the net flip loss (S0), the gross flip rate (S0b), the mean change in the correct
         answer's log-probability (S1) and the mean KL(dense || pruned) over A-D (S2); descriptive Spearman / Pearson
         correlations with the exploratory audit losses at three levels (per category, worst category, overall).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("pool_signals")


def _ps(cfg: dict) -> dict:
    return cfg["pool_signals"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _ps(cfg)["out_dir"])


def plan_params(cfg: dict) -> tuple[dict, str]:
    p = resolve_path(cfg, _ps(cfg)["plan_file"])
    text = p.read_bytes().replace(b"\r\n", b"\n")
    m = re.search(rb"```json\s*(\{.*?\})\s*```", text, re.S)
    return json.loads(m.group(1)), hashlib.sha256(text).hexdigest()


# ============================================================================ log-probability scoring
class LogprobScorer:
    """MMLUScorer's prompt, sort order and token-budget batching, returning the log-softmax of the four letters."""

    def __init__(self, model, tok, fewshot, shots: int, max_len: int, max_batch_tokens: int):
        from .eval.mmlu import MMLUScorer

        self.base = MMLUScorer(model, tok, fewshot, shots, max_len, {}, "logprob", max_batch_tokens)

    def score(self, rows: list[dict], budget: int) -> tuple[dict, int]:
        import torch

        from .utils.gpu import free_memory, is_oom

        b = self.base
        b.budget = budget
        enc = [(r, b.encode(r)) for r in rows]
        enc.sort(key=lambda x: len(x[1]), reverse=True)  # = MMLUScorer.score
        out, i = {}, 0
        while i < len(enc):
            L = len(enc[i][1])
            n = max(1, b.budget // max(L, 1))
            batch = enc[i : i + n]
            try:
                seqs = [s for _, s in batch]
                Lm = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), Lm), b.pad_id, dtype=torch.long)
                att = torch.zeros((len(seqs), Lm), dtype=torch.long)
                for j, s in enumerate(seqs):
                    ids[j, Lm - len(s):] = torch.tensor(s)
                    att[j, Lm - len(s):] = 1
                pos = (att.cumsum(-1) - 1).clamp_min(0)
                with torch.no_grad():
                    try:
                        o = b.model(input_ids=ids.to(b.dev), attention_mask=att.to(b.dev), position_ids=pos.to(b.dev),
                                    use_cache=False, logits_to_keep=1)
                    except TypeError:
                        o = b.model(input_ids=ids.to(b.dev), attention_mask=att.to(b.dev), position_ids=pos.to(b.dev), use_cache=False)
                lp = o.logits[:, -1, :].float().log_softmax(-1)[:, b.letter_ids].cpu().numpy()
            except BaseException as e:
                if not is_oom(e) or n == 1:
                    raise
                free_memory()
                b.budget = max(L, b.budget // 2)
                continue
            for (r, _), row in zip(batch, lp):
                pred = int(np.argmax(row))
                out[r["qid"]] = {"logp": [float(x) for x in row], "pred": pred, "correct": int(pred == r["answer"])}
            i += len(batch)
        return out, b.budget


def score(cfg: dict) -> Path:
    import torch

    from .adaptive.evaluator import question_order
    from .config import data_dir
    from .data.prepare import PreparedData
    from .gradient_audit import _clock, run_paths
    from .gradient_live import gate_pool, load_dense_copy
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pruning.masks import DenseStore, PruneState
    from .ranker import mask_hash
    from .utils.cache import KVCache

    t_all = time.time()
    P, plan_sha = plan_params(cfg)
    root = resolve_path(cfg, _ps(cfg)["runs_root"])
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rc, _ = run_paths(cfg)
    states = read_json(root / P["states_file"])["states"]
    for s in states:  # re-verify every mask before scoring
        if sha256_file(Path(s["mask_file"])) != s["mask_sha256"]:
            raise RuntimeError(f"{s['name']}: mask file changed")
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    tok = load_tokenizer(rc)
    model = load_model(rc)
    model.eval()
    units = build_units(model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))
    dense = DenseStore(units)
    dm = load_dense_copy(rc, model, units, dense)
    dev = next(model.parameters()).device
    tt = rc["testing"]
    shots, max_len, mbt = int(tt.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)), int(tt.get("max_batch_tokens", 32768))
    pool, _ = gate_pool(rc, data)
    field = rc["gate"]["capability"].get("domain_field", "category")
    by_cat: dict[str, list[dict]] = {}
    for r in pool:
        by_cat.setdefault(r[field], []).append(r)
    batches = []
    for cat in sorted(by_cat):
        rmap = {r["qid"]: r for r in by_cat[cat]}
        order = question_order(list(rmap), int(P["order_seed"]), cat)
        for i in range(0, len(order), int(P["batch_size"])):
            batches.append((cat, i, [rmap[q] for q in order[i : i + int(P["batch_size"])]]))
    cache = KVCache(out / "scores.sqlite")
    pool_sha = hashlib.sha256("\n".join(r["qid"] for r in pool).encode()).hexdigest()[:16]
    models = [("dense", None)] + [(s["state_id"], s) for s in states]
    res, timing, evals = {}, {}, {}
    budgets: dict[tuple, int] = {}
    for mid, s in models:
        if s is None:
            sc = LogprobScorer(dm, tok, data.mmlu_fewshot, shots, max_len, mbt)
        else:
            st = PruneState(units, rc["pruning"]["mode"])
            st.load(s["mask_file"])
            if mask_hash(st) != s["mask_hash"]:
                raise RuntimeError(f"{s['name']}: mask hash differs from the boundary's verified record")
            st.apply(dense)
            sc = LogprobScorer(model, tok, data.mmlu_fewshot, shots, max_len, mbt)
        t0, n_run = _clock(dev), 0
        res[mid] = {}
        for cat, i, rows in batches:
            key = f"{mid}|{pool_sha}|{cat}|{i}|{len(rows)}"
            rec = cache.get(key)
            if rec is None:
                b0 = budgets.get((cat, i), mbt)
                r_, b = sc.score(rows, b0)
                if (cat, i) in budgets and b != budgets[(cat, i)]:
                    raise RuntimeError(f"batch layout changed for {mid} {cat} {i} (OOM); rerun with a smaller token budget")
                budgets[(cat, i)] = b
                rec = {"items": r_, "budget": b}
                cache.put(key, rec)
                n_run += len(rows)
            else:
                budgets.setdefault((cat, i), int(rec["budget"]))
            res[mid].update(rec["items"])
        timing[mid] = _clock(dev) - t0
        evals[mid] = n_run
        log.info("%s: %d items (%d scored now) in %.1f s", mid, len(res[mid]), n_run, timing[mid])
    cache.close()
    qids = [r["qid"] for _, _, rows in batches for r in rows]
    cats = [r[field] for _, _, rows in batches for r in rows]
    ans = [int(r["answer"]) for _, _, rows in batches for r in rows]
    arrays = {"qid": np.array(qids), "category": np.array(cats), "answer": np.array(ans, dtype=np.int8)}
    for mid in res:
        arrays[f"logp__{mid}"] = np.array([res[mid][q]["logp"] for q in qids], dtype=np.float32)
        arrays[f"correct__{mid}"] = np.array([res[mid][q]["correct"] for q in qids], dtype=np.int8)
    np.savez_compressed(out / "logprobs.npz", **arrays)
    from .results_index import git

    man = {"plan_sha256_lf": plan_sha, "commit": git("rev-parse", "HEAD"), "clean_tree": git("status", "--porcelain") == "",
           "states": [{k: s[k] for k in ("name", "state_id", "mask_hash", "mask_sha256", "mlp_sparsity")} for s in states],
           "pool_items": len(qids), "pool_sha16": pool_sha, "protocol": {"shots": shots, "max_seq_len": max_len,
           "max_batch_tokens": mbt, "batch": P["batch_size"], "order_seed": P["order_seed"], "weights": "masked bf16",
           "logprob": "full-vocabulary log-softmax at the last position, letters A-D"},
           "item_model_evaluations_run": evals, "gpu_seconds": timing, "wall_s": time.time() - t_all,
           "npz_sha256": sha256_file(out / "logprobs.npz")}
    atomic_write_json(out / "manifest.json", man)
    del model, dm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out / "manifest.json"


# ============================================================================ signals and comparison
def signals(lp_dense: np.ndarray, lp_pruned: np.ndarray, answer: np.ndarray, c_dense: np.ndarray, c_pruned: np.ndarray) -> dict:
    """The plan's four signals over a set of items."""
    x = c_dense.astype(int) - c_pruned.astype(int)
    idx = np.arange(len(answer))
    d_corr = lp_dense[idx, answer] - lp_pruned[idx, answer]

    def norm(lp):
        z = lp - lp.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    a, b = norm(lp_dense.astype(np.float64)), norm(lp_pruned.astype(np.float64))
    kl = (np.exp(a) * (a - b)).sum(1)
    return {"S0": float(x.mean()), "S0b": float(np.abs(x).mean()), "S1": float(d_corr.mean()), "S2": float(kl.mean()), "n": int(len(x))}


def _rank(v: np.ndarray) -> np.ndarray:
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    for val in np.unique(v):  # average ranks for ties
        m = v == val
        r[m] = r[m].mean()
    return r


def corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "n": len(x)}
    return {"pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]), "n": len(x)}


def analyze(cfg: dict) -> Path:
    P, plan_sha = plan_params(cfg)
    root = resolve_path(cfg, _ps(cfg)["runs_root"])
    out = out_dir(cfg)
    man = read_json(out / "manifest.json")
    if man["plan_sha256_lf"] != plan_sha:
        raise SystemExit("the analysis plan changed after scoring")
    Z = np.load(out / "logprobs.npz")
    B = read_json(root / P["audit_file"])
    cats = sorted(set(Z["category"].tolist()))
    states = [s for s in B["states"]]
    sig = {}
    for s in states:
        sid = s["state_id"]
        sig[sid] = {}
        for c in [*cats, "overall"]:
            m = np.ones(len(Z["qid"]), bool) if c == "overall" else Z["category"] == c
            sig[sid][c] = signals(Z["logp__dense"][m], Z[f"logp__{sid}"][m], Z["answer"][m].astype(int),
                                  Z["correct__dense"][m], Z[f"correct__{sid}"][m])
    comp = {}
    for tgt in P["targets"]:
        comp[tgt] = {}
        for sname in P["signals"]:
            pc_x, pc_y, w_x, w_y, o_x, o_y = [], [], [], [], [], []
            for s in states:
                st = s["sets"][tgt]["stats"]
                for c in cats:
                    pc_x.append(sig[s["state_id"]][c][sname])
                    pc_y.append(st[c]["loss"])
                w_x.append(max(sig[s["state_id"]][c][sname] for c in cats))
                w_y.append(st["conclusion"]["worst_loss"])
                o_x.append(sig[s["state_id"]]["overall"][sname])
                o_y.append(st["overall"]["loss"])
            comp[tgt][sname] = {"per_category": corr(pc_x, pc_y), "worst": corr(w_x, w_y), "overall": corr(o_x, o_y)}
    S = {"plan_sha256_lf": plan_sha, "manifest": man, "signals": sig, "comparison": comp,
         "states": [{k: s[k] for k in ("name", "state_id", "mlp_sparsity")} | {"params_fraction": s["size"]["fraction_of_total_parameters_removed"],
                    "audit": {t: {"worst": s["sets"][t]["stats"]["conclusion"]["worst_loss"], "overall": s["sets"][t]["stats"]["overall"]["loss"],
                                  **{c: s["sets"][t]["stats"][c]["loss"] for c in cats}} for t in P["targets"]}} for s in states],
         "label": "EXPLORATORY: 7 checkpoints of one trajectory; audit sets inspected before; descriptive correlations only"}
    atomic_write_json(out / "signals.json", S)
    atomic_write_text(out / "report.md", render(S, cats, P))
    return out / "report.md"


def render(S: dict, cats: list[str], P: dict) -> str:
    names = {"S0": "net flip loss (gate statistic)", "S0b": "gross flip rate", "S1": "Δ log p(correct)", "S2": "KL(dense‖pruned) over A-D"}
    tn = {"comparison": "MMLU-CF validation", "exploratory": "MMLU held-out"}
    f = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
    m = S["manifest"]
    L = ["# Continuous paired signals on the gate pool vs. exploratory audit losses (Phase 2, EXPLORATORY)", "",
         f"Plan `docs/pool_signals_plan.md` (sha256 LF `{S['plan_sha256_lf'][:16]}...`, committed before scoring); scoring commit "
         f"`{m['commit'][:12]}` (clean: {m['clean_tree']}); {m['pool_items']:,} gate-pool items x {len(m['states']) + 1} models, "
         f"identical batches; {sum(m['item_model_evaluations_run'].values()):,} item-model evaluations, "
         f"{sum(m['gpu_seconds'].values()):.0f} GPU s. Seven checkpoints of one trajectory: descriptive only.", "",
         "## Signals per checkpoint (pool, all categories) and audit losses (pts)", "",
         "| state | params removed | S0 net flip (pts) | S0b gross flip (%) | S1 Δ log p(correct) | S2 KL (nats) | audit worst / overall: MMLU-CF | MMLU held-out |",
         "|---|---|---|---|---|---|---|---|"]
    for s in S["states"]:
        g = S["signals"][s["state_id"]]["overall"]
        a = s["audit"]
        L.append(f"| {s['name']} | {100 * s['params_fraction']:.2f} % | {100 * g['S0']:+.2f} | {100 * g['S0b']:.2f} | {g['S1']:+.4f} | "
                 f"{g['S2']:.4f} | {100 * a['comparison']['worst']:+.2f} / {100 * a['comparison']['overall']:+.2f} | "
                 f"{100 * a['exploratory']['worst']:+.2f} / {100 * a['exploratory']['overall']:+.2f} |")
    L += ["", "## Correlation with the audit losses (Spearman ρ / Pearson r)", "",
          "| audit set | signal | per category (28 pairs) | worst category (7) | overall (7) |", "|---|---|---|---|---|"]
    for t, d in S["comparison"].items():
        for sname in P["signals"]:
            c = d[sname]
            L.append(f"| {tn.get(t, t)} | {sname}: {names[sname]} | {f(c['per_category']['spearman'])} / {f(c['per_category']['pearson'])} | "
                     f"{f(c['worst']['spearman'])} / {f(c['worst']['pearson'])} | {f(c['overall']['spearman'])} / {f(c['overall']['pearson'])} |")
    L += ["", "Per category signals (pts for S0/S0b; nats for S1/S2):", ""]
    for c in cats:
        L.append(f"* {c}: " + "; ".join(f"{s['name']}: S0 {100 * S['signals'][s['state_id']][c]['S0']:+.2f}, S1 "
                                         f"{S['signals'][s['state_id']][c]['S1']:+.4f}, S2 {S['signals'][s['state_id']][c]['S2']:.4f}"
                                         for s in S["states"]))
    return "\n".join(L) + "\n"
