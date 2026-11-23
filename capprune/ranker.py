"""Learned ranking of structured pruning actions (``capprune ranker collect|evaluate``).

Data (``collect``, GPU)
    Questions: the pruning-time pool is split ONCE (seeded) into a small fixed PROBE set (``probe_per_domain``
    per category) and the LABEL set (the rest).  Labels and the capability gate use only the label set; the
    probe set is used only by probe-based selectors; the held-out audit split is not touched.
    Trajectories: each starts at the dense model; at every parent the candidate set (equal removal: every
    action = ``units_per_action`` x ``blocks_per_unit`` blocks) is the greedy-risk set, lowest z_sens, lowest
    z_grad, depth windows and ``random_actions`` seeded random unit sets.  Every candidate is scored once on the
    probe set and on the label set; the gate (threshold-aware, tolerance from the config, budget = label set)
    is replayed with one question order per parent.  The next parent is a seeded random ACCEPTED candidate that
    is not already a parent elsewhere, so no parent mask appears in two trajectories.  The dense parent is
    shared by every trajectory and forms its own group ("D0", training only).
    Development: the fixed-state pilot's four checkpoints and candidates, rescored under this protocol.
    Cache: every score is keyed by (parent-mask hash, action, evaluation-config hash).

Selectors (``evaluate``, CPU) on the same candidate sets: random (expected value), greedy risk, gradient-only,
probe-only (measured probe loss), ridge on pre-scoring features, ridge + probe features; on the development
checkpoints also the pilot's LLM shortlist choices.  Ridge: one model per category predicting the increment
over the parent's measured loss (cumulative = parent + increment); features scaled and alpha tuned by
leave-one-trajectory-out CV inside the training trajectories only.  Test: leave one NEW trajectory out.
Costs per decision: profiling (shared), probe questions on EVERY candidate for probe-based selectors,
gate questions for the chosen candidate; GPU seconds from measured timings.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json
from .utils.log import get_logger

log = get_logger("ranker")


def _rk(cfg: dict) -> dict:
    return cfg["ranker"]


def rk_out(cfg: dict) -> Path:
    return resolve_path(cfg, _rk(cfg)["out_dir"])


# ============================================================================ helpers
def split_pool(rows: list[dict], field: str, domains: list[str], n_per: int, seed: int):
    rng = np.random.default_rng(int(seed))
    P, L = [], []
    for d in domains:
        dr = sorted((r for r in rows if r[field] == d), key=lambda r: r["qid"])
        pick = set(rng.choice(len(dr), size=min(n_per, len(dr) - 1), replace=False).tolist())
        P += [r for i, r in enumerate(dr) if i in pick]
        L += [r for i, r in enumerate(dr) if i not in pick]
    return P, L


def mask_hash(st) -> str:
    h = hashlib.sha256()
    for n in sorted(st.masks):
        h.update(n.encode())
        h.update(np.packbits(st.masks[n].detach().to("cpu").numpy().reshape(-1)).tobytes())
    return h.hexdigest()[:20]


def canon(action: dict) -> str:
    return ",".join(f"{u}:{b}" for u, b in sorted(action.items(), key=lambda kv: (int(kv[0].split(".")[0][1:]), kv[0])))


def _loss(dense: dict, ans: dict, rows: list[dict], field: str) -> dict:
    acc: dict[str, list[int]] = {}
    for r in rows:
        a = acc.setdefault(r[field], [0, 0, 0])
        a[0] += dense[r["qid"]]
        a[1] += ans[r["qid"]]
        a[2] += 1
    return {c: (v[0] - v[1]) / v[2] for c, v in sorted(acc.items())}


class Store:
    """Append-only cache of scored states: key = parent-mask hash | action | evaluation-config hash."""

    def __init__(self, path: Path):
        self.path = path
        self.d = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.d[r["key"]] = r

    def get(self, key):
        return self.d.get(key)

    def put(self, rec):
        self.d[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# ============================================================================ candidates and features
def candidates(profiles: list[dict], geo: dict, sl: dict, n_layers: int, n_random: int, rng) -> list[dict]:
    from .fixed_state import build_shortlist

    base = build_shortlist(profiles, geo, sl, n_layers)
    k, b = int(sl["units_per_action"]), int(sl["blocks_per_unit"])
    room = {p["name"]: geo["cap_blocks"] - int(round(p["sparsity"] * geo["B"])) for p in profiles}
    names = sorted((p["name"] for p in profiles if room[p["name"]] >= b), key=lambda n: int(n.split(".")[0][1:]))
    have = {canon(a["units"]) for a in base}
    out = [{"kind": a["kind"], "units": a["units"]} for a in base]
    tries = 0
    while sum(c["kind"].startswith("random") for c in out) < n_random and tries < 100:
        tries += 1
        pick = sorted(rng.choice(len(names), size=k, replace=False).tolist())
        units = {names[i]: b for i in pick}
        if canon(units) not in have:
            have.add(canon(units))
            out.append({"kind": f"random {sum(c['kind'].startswith('random') for c in out) + 1}", "units": units})
    return out


FEATURES = ["parent_sparsity", "z_s_mean", "z_s_min", "z_s_max", "z_g_mean", "z_g_min", "z_g_max", "risk_mean",
            "depth_mean", "depth_min", "depth_max", "frac_early", "frac_mid", "frac_late",
            "unit_sp_mean", "unit_sp_max", "share_sum"]


def action_features(profiles: list[dict], action: dict, parent_sparsity: float, n_layers: int, B: int) -> dict:
    """Features available BEFORE the candidate is scored: the parent's profile and sparsity, and the action."""
    by = {p["name"]: p for p in profiles}
    ps = [by[u] for u in action]
    z = lambda v: v if v == v else 0.0  # noqa: E731
    zs = np.array([z(p["z_s"]) for p in ps])
    zg = np.array([z(p["z_g"]) for p in ps])
    dep = np.array([int(u.split(".")[0][1:]) / (n_layers - 1) for u in action])
    sp = np.array([p["sparsity"] for p in ps])
    after = np.array([p["sparsity"] + b / B for p, b in zip(ps, action.values())])
    return {"parent_sparsity": parent_sparsity, "z_s_mean": zs.mean(), "z_s_min": zs.min(), "z_s_max": zs.max(),
            "z_g_mean": zg.mean(), "z_g_min": zg.min(), "z_g_max": zg.max(), "risk_mean": float(0.5 * (zs + zg).mean()),
            "depth_mean": dep.mean(), "depth_min": dep.min(), "depth_max": dep.max(),
            "frac_early": float((dep < 1 / 3).mean()), "frac_mid": float(((dep >= 1 / 3) & (dep < 2 / 3)).mean()),
            "frac_late": float((dep >= 2 / 3).mean()), "unit_sp_mean": sp.mean(), "unit_sp_max": float(after.max()),
            "share_sum": float(sum(p["share"] for p in ps))}


# ============================================================================ collect (GPU)
def collect(cfg: dict) -> Path:
    import torch

    from .adaptive.evaluator import evaluate_candidate
    from .adaptive.live import CapabilityGate, LiveSource, attempt_order_seed
    from .adaptive.policies import make_policy
    from .config import deep_merge
    from .eval.mmlu import MMLUScorer
    from .fixed_state import _geometry, _rc, _setup
    from .pruning.masks import PruneState
    from .pruning.profile import collect_act_sqnorm, collect_grad_importance, sensitivity, zscores
    from .pruning.wanda import raise_unit

    rk = _rk(cfg)
    out = rk_out(cfg)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    rc, data, tok, model, units, dense_store, mode = _setup(cfg)
    dev = next(model.parameters()).device
    geo = _geometry(rc, units[0].channels)
    field = rc["gate"]["capability"].get("domain_field", "category")
    domains = list(rc["gate"]["capability"]["domains"])
    P, L = split_pool(data.mmlu_probe, field, domains, int(rk["probe_per_domain"]), int(rk["probe_seed"]))
    chunk = int(rk.get("chunk_size", 512))
    t_cfg = rc["testing"]
    scorer_cfg = {"shots": int(t_cfg.get("shots", 0)), "max_len": int(rc["model"].get("max_seq_len", 2048)),
                  "max_batch_tokens": int(t_cfg.get("max_batch_tokens", 32768)), "chunk": chunk,
                  "model": rc["model"]["name_or_path"], "dtype": rc["model"].get("dtype"),
                  "probe_ids": hashlib.sha256("\n".join(r["qid"] for r in P).encode()).hexdigest()[:16],
                  "label_ids": hashlib.sha256("\n".join(r["qid"] for r in L).encode()).hexdigest()[:16],
                  "pool_sha": data.manifest["files"]["mmlu_probe.jsonl"]["sha256"][:16]}
    eval_key = hashlib.sha256(json.dumps(scorer_cfg, sort_keys=True).encode()).hexdigest()[:12]
    atomic_write_json(out / "split.json", {"probe": [r["qid"] for r in P], "label": [r["qid"] for r in L],
                                           "eval_key": eval_key, "scorer": scorer_cfg})
    sc = MMLUScorer(model, tok, data.mmlu_fewshot, scorer_cfg["shots"], scorer_cfg["max_len"], {}, "ranker",
                    scorer_cfg["max_batch_tokens"])

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def score(rows):
        sync()
        t0 = time.perf_counter()
        res = {}
        for i in range(0, len(rows), chunk):
            res.update({q: int(r["correct"]) for q, r in sc.score(rows[i : i + chunk]).items()})
        sync()
        return res, time.perf_counter() - t0

    store = Store(out / "scores.jsonl")
    PruneState(units, mode).apply(dense_store)
    dense_st = PruneState(units, mode)
    dkey = f"{mask_hash(dense_st)}||{eval_key}"
    drec = store.get(dkey)
    if drec is None:
        aP, tP = score(P)
        aL, tL = score(L)
        drec = {"key": dkey, "parent": mask_hash(dense_st), "action": "", "P": aP, "L": aL, "t_probe": tP, "t_label": tL}
        store.put(drec)
    denseP, denseL = drec["P"], drec["L"]
    rc_g = deep_merge(rc, {"gate": {"capability": {"budget": len(L)}}})
    gate = CapabilityGate(rc_g, L)

    calib = data.calibration_pool[data.calibration_indices(int(rc["run"]["seed"]), int(rc["data"]["calibration"]["num_samples"]))]
    pp = rc["profiling"]
    prng = np.random.default_rng([int(rk.get("profile_seed", 0)), 1, 17])  # identical to fixed_state.prepare
    a_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["activation_samples"]), calib.shape[0]), replace=False))
    g_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["gradient_samples"]), calib.shape[0]), replace=False))

    def profile(st):
        st.apply(dense_store)
        sync()
        t0 = time.perf_counter()
        act = collect_act_sqnorm(model, st, calib[a_idx], {}, None, int(pp.get("activation_batch_size", 4)))
        sens = sensitivity(st, act, float(rc["pruning"].get("sensitivity_percentile", 10)))
        grad = collect_grad_importance(model, st, calib[g_idx])
        sync()
        zs, zg = zscores(sens), zscores(grad)
        profs = [{"name": u.name, "share": st.share(u.name), "z_s": zs[u.name], "z_g": zg[u.name],
                  "sparsity": st.unit_sparsity(u.name)} for u in units]
        return act, profs, time.perf_counter() - t0

    def state_of(parent_masks, act, action):
        st = PruneState(units, mode)
        st.masks = {n: m.clone() for n, m in parent_masks.items()}
        before = {u: int((~st.masks[u]).sum().item()) for u in action}
        for u, b in action.items():
            raise_unit(st, u, (before[u] / geo["mult"] + b) / geo["B"], act, rc)
        real = {u: (int((~st.masks[u]).sum().item()) - before[u]) // geo["mult"] for u in action}
        if real != action:
            raise RuntimeError(f"realised {real} != requested {action}")
        return st

    def score_state(st, parent_h, action):
        key = f"{parent_h}|{canon(action)}|{eval_key}"
        rec = store.get(key)
        if rec is None:
            st.apply(dense_store)
            aP, tP = score(P)
            aL, tL = score(L)
            rec = {"key": key, "parent": parent_h, "action": canon(action), "mask": mask_hash(st),
                   "sparsity": st.global_sparsity(), "P": aP, "L": aL, "t_probe": tP, "t_label": tL}
            store.put(rec)
        return rec

    def gate_replay(ans_L, seed):
        src = LiveSource(gate.pool, {q: bool(v) for q, v in denseL.items()},
                         lambda rows: {r["qid"]: bool(ans_L[r["qid"]]) for r in rows}, seed, gate.budget)
        run = evaluate_candidate(src, make_policy(gate.policy_name), gate.plan, "x")
        return {"decision": run.decision, "questions": run.questions}

    decisions = []
    by_unit = {u.name: u for u in units}

    def decide(group, traj, step, st, parent_rec, cands, order_seed, meta=None):
        act, profs, t_prof = profile(st)
        ph = mask_hash(st)
        if cands is None:
            cands = candidates(profs, geo, rk["shortlist"], len(units), int(rk.get("random_actions", 2)),
                               np.random.default_rng([int(rk["traj_seed"]), 7, traj_index(traj), step]))
        parentL = _loss(denseL, parent_rec["L"], L, field) if parent_rec else {d: 0.0 for d in domains}
        parentP = _loss(denseP, parent_rec["P"], P, field) if parent_rec else {d: 0.0 for d in domains}
        rows = []
        for i, c in enumerate(cands):
            cst = state_of(st.masks, act, c["units"])
            rec = score_state(cst, ph, c["units"])
            lab = _loss(denseL, rec["L"], L, field)
            prb = _loss(denseP, rec["P"], P, field)
            g = gate_replay(rec["L"], order_seed)
            params = sum(b * geo["mult"] * sum(l_.weight.shape[1] if p_ != "down" else l_.weight.shape[0]
                                               for p_, l_ in by_unit[u].linears.items()) for u, b in c["units"].items())
            rows.append({"idx": i, "kind": c["kind"], "action": c["units"], "action_key": canon(c["units"]),
                         "mask": rec["mask"], "sparsity_after": rec["sparsity"], "params_removed": params,
                         "channels_removed": sum(c["units"].values()) * geo["mult"],
                         "layers": sorted(int(u.split(".")[0][1:]) for u in c["units"]),
                         "features": action_features(profs, c["units"], st.global_sparsity(), len(units), geo["B"]),
                         "label": lab, "worst": max(lab.values()), "worst_domain": max(lab, key=lab.get),
                         "probe": prb, "probe_worst": max(prb.values()), "gate": g,
                         "t_probe": rec["t_probe"], "t_label": rec["t_label"],
                         "probe_questions": len(P), "label_questions": len(L)})
            log.info("%s %s step %d cand %d %-22s worst %+.2f probe %+.2f gate %s q=%d", group, traj, step, i, c["kind"],
                     100 * rows[-1]["worst"], 100 * rows[-1]["probe_worst"], g["decision"], g["questions"])
        d = {"group": group, "trajectory": traj, "step": step, "parent_mask": ph, "parent_sparsity": st.global_sparsity(),
             "parent_loss": parentL, "parent_probe_loss": parentP, "profile_seconds": t_prof, "order_seed": order_seed,
             "candidates": rows, **(meta or {})}
        decisions.append(d)
        return d, act

    def traj_index(t):
        return 0 if t in ("G", "D0") else int(str(t)[1:])

    # ---- development: the fixed-state pilot's checkpoints and candidates
    if rk.get("dev_source"):
        src = resolve_path(cfg, rk["dev_source"])
        info = read_json(src / "states.json")
        props = [json.loads(l_) for p in sorted(src.glob("proposals-*.jsonl")) for l_ in p.read_text(encoding="utf-8").splitlines()]
        for i, s in enumerate(info["states"]):
            st = PruneState(units, mode)
            st.load(src / f"mask-{s['name']}.safetensors")
            prec = None if s["state_id"] is None else score_state(st, mask_hash(st), {})
            llm = [{"model": r["model"], "condition": r["condition"], "sample": r["sample"], "outcome": r["outcome"],
                    "action_key": canon(r["action"]) if r.get("outcome") == "valid" else None, "action": r.get("action"),
                    "latency_s": r.get("latency_s")} for r in props if r["state"] == s["name"]]
            cands = [{"kind": a["kind"], "units": a["units"]} for a in s["shortlist"]]
            extra = {canon(r["action"]): r["action"] for r in llm if r["action_key"] and r["action_key"] not in {canon(c["units"]) for c in cands}}
            d, _ = decide("dev", "G", i, st, prec, cands + [{"kind": "llm free-form", "units": a} for a in extra.values()],
                          attempt_order_seed(int(rk.get("order_seed", 0)), int(rc["run"]["seed"]), i + 1),
                          {"state_name": s["name"], "llm": llm, "n_shortlist": len(cands)})
    # ---- new trajectories
    used = set()
    for k in range(1, int(rk["trajectories"]) + 1):
        traj = f"T{k}"
        rng = np.random.default_rng([int(rk["traj_seed"]), k])
        st, prec = PruneState(units, mode), None
        for step in range(0, int(rk["steps"]) + 1):
            seed = attempt_order_seed(int(rk.get("order_seed", 0)), int(rc["run"]["seed"]), 10_000 + 100 * k + step)
            d, act = decide("D0" if step == 0 else "new", traj, step, st, prec, None, seed)
            if step == int(rk["steps"]):
                break
            ok = [c for c in d["candidates"] if c["gate"]["decision"] == "accept" and c["mask"] not in used]
            if not ok:
                log.warning("%s: no unused accepted candidate at step %d; trajectory ends", traj, step)
                break
            nxt = ok[int(rng.integers(len(ok)))]
            used.add(nxt["mask"])
            st = state_of(st.masks, act, nxt["action"])
            prec = store.get(f"{d['parent_mask']}|{nxt['action_key']}|{eval_key}")
    PruneState(units, mode).apply(dense_store)
    atomic_write_json(out / "decisions.json", {"eval_key": eval_key, "probe_questions": len(P), "label_questions": len(L),
                                               "domains": domains, "tolerance": rc["gate"]["capability"]["tolerance"],
                                               "collect_wall_seconds": time.perf_counter() - t_all,
                                               "decisions": decisions})
    return out / "decisions.json"


# ============================================================================ ridge
def _ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return {"mu": mu, "sd": sd, "ym": ym, "W": W}


def _ridge_predict(m, X):
    return (X - m["mu"]) / m["sd"] @ m["W"] + m["ym"]


def _matrices(decs, domains, with_probe):
    X, Y, G = [], [], []
    for d in decs:
        for c in d["candidates"]:
            f = [c["features"][k] for k in FEATURES] + [d["parent_loss"][dm] for dm in domains]
            if with_probe:
                f += [c["probe"][dm] for dm in domains]
            X.append(f)
            Y.append([c["label"][dm] - d["parent_loss"][dm] for dm in domains])
            G.append(d["trajectory"] if d["group"] != "D0" else "D0")
    return np.array(X, float), np.array(Y, float), np.array(G)


def fit_selector(train, domains, with_probe, alphas):
    """Alpha by leave-one-group-out CV inside the training data only; then refit on all training data."""
    X, Y, G = _matrices(train, domains, with_probe)
    groups = sorted(set(G))
    cv = {}
    for a in alphas:
        err = []
        for g in groups:
            tr, te = G != g, G == g
            if tr.sum() < 5 or te.sum() == 0:
                continue
            m = _ridge_fit(X[tr], Y[tr], a)
            err.append(float(((_ridge_predict(m, X[te]) - Y[te]) ** 2).mean()))
        cv[a] = float(np.mean(err)) if err else float("inf")
    best = min(cv, key=lambda a: (cv[a], a))
    return _ridge_fit(X, Y, best) | {"alpha": best, "cv": cv}


# ============================================================================ evaluate
SELECTORS = ["random", "greedy", "gradient", "probe", "ridge", "ridge+probe"]


def _rank(sel, d, model, domains):
    cs = d["candidates"]
    if sel == "greedy":
        key = [c["features"]["risk_mean"] for c in cs]
    elif sel == "gradient":
        key = [c["features"]["z_g_mean"] for c in cs]
    elif sel == "probe":
        key = [c["probe_worst"] for c in cs]
    elif sel in ("ridge", "ridge+probe"):
        X, _, _ = _matrices([d], domains, sel == "ridge+probe")
        pred = _ridge_predict(model, X) + np.array([[d["parent_loss"][dm] for dm in domains]])
        key = pred.max(1).tolist()
    else:
        raise ValueError(sel)
    return sorted(range(len(cs)), key=lambda i: (round(key[i], 12), i))


def _outcome(sel, d, order, label_s_per_q, probe_cost):
    cs = d["candidates"]
    best = min(c["worst"] for c in cs)
    if sel == "random":  # expected value of a uniformly random pick
        return {"worst": float(np.mean([c["worst"] for c in cs])),
                "accept": float(np.mean([c["gate"]["decision"] == "accept" for c in cs])),
                "regret": float(np.mean([c["worst"] for c in cs])) - best,
                "questions": float(np.mean([c["gate"]["questions"] for c in cs])),
                "gate_s": float(np.mean([c["gate"]["questions"] * c["t_label"] / c["label_questions"] for c in cs])),
                "probe_q": 0, "probe_s": 0.0, "choice": "random"}
    c = cs[order[0]]
    pq, ps = probe_cost if sel in ("probe", "ridge+probe") else (0, 0.0)
    return {"worst": c["worst"], "accept": float(c["gate"]["decision"] == "accept"), "regret": c["worst"] - best,
            "questions": c["gate"]["questions"] + pq, "gate_s": c["gate"]["questions"] * c["t_label"] / c["label_questions"],
            "probe_q": pq, "probe_s": ps, "choice": f"{c['kind']} [{','.join(map(str, c['layers']))}]", "pick": order[0]}


def _sequential(order, d, budget_q):
    """Gate the ranked candidates one by one until an ACCEPT or the question budget runs out."""
    spent, tried, sec = 0, 0, 0.0
    for i in order:
        c = d["candidates"][i]
        q = c["gate"]["questions"]
        if spent + q > budget_q and tried:
            break
        spent += q
        tried += 1
        sec += q * c["t_label"] / c["label_questions"]
        if c["gate"]["decision"] == "accept":
            return {"worst": c["worst"], "accept": 1.0, "questions": spent, "gate_s": sec, "tried": tried, "pick": i}
    c = d["candidates"][order[0]]
    return {"worst": c["worst"], "accept": 0.0, "questions": spent, "gate_s": sec, "tried": tried, "pick": order[0]}


BOOT_PAIRS = [("ridge", "gradient"), ("ridge", "probe"), ("ridge+probe", "ridge"), ("probe", "gradient"),
              ("ridge+probe", "gradient"), ("gradient", "greedy"), ("probe", "greedy"), ("greedy", "random")]


def _bootstrap(cfg: dict, D: dict, results: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over LABEL questions (resampled within category; one draw shared by every candidate and
    the dense reference) of differences in mean worst-category loss between selectors on the unseen-trajectory
    test decisions.  The choices do not depend on label answers, so they stay fixed and only their measured
    losses are resampled.  Covers question sampling only, not variation between trajectories."""
    from .config import data_dir
    from .data.prepare import PreparedData
    from .fixed_state import _rc
    from .model import tokenizer_id

    rc = _rc(cfg)
    field = rc["gate"]["capability"].get("domain_field", "category")
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    cat = {r["qid"]: r[field] for r in data.mmlu_probe}
    split = read_json(rk_out(cfg) / "split.json")
    L, P, domains, ek = split["label"], split["probe"], D["domains"], D["eval_key"]
    store = Store(rk_out(cfg) / "scores.jsonl")
    dense = store.get(f"{next(d['parent_mask'] for d in D['decisions'] if d['parent_sparsity'] == 0)}||{ek}")
    test = [d for d in D["decisions"] if d["group"] == "new"]
    pos = {(d["trajectory"], d["step"]): i for i, d in enumerate(test)}
    row, dl, dp = {}, [], []
    for i, d in enumerate(test):
        for j, c in enumerate(d["candidates"]):
            rec = store.get(f"{d['parent_mask']}|{c['action_key']}|{ek}")
            row[(i, j)] = len(dl)
            dl.append([dense["L"][q] - rec["L"][q] for q in L])
            dp.append([dense["P"][q] - rec["P"][q] for q in P])
    dl, dp = np.array(dl, np.float32), np.array(dp, np.float32)
    cl = np.array([domains.index(cat[q]) for q in L])
    cp = np.array([domains.index(cat[q]) for q in P])
    rng = np.random.default_rng(seed)
    per = np.zeros((len(dl), len(domains), n_boot + 1), np.float32)
    se_l, se_p = [], []
    for k in range(len(domains)):
        cols = np.flatnonzero(cl == k)
        W = np.ones((n_boot + 1, len(cols)), np.float32)  # draw 0 = the point estimate
        W[1:] = rng.multinomial(len(cols), np.full(len(cols), 1 / len(cols)), size=n_boot)
        per[:, k, :] = dl[:, cols] @ W.T / len(cols)
        se_l += (dl[:, cols].std(1) / np.sqrt(len(cols))).tolist()
        pc = np.flatnonzero(cp == k)
        se_p += (dp[:, pc].std(1) / np.sqrt(len(pc))).tolist()
    worst = per.max(1)
    for (i, j), r in row.items():  # the resampling reproduces the stored labels at draw 0
        assert abs(worst[r, 0] - test[i]["candidates"][j]["worst"]) < 1e-5

    def series(s):
        rs = [r for r in results if r["selector"] == s and r["fold"] == r["trajectory"]]
        assert len(rs) == len(test), s
        v = np.zeros(n_boot + 1)
        for r in rs:
            i = pos[(r["trajectory"], r["step"])]
            v += (worst[[row[(i, j)] for j in range(len(test[i]["candidates"]))]].mean(0) if s == "random"
                  else worst[row[(i, r["pick"])]])
        return v / len(rs)

    out = []
    for a, b in BOOT_PAIRS:
        diff = series(a) - series(b)
        lo, hi = np.percentile(diff[1:], [2.5, 97.5])
        out.append({"a": a, "b": b, "diff": float(diff[0]), "lo": float(lo), "hi": float(hi),
                    "p_a_better": float((diff[1:] < 0).mean())})
    return {"n_boot": n_boot, "decisions": len(test), "pairs": out,
            "median_se_category_loss_label": float(np.median(se_l)), "median_se_category_loss_probe": float(np.median(se_p)),
            "median_sd_candidate_worst_label": float(np.median(worst[:, 1:].std(1)))}


def evaluate(cfg: dict) -> Path:
    rk = _rk(cfg)
    out = rk_out(cfg)
    t0 = time.perf_counter()
    D = read_json(out / "decisions.json")
    decs, domains = D["decisions"], D["domains"]
    alphas = [float(a) for a in rk.get("ridge_alphas", [0.01, 0.1, 1, 10, 100, 1000])]
    new = [d for d in decs if d["group"] == "new"]
    d0 = [d for d in decs if d["group"] == "D0"]
    dev = [d for d in decs if d["group"] == "dev"]
    trajs = sorted({d["trajectory"] for d in new}, key=lambda t: int(t[1:]))
    # development decisions also hold the LLMs' free-form actions (scored for their own rows); selectors
    # choose among the pilot's shortlist only, as the LLM shortlist conditions did
    dev = [dict(d, all_candidates=d["candidates"], candidates=d["candidates"][: d["n_shortlist"]]) for d in dev]

    def probe_cost(d):  # every candidate is probed, chosen or not
        return (len(d["candidates"]) * D["probe_questions"], float(sum(c["t_probe"] for c in d["candidates"])))

    folds = [(t, [d for d in new if d["trajectory"] != t] + d0, [d for d in new if d["trajectory"] == t]) for t in trajs]
    if dev:  # development: train on new trajectories only (not the dense group, which duplicates the dev dense parent)
        folds.append(("dev (pilot G)", new, dev))
    results, fold_models, spear = [], {}, {}
    for name, train, test in folds:
        # split-integrity checks: no parent mask on both sides
        tr_masks = {d["parent_mask"] for d in train}
        te_masks = {d["parent_mask"] for d in test}
        overlap = tr_masks & te_masks
        if overlap:
            raise RuntimeError(f"fold {name}: parent masks on both sides of the split: {sorted(overlap)[:3]}")
        models = {s: fit_selector(train, domains, s == "ridge+probe", alphas) for s in ("ridge", "ridge+probe")}
        fold_models[name] = {s: {"alpha": m["alpha"], "cv": m["cv"]} for s, m in models.items()}
        for d in test:
            true = np.array([c["worst"] for c in d["candidates"]])
            for sel in ("greedy", "gradient", "probe", "ridge", "ridge+probe"):
                rank_pos = np.empty(len(true))
                rank_pos[_rank(sel, d, models.get(sel), domains)] = np.arange(len(true))
                spear.setdefault((name, sel), []).append(_spearman(rank_pos, true))
            pc = probe_cost(d)
            for sel in SELECTORS:
                order = None if sel == "random" else _rank(sel, d, models.get(sel), domains)
                o = _outcome(sel, d, order, None, pc)
                o.update(selector=sel, fold=name, trajectory=d["trajectory"], step=d["step"],
                         state=d.get("state_name", f"{d['trajectory']}/{d['step']}"),
                         parent_sparsity=d["parent_sparsity"], profile_s=d["profile_seconds"],
                         gpu_s=d["profile_seconds"] + o["gate_s"] + o["probe_s"], n_candidates=len(d["candidates"]),
                         best=min(c["worst"] for c in d["candidates"]),
                         params_removed=sorted({c["params_removed"] for c in d["candidates"]}))
                results.append(o)
            # same question budget as the probe selector, spent on gating more candidates instead
            pq = next(r["questions"] for r in results if r["selector"] == "probe" and r["state"] == results[-1]["state"]
                      and r["fold"] == name)
            for sel in ("gradient", "ridge"):
                order = _rank(sel, d, models.get(sel), domains)
                s = _sequential(order, d, pq)
                results.append({"selector": f"{sel} sequential @probe budget", "fold": name, "trajectory": d["trajectory"],
                                "step": d["step"], "state": d.get("state_name", f"{d['trajectory']}/{d['step']}"),
                                "worst": s["worst"], "accept": s["accept"], "regret": s["worst"] - min(c["worst"] for c in d["candidates"]),
                                "questions": s["questions"], "probe_q": 0, "probe_s": 0.0, "gate_s": s["gate_s"],
                                "profile_s": d["profile_seconds"], "gpu_s": d["profile_seconds"] + s["gate_s"],
                                "choice": f"{s['tried']} gated", "pick": s["pick"], "parent_sparsity": d["parent_sparsity"],
                                "n_candidates": len(d["candidates"]), "best": min(c["worst"] for c in d["candidates"]),
                                "params_removed": sorted({c["params_removed"] for c in d["candidates"]})})
            for r in d.get("llm", []):  # development only: the pilot's LLM choices, rescored under this protocol
                if r["outcome"] != "valid":
                    continue
                c = next((c for c in d["all_candidates"] if c["action_key"] == r["action_key"]), None)
                if c is None:
                    continue
                results.append({"selector": f"LLM {r['model']} {r['condition']}", "fold": name, "trajectory": d["trajectory"],
                                "step": d["step"], "state": d.get("state_name"), "worst": c["worst"],
                                "accept": float(c["gate"]["decision"] == "accept"),
                                "regret": c["worst"] - min(x["worst"] for x in d["candidates"][: d["n_shortlist"]]),
                                "questions": c["gate"]["questions"], "probe_q": 0, "probe_s": 0.0,
                                "gate_s": c["gate"]["questions"] * c["t_label"] / c["label_questions"],
                                "profile_s": d["profile_seconds"], "gpu_s": d["profile_seconds"] + c["gate"]["questions"] * c["t_label"] / c["label_questions"],
                                "choice": f"{c['kind']} [{','.join(map(str, c['layers']))}]", "latency_s": r.get("latency_s"),
                                "parent_sparsity": d["parent_sparsity"], "n_candidates": d["n_shortlist"],
                                "best": min(x["worst"] for x in d["candidates"][: d["n_shortlist"]]),
                                "params_removed": sorted({c["params_removed"]})})
    train_s = time.perf_counter() - t0
    boot = _bootstrap(cfg, D, results) if new else None
    # each cached score counted once (the dense parent's candidates are shared by every trajectory)
    uniq = {(d["parent_mask"], c["action_key"]): c["t_probe"] + c["t_label"] for d in decs for c in d["candidates"]}
    offline = {"collect_wall_seconds": D["collect_wall_seconds"],
               "scoring_gpu_seconds": float(sum(uniq.values())), "distinct_state_actions": len(uniq),
               "profiling_gpu_seconds": float(sum(d["profile_seconds"] for d in decs)),
               "pairs": sum(len(d["candidates"]) for d in decs),
               "distinct_scored": len({c["mask"] for d in decs for c in d["candidates"]}),
               "training_and_evaluation_cpu_seconds": train_s}
    S = {"results": results, "folds": fold_models, "offline": offline, "domains": domains, "bootstrap": boot,
         "spearman": {f"{k[0]}|{k[1]}": float(np.nanmean(v)) for k, v in spear.items()},
         "probe_questions": D["probe_questions"], "label_questions": D["label_questions"],
         "trajectories": {t: [{"step": d["step"], "parent_sparsity": d["parent_sparsity"], "parent_mask": d["parent_mask"],
                               "n": len(d["candidates"]), "accepted": sum(c["gate"]["decision"] == "accept" for c in d["candidates"])}
                              for d in decs if d["trajectory"] == t] for t in ["G", *trajs]}}
    atomic_write_json(out / "evaluation.json", S)
    atomic_write_text(out / "report.md", render(S))
    return out / "report.md"


def _spearman(a, b):
    if len(a) < 3 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def render(S: dict) -> str:
    pts = lambda x: f"{100 * x:+.2f}"  # noqa: E731
    R = S["results"]
    L = ["# Learned action ranking: offline comparison", "",
         f"Every candidate removes the same number of MLP channels. Worst-category loss = exact loss on the {S['label_questions']}-question "
         f"label set vs. the dense model (accuracy points). Probe selectors score {S['probe_questions']} separate probe questions on "
         "EVERY candidate (charged). Gate = threshold-aware, 3-point limit, budget = label set, one order per parent.", ""]
    for fold in dict.fromkeys(r["fold"] for r in R):
        rows = [r for r in R if r["fold"] == fold]
        sels = list(dict.fromkeys(r["selector"] for r in rows))
        n = len({r["state"] for r in rows})
        L += [f"## {fold} ({n} decision points)", "",
              "| selector | mean worst-category loss | accept rate | mean regret | questions / decision | of which probe | GPU s / decision | choices |",
              "|---|---|---|---|---|---|---|---|"]
        for s in sels:
            rs = [r for r in rows if r["selector"] == s]
            L.append(f"| {s} | {pts(np.mean([r['worst'] for r in rs]))} | {100 * np.mean([r['accept'] for r in rs]):.0f}% | "
                     f"{pts(np.mean([r['regret'] for r in rs])) if all('regret' in r for r in rs) else 'n/a'} | "
                     f"{np.mean([r['questions'] for r in rs]):,.0f} | {np.mean([r['probe_q'] for r in rs]):,.0f} | "
                     f"{np.mean([r['gpu_s'] for r in rs]):.1f} | " + "; ".join(f"{r['state']}: {r['choice']}" for r in rs[:4]) + " |")
        best = [r for r in rows if r["selector"] == "random"]
        L += ["", "Best candidate per decision (oracle): " + ", ".join(f"{r['state']} {pts(r['best'])}" for r in best) + ".", ""]
    L += ["## Overall (unseen trajectories: mean over all leave-one-trajectory-out test decisions)", "",
          "| selector | mean worst | accept | mean regret | questions / decision | GPU s / decision | mean within-parent Spearman |", "|---|---|---|---|---|---|---|"]
    test = [r for r in R if r["fold"].startswith("T")]
    for s in dict.fromkeys(r["selector"] for r in test):
        rs = [r for r in test if r["selector"] == s]
        sp = [v for k, v in S["spearman"].items() if k.split("|")[0].startswith("T") and k.split("|")[1] == s]
        L.append(f"| {s} | {pts(np.mean([r['worst'] for r in rs]))} | {100 * np.mean([r['accept'] for r in rs]):.0f}% | "
                 f"{pts(np.mean([r['regret'] for r in rs]))} | {np.mean([r['questions'] for r in rs]):,.0f} | "
                 f"{np.mean([r['gpu_s'] for r in rs]):.1f} | {np.nanmean(sp) if sp else float('nan'):+.2f} |")
    b = S.get("bootstrap")
    if b:
        L += ["", f"### Measurement uncertainty (paired bootstrap over label questions, {b['n_boot']} draws, "
              f"{b['decisions']} unseen decisions; question sampling only, not trajectory-to-trajectory variation)", "",
              f"Median standard error of one candidate's per-category loss: label set {100 * b['median_se_category_loss_label']:.2f} "
              f"points, probe set {100 * b['median_se_category_loss_probe']:.2f} points. Median bootstrap SD of one "
              f"candidate's worst-category loss: {100 * b['median_sd_candidate_worst_label']:.2f} points.", "",
              "| comparison (A - B, mean worst-category loss) | difference | 95% interval | share of draws where A is lower |",
              "|---|---|---|---|"]
        for p in b["pairs"]:
            L.append(f"| {p['a']} - {p['b']} | {pts(p['diff'])} | [{pts(p['lo'])}, {pts(p['hi'])}] | {100 * p['p_a_better']:.0f}% |")
    o = S["offline"]
    L += ["", f"Offline cost (separate): {o['pairs']} candidate records ({o.get('distinct_state_actions', '?')} distinct "
          f"state-action scorings, {o['distinct_scored']} distinct masks), scoring GPU {o['scoring_gpu_seconds'] / 60:.1f} min, "
          f"profiling {o['profiling_gpu_seconds'] / 60:.1f} min, collection wall {o['collect_wall_seconds'] / 60:.1f} min "
          f"(includes model loading and gate replays); training + evaluation {o['training_and_evaluation_cpu_seconds']:.2f} s CPU.",
          "", "Ridge alpha per fold (chosen by leave-one-trajectory-out CV inside the training data): " + "; ".join(
              f"{f}: " + ", ".join(f"{s} {m['alpha']:g}" for s, m in v.items()) for f, v in S["folds"].items()), "",
          "## Trajectories", ""]
    for t, steps in S["trajectories"].items():
        L.append(f"* {t}: " + "; ".join(f"step {s['step']} parent {100 * s['parent_sparsity']:.2f} % ({s['accepted']}/{s['n']} candidates accepted)"
                                        for s in steps))
    return "\n".join(L) + "\n"
