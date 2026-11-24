"""Gradient-only top-1 selection in the live pruning-and-rollback loop (method kind ``ranked_search``).

One run = one selector x one seed, started from the dense checkpoint.  Per attempt t at parent P:

1. Profile P with the ranker experiment's protocol (fixed calibration subsets drawn from
   ``ranker.profile_seed``; activations AND gradients at every parent).  The profile is persisted before use.
2. Candidate set = ``ranker.candidates``: the greedy-risk set, lowest z_sens, lowest z_grad, depth windows and
   seeded random unit sets, every action = ``units_per_action`` x ``blocks_per_unit`` blocks (the matched step).
   Actions that are illegal at P (a unit without room under the per-unit cap, an increment outside the search
   bounds, a different step size) are dropped and logged.
3. Rank by the offline gradient-only rule (``ranker._rank('gradient')``): ascending mean z_grad of the action's
   units, ties broken by generator index.  The whole ordered shortlist is logged.
4. Build the top-ranked trial from P with the structured MLP-channel operator (``raise_unit``).
5. Gate it with the threshold-aware capability gate: paired loss vs. the ORIGINAL dense model per category,
   exact finite-pool bounds, the configured tolerance, one fresh question order per attempt, on the ranker's
   label set (its probe set stays unused).  Every batch the gate reveals is scored on a never-modified dense
   copy of the model and on the trial with the SAME rows and batch layout (bf16 batch composition alone flips
   ~1 % of answers, so a one-pass dense reference would add churn to the paired outcomes).
6. ACCEPT -> the trial becomes the next parent.  REJECT or INCONCLUSIVE -> P is restored bit-for-bit and the run
   ENDS (a candidate failure).  The cumulative question cap is enforced across attempts: a reveal that would
   exceed it stops the gate before anything is scored, P is restored and the run ends with a BUDGET stop (not a
   candidate failure).  The proposal cap ends the run after ``max_proposals`` attempts.

7. Fallback after a rank-1 REJECT/INCONCLUSIVE (``ranked_search.fallback_policy``):
   * ``topk`` (default, ``fallback_k``): the parent's next-ranked actions in gradient order;
   * ``kl_probe`` (screen-then-confirm protocol, amendment 2): the next attempt reloads the parent's profile and
     shortlist, scores ranks 2-11 and the dense copy on the probe items (``ranked_search.kl_probe``: the screening
     pool's probe split, disjoint from the gate's items and any sealed confirmation set), picks the action with
     ``signals.pick_action`` on KL (the matched-signal rule) and gates it. ACCEPT -> top-1 resumes at the new parent;
     REJECT/INCONCLUSIVE -> the run ends (``fallback_exhausted``). The probe's paired items count against the question
     cap (checked before it starts) and are reported separately.
   * ``ranked_search.counterfactual`` (with ``kl_probe``): at each probe, the KL, S1c and TV picks and gradient rank 2
     are scored on a hold-out slice carved out of the gate's items (never gated, never probed); recorded per rejection
     with costs, never used for a decision, not charged to the cap unless ``charge_to_cap``.

Crash safety: profiles, every paired gate batch and every probe / counterfactual batch are persisted before use under
deterministic keys, and the search state is checkpointed after every attempt, so an interrupted run replays to the
identical trajectory (an interrupted probe replays to the same choice).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .adaptive.evaluator import ACCEPT, INCONCLUSIVE, REJECT, domain_state, evaluate_candidate, question_order
from .adaptive.live import CapabilityGate, attempt_order_seed, domain_summary, weights_digest
from .adaptive.policies import make_policy
from .config import deep_merge
from .utils.faults import fault_point
from .utils.io import short_hash
from .utils.log import get_logger
from .utils.runtime import check_stop

log = get_logger("ranked")
BUDGET_STOP = "budget_stop"
STOP_KINDS = {"candidate_rejected": "candidate failure", "candidate_inconclusive": "candidate failure",
              "question_cap": "budget stop", "proposal_cap": "budget stop", "no_legal_candidate": "no legal action",
              "prefix_mismatch": "replay check failed", "fallback_exhausted": "candidate failure (fallback exhausted: top-k, or the KL pick)"}


class BudgetStop(Exception):
    """The next gate batch would exceed the run's cumulative question cap (raised BEFORE scoring it)."""


def _sync(dev) -> None:
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ============================================================================ paired scoring
class PairedScorer:
    """Scores the same rows on the dense model and on the trial model with one identical batch layout.

    Both use ``MMLUScorer`` with the gate's prompt (shots, max_len) and token budget.  If a CUDA OOM shrinks
    either layout, both sides are rescored with the smaller budget, so a pair is never split across layouts.
    """

    def __init__(self, dense_model, model, tok, fewshot, shots: int, max_len: int, max_batch_tokens: int, hb=None):
        from .eval.mmlu import MMLUScorer

        store: dict = {}
        self.d = MMLUScorer(dense_model, tok, fewshot, shots, max_len, store, "pair_dense", max_batch_tokens, hb)
        self.t = MMLUScorer(model, tok, fewshot, shots, max_len, store, "pair_trial", max_batch_tokens, hb)
        self.budget = int(max_batch_tokens)
        self.dev = next(model.parameters()).device
        self.evals = {"dense": 0, "trial": 0}  # item-model evaluations actually executed (incl. OOM re-runs)

    def _run(self, sc, rows, budget):
        sc.budget = budget
        self.evals["dense" if sc is self.d else "trial"] += len(rows)
        _sync(self.dev)
        t0 = time.perf_counter()
        res = sc.score(rows)
        _sync(self.dev)
        return {q: int(r["correct"]) for q, r in res.items()}, time.perf_counter() - t0, sc.budget

    def score(self, rows: list[dict], dense: dict | None = None) -> dict:
        """``dense``: a cached dense result for exactly these rows ({d, sd, budget}) or None."""
        while True:
            b = self.budget
            if dense is not None and int(dense["budget"]) == b:
                d, sd, bd, cached = dense["d"], 0.0, b, True
            else:
                (d, sd, bd), cached = self._run(self.d, rows, b), False
            t, st, bt = self._run(self.t, rows, b)
            if bd == b and bt == b:
                return {"d": d, "t": t, "sd": sd, "st": st, "budget": b, "dense_cached": cached}
            self.budget = min(bd, bt)
            log.warning("OOM changed the batch layout; rescoring the pair with token budget %d", self.budget)


class PairedLogprobScorer:
    """The four letters' log-probabilities of the dense copy and the trial on the same rows with one identical batch
    layout (``pool_signals.LogprobScorer``: MMLUScorer's prompt, sort and token-budget batching). If a CUDA OOM shrinks
    either layout, both sides are rescored with the smaller budget."""

    def __init__(self, dense_model, model, tok, fewshot, shots: int, max_len: int, max_batch_tokens: int):
        from .pool_signals import LogprobScorer

        self.d = LogprobScorer(dense_model, tok, fewshot, shots, max_len, max_batch_tokens)
        self.t = LogprobScorer(model, tok, fewshot, shots, max_len, max_batch_tokens)
        self.budget = int(max_batch_tokens)
        self.dev = next(model.parameters()).device
        self.evals = {"dense": 0, "trial": 0}

    def _run(self, sc, rows, budget):
        self.evals["dense" if sc is self.d else "trial"] += len(rows)
        _sync(self.dev)
        t0 = time.perf_counter()
        res, b = sc.score(rows, budget)
        _sync(self.dev)
        return res, time.perf_counter() - t0, b

    def score(self, rows: list[dict], dense: dict | None = None) -> dict:
        """``dense``: a cached dense result for exactly these rows ({items, budget}) or None."""
        while True:
            b = self.budget
            if dense is not None and int(dense["budget"]) == b:
                d, sd, bd, cached = dense["items"], 0.0, b, True
            else:
                (d, sd, bd), cached = self._run(self.d, rows, b), False
            t, st, bt = self._run(self.t, rows, b)
            if bd == b and bt == b:
                return {"d": d, "t": t, "sd": sd, "st": st, "budget": b, "dense_cached": cached}
            self.budget = min(bd, bt)
            log.warning("OOM changed the batch layout; rescoring the log-prob pair with token budget %d", self.budget)


class PairedSource:
    """Gate item source: a domain's next items in a fixed order, each requested batch scored as a dense/trial pair.

    Cache keys depend only on (order seed, domain, position, size) and the trial's mask hash, so the same batch of
    the same trial is never scored twice (crash replay), and dense answers are shared by every trial gated with the
    same question order (stall diagnostic).  ``cap_left`` = questions left under the run's cumulative cap.
    """

    def __init__(self, pool: dict[str, list[dict]], seed: int, trial_key: str, scorer: PairedScorer, cache,
                 cap_left: int | None = None, ro_caches=(), hb=None):
        self.pool_sizes = {d: len(r) for d, r in pool.items()}
        self._rows = {r["qid"]: r for rows in pool.values() for r in rows}
        self._order = {d: question_order([r["qid"] for r in rows], seed, d) for d, rows in pool.items()}
        self._cursor = {d: 0 for d in pool}
        self.seed, self.trial_key, self.scorer, self.cache = int(seed), trial_key, scorer, cache
        self.cap_left = None if cap_left is None else int(cap_left)
        self.ro = list(ro_caches)
        self.hb = hb
        self.used = 0                 # questions revealed to the gate (trial evaluations)
        self.items = {d: [] for d in pool}
        self.t_trial = 0.0            # measured trial scoring seconds (replayed from the record after a crash)
        self.t_dense = 0.0            # measured dense-reference scoring seconds
        self.dense_q = 0              # dense-reference question evaluations run for this trial
        self.replayed_q = 0           # questions whose pair came from the cache

    def _get(self, key):
        v = self.cache.get(key)
        for c in self.ro:
            if v is not None:
                break
            v = c.get(key)
        return v

    def reveal(self, domain: str, count: int) -> list[tuple[str, int, int]]:
        if self.cap_left is not None and self.used + count > self.cap_left:
            raise BudgetStop(f"{domain}: {self.used} + {count} questions > {self.cap_left} left under the cap")
        check_stop()
        c = self._cursor[domain]
        ids = self._order[domain][c : c + count]
        if len(ids) != count:
            raise IndexError(f"{domain}: only {len(self._order[domain]) - c} items left")
        bkey = f"{self.seed}|{domain}|{c}|{count}"
        pkey = f"pair|{self.trial_key}|{bkey}"
        rec = self._get(pkey)
        if rec is None:
            res = self.scorer.score([self._rows[i] for i in ids], self._get(f"dense|{bkey}"))
            rec = {"d": res["d"], "t": res["t"], "sd": res["sd"], "st": res["st"], "budget": res["budget"],
                   "dense_q": 0 if res["dense_cached"] else count}
            put = {pkey: rec}
            if not res["dense_cached"]:
                put[f"dense|{bkey}"] = {"d": res["d"], "sd": res["sd"], "budget": res["budget"]}
            self.cache.put_many(put)  # one transaction: pair and dense reference together
            fault_point("mid_ranked_gate")
        else:
            self.replayed_q += count
        self.t_trial += float(rec["st"])
        self.t_dense += float(rec["sd"])
        self.dense_q += int(rec.get("dense_q", 0))
        self._cursor[domain] = c + count
        self.used += count
        out = [(i, int(rec["d"][i]), int(rec["t"][i])) for i in ids]
        self.items[domain] += out
        if self.hb:
            self.hb.beat("gate", f"{domain} n={self._cursor[domain]} used={self.used}")
        return out


def partial_summary(plan, items: dict[str, list]) -> dict[str, dict]:
    """Per-category estimate and bounds at the last completed look (for a gate stopped by the question cap)."""
    out = {}
    for dp in plan.domains:
        its = items.get(dp.name, [])
        n = len(its)
        kp = sum(1 for _, d, t in its if d - t == 1)
        km = sum(1 for _, d, t in its if d - t == -1)
        sizes = [lk.n for lk in dp.looks]
        if n in sizes:
            s = domain_state(dp, n, kp, km, sizes.index(n))
            out[dp.name] = {"n": n, "est": round(s.estimate, 5), "lo": round(s.lo, 5), "hi": round(s.hi, 5),
                            "status": s.status, "broken": kp, "fixed": km}
        else:
            out[dp.name] = {"n": n, "est": round((kp - km) / n, 5) if n else None, "lo": None, "hi": None,
                            "status": "open", "broken": kp, "fixed": km}
    return out


def sealed_qids(rc: dict) -> set[str]:
    """``ranked_search.sealed_confirmation``: files listing the sealed confirmation items (a JSON list of qids, a JSON
    object with ``qids``, or JSONL rows with ``qid``). Screening must never read them; only their ids are loaded."""
    import json
    from pathlib import Path

    from .config import resolve_path

    out: set[str] = set()
    for p in (rc.get("ranked_search") or {}).get("sealed_confirmation") or []:
        path = Path(resolve_path(rc, p))
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":  # split on newlines only: question text may hold U+2028 and other line separators
            out |= {json.loads(line)["qid"] for line in text.split(chr(10)) if line.strip()}
        else:
            v = json.loads(text)
            out |= set(v["qids"] if isinstance(v, dict) else v)
    return out


def screening_split(rc: dict, data) -> dict:
    """The pruning-time (screening) pool in disjoint parts:

    * ``gate``    the capability gate's items: the ranker's label set (default) or the whole pool;
    * ``probe``   the ranker's probe split (``ranker.probe_per_domain`` per category, ``ranker.probe_seed``), used by
                  the KL probe after a rank-1 rejection (protocol amendment 2);
    * ``holdout`` (``ranked_search.counterfactual.enabled``) ``holdout_per_domain`` items per category carved out of
                  the label set by seed for the counterfactual log; the gate and the probe never see them.

    Every part is checked against the sealed confirmation set (``sealed_qids``); any overlap stops the run."""
    from .adaptive.evaluator import question_order
    from .ranker import split_pool

    g = rc["gate"]["capability"]
    field, domains = g.get("domain_field", "category"), list(g["domains"])
    rk, rs = rc["ranker"], rc.get("ranked_search") or {}
    kind = rs.get("gate_pool", "label")
    if kind == "full":
        G, P, info = list(data.mmlu_probe), [], {"kind": "full", "probe_excluded": 0}
    elif kind == "label":
        P, G = split_pool(data.mmlu_probe, field, domains, int(rk["probe_per_domain"]), int(rk["probe_seed"]))
        info = {"kind": "label", "probe_excluded": len(P)}
    else:
        raise ValueError(f"ranked_search.gate_pool must be label|full, got {kind!r}")
    H: list[dict] = []
    cf = rs.get("counterfactual") or {}
    if cf.get("enabled"):
        if kind != "label":
            raise ValueError("the counterfactual hold-out needs ranked_search.gate_pool: label")
        n, seed = int(cf["holdout_per_domain"]), int(cf.get("seed", 9091))
        hold: set[str] = set()
        for d in domains:
            ids = [r["qid"] for r in G if r[field] == d]
            if n >= len(ids):
                raise ValueError(f"counterfactual hold-out {n} >= the {len(ids)} gate items of {d}")
            hold |= set(question_order(ids, seed, f"holdout:{d}")[:n])
        H, G = [r for r in G if r["qid"] in hold], [r for r in G if r["qid"] not in hold]
        info["holdout"] = len(H)
    sealed = sealed_qids(rc)
    if sealed:
        overlap = {name: len(sealed & {r["qid"] for r in part}) for name, part in (("gate", G), ("probe", P), ("holdout", H))}
        if any(overlap.values()):
            raise RuntimeError(f"screening items overlap the sealed confirmation set: {overlap}")
        info["sealed_confirmation_qids"] = len(sealed)
    return {"gate": G, "probe": P, "holdout": H, "info": info, "field": field, "domains": domains}


def gate_pool(rc: dict, data) -> tuple[list[dict], dict]:
    """The gate's question pool: the ranker's label set (default) or the whole pruning-time pool, minus the
    counterfactual hold-out when that is enabled."""
    s = screening_split(rc, data)
    return s["gate"], s["info"]


def gate_margin(rc: dict) -> float:
    return float((rc.get("ranked_search") or {}).get("gate_margin", 0.0) or 0.0)


def make_gate(rc: dict, pool: list[dict], margin: float | None = None) -> CapabilityGate:
    """The locked threshold-aware gate on ``pool``; budget = min(configured budget, pool size).

    ``margin`` (default ``ranked_search.gate_margin``, 0) lowers the PRUNING-TIME tolerance of every category to
    ``gate.capability.tolerance - margin``; the configured tolerance itself stays the audit target."""
    g = rc["gate"]["capability"]
    b = min(int(g["budget"]), len(pool))
    m = gate_margin(rc) if margin is None else float(margin)
    tol = g["tolerance"]
    eff = {d: float(v) - m for d, v in tol.items()} if isinstance(tol, dict) else float(tol) - m
    lowest = min(eff.values()) if isinstance(eff, dict) else eff
    if lowest <= 0:
        raise ValueError(f"gate margin {m} leaves no tolerance ({tol})")
    return CapabilityGate(deep_merge(rc, {"gate": {"capability": {"budget": b, "tolerance": eff}}}), pool)


# ============================================================================ candidates
def legal_shortlist(rc: dict, geo: dict, profs: list[dict], parent_sparsity: float, n_layers: int, step: int):
    """Generate the ranker's candidate set, drop illegal actions, rank by the gradient-only rule."""
    from .ranker import action_features, candidates

    rk, rs = rc["ranker"], rc.get("ranked_search") or {}
    sl = rk["shortlist"]
    rng = np.random.default_rng([int(rk["traj_seed"]), 7, int(rs.get("candidate_stream", 0)), int(step)])
    raw = candidates(profs, geo, sl, n_layers, int(rk.get("random_actions", 2)), rng)
    by = {p["name"]: p for p in profs}
    lo, hi = float(rc["search"]["min_increment"]), float(rc["search"]["max_increment"])
    want = int(sl["units_per_action"]) * int(sl["blocks_per_unit"])
    legal, illegal = [], []
    for c in raw:
        why = []
        for u, b in c["units"].items():
            cur = int(round(by[u]["sparsity"] * geo["B"]))
            if cur + b > geo["cap_blocks"]:
                why.append(f"{u}: {cur}+{b} blocks > cap {geo['cap_blocks']}")
            if not lo - 1e-9 <= b / geo["B"] <= hi + 1e-9:
                why.append(f"{u}: increment {b / geo['B']:.4f} outside [{lo}, {hi}]")
        if sum(c["units"].values()) != want:
            why.append(f"{sum(c['units'].values())} blocks != matched step {want}")
        (illegal if why else legal).append({**c, "why": why} if why else dict(c))
    for i, c in enumerate(legal):
        c["idx"] = i
        c["features"] = action_features(profs, c["units"], parent_sparsity, n_layers, geo["B"])
    order = rank_gradient(legal)
    return [legal[i] for i in order], illegal


def rank_gradient(cands: list[dict]) -> list[int]:
    """Indices of ``cands`` in the offline gradient-only order: ascending mean z_grad, ties by generator index
    (the key of ``ranker._rank('gradient')``)."""
    return sorted(range(len(cands)), key=lambda i: (round(cands[i]["features"]["z_g_mean"], 12), i))


def shortlist_record(ranked: list[dict], mult: int) -> list[dict]:
    from .ranker import canon

    return [{"rank": r + 1, "idx": c["idx"], "kind": c["kind"], "action": c["units"], "action_key": canon(c["units"]),
             "layers": sorted(int(u.split(".")[0][1:]) for u in c["units"]),
             "z_g_mean": round(float(c["features"]["z_g_mean"]), 6), "z_s_mean": round(float(c["features"]["z_s_mean"]), 6),
             "risk_mean": round(float(c["features"]["risk_mean"]), 6), "channels": sum(c["units"].values()) * mult}
            for r, c in enumerate(ranked)]


def profile_parent(model, state, calib, rc: dict, batch_store: dict, hb=None):
    """The ranker experiment's profiling: fixed activation/gradient subsets from ``ranker.profile_seed``."""
    from .pruning.profile import collect_act_sqnorm, collect_grad_importance, sensitivity

    pp = rc["profiling"]
    prng = np.random.default_rng([int(rc["ranker"].get("profile_seed", 0)), 1, 17])  # = ranker.collect
    a_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["activation_samples"]), calib.shape[0]), replace=False))
    g_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["gradient_samples"]), calib.shape[0]), replace=False))
    act = collect_act_sqnorm(model, state, calib[a_idx], batch_store, hb, int(pp.get("activation_batch_size", 4)))
    sens = sensitivity(state, act, float(rc["pruning"].get("sensitivity_percentile", 10)))
    grad = collect_grad_importance(model, state, calib[g_idx], hb)
    return act, sens, grad


def unit_profiles(state, sens: dict, grad: dict) -> list[dict]:
    from .pruning.profile import zscores

    zs, zg = zscores(sens), zscores(grad)
    return [{"name": u.name, "share": state.share(u.name), "z_s": zs[u.name], "z_g": zg[u.name],
             "sparsity": state.unit_sparsity(u.name)} for u in state.units]


def build_trial(state, dense_store, action: dict, act: dict, rc: dict, geo: dict):
    """Raise every unit of ``action`` by its blocks from the current (parent) masks; returns (changed units,
    actual channels removed per unit)."""
    from .pruning.wanda import raise_unit

    before = {u: int((~state.masks[u]).sum().item()) for u in action}
    changed = []
    for u, b in action.items():
        if raise_unit(state, u, (before[u] // geo["mult"] + b) / geo["B"], act, rc):
            changed.append(u)
    actual = {u: int((~state.masks[u]).sum().item()) - before[u] for u in action}
    if changed:
        state.apply(dense_store, only=changed)
    return changed, actual


def load_dense_copy(rc: dict, model, units, dense_store):
    """A second, never-modified copy of the original checkpoint, verified bit-identical to the dense weights."""
    from .model import build_units, load_model

    dm = load_model(rc)
    dm.eval()
    for p in dm.parameters():
        p.requires_grad_(False)
    dunits = {u.name: u for u in build_units(dm, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))}
    prunable = set()
    with torch.no_grad():
        for u in units:
            for part, lin in dunits[u.name].linears.items():
                prunable.add(id(lin.weight))
                if not torch.equal(lin.weight, dense_store.get(u.name, part)):
                    raise RuntimeError(f"dense copy differs from the dense store at {u.name}.{part}")
        work = dict(model.named_parameters())
        for n, p in dm.named_parameters():
            if id(p) not in prunable and not torch.equal(p, work[n]):
                raise RuntimeError(f"dense copy differs from the working model at {n}")
    return dm


def _test_forced_reject(t: int, rank: int, decision: str) -> bool:
    """CAPPRUNE_TEST_FORCE_REJECT="2,5": a rank-1 ACCEPT at those attempts is treated as REJECT (tests only)."""
    import os

    spec = os.environ.get("CAPPRUNE_TEST_FORCE_REJECT")
    return bool(spec) and rank == 1 and decision == ACCEPT and str(t) in {x.strip() for x in spec.split(",")}


# ============================================================================ replay prefix
def load_prefix(rc: dict) -> dict | None:
    """``ranked_search.verify_prefix``: an earlier run whose attempts this run must reproduce exactly before it
    continues (a replay from dense, used instead of branching a finished run directory).  Read-only."""
    vp = (rc.get("ranked_search") or {}).get("verify_prefix")
    if not vp:
        return None
    import json
    from pathlib import Path

    from .config import resolve_path
    from .utils.io import read_json

    jp = resolve_path(rc, vp["journal"])
    last = {}
    for line in Path(jp).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    att = [last[k] for k in sorted(last)]
    fr = read_json(resolve_path(rc, vp["frozen"]))
    if len(att) != int(vp["attempts"]) or fr["state_id"] != vp["state_id"] or fr["mask_sha256"] != vp["mask_sha256"]:
        raise RuntimeError(f"verify_prefix source does not match its recorded provenance ({len(att)} attempts, "
                           f"{fr['state_id']}, {fr['mask_sha256'][:12]})")
    return {"attempts": {a["iteration"]: a for a in att}, "n": len(att), "state_id": fr["state_id"],
            "mask_sha256": fr["mask_sha256"], "mask_file": fr["mask_file"], "journal": str(jp)}


def prefix_view(a: dict) -> dict:
    """The parts of an attempt record that a replay must reproduce exactly."""
    return {"parent_mask_hash": a["parent_mask_hash"], "shortlist": [s["action_key"] for s in a["shortlist"]],
            "chosen": a["chosen"]["action_key"], "cand_id": a.get("cand_id"), "trial_mask_hash": a.get("trial_mask_hash"),
            "decision": a.get("gate_decision"), "questions": (a.get("gate") or {}).get("questions"),
            "order_seed": (a.get("gate") or {}).get("order_seed"),
            "domains": {d: [v["n"], v["broken"], v["fixed"], v["status"]] for d, v in ((a.get("gate") or {}).get("domains") or {}).items()}}


# ============================================================================ the live loop
def initial_state() -> dict:
    return {"iteration": 1, "state_id": None, "rho": 0.0, "accepts": 0, "rollbacks": 0, "stop_reason": None,
            "questions": 0, "dense_reference_questions": 0, "parent_digest": None, "budget_stop": None,
            "gate_counts": {k: 0 for k in ("accept", "reject", "inconclusive", BUDGET_STOP)},
            "budget": {"probe_questions": 0, "probe_tests": 0, "profile_seconds": 0.0, "build_seconds": 0.0,
                       "gate_seconds": 0.0, "gate_scoring_seconds": 0.0, "gate_dense_reference_seconds": 0.0,
                       "rollback_seconds": 0.0, "search_wall_seconds": 0.0, "controller_seconds": 0.0},
            "history": []}


class RankedSearch:
    def __init__(self, ctx):
        from .fixed_state import _geometry

        self.ctx, self.rc, self.S = ctx, ctx.rc, ctx.S["search"]
        rs = self.rc["ranked_search"]
        if rs.get("selector", "gradient") != "gradient":
            raise ValueError(f"ranked_search.selector {rs.get('selector')!r}: only 'gradient' is implemented")
        self.max_props = int(rs["max_proposals"])
        self.cap = None if rs.get("question_cap") in (None, "null") else int(rs["question_cap"])
        self.geo = _geometry(self.rc, ctx.units[0].channels)
        self.pool, self.pool_info = gate_pool(self.rc, ctx.data)
        self.gate = make_gate(self.rc, self.pool)
        self.min_total = sum(dp.looks[0].n for dp in self.gate.plan.domains)
        self.dev = next(ctx.model.parameters()).device
        t = self.rc["testing"]
        self.dense_model = load_dense_copy(self.rc, ctx.model, ctx.units, ctx.dense)
        self.scorer = PairedScorer(self.dense_model, ctx.model, ctx.tok, ctx.data.mmlu_fewshot, int(t.get("shots", 0)),
                                   int(self.rc["model"].get("max_seq_len", 2048)), int(t.get("max_batch_tokens", 32768)),
                                   ctx.hb)
        # parameters per MLP channel (gate row + up row + down column, + biases): physical removal per channel
        self.ch_params = {u.name: sum(lin.weight.shape[1] if p != "down" else lin.weight.shape[0] for p, lin in u.linears.items())
                          + sum(1 for p, lin in u.linears.items() if p != "down" and lin.bias is not None) for u in ctx.units}
        self.prefix = load_prefix(self.rc)
        self.fallback_k = int(rs.get("fallback_k", 1))  # candidates tried per parent (1 = top-1 only)
        if self.fallback_k < 1:
            raise ValueError("ranked_search.fallback_k must be >= 1")
        # ---- protocol amendment 2: after a rank-1 rejection, a KL probe over ranks 2-11 picks the fallback
        self.fallback_policy = rs.get("fallback_policy", "topk")
        if self.fallback_policy not in ("topk", "kl_probe"):
            raise ValueError(f"ranked_search.fallback_policy must be topk|kl_probe, got {self.fallback_policy!r}")
        self.split = screening_split(self.rc, ctx.data)
        self.kl = self.cf = None
        if self.fallback_policy == "kl_probe":
            if self.fallback_k != 1:
                raise ValueError("ranked_search.fallback_k must be 1 with fallback_policy: kl_probe")
            kp = rs.get("kl_probe") or {}
            self.kl = {"per_domain": int(kp.get("per_domain", 100)), "ranks": [int(x) for x in kp.get("ranks", [2, 11])],
                       "batch_size": int(kp.get("batch_size", 100)), "order_seed": int(kp.get("order_seed", 2026))}
            self.probe_rows, self.probe_sha = self._item_set(self.split["probe"], self.kl["per_domain"], self.kl["order_seed"])
            self.n_probe = sum(len(v) for v in self.probe_rows.values())
            self.lp_scorer = PairedLogprobScorer(self.dense_model, ctx.model, ctx.tok, ctx.data.mmlu_fewshot,
                                                 int(t.get("shots", 0)), int(self.rc["model"].get("max_seq_len", 2048)),
                                                 int(t.get("max_batch_tokens", 32768)))
        cf = rs.get("counterfactual") or {}
        if cf.get("enabled"):
            if self.kl is None:
                raise ValueError("ranked_search.counterfactual needs fallback_policy: kl_probe")
            self.cf = {"batch_size": int(cf.get("batch_size", 100)), "charge_to_cap": bool(cf.get("charge_to_cap", False)),
                       "seed": int(cf.get("seed", 9091))}
            self.cf_rows, self.cf_sha = self._item_set(self.split["holdout"], None, self.cf["seed"])
        if self.kl is not None:
            self._write_split_record()

    # ---------------------------------------------------------------- amendment 2: KL probe and counterfactual log
    def _item_set(self, rows: list[dict], per_domain: int | None, seed: int) -> tuple[dict[str, list[dict]], str]:
        """Rows by category in a seeded order (the first ``per_domain`` of each category if given) and a hash of the
        resulting id lists."""
        import hashlib
        import json

        field = self.split["field"]
        out = {}
        for d in self.split["domains"]:
            by = {r["qid"]: r for r in rows if r[field] == d}
            ids = question_order(sorted(by), seed, d)
            if per_domain is not None:
                if per_domain > len(ids):
                    raise ValueError(f"{d}: {per_domain} probe items requested, {len(ids)} in the probe split")
                ids = ids[:per_domain]
            out[d] = [by[q] for q in ids]
        sha = hashlib.sha256(json.dumps({d: [r["qid"] for r in v] for d, v in out.items()}, sort_keys=True).encode())
        return out, sha.hexdigest()[:16]

    def _write_split_record(self) -> None:
        from .utils.io import atomic_write_json

        p = self.ctx.run_dir / "screening_split.json"
        rec = {"gate_qids": sorted(r["qid"] for r in self.split["gate"]), "info": self.split["info"],
               "probe": {d: [r["qid"] for r in v] for d, v in self.probe_rows.items()}, "probe_set_sha16": self.probe_sha,
               "holdout": {d: [r["qid"] for r in v] for d, v in self.cf_rows.items()} if self.cf else None,
               "holdout_set_sha16": self.cf_sha if self.cf else None,
               "sealed_confirmation": (self.rc.get("ranked_search") or {}).get("sealed_confirmation") or []}
        if not p.exists():
            atomic_write_json(p, rec)

    def _score_set(self, tag: str, trial_hash: str, rows_by_cat: dict[str, list[dict]], set_sha: str, batch: int):
        """Score the candidate currently applied to the model and the dense copy on a fixed item set. Every batch is
        persisted (dense and candidate answers in one record) BEFORE it is used, under keys that depend only on the
        candidate's mask, the item set and the batch, so an interrupted probe replays to the same answers."""
        cache = self.ctx.eval_cache
        out, st = {}, {"items": 0, "scored": 0, "replayed": 0, "dense_scored": 0, "seconds": 0.0, "dense_seconds": 0.0}
        for cat in sorted(rows_by_cat):
            rows = rows_by_cat[cat]
            for i in range(0, len(rows), batch):
                b_rows = rows[i : i + batch]
                bkey = f"{set_sha}|{cat}|{i}|{len(b_rows)}"
                pk, dk = f"{tag}|{trial_hash}|{bkey}", f"{tag}_dense|{bkey}"
                rec = cache.get(pk)
                if rec is None:
                    check_stop()
                    res = self.lp_scorer.score(b_rows, cache.get(dk))
                    rec = {"d": {q: v["logp"] for q, v in res["d"].items()}, "dc": {q: v["correct"] for q, v in res["d"].items()},
                           "t": {q: v["logp"] for q, v in res["t"].items()}, "tc": {q: v["correct"] for q, v in res["t"].items()},
                           "budget": res["budget"], "st": res["st"], "sd": res["sd"]}
                    put = {pk: rec}
                    if not res["dense_cached"]:
                        put[dk] = {"items": res["d"], "budget": res["budget"]}
                        st["dense_scored"] += len(b_rows)
                    cache.put_many(put)  # one transaction: the pair (with its dense side) and the dense reference
                    fault_point(f"mid_{tag}")
                    st["scored"] += len(b_rows)
                    st["seconds"] += float(res["st"])
                    st["dense_seconds"] += float(res["sd"])
                else:
                    st["replayed"] += len(b_rows)
                st["items"] += len(b_rows)
                for r in b_rows:
                    q = r["qid"]
                    out[q] = (rec["d"][q], rec["t"][q], int(rec["dc"][q]), int(rec["tc"][q]))
        return out, st

    def _score_candidates(self, ranks: list[int], ranked: list[dict], act, tag: str, rows_by_cat, set_sha: str, batch: int):
        """Build each ranked action on the parent, score it on the item set, restore the parent (verified at the end)."""
        from .ranker import canon, mask_hash

        ctx = self.ctx
        digest = weights_digest(ctx.state)
        out = []
        tot = {"items": 0, "scored": 0, "replayed": 0, "dense_scored": 0, "seconds": 0.0, "dense_seconds": 0.0}
        for r in ranks:
            c = ranked[r - 1]
            parent_masks = dict(ctx.state.masks)
            changed, _ = build_trial(ctx.state, ctx.dense, c["units"], act, self.rc, self.geo)
            h = mask_hash(ctx.state)
            res, st = self._score_set(tag, h, rows_by_cat, set_sha, batch)
            for n in changed:
                ctx.state.masks[n] = parent_masks[n]
            if changed:
                ctx.state.apply(ctx.dense, only=changed)
            for k in tot:
                tot[k] += st[k]
            out.append({"rank": r, "kind": c["kind"], "action_key": canon(c["units"]), "mask_hash": h, "res": res})
        if weights_digest(ctx.state) != digest:
            raise RuntimeError("the parent was not restored exactly after scoring the probe candidates")
        return out, tot

    @staticmethod
    def _category_signals(res: dict, rows_by_cat: dict[str, list[dict]]) -> dict:
        from .signals import item_terms, signal_means

        out = {}
        for cat, rows in rows_by_cat.items():
            q = [r["qid"] for r in rows]
            t = item_terms(np.array([res[x][0] for x in q]), np.array([res[x][1] for x in q]),
                           np.array([int(r["answer"]) for r in rows]), np.array([res[x][2] for x in q]),
                           np.array([res[x][3] for x in q]))
            out[cat] = signal_means(t, np.ones(len(q), bool))
        return out

    def _kl_probe(self, ranks: list[int], ranked: list[dict], act) -> dict:
        """Score ranks 2-11 on the probe items and pick by ``signals.pick_action`` on KL (the matched-signals rule);
        the S1c, TV and net-flip picks are recorded too (secondary, not used)."""
        from .signals import pick_action, worst_category

        cands, st = self._score_candidates(ranks, ranked, act, "kl_probe", self.probe_rows, self.probe_sha,
                                           self.kl["batch_size"])
        vals = [self._category_signals(c["res"], self.probe_rows) for c in cands]
        picks = {s: ranks[pick_action(vals, ranks, s)] for s in ("S2", "S1c", "TV", "S0")}
        return {"ranks": ranks, "choice_rank": picks["S2"], "signal": "S2 (KL over A-D)",
                "selection_rule": "capprune.signals.pick_action", "probe_set_sha16": self.probe_sha,
                "items_per_category": {d: len(v) for d, v in self.probe_rows.items()}, "picks": picks,
                "candidates": [{"rank": c["rank"], "kind": c["kind"], "action_key": c["action_key"], "mask_hash": c["mask_hash"],
                                "worst": {s: round(worst_category(v, s), 6) for s in ("S0", "S2", "S1c", "TV")},
                                "by_category": {d: {s: round(x, 6) for s, x in sv.items()} for d, sv in v.items()}}
                               for c, v in zip(cands, vals)],
                "cost": {"paired_items_charged": len(ranks) * self.n_probe, **st}}

    def _counterfactual(self, ranked: list[dict], act, probe: dict) -> dict:
        """The KL, S1c and TV picks and gradient rank 2, each scored on the held-out slice that neither the gate nor the
        probe uses (net flip loss per category vs. the dense copy). Recorded only; never used for a decision."""
        picks = {"KL": probe["picks"]["S2"], "S1c": probe["picks"]["S1c"], "TV": probe["picks"]["TV"], "gradient rank 2": 2}
        ranks = sorted(set(picks.values()))
        cands, st = self._score_candidates(ranks, ranked, act, "counterfactual", self.cf_rows, self.cf_sha,
                                           self.cf["batch_size"])
        by_rank = {}
        for c in cands:
            per = {d: float(np.mean([c["res"][r["qid"]][2] - c["res"][r["qid"]][3] for r in rows]))
                   for d, rows in self.cf_rows.items()}
            by_rank[str(c["rank"])] = {"kind": c["kind"], "action_key": c["action_key"], "mask_hash": c["mask_hash"],
                                       "holdout_net_flip": {d: round(v, 6) for d, v in per.items()},
                                       "worst": round(max(per.values()), 6)}
        n = sum(len(v) for v in self.cf_rows.values())
        return {"picks": picks, "by_rank": by_rank, "holdout_set_sha16": self.cf_sha,
                "items_per_category": {d: len(v) for d, v in self.cf_rows.items()},
                "charged_to_cap": self.cf["charge_to_cap"], "cost": {"paired_items": len(ranks) * n, **st}}

    def _fb(self) -> dict:
        return self.S.setdefault("fallback", {"k": self.fallback_k, "triggers": 0, "rescued": 0, "failed": 0,
                                              "blocked_by_cap": 0, "extra_questions": 0, "extra_evaluations": 0,
                                              "extra_seconds": 0.0})

    def params_removed(self) -> int:
        st = self.ctx.state
        return int(sum(int((~st.masks[u.name]).sum().item()) * self.ch_params[u.name] for u in st.units))

    def _journal(self, rec: dict) -> None:
        from .utils.io import append_jsonl

        append_jsonl(self.ctx.run_dir / "journal.jsonl", {"ts": time.time(), **rec})

    def _clock(self) -> float:
        _sync(self.dev)
        return time.perf_counter()

    def _lap(self, t0: float) -> float:
        _sync(self.dev)
        return time.perf_counter() - t0

    def _done(self) -> str | None:
        S = self.S
        if S["stop_reason"]:
            return S["stop_reason"]
        capped = S["iteration"] > self.max_props or (self.cap is not None and self.cap - S["questions"] < self.min_total)
        if capped and S.get("pending"):  # a queued fallback candidate could not be gated
            self._fb()["blocked_by_cap"] += 1
            S["pending"] = None
        if S["iteration"] > self.max_props:
            return "proposal_cap"
        if self.cap is not None and self.cap - S["questions"] < self.min_total:
            S["budget_stop"] = {"when": "before_attempt", "attempt": S["iteration"], "left": self.cap - S["questions"],
                                "needed_for_initial_sample": self.min_total}
            return "question_cap"
        return None

    def _profile(self, t: int):
        from .pruning.profile import load_profile, save_profile

        path = self.ctx.run_dir / "profiles" / f"iter-{t:06d}.safetensors"
        if path.exists():
            try:
                return load_profile(path, self.dev)
            except Exception as e:
                log.warning("profile for attempt %d unreadable (%s); recomputing", t, e)
        self.ctx.hb.beat("profile", f"attempt {t}")
        act, sens, grad = profile_parent(self.ctx.model, self.ctx.state, self.ctx.calib, self.rc, self.ctx.S["batch"],
                                         self.ctx.hb)
        save_profile(path, act, sens, grad)  # persist BEFORE use -> identical replay after a crash
        return act, sens, grad

    def run(self) -> str:
        ctx, S = self.ctx, self.S
        while True:
            reason = self._done()
            if reason:
                S["stop_reason"] = reason
                ctx.checkpoint()
                log.info("ranked search finished: %s (%s; MLP sparsity %.4f, %d accepts, %d questions)", reason,
                         STOP_KINDS.get(reason, "?"), S["rho"], S["accepts"], S["questions"])
                return reason
            check_stop()
            self._attempt(S["iteration"])
            ctx.checkpoint()

    def _attempt(self, t: int) -> None:
        from .ranker import canon, mask_hash

        ctx, S, rc, geo = self.ctx, self.S, self.rc, self.geo
        t_att = time.perf_counter()
        tim: dict[str, float] = {}
        ctx.hb.beat("search", f"attempt {t}")
        parent_id = S["state_id"] or "dense"
        dg = weights_digest(ctx.state)
        if S.get("parent_digest") is None:
            S["parent_digest"] = dg
        parent_ok = dg == S["parent_digest"]
        if not parent_ok:
            log.error("attempt %d: weights of parent %s differ from the accepted checkpoint", t, parent_id)
        rho_prev = ctx.state.global_sparsity()
        parent_hash = mask_hash(ctx.state)
        pending = S.get("pending")  # a fallback: the next-ranked action of the SAME parent, profile and shortlist
        kl_pending = bool(pending) and pending.get("mode") == "kl_probe"  # amendment 2: the probe picks the rank
        rank = int(pending["rank"]) if pending and not kl_pending else 1
        prof_t = int(pending["profile_attempt"]) if pending else t
        if pending and pending["parent_mask_hash"] != parent_hash:
            raise RuntimeError(f"attempt {t}: fallback parent {pending['parent_mask_hash']} != restored parent {parent_hash}")
        t0 = self._clock()
        act, sens, grad = self._profile(prof_t)  # a fallback reloads its parent's persisted profile (no new profiling)
        tim["profile_s"] = self._lap(t0)
        fault_point("ranked_after_profile")
        profs = unit_profiles(ctx.state, sens, grad)
        ranked, illegal = legal_shortlist(rc, geo, profs, rho_prev, len(ctx.units), S["accepts"])
        short = shortlist_record(ranked, geo["mult"])
        if pending and [s["action_key"] for s in short] != pending["shortlist"]:
            raise RuntimeError(f"attempt {t}: fallback shortlist differs from the parent's ranked shortlist")
        rec = {"type": "iteration", "iteration": t, "parent_id": parent_id, "parent_mask_hash": parent_hash,
               "parent_sparsity": round(rho_prev, 6), "rho_before": round(rho_prev, 6), "selector": "gradient",
               "candidate_step": S["accepts"], "shortlist": short, "illegal": illegal, "rank": rank,
               "profile_attempt": prof_t, "fallback_of": pending["trigger_attempt"] if pending else None}
        if not ranked:
            S["stop_reason"] = "no_legal_candidate"
            S["iteration"] = t + 1
            self._journal({**rec, "outcome": "stop", "gate_decision": "none", "stop_reason": "no_legal_candidate"})
            return
        probe_rec = cf_rec = None
        if kl_pending:
            lo, hi = self.kl["ranks"]
            ranks = list(range(max(lo, 2), min(hi, len(ranked)) + 1))
            charge = len(ranks) * self.n_probe
            left = None if self.cap is None else self.cap - S["questions"]
            if not ranks or (left is not None and charge > left):
                S["pending"] = None
                if ranks:  # the probe's items count against the question cap: a budget stop before anything is scored
                    self._fb()["blocked_by_cap"] += 1
                    S["stop_reason"] = "question_cap"
                    S["budget_stop"] = {"when": "before_kl_probe", "attempt": t, "left": left, "needed": charge}
                else:
                    S["stop_reason"] = "fallback_exhausted"
                S["iteration"] = t + 1
                self._journal({**rec, "rank": None, "outcome": BUDGET_STOP if ranks else "stop", "gate_decision": "none",
                               "stop_reason": S["stop_reason"], "kl_probe": {"blocked": S.get("budget_stop") or "no ranks"}})
                return
            t0 = self._clock()
            probe_rec = self._kl_probe(ranks, ranked, act)
            tim["kl_probe_s"] = self._lap(t0)
            S["questions"] += charge  # charged to the arm's question cap; reported separately below
            kp = S.setdefault("kl_probe", {"probes": 0, "paired_items_charged": 0, "dense_items_scored": 0,
                                           "items_scored": 0, "items_replayed": 0, "seconds": 0.0, "choices": []})
            kp["probes"] += 1
            kp["paired_items_charged"] += charge
            kp["dense_items_scored"] += probe_rec["cost"]["dense_scored"]
            kp["items_scored"] += probe_rec["cost"]["scored"]
            kp["items_replayed"] += probe_rec["cost"]["replayed"]
            kp["seconds"] += tim["kl_probe_s"]
            kp["choices"].append({"attempt": t, "trigger_attempt": pending["trigger_attempt"], "rank": probe_rec["choice_rank"]})
            rank = probe_rec["choice_rank"]
            rec["rank"] = rank
            fault_point("ranked_after_kl_probe")
            if self.cf is not None:
                t0 = self._clock()
                cf_rec = self._counterfactual(ranked, act, probe_rec)
                tim["counterfactual_s"] = self._lap(t0)
                cs = S.setdefault("counterfactual", {"logs": 0, "paired_items": 0, "items_scored": 0, "dense_items_scored": 0,
                                                     "seconds": 0.0, "charged_to_cap": self.cf["charge_to_cap"]})
                cs["logs"] += 1
                cs["paired_items"] += cf_rec["cost"]["paired_items"]
                cs["items_scored"] += cf_rec["cost"]["scored"]
                cs["dense_items_scored"] += cf_rec["cost"]["dense_scored"]
                cs["seconds"] += tim["counterfactual_s"]
                if self.cf["charge_to_cap"]:
                    S["questions"] += cf_rec["cost"]["paired_items"]
        top = ranked[rank - 1]
        action = top["units"]
        want = self.prefix["attempts"].get(t) if self.prefix else None
        if want is not None:  # replay of an earlier run: the same parent, shortlist and choice before anything is gated
            pv = prefix_view(want)
            diffs = [k for k, v in (("parent_mask_hash", parent_hash), ("shortlist", [s["action_key"] for s in short]),
                                    ("chosen", canon(action))) if pv[k] != v]
            if diffs:
                S["stop_reason"] = "prefix_mismatch"
                S["prefix"] = {"n": self.prefix["n"], "verified_through": t - 1, "ok": False, "attempt": t, "diffs": diffs}
                S["iteration"] = t + 1
                self._journal({**rec, "outcome": "stop", "gate_decision": "none", "stop_reason": "prefix_mismatch",
                               "prefix_check": S["prefix"]})
                log.error("attempt %d does not reproduce the prefix run (%s); stopping before gating", t, diffs)
                return
        # ---- trial = parent + the top-ranked action
        t0 = self._clock()
        parent_masks = dict(ctx.state.masks)  # raise_unit replaces a unit's mask tensor, never mutates it
        parent_fp = ctx.state.fingerprint()
        changed, actual = build_trial(ctx.state, ctx.dense, action, act, rc, geo)
        tim["build_s"] = self._lap(t0)
        cand_id = short_hash(S["state_id"], t, canon(action))
        trial_hash = mask_hash(ctx.state)
        rho_c = ctx.state.global_sparsity()
        trial_params = self.params_removed()
        fault_point("ranked_after_build")
        ev0 = dict(self.scorer.evals)
        # ---- gate (cumulative question cap enforced inside the item source)
        seed = attempt_order_seed(self.gate.base_seed, int(rc["run"]["seed"]), t)
        cap_left = None if self.cap is None else self.cap - S["questions"]
        src = PairedSource(self.gate.pool, seed, trial_hash, self.scorer, ctx.eval_cache, cap_left, hb=ctx.hb)
        t0 = self._clock()
        looks = steps = trace = None
        stop_note = None
        try:
            run = evaluate_candidate(src, make_policy(self.gate.policy_name), self.gate.plan, cand_id)
            decision, doms = run.decision, domain_summary(run)
            looks, steps, trace = run.looks_reached, run.steps, run.trace
        except BudgetStop as e:
            decision, doms, stop_note = BUDGET_STOP, partial_summary(self.gate.plan, src.items), str(e)
        tim["gate_s"] = self._lap(t0)
        forced = _test_forced_reject(t, rank, decision)
        if forced:  # test hook (inert unless CAPPRUNE_TEST_FORCE_REJECT is set): exercises the fallback's rescue path
            decision, stop_note = REJECT, "test hook: rank-1 ACCEPT forced to REJECT"
        tim["gate_trial_scoring_s"] = src.t_trial
        tim["gate_dense_reference_s"] = src.t_dense
        S["questions"] += src.used
        S["dense_reference_questions"] += src.dense_q
        # ---- accept, or restore the parent and either queue the next-ranked action (top-k) or end the run
        rollback_ok = None
        fallback_triggered = rescued = False
        if decision == ACCEPT:
            ctx.save_state_masks(cand_id)
            fault_point("ranked_after_masks")
            S.update(state_id=cand_id, rho=rho_c, accepts=S["accepts"] + 1)
            S["parent_digest"] = weights_digest(ctx.state)  # the trial is the next parent
            outcome = "accept"
            S["pending"] = None
            if rank > 1:
                rescued = True
                self._fb()["rescued"] += 1
        else:
            t0 = self._clock()
            for n in changed:
                ctx.state.masks[n] = parent_masks[n]
            if changed:
                ctx.state.apply(ctx.dense, only=changed)
            tim["rollback_s"] = self._lap(t0)
            rollback_ok = weights_digest(ctx.state) == S["parent_digest"] and ctx.state.fingerprint() == parent_fp
            if not rollback_ok:
                log.error("attempt %d: rollback did not restore parent %s exactly", t, parent_id)
            if decision == BUDGET_STOP:
                outcome = BUDGET_STOP
                S["stop_reason"] = "question_cap"
                S["pending"] = None
                S["budget_stop"] = {"when": "during_attempt", "attempt": t, "left": cap_left, "used": src.used,
                                    "note": stop_note}
            else:
                outcome = "rollback"
                S["rollbacks"] += 1
                if self.kl is not None and rank == 1 and len(ranked) >= 2:
                    fallback_triggered = True  # amendment 2: the next proposal is the KL probe's pick among ranks 2-11
                    self._fb()["triggers"] += 1
                    S["pending"] = {"mode": "kl_probe", "rank": None, "profile_attempt": prof_t, "parent_mask_hash": parent_hash,
                                    "shortlist": [s["action_key"] for s in short], "trigger_attempt": t}
                elif rank < self.fallback_k and rank < len(ranked):
                    # top-k: the parent is restored (verified above); the next-ranked action becomes the next proposal
                    fallback_triggered = True
                    if rank == 1:
                        self._fb()["triggers"] += 1
                    S["pending"] = {"rank": rank + 1, "profile_attempt": prof_t, "parent_mask_hash": parent_hash,
                                    "shortlist": [s["action_key"] for s in short],
                                    "trigger_attempt": pending["trigger_attempt"] if pending else t}
                else:
                    S["pending"] = None
                    if rank > 1:
                        self._fb()["failed"] += 1
                        S["stop_reason"] = "fallback_exhausted"
                    else:
                        S["stop_reason"] = "candidate_rejected" if decision == REJECT else "candidate_inconclusive"
        S["gate_counts"][decision] = S["gate_counts"].get(decision, 0) + 1
        evals = {"paired_questions": src.used, "dense_run": self.scorer.evals["dense"] - ev0["dense"],
                 "trial_run": self.scorer.evals["trial"] - ev0["trial"], "pairs_replayed_from_cache": src.replayed_q}
        evals["item_model_evaluations_run"] = evals["dense_run"] + evals["trial_run"]
        ev = S.setdefault("evaluations", {"paired_questions": 0, "dense_run": 0, "trial_run": 0,
                                          "pairs_replayed_from_cache": 0, "item_model_evaluations_run": 0})
        for k, v in evals.items():
            ev[k] = ev.get(k, 0) + v
        prefix_check = None
        if want is not None and decision == BUDGET_STOP:  # the cap cut the replayed attempt short: a cap stop, not a mismatch
            prefix_check = {"n": self.prefix["n"], "verified_through": t - 1, "ok": None, "attempt": t, "diffs": [],
                            "final_mask_verified": False, "note": "question cap reached inside the replayed prefix"}
            S["prefix"] = prefix_check
        elif want is not None:  # the gate must reproduce the earlier run too; at its last attempt, so must the mask file
            from .utils.io import sha256_file

            pv = prefix_view(want)
            got = {"cand_id": cand_id, "trial_mask_hash": trial_hash, "decision": decision, "questions": src.used,
                   "order_seed": seed, "domains": {d: [v["n"], v["broken"], v["fixed"], v["status"]] for d, v in doms.items()}}
            diffs = [k for k, v in got.items() if pv[k] != v]
            file_bytes_equal = None
            if t == self.prefix["n"] and not diffs:
                # mask CONTENT must match the frozen checkpoint (packed-bit hash); the reference file itself is checked
                # against its recorded sha256.  Byte equality of the two files is informational only: safetensors writes
                # the metadata keys in an unspecified order, so identical masks can give different file hashes.
                from .pruning.masks import PruneState

                ref = PruneState(ctx.units, ctx.state.mode)
                ref.load(self.prefix["mask_file"])
                if S["state_id"] != self.prefix["state_id"]:
                    diffs.append("final_state_id")
                elif sha256_file(self.prefix["mask_file"]) != self.prefix["mask_sha256"]:
                    diffs.append("frozen_mask_file_changed")
                elif mask_hash(ref) != mask_hash(ctx.state):
                    diffs.append("final_mask_content")
                file_bytes_equal = sha256_file(ctx.store.mask_path(S["state_id"])) == self.prefix["mask_sha256"]
            prefix_check = {"n": self.prefix["n"], "verified_through": t if not diffs else t - 1, "ok": not diffs,
                            "attempt": t, "diffs": diffs, "final_mask_verified": t == self.prefix["n"] and not diffs,
                            "final_mask_file_bytes_equal": file_bytes_equal}
            S["prefix"] = prefix_check
            if diffs:
                S["stop_reason"] = "prefix_mismatch"
                log.error("attempt %d does not reproduce the prefix run (%s); stopping", t, diffs)
            elif t == self.prefix["n"]:
                log.info("prefix verified: attempts 1-%d and the final mask (sha256 %s...) match %s", t,
                         self.prefix["mask_sha256"][:12], self.prefix["journal"])
        tim["attempt_s"] = time.perf_counter() - t_att
        b = S["budget"]
        b["probe_questions"] = S["questions"]
        b["probe_tests"] += 1
        if probe_rec is not None:
            b["kl_probe_paired_items"] = b.get("kl_probe_paired_items", 0) + probe_rec["cost"]["paired_items_charged"]
            b["kl_probe_seconds"] = b.get("kl_probe_seconds", 0.0) + tim.get("kl_probe_s", 0.0)
        if cf_rec is not None:
            b["counterfactual_paired_items"] = b.get("counterfactual_paired_items", 0) + cf_rec["cost"]["paired_items"]
            b["counterfactual_seconds"] = b.get("counterfactual_seconds", 0.0) + tim.get("counterfactual_s", 0.0)
        for k, key in (("profile_s", "profile_seconds"), ("build_s", "build_seconds"), ("gate_s", "gate_seconds"),
                       ("gate_trial_scoring_s", "gate_scoring_seconds"),
                       ("gate_dense_reference_s", "gate_dense_reference_seconds"), ("rollback_s", "rollback_seconds"),
                       ("attempt_s", "search_wall_seconds")):
            b[key] = b.get(key, 0.0) + tim.get(k, 0.0)
        gpu_s = sum(tim.get(k, 0.0) for k in ("profile_s", "build_s", "gate_trial_scoring_s", "gate_dense_reference_s",
                                               "rollback_s", "kl_probe_s", "counterfactual_s"))
        if rank > 1:  # the extra cost of the fallback candidates
            fb = self._fb()
            fb["extra_questions"] += src.used
            fb["extra_evaluations"] += evals["item_model_evaluations_run"]
            fb["extra_seconds"] += tim["attempt_s"]
        mult = geo["mult"]
        rec.update(
            chosen={"rank": rank, "idx": top["idx"], "kind": top["kind"], "action": action, "action_key": canon(action),
                    "layers": short[rank - 1]["layers"], "z_g_mean": short[rank - 1]["z_g_mean"]},
            fallback_triggered=fallback_triggered, rescued=rescued, gate_margin=gate_margin(rc), forced_by_test_hook=forced,
            cand_id=cand_id, trial_id=cand_id, trial_mask_hash=trial_hash,
            decisions=[(u, round(bk / geo["B"], 6)) for u, bk in action.items()],
            proposed_channels=sum(action.values()) * mult, actual_channels=sum(actual.values()),
            proposed_by_unit={u: bk * mult for u, bk in action.items()}, actual_by_unit=actual,
            channels_match=all(actual[u] == action[u] * mult for u in action),
            rho_candidate=round(rho_c, 6), rho_after=round(S["rho"], 6),
            outcome=outcome, gate_decision=decision,
            gate={"kind": "capability", "policy": self.gate.policy_name, "decision": decision, "questions": src.used,
                  "budget": self.gate.budget, "cap_left": cap_left, "order_seed": seed, "domains": doms,
                  "tolerance": self.gate.settings.tolerance, "looks": looks, "steps": steps, "trace": trace,
                  "dense_reference_questions": src.dense_q, "replayed_questions": src.replayed_q,
                  "pool": {**self.pool_info, "size": len(self.pool)}, "stop_note": stop_note},
            questions_cumulative=S["questions"], question_cap=self.cap, timings={k: round(v, 4) for k, v in tim.items()},
            gpu_seconds=round(gpu_s, 4), parent_verified=parent_ok, rollback_verified=rollback_ok,
            trial_params_removed=trial_params, params_removed=self.params_removed(), evaluations=evals,
            evaluations_cumulative=dict(ev), prefix_check=prefix_check, stop_reason=S["stop_reason"])
        if self.kl is not None:
            rec.update(fallback_policy="kl_probe", kl_probe=probe_rec, counterfactual=cf_rec,
                       gate_questions=src.used, kl_probe_items_charged=probe_rec["cost"]["paired_items_charged"] if probe_rec else 0)
        S["history"].append({k: rec[k] for k in ("iteration", "outcome", "gate_decision", "cand_id", "rho_candidate",
                                                 "questions_cumulative")})
        self._journal(rec)
        failed = [d for d in doms if doms[d]["status"] == "fail"]
        worst = failed[0] if failed else max(doms, key=lambda d: doms[d]["est"] if doms[d]["est"] is not None else -9)
        log.info("attempt %d: %s | rank %d %s %s | MLP sparsity %.4f -> %.4f | gate %s n=%d (cum %d) | %s %s %+.2f pts%s",
                 t, outcome.upper(), rank, top["kind"], short[rank - 1]["layers"], rho_prev, rho_c, decision, src.used,
                 S["questions"], "blocked by" if failed else "largest estimate", worst, 100 * (doms[worst]["est"] or 0.0),
                 (" -> KL probe queued" if self.kl is not None else " -> fallback to rank %d queued" % (rank + 1))
                 if fallback_triggered else (" (rescued)" if rescued else ""))
        S["iteration"] = t + 1
