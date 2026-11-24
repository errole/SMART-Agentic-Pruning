"""Retrospective population-level re-gating of finished gradient runs (``capprune population-regate``).

register  Records the SHA-256 (line endings normalised to LF) and git blob id of the pre-registered rule
          (docs/population_gate_rule.md) before anything is computed.
run       CPU only.  Re-verifies every attempt's trial mask (rebuilt from the parent chain, the recorded action and
          the persisted profile) against its journal, loads its cached paired gate batches, checks them against the
          journal's counts, replays the gate's recorded reveal sequence and re-decides every candidate with exact
          population (binomial, no finite-pool correction) bounds and the rule's error spending.  Writes the
          per-attempt decisions, the run-level stopping points and the required-pool-size analysis.

The rule's parameters are read from the JSON block of the rule file itself, so the hashed file is the only source.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json
from .utils.log import get_logger

log = get_logger("population_regate")
REPO = Path(__file__).resolve().parents[1]
ACCEPT, REJECT, INCONCLUSIVE = "accept", "reject", "inconclusive (revealed items)"
_EPS = 1e-12


# ============================================================================ exact binomial (Clopper-Pearson) bounds
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 100000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise RuntimeError("incomplete beta continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - betainc(k + 1, n - k, p)


@lru_cache(maxsize=None)
def cp_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound: the p with P_p(X <= k) = delta (1 if k = n)."""
    if not (0 <= k <= n and n >= 1 and 0 < delta < 1):
        raise ValueError(f"cp_upper: k={k} n={n} delta={delta}")
    if k == n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return hi


def cp_lower(k: int, n: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson lower bound (0 if k = 0)."""
    return 0.0 if k == 0 else 1.0 - cp_upper(n - k, n, delta)


def population_bounds(n: int, k_plus: int, k_minus: int, d_hi: float, d_lo: float) -> tuple[float, float]:
    """(LB, UB) on the population paired loss p+ - p-; P(UB < mu) <= d_hi, P(LB > mu) <= d_lo."""
    ub = cp_upper(k_plus, n, d_hi / 2) - cp_lower(k_minus, n, d_hi / 2)
    lb = cp_lower(k_plus, n, d_lo / 2) - cp_upper(k_minus, n, d_lo / 2)
    return lb, ub


# ============================================================================ the rule
def _norm_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rule_params(path: Path) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", path.read_text(encoding="utf-8"), re.S)
    if not m:
        raise ValueError(f"{path}: no JSON parameter block")
    return json.loads(m.group(1))


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def _pr(cfg: dict) -> dict:
    return cfg["population_regate"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _pr(cfg)["out_dir"])


def register(cfg: dict) -> Path:
    rule = resolve_path(cfg, _pr(cfg)["rule_file"])
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "rule_registration.json"
    rec = {"rule_file": str(rule), "sha256_lf_normalised": _norm_sha(rule), "git_blob": _git("hash-object", str(rule)),
           "committed_in": _git("log", "-1", "--format=%H", "--", str(rule)),
           "rule_file_uncommitted_changes": bool(_git("status", "--porcelain", "--", str(rule))),
           "params": rule_params(rule)}
    if p.exists():
        old = read_json(p)
        if old["sha256_lf_normalised"] != rec["sha256_lf_normalised"]:
            raise SystemExit("a different rule was registered for this output folder; use a new out_dir")
        return p
    if any(out.glob("regate*.json")):
        raise SystemExit("results already exist; registration must precede computation")
    rec["registered_at"] = time.time()
    atomic_write_json(p, rec)
    return p


def check_rule(cfg: dict) -> tuple[dict, dict]:
    reg = read_json(out_dir(cfg) / "rule_registration.json")
    rule = Path(reg["rule_file"])
    if _norm_sha(rule) != reg["sha256_lf_normalised"]:
        raise SystemExit("the rule file changed after registration; refusing to compute")
    return reg["params"], reg


# ============================================================================ spending
def look_deltas(pool_sizes: dict[str, int], P: dict) -> dict[str, dict]:
    """Per category: planned look sizes (the gate's plan, budget = pool) and the per-look error for accept / reject."""
    from .adaptive.evaluator import look_sizes

    total_min = P["min_sample"] * len(pool_sizes)
    budget = sum(pool_sizes.values())
    out = {}
    for c, N in pool_sizes.items():
        sizes = look_sizes(N, P["min_sample"], P["batch_size"], min(N, budget - (total_min - P["min_sample"])))
        J = len(sizes)
        div = P["candidates_per_run"] * len(pool_sizes) * J
        out[c] = {"looks": sizes, "J": J, "d_hi": P["alpha_false_accept"] / div, "d_lo": P["alpha_false_reject"] / div}
    return out


def total_spent(deltas: dict[str, dict], P: dict) -> tuple[float, float]:
    """Error spent over every candidate, category and planned look of a run (must be <= alpha)."""
    hi = P["candidates_per_run"] * sum(v["d_hi"] * v["J"] for v in deltas.values())
    lo = P["candidates_per_run"] * sum(v["d_lo"] * v["J"] for v in deltas.values())
    return hi, lo


# ============================================================================ one candidate
def reveal_sequence(gate: dict, categories: list[str]) -> list[tuple[str, int, int]]:
    """(category, cursor, size) in the order the live gate revealed them: every category's minimum sample, then the
    batches named by the trace (trace[i]['chosen'] is the category revealed before step i was recorded)."""
    trace = gate.get("trace") or []
    n = {c: trace[0]["domains"][c][0] for c in categories} if trace else {}
    seq = [(c, 0, n[c]) for c in categories]
    for t in trace[1:]:
        c = t["chosen"]
        new = t["domains"][c][0]
        seq.append((c, n[c], new - n[c]))
        n[c] = new
    return seq


def redecide(batches: list[tuple[str, list[int]]], deltas: dict, P: dict, categories: list[str]) -> dict:
    """Replay revealed batches (category, x values) and apply the population rule after every reveal (the first
    evaluation after every category's minimum sample)."""
    tol = P["tolerance"]
    cnt = {c: [0, 0, 0] for c in categories}  # n, lost, gained
    seen_min = set()
    steps = 0
    last = None
    for c, xs in batches:
        a = cnt[c]
        a[0] += len(xs)
        a[1] += sum(1 for x in xs if x == 1)
        a[2] += sum(1 for x in xs if x == -1)
        seen_min.add(c)
        if len(seen_min) < len(categories):
            continue
        steps += 1
        doms = {}
        for d in categories:
            n, kp, km = cnt[d]
            if n not in deltas[d]["looks"]:
                raise RuntimeError(f"{d}: {n} items is not a planned look")
            lb, ub = population_bounds(n, kp, km, deltas[d]["d_hi"], deltas[d]["d_lo"])
            doms[d] = {"n": n, "lost": kp, "gained": km, "est": (kp - km) / n, "lb": lb, "ub": ub,
                       "status": "pass" if ub <= tol + _EPS else ("fail" if lb > tol + _EPS else "open")}
        last = {"step": steps, "questions": sum(v[0] for v in cnt.values()), "domains": doms}
        if any(v["status"] == "fail" for v in doms.values()):
            return {"decision": REJECT, **last}
        if all(v["status"] == "pass" for v in doms.values()):
            return {"decision": ACCEPT, **last}
    return {"decision": INCONCLUSIVE, **(last or {"step": 0, "questions": 0, "domains": {}})}


def replay_live(batches: list[tuple[str, list[int]]], plan, categories: list[str]) -> dict:
    """Recompute the LIVE gate's finite-pool decision from the same reconstructed reveal sequence (a check that the
    sequence and the cached pairs are exactly what the gate saw)."""
    from .adaptive.evaluator import decide, domain_state

    cnt = {c: [0, 0, 0] for c in categories}
    look = {c: -1 for c in categories}
    seen = set()
    for c, xs in batches:
        cnt[c][0] += len(xs)
        cnt[c][1] += xs.count(1)
        cnt[c][2] += xs.count(-1)
        look[c] += 1
        seen.add(c)
        if len(seen) < len(categories):
            continue
        d = decide(tuple(domain_state(plan[x], *cnt[x], look[x]) for x in categories))
        if d is not None:
            return {"decision": d, "questions": sum(v[0] for v in cnt.values())}
    return {"decision": "inconclusive", "questions": sum(v[0] for v in cnt.values())}


# ============================================================================ the CPU re-gating run
def _attempts(run_dir: Path) -> list[dict]:
    last = {}
    for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return [last[k] for k in sorted(last)]


def _load_cpu_model(rc: dict):
    from .model import build_units, load_model

    rc = json.loads(json.dumps(rc))
    rc["model"]["device"] = "cpu"
    model = load_model(rc)
    return model, build_units(model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))


def verify_masks(run_dir: Path, rc: dict, units, archive_dir: Path | None) -> list[dict]:
    """Rebuild every attempt's trial mask from the verified parent chain (CPU) and compare with the journal."""
    from .fixed_state import _geometry
    from .pruning.masks import PruneState
    from .pruning.profile import load_profile
    from .pruning.wanda import raise_unit
    from .ranker import canon, mask_hash

    geo = _geometry(rc, units[0].channels)
    arch = {}
    if archive_dir and (archive_dir / "manifest.json").exists():
        arch = {s["state_id"]: s["mask_hash"] for s in read_json(archive_dir / "manifest.json")["states"]}
    parent = PruneState(units, rc["pruning"]["mode"])
    out = []
    for a in _attempts(run_dir):
        if a.get("gate_decision") in (None, "none"):
            out.append({"attempt": a["iteration"], "skipped": "not gated"})
            continue
        ph = mask_hash(parent)
        if ph != a["parent_mask_hash"]:
            raise RuntimeError(f"{run_dir.name} attempt {a['iteration']}: parent hash {ph} != journal {a['parent_mask_hash']}")
        act, _, _ = load_profile(run_dir / "profiles" / f"iter-{int(a.get('profile_attempt') or a['iteration']):06d}.safetensors", "cpu")
        trial = parent.clone()
        action = {u: int(b) for u, b in a["chosen"]["action"].items()}
        if canon(action) != a["chosen"]["action_key"]:
            raise RuntimeError(f"attempt {a['iteration']}: recorded action and key disagree")
        for u, b in action.items():
            before = int((~trial.masks[u]).sum().item()) // geo["mult"]
            raise_unit(trial, u, (before + b) / geo["B"], act, rc)  # dense weights: same channel scores as the run
        th = mask_hash(trial)
        ok = th == a["trial_mask_hash"]
        rec = {"attempt": a["iteration"], "trial_mask_hash": th, "journal_trial_mask_hash": a["trial_mask_hash"],
               "matches_journal": ok}
        if a["gate_decision"] == "accept" and a.get("cand_id") in arch:
            rec["matches_verified_archive"] = arch[a["cand_id"]] == th
            ok &= rec["matches_verified_archive"]
        out.append(rec)
        if not ok:
            raise RuntimeError(f"{run_dir.name} attempt {a['iteration']}: rebuilt trial mask does not match: {rec}")
        if a["gate_decision"] == "accept":
            parent = trial
    return out


def regate_run(run_dir: Path, P: dict, data, units, archive_dir: Path | None, verify: bool = True) -> dict:
    from .gradient_live import gate_pool, make_gate

    rj = read_json(run_dir / "run.json")
    rc = rj["config"]
    cats = list(P["categories"])
    pool, _ = gate_pool(rc, data)
    live_plan = make_gate(rc, pool).plan  # the run's own gate (its tolerance and margin)
    g0 = rc["gate"]["capability"]
    if (int(g0["min_sample"]), int(g0["batch_size"])) != (P["min_sample"], P["batch_size"]):
        raise RuntimeError(f"{run_dir.name}: the rule's reveal schedule ({P['min_sample']}/{P['batch_size']}) is not the run's "
                           f"gate schedule ({g0['min_sample']}/{g0['batch_size']})")
    sizes = {c: sum(1 for r in pool if r[rc["gate"]["capability"].get("domain_field", "category")] == c) for c in cats}
    deltas = look_deltas(sizes, P)
    spent = total_spent(deltas, P)
    checks = verify_masks(run_dir, rc, units, archive_dir) if verify else []
    con = sqlite3.connect(f"file:{(run_dir / 'caches' / 'eval.sqlite').as_posix()}?mode=ro", uri=True)
    rows = []
    for a in _attempts(run_dir):
        g = a.get("gate") or {}
        if not g.get("trace"):
            continue
        seq = reveal_sequence(g, cats)
        batches = []
        for c, cur, size in seq:
            key = f"pair|{a['trial_mask_hash']}|{g['order_seed']}|{c}|{cur}|{size}"
            row = con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
            if row is None:
                raise RuntimeError(f"{run_dir.name} attempt {a['iteration']}: cached batch {key} missing")
            v = json.loads(row[0])
            if set(v["d"]) != set(v["t"]) or len(v["t"]) != size:
                raise RuntimeError(f"attempt {a['iteration']}: malformed batch {key}")
            batches.append((c, [int(v["d"][q]) - int(v["t"][q]) for q in sorted(v["t"])]))
        # the cached pairs must reproduce the journal's final counts (n, lost, gained) of every category
        tot = {c: [0, 0, 0] for c in cats}
        for c, xs in batches:
            tot[c][0] += len(xs)
            tot[c][1] += xs.count(1)
            tot[c][2] += xs.count(-1)
        jd = g["domains"]
        counts_ok = all(tot[c] == [jd[c]["n"], jd[c]["broken"], jd[c]["fixed"]] for c in cats)
        if not counts_ok:
            raise RuntimeError(f"{run_dir.name} attempt {a['iteration']}: cached pairs do not reproduce the journal counts")
        live = replay_live(batches, live_plan, cats)
        live_ok = live["decision"] == g["decision"] and live["questions"] == g["questions"]
        if not live_ok and not a.get("forced_by_test_hook"):
            raise RuntimeError(f"{run_dir.name} attempt {a['iteration']}: replay gives {live}, journal {g['decision']} / {g['questions']}")
        pop = redecide(batches, deltas, P, cats)
        proj = {}
        for c in cats:
            n, kp, km = tot[c]
            N = sizes[c]
            pp, pm = kp / n, km / n
            _, ub_full = population_bounds(N, round(pp * N), round(pm * N), deltas[c]["d_hi"], deltas[c]["d_lo"])
            proj[c] = {"q_disagree": (kp + km) / n, "p_lost": pp, "p_gained": pm, "est": pp - pm, "n_revealed": n,
                       "pool": N, "ub_full_pool_at_observed_rates": ub_full, "width_full_pool": ub_full - (pp - pm)}
        rows.append({"attempt": a["iteration"], "rank": (a.get("chosen") or {}).get("rank", 1), "cand_id": a.get("cand_id"),
                     "parent_id": a["parent_id"], "trial_mask_hash": a["trial_mask_hash"],
                     "parent_sparsity": a["parent_sparsity"], "rho_candidate": a["rho_candidate"],
                     "params_removed_trial": a.get("trial_params_removed"), "params_removed_after": a.get("params_removed"),
                     "live_decision": a["gate_decision"], "live_questions": g["questions"],
                     "live_tolerance": g.get("tolerance"), "population": pop, "counts_match_journal": counts_ok,
                     "live_decision_reproduced": live_ok,
                     "full_pool_projection": proj})
    con.close()
    # run-level: the first candidate the population rule does not accept, along the live trajectory
    stop = next((r for r in rows if r["population"]["decision"] != ACCEPT), None)
    params_before = {}
    prev = 0
    for r in rows:
        params_before[r["attempt"]] = prev
        if r["live_decision"] == "accept":
            prev = r["params_removed_after"]
    res = {"run_dir": str(run_dir), "pool_sizes": sizes, "deltas": deltas, "error_spent": {"accept": spent[0], "reject": spent[1]},
           "mask_checks": checks, "attempts": rows}
    if stop:
        res["first_stop"] = {"attempt": stop["attempt"], "rank": stop["rank"], "decision": stop["population"]["decision"],
                             "live_decision": stop["live_decision"], "parent_id": stop["parent_id"],
                             "params_removed_at_stop": params_before[stop["attempt"]],
                             "accepted_before_stop": sum(1 for r in rows if r["attempt"] < stop["attempt"] and r["live_decision"] == "accept")}
    return res


def required_items(q: float, mu: float, P: dict, power: float | None = None, n_max: int = 400_000) -> int | None:
    """Smallest 100-item look n at which the tolerance is certified (expected counts, or simulated power)."""
    tol, C = P["tolerance"], P["candidates_per_run"]
    pp, pm = (q + mu) / 2, (q - mu) / 2
    if pm < 0:
        pp, pm = mu, 0.0
    rng = np.random.default_rng(P["n80_seed"])

    def ok(n):
        d = P["alpha_false_accept"] / (C * len(P["categories"]) * (n // P["batch_size"]))
        if power is None:
            return population_bounds(n, round(pp * n), round(pm * n), d, d)[1] <= tol + _EPS
        draws = rng.multinomial(n, [pp, pm, 1 - pp - pm], size=P["n80_draws"])
        return np.mean([population_bounds(n, int(a), int(b), d, d)[1] <= tol + _EPS for a, b, _ in draws]) >= power

    lo, hi = 1, n_max // P["batch_size"]
    if not ok(hi * P["batch_size"]):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid * P["batch_size"]):
            hi = mid
        else:
            lo = mid + 1
    return lo * P["batch_size"]


def run(cfg: dict) -> Path:
    from .config import data_dir
    from .data.prepare import PreparedData
    from .model import tokenizer_id

    P, reg = check_rule(cfg)
    pr = _pr(cfg)
    root = resolve_path(cfg, pr["runs_root"])
    out = out_dir(cfg)
    t0 = time.time()
    data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
    rc0 = read_json(root / P["runs"]["A"] / "run.json")["config"]
    model, units = _load_cpu_model(rc0)
    del model
    arch_root = resolve_path(cfg, pr["archive"]) if pr.get("archive") else None
    results = {}
    for name, rel in P["runs"].items():
        rd = root / rel
        results[name] = regate_run(rd, P, data, units, (arch_root / rel) if arch_root else None)
        fs = results[name].get("first_stop")
        log.info("%s: %d candidates re-decided; first stop %s", name, len(results[name]["attempts"]),
                 fs and f"attempt {fs['attempt']} ({fs['decision']}), {fs['params_removed_at_stop']:,} params")
    # required pool sizes from run A's observed disagreement rates
    A = results["A"]["attempts"]
    sizes = {}
    for c in P["categories"]:
        qs = [r["full_pool_projection"][c]["q_disagree"] for r in A]
        widths = [r["full_pool_projection"][c]["width_full_pool"] for r in A]
        qm, qx = float(np.median(qs)), float(max(qs))
        sizes[c] = {"pool": results["A"]["pool_sizes"][c], "q_median": qm, "q_max": qx, "q_last": qs[-1],
                    "width_full_pool_median": float(np.median(widths)), "width_full_pool_max": float(max(widths)),
                    "width_full_pool_last": widths[-1],
                    "needed": {f"{100 * mu:.1f}": {"n50_q_median": required_items(qm, mu, P), "n50_q_max": required_items(qx, mu, P),
                                                   "n80_q_median": required_items(qm, mu, P, 0.8), "n80_q_max": required_items(qx, mu, P, 0.8)}
                               for mu in P["sensitivity_true_loss"]}}
        log.info("%s: q median %.3f max %.3f; needed %s", c, qm, qx, sizes[c]["needed"])
    S = {"rule": reg, "commit": _git("rev-parse", "HEAD"), "working_tree_clean": _git("status", "--porcelain") == "",
         "runs": results, "required_pool_sizes": sizes, "dense_parameters": pr.get("dense_parameters"),
         "label": "retrospective; audit comparisons exploratory", "cpu_seconds": time.time() - t0}
    atomic_write_json(out / "regate.json", S)
    atomic_write_text(out / "report.md", render(S, P))
    return out / "report.md"


# ============================================================================ report
def render(S: dict, P: dict) -> str:
    pts = lambda x: "n/a" if x is None else f"{100 * x:+.2f}"  # noqa: E731
    D = S["dense_parameters"] or 1
    L = ["# Retrospective population-level re-gating (Phase 1, CPU)", "",
         f"Rule `{Path(S['rule']['rule_file']).name}` (sha256 LF-normalised `{S['rule']['sha256_lf_normalised'][:16]}...`, committed in "
         f"`{S['rule']['committed_in'][:12]}`), registered before computation; code commit `{S['commit'][:12]}` (clean tree: "
         f"{S['working_tree_clean']}). Tolerance {100 * P['tolerance']:.1f} pts; population bounds = exact Clopper-Pearson, no "
         f"finite-pool correction; alpha {P['alpha_false_accept']} per run over {P['candidates_per_run']} candidates x 4 "
         "categories x planned looks. Audit comparisons are exploratory.", ""]
    for name, R in S["runs"].items():
        fs = R.get("first_stop")
        L += [f"## Run {name}: `{R['run_dir']}`", "",
              f"Error spent over the run's plan: accept {R['error_spent']['accept']:.4f}, reject {R['error_spent']['reject']:.4f}. "
              f"Trial masks rebuilt on CPU and matched to the journal: {sum(1 for c in R['mask_checks'] if c.get('matches_journal'))}"
              f"/{sum(1 for c in R['mask_checks'] if 'matches_journal' in c)}; cached pairs reproduce the journal counts for every attempt.", ""]
        if fs:
            L.append(f"**First population non-accept: attempt {fs['attempt']} (rank {fs['rank']}), {fs['decision']}; live: "
                     f"{fs['live_decision']}.** The run would stop at its parent `{fs['parent_id']}` after {fs['accepted_before_stop']} "
                     f"accepted steps, {fs['params_removed_at_stop']:,} parameters removed = "
                     f"{100 * fs['params_removed_at_stop'] / D:.2f} % (audit boundary {100 * P['audit_boundary_params_fraction']:.3f} %, "
                     f"arm C {100 * P['arm_c_params_fraction']:.2f} %).")
        L += ["", "| attempt | rank | live (tolerance) | population decision | at questions (live) | per category: est [population LB, UB] at the decision | full-pool UB at observed rates |",
              "|---|---|---|---|---|---|---|"]
        for r in R["attempts"]:
            p = r["population"]
            dl = "; ".join(f"{c} {pts(v['est'])} [{pts(v['lb'])}, {pts(v['ub'])}] n={v['n']}" for c, v in p["domains"].items())
            fp = "; ".join(f"{c} {pts(v['ub_full_pool_at_observed_rates'])}" for c, v in r["full_pool_projection"].items())
            tol = r.get("live_tolerance") or {}
            t = next(iter(tol.values())) if isinstance(tol, dict) and tol else tol
            L.append(f"| {r['attempt']} | {r['rank']} | {r['live_decision']} ({100 * t:.1f}) | **{p['decision']}** | "
                     f"{p['questions']:,} ({r['live_questions']:,}) | {dl} | {fp} |")
        L.append("")
    L += ["## Required pool sizes (run A's observed paired disagreement rates)", "",
          "| category | pool | q median / max | full-pool width (median / max over attempts) | items to certify 3 pts: true loss 0 / 1.0 / 1.5 (n50 at median q; n80 at max q) |",
          "|---|---|---|---|---|"]
    for c, v in S["required_pool_sizes"].items():
        nd = v["needed"]
        cells = " / ".join(f"{nd[k]['n50_q_median'] or '>400k'} ({nd[k]['n80_q_max'] or '>400k'})" for k in nd)
        L.append(f"| {c} | {v['pool']:,} | {100 * v['q_median']:.1f} % / {100 * v['q_max']:.1f} % | "
                 f"{100 * v['width_full_pool_median']:.2f} / {100 * v['width_full_pool_max']:.2f} pts | {cells} |")
    return "\n".join(L) + "\n"
