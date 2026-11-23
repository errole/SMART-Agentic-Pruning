"""Fixed-state comparison of pruning controllers (``capprune fixed-state prepare|query|score|report``).

Every controller sees the SAME frozen parent checkpoints, with the same layer profiles, capability feedback
and constraints, and every candidate action removes the SAME number of MLP channels.  Only the choice of
units differs, so differences in outcome are differences in action quality.

States     dense + accepted checkpoints of a source run nearest preset MLP sparsities (chosen by sparsity only).
           Checkpoints whose mask files were pruned from storage are rebuilt by replaying the run's accepted
           decisions with its saved per-iteration profiles (deterministic structured Wanda); the replay is
           checked against every mask that is still stored.
Step       every action removes exactly ``step_blocks`` blocks of ``channel_multiple`` channels.
Conditions per state and model: free-form (``layer_decisions``, rescaled to exactly the step) and shortlist
           (``action_id`` from a fixed list, identical order and information for every controller); plus the
           greedy sensitivity heuristic's pick on the shortlist (lowest mean 0.5*(z_sens + z_grad)).
Shortlist  built from the state's profiles before anything is scored: the greedy heuristic's lowest-risk set,
           lowest z_sens, lowest z_grad, and contiguous depth windows; each = ``units_per_action`` units x
           ``blocks_per_unit`` blocks, de-duplicated, ordered by first layer.
Scoring    each distinct (state, action) once: the trial = frozen parent + action; the full pruning-time pool
           is scored 0-shot with the same chunking as the dense reference; the capability gate (threshold-
           aware, configured tolerance and budget) is replayed on those answers with ONE question order per
           state -> decision, questions the live gate would score, bounds; plus exact pool losses.

Phases run in separate processes so the pruned model and one Ollama controller never share the GPU.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .utils.io import atomic_write_json, atomic_write_text, read_json
from .utils.log import get_logger

log = get_logger("fixed_state")
CONDITIONS = ("free", "shortlist")


def _fs(cfg: dict) -> dict:
    return cfg["fixed_state"]


def out_dir(cfg: dict) -> Path:
    return resolve_path(cfg, _fs(cfg)["out_dir"])


def _rc(cfg: dict) -> dict:
    from .config import run_config

    return run_config(cfg, "llm_cap", float(cfg["matrix"]["targets"][0]), int(cfg["matrix"]["seeds"][0]))


def _geometry(rc: dict, channels: int) -> dict:
    mult = int(rc["pruning"]["structured"]["channel_multiple"])
    B = channels // mult
    return {"mult": mult, "B": B, "channels": channels,
            "maxb": int(math.floor(float(rc["search"]["max_increment"]) * B + 1e-9)),
            "minb": max(1, int(math.ceil(float(rc["search"]["min_increment"]) * B - 1e-9))),
            "cap_blocks": int(math.floor(float(rc["pruning"]["structured"]["max_unit_sparsity"]) * B + 1e-9))}


# ============================================================================ states
def _accepted(src: Path) -> list[dict]:
    last = {}
    for line in (src / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return [last[k] for k in sorted(last) if last[k].get("gate_decision") == "accept"]


def select_states(cfg: dict) -> list[dict]:
    """dense + the accepted checkpoint nearest each target sparsity (selection by sparsity only)."""
    src = resolve_path(cfg, _fs(cfg)["state_source"])
    acc = _accepted(src)
    chosen = []
    for t in _fs(cfg)["state_targets"]:
        best = min(acc, key=lambda a: (abs(a["rho_after"] - float(t)), a["iteration"]))
        if best["trial_id"] not in [c["state_id"] for c in chosen]:
            chosen.append({"state_id": best["trial_id"], "rho": best["rho_after"], "iteration": best["iteration"],
                           "target": float(t)})
    return [{"name": "dense", "state_id": None, "rho": 0.0, "iteration": 0, "target": 0.0}] + \
        [{"name": f"s{i + 1}", **c} for i, c in enumerate(chosen)]


def rebuild_states(cfg, rc, units, mode, states):
    """Masks of every chosen state: from storage when present, else by replaying the accepted chain; the replay
    is verified against every stored mask of the chain."""
    from .pruning.masks import PruneState
    from .pruning.profile import load_profile
    from .pruning.wanda import raise_unit
    from .utils.checkpoint import CheckpointStore

    src = resolve_path(cfg, _fs(cfg)["state_source"])
    store = CheckpointStore(src)
    acc = _accepted(src)
    dev = next(iter(units[0].linears.values())).weight.device
    st = PruneState(units, mode)
    masks, checks = {}, []
    for a in acc:
        act, _, _ = load_profile(src / "profiles" / f"iter-{a['iteration']:06d}.safetensors", dev)
        for name, inc in a["decisions"]:
            raise_unit(st, name, st.unit_sparsity(name) + inc, act, rc)
        if abs(st.global_sparsity() - a["rho_after"]) > 1e-6:
            raise RuntimeError(f"replay of {a['trial_id']} gives sparsity {st.global_sparsity()} != {a['rho_after']}")
        stored = store.mask_path(a["trial_id"])
        if stored.exists():
            ref = PruneState(units, mode)
            ref.load(stored)
            same = all(bool((ref.masks[n] == st.masks[n]).all()) for n in st.masks)
            checks.append({"state_id": a["trial_id"], "stored_mask_equals_replay": same})
            if not same:
                raise RuntimeError(f"replayed masks differ from stored mask {a['trial_id']}")
        masks[a["trial_id"]] = {n: m.clone() for n, m in st.masks.items()}
    out = {}
    for s in states:
        out[s["name"]] = None if s["state_id"] is None else masks[s["state_id"]]
    return out, checks


# ============================================================================ prompts and shortlist
def _risk(p) -> float:  # the existing HeuristicController's risk score
    zs = p["z_s"] if p["z_s"] == p["z_s"] else 0.0
    zg = p["z_g"] if p["z_g"] == p["z_g"] else 0.0
    return 0.5 * (zs + zg)


def build_shortlist(profiles: list[dict], geo: dict, sl: dict, n_layers: int) -> list[dict]:
    k, b = int(sl["units_per_action"]), int(sl["blocks_per_unit"])
    room = {p["name"]: geo["cap_blocks"] - int(round(p["sparsity"] * geo["B"])) for p in profiles}
    ok = [p for p in profiles if room[p["name"]] >= b]
    by = {p["name"]: p for p in profiles}
    sets = []

    def add(kind, names):
        names = sorted(names, key=lambda n: int(n.split(".")[0][1:]))
        if len(names) == k and all(n in by and room[n] >= b for n in names) and names not in [s[1] for s in sets]:
            sets.append((kind, names))

    add("greedy (lowest risk)", [p["name"] for p in sorted(ok, key=lambda p: (_risk(p), p["name"]))[:k]])
    add("lowest z_sens", [p["name"] for p in sorted(ok, key=lambda p: (p["z_s"] if p["z_s"] == p["z_s"] else 1e9, p["name"]))[:k]])
    add("lowest z_grad", [p["name"] for p in sorted(ok, key=lambda p: (p["z_g"] if p["z_g"] == p["z_g"] else 1e9, p["name"]))[:k]])
    suffix = profiles[0]["name"].split(".", 1)[1]
    for s0 in sl["window_starts"]:
        if s0 + k <= n_layers:
            add(f"layers {s0}-{s0 + k - 1}", [f"L{i}.{suffix}" for i in range(s0, s0 + k)])
    sets.sort(key=lambda s: (int(s[1][0].split(".")[0][1:]), s[1]))
    out = []
    for i, (kind, names) in enumerate(sets):
        ps = [by[n] for n in names]
        out.append({"id": f"A{i + 1}", "kind": kind, "units": {n: b for n in names},
                    "mean_z_s": float(np.mean([p["z_s"] for p in ps])), "mean_z_g": float(np.mean([p["z_g"] for p in ps])),
                    "mean_risk": float(np.mean([_risk(p) for p in ps]))})
    return out


def greedy_pick(shortlist: list[dict]) -> str:
    return min(shortlist, key=lambda a: (round(a["mean_risk"], 12), int(a["id"][1:])))["id"]


def _step_text(geo: dict, step: int) -> dict:
    per = 1.0 / geo["B"]
    return {"blocks": step, "mult": geo["mult"], "ch": step * geo["mult"], "per": per, "sum": step * per,
            "min_units": int(math.ceil(step / geo["maxb"])), "maxb": geo["maxb"], "maxinc": geo["maxb"] * per}


def prompts(ctx: dict, geo: dict, step: int, shortlist: list[dict], total_channels: int) -> dict:
    from .search.controller import system_prompt, user_prompt

    base = system_prompt(ctx)
    head, tail = base.split("Respond with ONLY", 1)
    t = _step_text(geo, step)
    pct = 100.0 * t["ch"] / total_channels
    free_rule = (f"Fixed step size for this decision: remove exactly {t['blocks']} blocks of {t['mult']} channels "
                 f"({t['ch']} channels, {pct:.3f}% of all MLP channels) in total. One block of one unit is "
                 f"additional_sparsity {t['per']:.4f}, so your additional_sparsity values must sum to {t['sum']:.4f}, "
                 f"spread over at least {t['min_units']} units (at most {t['maxb']} blocks = {t['maxinc']:.4f} per unit). "
                 "A proposal with a different total is rescaled to exactly this step. The step size is fixed; your "
                 "decision is which units to prune.\n\n")
    short_rule = (f"Fixed step size for this decision: every action in the shortlist below removes exactly {t['ch']} "
                  f"channels ({pct:.3f}% of all MLP channels). The step size is fixed; your decision is which action "
                  "to take. Choose exactly one action by its ID.\n\n"
                  'Respond with ONLY a JSON object of the form {"reasoning": "...", "action_id": "A1"}. Use '
                  '"action_id": null to abstain.')
    user = user_prompt(ctx)
    lines = ["", f"Shortlist (fixed order; every action removes exactly {t['ch']} channels):"]
    for a in shortlist:
        u = ", ".join(a["units"])
        per_unit = next(iter(a["units"].values())) * geo["mult"]
        lines.append(f"{a['id']}: {u} (+{per_unit} channels each) | mean z_sens {a['mean_z_s']:+.2f} | "
                     f"mean z_grad {a['mean_z_g']:+.2f}")
    return {"free": {"system": head + free_rule + "Respond with ONLY" + tail, "user": user},
            "shortlist": {"system": head + short_rule, "user": user + "\n" + "\n".join(lines)}}


# ============================================================================ phase 1: prepare (GPU)
def _score_pool(model, tok, data, rc, chunk) -> dict[str, int]:
    from .eval.mmlu import MMLUScorer

    t = rc["testing"]
    sc = MMLUScorer(model, tok, data.mmlu_fewshot, int(t.get("shots", 0)), int(rc["model"].get("max_seq_len", 2048)),
                    {}, "fixed_state", int(t.get("max_batch_tokens", 32768)))
    rows = data.mmlu_probe
    res = {}
    for i in range(0, len(rows), chunk):
        res.update({q: int(r["correct"]) for q, r in sc.score(rows[i : i + chunk]).items()})
    return res


def _gate_replay(rc, data, dense: dict, trial: dict, seed: int, key: str) -> dict:
    from .adaptive.evaluator import evaluate_candidate
    from .adaptive.live import CapabilityGate, LiveSource, domain_summary
    from .adaptive.policies import make_policy

    gate = CapabilityGate(rc, data.mmlu_probe)
    src = LiveSource(gate.pool, {q: bool(v) for q, v in dense.items()},
                     lambda rows: {r["qid"]: bool(trial[r["qid"]]) for r in rows}, seed, gate.budget)
    run = evaluate_candidate(src, make_policy(gate.policy_name), gate.plan, key)
    return {"decision": run.decision, "questions": run.questions, "domains": domain_summary(run)}


def _exact(data, field, dense, trial) -> dict:
    acc: dict[str, list[int]] = {}
    for r in data.mmlu_probe:
        a = acc.setdefault(r[field], [0, 0, 0])
        a[0] += dense[r["qid"]]
        a[1] += trial[r["qid"]]
        a[2] += 1
    return {c: (v[0] - v[1]) / v[2] for c, v in sorted(acc.items())}


def _setup(cfg):
    import torch

    from .config import data_dir
    from .data.prepare import PreparedData
    from .model import build_units, load_model, load_tokenizer, tokenizer_id
    from .pruning.masks import DenseStore

    rc = _rc(cfg)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    tok = load_tokenizer(rc)
    model = load_model(rc)
    mode = rc["pruning"]["mode"]
    units = build_units(model, mode, rc["pruning"].get("eligible", []))
    torch.manual_seed(0)
    return rc, data, tok, model, units, DenseStore(units), mode


def prepare(cfg: dict) -> Path:
    import torch

    from .adaptive.live import attempt_order_seed
    from .eval.perplexity import perplexity
    from .pruning.masks import PruneState
    from .pruning.profile import collect_act_sqnorm, collect_grad_importance, save_profile, sensitivity, zscores

    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    fs = _fs(cfg)
    rc, data, tok, model, units, dense_store, mode = _setup(cfg)
    geo = _geometry(rc, units[0].channels)
    states = select_states(cfg)
    masks, checks = rebuild_states(cfg, rc, units, mode, states)
    chunk = int(fs.get("chunk_size", 512))
    field = rc["gate"]["capability"].get("domain_field", "category")
    calib = data.calibration_pool[data.calibration_indices(int(rc["run"]["seed"]), int(rc["data"]["calibration"]["num_samples"]))]
    p = rc["profiling"]
    rng = np.random.default_rng([int(fs.get("profile_seed", 0)), 1, 17])
    a_idx = np.sort(rng.choice(calib.shape[0], size=min(int(p["activation_samples"]), calib.shape[0]), replace=False))
    g_idx = np.sort(rng.choice(calib.shape[0], size=min(int(p["gradient_samples"]), calib.shape[0]), replace=False))
    PruneState(units, mode).apply(dense_store)
    dense = _score_pool(model, tok, data, rc, chunk)
    atomic_write_json(out / "answers-dense.json", dense, indent=None)
    ppl0 = perplexity(model, data.ppl_gate, {}, "fs_ppl", 4)
    total_channels = sum(u.channels for u in units)
    step = int(fs["step_blocks"])
    info = {"geometry": geo, "step_blocks": step, "step_channels": step * geo["mult"], "total_mlp_channels": total_channels,
            "mask_replay_checks": checks, "ppl0": ppl0, "states": []}
    for i, s in enumerate(states):
        st = PruneState(units, mode)
        if masks[s["name"]] is not None:
            st.masks = {n: m.clone() for n, m in masks[s["name"]].items()}
        st.apply(dense_store)
        act = collect_act_sqnorm(model, st, calib[a_idx], {}, None, int(p.get("activation_batch_size", 4)))
        sens = sensitivity(st, act, float(rc["pruning"].get("sensitivity_percentile", 10)))
        grad = collect_grad_importance(model, st, calib[g_idx])
        save_profile(out / f"profile-{s['name']}.safetensors", act, sens, grad)
        st.save(out / f"mask-{s['name']}.safetensors")
        ppl = perplexity(model, data.ppl_gate, {}, "fs_ppl", 4)
        ans = dense if s["state_id"] is None else _score_pool(model, tok, data, rc, chunk)
        atomic_write_json(out / f"answers-{s['name']}.json", ans, indent=None)
        seed = attempt_order_seed(int(fs.get("order_seed", 0)), int(rc["run"]["seed"]), i + 1)
        view = None if s["state_id"] is None else _gate_replay(rc, data, dense, ans, seed, f"state-{s['name']}")
        zs, zg = zscores(sens), zscores(grad)
        profiles = sorted(({"name": u.name, "share": st.share(u.name), "z_s": zs[u.name], "z_g": zg[u.name],
                            "sparsity": st.unit_sparsity(u.name)} for u in units),
                          key=lambda q: (q["z_s"] if q["z_s"] == q["z_s"] else 1e9, q["name"]))
        from .adaptive.live import CapabilityGate

        gate = CapabilityGate(rc, data.mmlu_probe)
        ctx = {"gate": gate.describe(), "iteration": 1, "max_iterations": int(rc["search"]["max_iterations"]),
               "model_name": rc["model"]["name_or_path"], "mode": mode, "nm": (2, 4), "channel_multiple": geo["mult"],
               "target": float(rc["run"]["target"]), "rho": st.global_sparsity(), "ppl0": ppl0, "ppl": ppl,
               "tau": float(rc["search"]["ppl_rollback_threshold"]), "min_inc": rc["search"]["min_increment"],
               "max_inc": rc["search"]["max_increment"],
               "max_unit": float(rc["pruning"]["structured"]["max_unit_sparsity"]), "profiles": profiles,
               "feedback": None, "history_compact": None,
               "domain_view": {"kind": "capability", "domains": view["domains"] if view else None,
                               "tolerance": gate.settings.tolerance}}
        sl = build_shortlist(profiles, geo, fs["shortlist"], len(units))
        pr = prompts(ctx, geo, step, sl, total_channels)
        info["states"].append({**s, "rho": st.global_sparsity(), "order_seed": seed, "ppl": ppl, "view": view,
                               "exact_loss": _exact(data, field, dense, ans), "profiles": profiles, "shortlist": sl,
                               "greedy_pick": greedy_pick(sl), "prompts": pr})
        log.info("state %s: sparsity %.4f, ppl %.3f, %d shortlist actions, greedy picks %s", s["name"],
                 st.global_sparsity(), ppl, len(sl), greedy_pick(sl))
    PruneState(units, mode).apply(dense_store)
    atomic_write_json(out / "states.json", info)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out / "states.json"


# ============================================================================ controller provenance
def _ollama_base(rc: dict) -> str:
    return rc["controller"]["endpoints"][0]["base_url"].rstrip("/").removesuffix("/v1")


def _get(url: str, body: dict | None = None):
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def record_models(cfg: dict) -> Path:
    """Digest, quantization, size and maximum context of every configured controller (Ollama API)."""
    rc = _rc(cfg)
    base = _ollama_base(rc)
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    try:
        tags = {m["name"]: m for m in _get(f"{base}/api/tags")["models"]}
        server = _get(f"{base}/api/version").get("version")
    except Exception as e:
        tags, server = {}, f"unavailable: {e}"
    for m in _fs(cfg)["controllers"]:
        rec = {"model": m, "server": f"Ollama {server}", "digest": None, "quantization": None, "parameters": None}
        try:
            show = _get(f"{base}/api/show", {"model": m})
            mi = show.get("model_info", {})
            t = tags.get(m) or tags.get(f"{m}:latest") or {}
            rec.update(digest=t.get("digest"), size_bytes=t.get("size"), quantization=show["details"].get("quantization_level"),
                       parameters=show["details"].get("parameter_size"), format=show["details"].get("format"),
                       family=show["details"].get("family"),
                       max_context=next((v for k, v in mi.items() if k.endswith(".context_length")), None))
        except Exception as e:
            rec["error"] = str(e)[:300]
        recs.append(rec)
    atomic_write_json(out / "controller_models.json", recs)
    return out / "controller_models.json"


def _loaded_context(rc: dict, model: str) -> dict | None:
    try:
        for m in _get(f"{_ollama_base(rc)}/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return {k: m.get(k) for k in ("context_length", "size_vram", "size", "digest") if k in m}
    except Exception:
        return None
    return None


# ============================================================================ phase 2: query (one controller)
def parse_choice(text: str) -> dict:
    from .search.controller import _THINK, ParseError

    t = _THINK.sub("", text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    s0, s1 = t.find("{"), t.rfind("}")
    if s0 < 0 or s1 <= s0:
        raise ParseError("no JSON object in response")
    obj = json.loads(t[s0 : s1 + 1])
    return {"reasoning": str(obj.get("reasoning", ""))[:2000], "action_id": obj.get("action_id")}


def normalize_free(decisions: list[tuple[str, float]], profiles: list[dict], geo: dict, step: int) -> dict | None:
    """Rescale sanitized increments to exactly ``step`` blocks (largest remainder, per-unit caps)."""
    cur = {p["name"]: int(round(p["sparsity"] * geo["B"])) for p in profiles}
    w = {u: max(float(inc), 0.0) * geo["B"] for u, inc in decisions if u in cur}
    capu = {u: max(0, min(geo["maxb"], geo["cap_blocks"] - cur[u])) for u in w}
    if sum(capu.values()) < step or not w:
        return None
    alloc = {u: 0 for u in w}
    active = {u for u in w if capu[u] > 0 and w[u] > 0}
    while sum(alloc.values()) < step and active:
        left = step - sum(alloc.values())
        tot = sum(w[u] for u in active)
        quota = {u: left * w[u] / tot for u in active}
        base = {u: min(int(math.floor(quota[u])), capu[u] - alloc[u]) for u in active}
        extra = left - sum(base.values())
        for u in sorted(active, key=lambda u: (-(quota[u] - math.floor(quota[u])), u)):
            if extra <= 0:
                break
            if alloc[u] + base[u] < capu[u]:
                base[u] += 1
                extra -= 1
        for u in active:
            alloc[u] += base[u]
        active = {u for u in active if alloc[u] < capu[u]}
    return {u: b for u, b in alloc.items() if b > 0} if sum(alloc.values()) == step else None


def query(cfg: dict, model: str) -> Path:
    from .config import deep_merge
    from .controller_check import _StubState
    from .search.controller import LLMController, sanitize
    from .utils.cache import KVCache
    from .utils.runtime import ControllerUnavailable

    out = out_dir(cfg)
    info = read_json(out / "states.json")
    fs = _fs(cfg)
    rc = deep_merge(_rc(cfg), {"controller": {"model": model}})
    geo, step = info["geometry"], info["step_blocks"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    cache = KVCache(out / f"controller-{safe}.sqlite")
    ctl = LLMController(rc, cache)
    res_p = out / f"proposals-{safe}.jsonl"
    done = set()
    if res_p.exists():
        done = {(r["state"], r["condition"], r["sample"]) for r in map(json.loads, res_p.read_text(encoding="utf-8").splitlines())}
    t0 = time.time()
    try:  # warm-up so model loading is not counted as decision latency
        ctl.propose(999_999, "Reply with only a JSON object.", 'Return {"ok": true}.')
    except ControllerUnavailable as e:
        raise SystemExit(f"controller {model} unavailable: {e}")
    warm = time.time() - t0
    atomic_write_json(out / f"loaded-{safe}.json", {"model": model, "warmup_seconds": warm,
                                                    "loaded": _loaded_context(rc, model)})
    for si, s in enumerate(info["states"]):
        table = {p["name"]: (p["share"], p["sparsity"]) for p in s["profiles"]}
        ids = {a["id"] for a in s["shortlist"]}
        for ci, cond in enumerate(CONDITIONS):
            pr = s["prompts"][cond]
            for k in range(int(fs["samples"])):
                if (s["name"], cond, k) in done:
                    continue
                key = 1000 * (si + 1) + 100 * ci + k
                rec = {"model": model, "state": s["name"], "condition": cond, "sample": k, "warmup_seconds": warm}
                try:
                    parsed, meta = ctl.propose(key, pr["system"], pr["user"])
                    raw = cache.get(f"iter{key:06d}/a0")["raw"]
                    usage = meta.get("usage") or {}
                    rec.update(latency_s=meta.get("latency_s"), repaired="first" in usage, usage=usage, raw=raw[:3000],
                               replayed=bool(meta.get("replayed")))
                except ControllerUnavailable as e:
                    rec.update(outcome="invalid", reason=f"no parseable reply: {str(e)[:300]}", latency_s=None)
                    _append(res_p, rec)
                    continue
                if cond == "free":
                    if parsed["stop_pruning"] or not parsed["layer_decisions"]:
                        rec.update(outcome="abstain", reason="stop_pruning or no decisions")
                    else:
                        decs, notes = sanitize(parsed, _StubState(table, "structured"), rc, float(rc["run"]["target"]))
                        act = normalize_free(decs, s["profiles"], geo, step)
                        raw_blocks = sum(d["additional_sparsity"] for d in parsed["layer_decisions"]) * geo["B"]
                        rec.update(proposed=parsed["layer_decisions"], notes=notes, proposed_blocks=raw_blocks,
                                   reasoning=parsed["reasoning"][:800])
                        if act is None:
                            rec.update(outcome="invalid", reason=f"cannot realise {step} blocks with <= {geo['maxb']} "
                                                                 f"blocks per known unit ({len(decs)} usable units)")
                        else:
                            rec.update(outcome="valid", action=act, rescale=step / raw_blocks if raw_blocks else None)
                else:
                    try:
                        ch = parse_choice(raw)
                    except Exception as e:
                        rec.update(outcome="invalid", reason=f"unparseable choice: {e}")
                        _append(res_p, rec)
                        continue
                    aid = ch["action_id"]
                    rec["reasoning"] = ch["reasoning"][:800]
                    if aid is None or (isinstance(aid, str) and aid.strip().lower() in ("", "none", "null", "abstain")):
                        rec.update(outcome="abstain", reason="action_id null")
                    elif str(aid).strip() in ids:
                        a = next(x for x in s["shortlist"] if x["id"] == str(aid).strip())
                        rec.update(outcome="valid", action_id=a["id"], action=a["units"])
                    else:
                        rec.update(outcome="invalid", reason=f"unknown action_id {aid!r}")
                _append(res_p, rec)
                log.info("%s %s %s #%d: %s", model, s["name"], cond, k, rec["outcome"])
    cache.close()
    return res_p


def _append(p: Path, rec: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================================ phase 3: score (GPU)
def _canon(action: dict) -> str:
    return ",".join(f"{u}:{b}" for u, b in sorted(action.items(), key=lambda kv: (int(kv[0].split(".")[0][1:]), kv[0])))


def needed_actions(out: Path, info: dict) -> dict[str, dict[str, dict]]:
    need = {s["name"]: {_canon(a["units"]): a["units"] for a in s["shortlist"]} for s in info["states"]}
    for p in sorted(out.glob("proposals-*.jsonl")):
        for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()):
            if r.get("outcome") == "valid":
                need[r["state"]][_canon(r["action"])] = r["action"]
    return need


def score(cfg: dict) -> Path:
    from .pruning.masks import PruneState
    from .pruning.profile import load_profile
    from .pruning.wanda import raise_unit

    out = out_dir(cfg)
    info = read_json(out / "states.json")
    fs = _fs(cfg)
    rc, data, tok, model, units, dense_store, mode = _setup(cfg)
    geo = info["geometry"]
    field = rc["gate"]["capability"].get("domain_field", "category")
    dense = read_json(out / "answers-dense.json")
    res_p = out / "scores.jsonl"
    have = {}
    if res_p.exists():
        for r in map(json.loads, res_p.read_text(encoding="utf-8").splitlines()):
            have[(r["state"], r["action_key"])] = r
    need = needed_actions(out, info)
    dev = next(model.parameters()).device
    by_unit = {u.name: u for u in units}
    for s in info["states"]:
        act, _, _ = load_profile(out / f"profile-{s['name']}.safetensors", dev)
        for key, action in need[s["name"]].items():
            if (s["name"], key) in have:
                continue
            st = PruneState(units, mode)
            st.load(out / f"mask-{s['name']}.safetensors")
            before = {u: int((~st.masks[u]).sum().item()) for u in action}
            for u, b in action.items():
                raise_unit(st, u, (before[u] / geo["mult"] + b) / geo["B"], act, rc)
            realized = {u: (int((~st.masks[u]).sum().item()) - before[u]) // geo["mult"] for u in action}
            if realized != action:
                raise RuntimeError(f"{s['name']} {key}: realised {realized} != requested {action}")
            st.apply(dense_store)
            t0 = time.time()
            ans = _score_pool(model, tok, data, rc, int(fs.get("chunk_size", 512)))
            g = _gate_replay(rc, data, dense, ans, s["order_seed"], f"{s['name']}|{key}")
            ex = _exact(data, field, dense, ans)
            ch = sum(b * geo["mult"] for b in action.values())
            params = sum(b * geo["mult"] * sum(l.weight.shape[1] if p_ != "down" else l.weight.shape[0]
                                               for p_, l in by_unit[u].linears.items()) for u, b in action.items())
            rec = {"state": s["name"], "action_key": key, "action": action, "channels": ch, "parameters": params,
                   "gate": g, "exact_loss": ex, "worst_category": max(ex, key=ex.get), "worst_loss": max(ex.values()),
                   "sparsity_after": st.global_sparsity(), "score_seconds": time.time() - t0}
            _append(res_p, rec)
            log.info("%s %s: %s q=%d worst %s %+.2f pts", s["name"], key, g["decision"], g["questions"],
                     rec["worst_category"], 100 * rec["worst_loss"])
        PruneState(units, mode).apply(dense_store)
    return res_p


# ============================================================================ phase 4: report
def report(cfg: dict) -> Path:
    out = out_dir(cfg)
    info = read_json(out / "states.json")
    scores = {(r["state"], r["action_key"]): r for r in map(json.loads, (out / "scores.jsonl").read_text(encoding="utf-8").splitlines())}
    props = [r for p in sorted(out.glob("proposals-*.jsonl")) for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())]
    seen_models = set(r["model"] for r in props)
    models = [m for m in _fs(cfg)["controllers"] if m in seen_models] + sorted(seen_models - set(_fs(cfg)["controllers"]))
    rows = []  # one per (condition, state, sample)
    for r in props:
        sc = scores.get((r["state"], _canon(r["action"]))) if r.get("outcome") == "valid" else None
        rows.append({**r, "score": sc})
    for s in info["states"]:  # greedy heuristic's pick on the shortlist (deterministic, one per state)
        a = next(x for x in s["shortlist"] if x["id"] == s["greedy_pick"])
        rows.append({"model": "greedy", "state": s["name"], "condition": "shortlist", "sample": 0, "outcome": "valid",
                     "action_id": a["id"], "action": a["units"], "latency_s": 0.0,
                     "score": scores.get((s["name"], _canon(a["units"])))})
    # per-state landscape of the shortlist (for rank / regret)
    land = {}
    for s in info["states"]:
        land[s["name"]] = sorted((scores[(s["name"], _canon(a["units"]))]["worst_loss"], a["id"]) for a in s["shortlist"]
                                 if (s["name"], _canon(a["units"])) in scores)

    def cond_key(r):
        return f"{r['model']} {r['condition']}"

    conds = list(dict.fromkeys(cond_key(r) for r in sorted(rows, key=lambda r: (r["model"] == "greedy", r["condition"], r["model"]))))
    summary = {}
    for c in conds:
        rs = [r for r in rows if cond_key(r) == c]
        valid = [r for r in rs if r["outcome"] == "valid" and r["score"]]
        regrets = []
        for r in valid:
            best = land[r["state"]][0][0] if land[r["state"]] else None
            if best is not None:
                regrets.append(r["score"]["worst_loss"] - best)
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None and r["model"] != "greedy"]
        summary[c] = {"trials": len(rs), "valid": len(valid), "abstain": sum(r["outcome"] == "abstain" for r in rs),
                      "invalid": sum(r["outcome"] == "invalid" for r in rs),
                      "accept": sum(r["score"]["gate"]["decision"] == "accept" for r in valid),
                      "reject": sum(r["score"]["gate"]["decision"] == "reject" for r in valid),
                      "inconclusive": sum(r["score"]["gate"]["decision"] == "inconclusive" for r in valid),
                      "mean_worst_loss": float(np.mean([r["score"]["worst_loss"] for r in valid])) if valid else None,
                      "mean_regret": float(np.mean(regrets)) if regrets else None,
                      "mean_questions": float(np.mean([r["score"]["gate"]["questions"] for r in valid])) if valid else None,
                      "parameters": sorted({r["score"]["parameters"] for r in valid}),
                      "median_latency_s": statistics.median(lat) if lat else None,
                      "layers": _layer_counts(valid)}
    S = {"states": [{k: s[k] for k in ("name", "state_id", "rho", "target", "order_seed", "ppl", "exact_loss", "greedy_pick")}
                    for s in info["states"]], "summary": summary, "rows": rows, "landscape": land,
         "mask_replay_checks": info["mask_replay_checks"], "geometry": info["geometry"], "step_blocks": info["step_blocks"],
         "models": models}
    controllers = out / "controller_models.json"
    S["controller_models"] = read_json(controllers) if controllers.exists() else None
    S["loaded"] = {read_json(p)["model"]: read_json(p) for p in sorted(out.glob("loaded-*.json"))}
    S["max_prompt_tokens"] = {}
    for r in props:
        pt = (r.get("usage") or {}).get("prompt_tokens")
        if pt:
            S["max_prompt_tokens"][r["model"]] = max(pt, S["max_prompt_tokens"].get(r["model"], 0))
    atomic_write_json(out / "summary.json", S)
    atomic_write_text(out / "report.md", render(S, info))
    return out / "report.md"


def compare(S: dict) -> dict:
    """Per state and condition: mean worst-category loss and accepts over the valid samples; plus the shortlist's
    best action and its mean (the expected result of a uniformly random shortlist pick)."""
    by: dict = {}
    for r in S["rows"]:
        if r["outcome"] == "valid" and r.get("score"):
            c = by.setdefault(r["state"], {}).setdefault(f"{r['model']} {r['condition']}", [])
            c.append((r["score"]["worst_loss"], r["score"]["gate"]["decision"] == "accept"))
    out = {}
    for st, conds in by.items():
        ws = [w for w, _ in S["landscape"][st]]
        out[st] = {"best shortlist action": {"loss": min(ws), "accept": None, "n": 1},
                   "random shortlist pick": {"loss": float(np.mean(ws)), "accept": None, "n": len(ws)},
                   **{c: {"loss": float(np.mean([w for w, _ in v])), "accept": sum(a for _, a in v), "n": len(v)}
                      for c, v in conds.items()}}
    return out


def _layer_counts(valid):
    c: dict[str, int] = {}
    for r in valid:
        for u in r["action"]:
            c[u.split(".")[0]] = c.get(u.split(".")[0], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], int(kv[0][1:]))))


def _num(x, fmt: str) -> str:
    return "n/a" if x is None else fmt.format(x)


def render(S: dict, info: dict) -> str:
    pts = lambda x: "n/a" if x is None else f"{100 * x:+.2f}"  # noqa: E731
    g = S["geometry"]
    L = ["# Fixed-state controller comparison", "",
         f"Every action removes exactly {S['step_blocks']} blocks x {g['mult']} = {S['step_blocks'] * g['mult']} MLP channels. "
         "Worst-category loss = exact loss on the full pruning-time pool vs. the dense model (accuracy points); gate "
         "decision and questions = the threshold-aware capability gate replayed with one fixed question order per state.", ""]
    cm = S.get("controller_models")
    if cm:
        L += ["Controllers: " + "; ".join(f"`{m['model']}` ({m.get('quantization') or 'n/a'}, {m.get('parameters') or 'n/a'}, "
                                         f"digest `{(m.get('digest') or 'n/a')[:12]}`, max context {m.get('max_context') or 'n/a'})"
                                         for m in cm) + ".", ""]
    loaded = S.get("loaded") or {}
    if loaded:
        L += ["Runtime context while queried / largest prompt: " + "; ".join(
            f"`{m}` {(v.get('loaded') or {}).get('context_length', 'n/a')} / {S['max_prompt_tokens'].get(m, 'n/a')} tokens "
            f"(load {v.get('warmup_seconds', 0):.0f} s, excluded from latency)" for m, v in loaded.items()) + ".", ""]
    L += ["Mask replay checks (rebuilt states vs. stored masks): " + (", ".join(
        f"{c['state_id']}: {'identical' if c['stored_mask_equals_replay'] else 'DIFFERENT'}" for c in S["mask_replay_checks"]) or "none") + ".", ""]
    L += ["## States", "", "| state | checkpoint | MLP sparsity | pool loss by category (pts) | greedy pick |", "|---|---|---|---|---|"]
    for s in S["states"]:
        L.append(f"| {s['name']} | {s['state_id'] or 'dense'} | {100 * s['rho']:.2f} % | "
                 + ", ".join(f"{c} {100 * v:+.1f}" for c, v in s["exact_loss"].items()) + f" | {s['greedy_pick']} |")
    L += ["", "## Summary over states and samples", "",
          "| condition | trials | valid / abstain / invalid | accept / reject / inconclusive | mean worst-category loss | mean regret vs best shortlist action | mean gate questions | params removed | median latency s | layers chosen (count) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in S["summary"].items():
        params = ", ".join(f"{p:,}" for p in v["parameters"]) or "n/a"
        layers = ", ".join(f"{k}:{n}" for k, n in list(v["layers"].items())[:8])
        L.append(f"| {c} | {v['trials']} | {v['valid']} / {v['abstain']} / {v['invalid']} | {v['accept']} / {v['reject']} / "
                 f"{v['inconclusive']} | {pts(v['mean_worst_loss'])} | {pts(v['mean_regret'])} | "
                 f"{_num(v['mean_questions'], '{:,.0f}')} | {params} | {_num(v['median_latency_s'], '{:.1f}')} | {layers} |")
    C = compare(S)
    states = [s["name"] for s in S["states"]]
    conds = ["best shortlist action", "random shortlist pick"] + list(S["summary"])
    L += ["", "## Mean worst-category loss by state (accuracy points; accepts / valid samples)", "",
          "| condition | " + " | ".join(states) + " | mean over states |", "|---|" + "---|" * (len(states) + 1)]
    for c in conds:
        cells, means = [], []
        for st in states:
            v = C.get(st, {}).get(c)
            if v is None:
                cells.append("n/a")
                continue
            means.append(v["loss"])
            cells.append(f"{pts(v['loss'])}" + ("" if v["accept"] is None else f" ({v['accept']}/{v['n']})"))
        L.append(f"| {c} | " + " | ".join(cells) + f" | {pts(float(np.mean(means))) if means else 'n/a'} |")

    def diff(a, b):
        ds = [C[st][a]["loss"] - C[st][b]["loss"] for st in states if a in C.get(st, {}) and b in C.get(st, {})]
        wins = sum(d < -1e-12 for d in ds), sum(d > 1e-12 for d in ds)
        return ds, wins

    m7, m14 = (S["models"] + [None, None])[:2]
    qs = [("14B beats 7B with the same shortlist?", f"{m14} shortlist", f"{m7} shortlist"),
          ("7B improves when given the shortlist?", f"{m7} shortlist", f"{m7} free"),
          ("14B free vs. 7B free", f"{m14} free", f"{m7} free"),
          ("7B shortlist vs. greedy (same shortlist)", f"{m7} shortlist", "greedy shortlist"),
          ("14B shortlist vs. greedy (same shortlist)", f"{m14} shortlist", "greedy shortlist"),
          ("7B free vs. greedy", f"{m7} free", "greedy shortlist"),
          ("14B free vs. greedy", f"{m14} free", "greedy shortlist"),
          ("greedy vs. random shortlist pick", "greedy shortlist", "random shortlist pick")]
    L += ["", "## Head-to-head (difference in mean worst-category loss, A - B; negative = A better)", "",
          "| question | A | B | " + " | ".join(states) + " | states A better / B better |", "|---|---|---|" + "---|" * (len(states) + 1)]
    for q, a, b in qs:
        if not a or not b:
            continue
        cells = []
        for st in states:
            if a in C.get(st, {}) and b in C.get(st, {}):
                cells.append(pts(C[st][a]["loss"] - C[st][b]["loss"]))
            else:
                cells.append("n/a")
        _, (wa, wb) = diff(a, b)
        L.append(f"| {q} | {a} | {b} | " + " | ".join(cells) + f" | {wa} / {wb} |")
    L += ["", "## By state", ""]
    for s in info["states"]:
        name = s["name"]
        L += [f"### {name} ({100 * s['rho']:.2f} % MLP sparsity)", "",
              "Shortlist landscape (every action scored): " + ", ".join(
                  f"{aid} {pts(w)}" for w, aid in S["landscape"][name]) + "  (worst-category loss, best first).", "",
              "| condition | sample | outcome | action | gate | questions | worst category | rank in shortlist | latency s |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in S["rows"] if x["state"] == name]:
            sc = r.get("score")
            act = r.get("action_id") or (", ".join(f"{u.split('.')[0]}+{b}" for u, b in r["action"].items()) if r.get("action") else
                                         r.get("reason", ""))
            rank = ""
            if sc:
                ws = [w for w, _ in S["landscape"][name]]
                k = 1 + sum(w < sc["worst_loss"] - 1e-12 for w in ws)
                rank = f"{k}/{len(ws)}" if k <= len(ws) else f"worse than all {len(ws)}"
            L.append(f"| {r['model']} {r['condition']} | {r['sample']} | {r['outcome']} | {act} | "
                     f"{sc['gate']['decision'] if sc else ''} | {sc['gate']['questions'] if sc else ''} | "
                     f"{(sc['worst_category'] + ' ' + pts(sc['worst_loss'])) if sc else ''} | {rank} | "
                     f"{'' if r.get('latency_s') is None else round(r['latency_s'], 1)} |")
        L.append("")
    return "\n".join(L) + "\n"
