"""Pruning controllers.

``LLMController`` reproduces the base paper's agent (Sec. 3.3-3.4): it receives
z-scored unit profiles, global/target sparsity, current vs. baseline perplexity
and self-reflection feedback on the previous iteration, and returns JSON
``{reasoning, stop_pruning, layer_decisions:[{layer, additional_sparsity}]}``.
CAP-Prune variants additionally show it the domain-loss estimates and why a
step was rejected.

Reliability:
* every response is stored in a replay cache under ``iter{t}/a{attempt}``; a
  resumed run replays it instead of re-querying a stochastic model;
* endpoints are tried in order with exponential backoff + jitter; by default a
  fallback endpoint must serve the *same* model (``allow_model_switch: false``),
  so the "fixed LLM controller" stays fixed;
* if every endpoint is down the run pauses (exit code 76) and the supervisor
  resumes it later - or, if configured, the deterministic heuristic controller
  stands in and the iteration is labelled ``heuristic_fallback``.

``HeuristicController`` is the non-LLM ablation: least-sensitive-first greedy
selection with a step size adapted from the outcome history (deterministic,
so it is resume-safe without extra state).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ..utils.cache import KVCache
from ..utils.io import sha256_text
from ..utils.log import get_logger
from ..utils.runtime import ControllerUnavailable, check_stop

log = get_logger("controller")


# ============================================================================ prompts
MODE_TEXT = {
    "unstructured": (
        "Unstructured Wanda pruning: additional_sparsity is the extra fraction of the unit's weights "
        "(per output row) to remove this iteration."
    ),
    "nm": (
        "Semi-structured {n}:{m} pruning (NVIDIA sparse tensor cores): listing a unit converts it from dense to "
        "{n}:{m} sparsity ({s:.0%} of its weights) in one step; additional_sparsity is ignored and already-converted "
        "units cannot be pruned further."
    ),
    "structured": (
        "Structured pruning of MLP intermediate channels (gate/up/down pruned jointly, physically shrinking the "
        "layer): additional_sparsity is the extra fraction of the block's channels to remove (rounded to multiples "
        "of {mult} channels)."
    ),
}


def system_prompt(ctx: dict) -> str:
    n, m = ctx["nm"]
    mode = MODE_TEXT[ctx["mode"]].format(n=n, m=m, s=1 - n / m, mult=ctx.get("channel_multiple", 64))
    cap_rule = ""
    if ctx.get("domain_view") is not None:
        cap_rule = (
            " A step is ALSO rejected if targeted knowledge tests show that any knowledge domain's accuracy loss "
            "versus the original model exceeds its limit (absolute limit grows with sparsity progress; a domain may "
            "also not be hurt much more than the others), if the tests are inconclusive within the testing budget, "
            "or if a deployment requirement (latency/memory) fails. Prefer choices that spread damage evenly and "
            "protect the most affected domain."
        )
    return (
        f"You are an expert model-compression agent guiding iterative post-training pruning of the language model "
        f"{ctx['model_name']}. At each iteration you choose which prunable units to prune further and by how much. "
        f"Weights inside a unit are selected by the Wanda score |W| * ||X||_2.\n\n"
        f"Pruning operation: {mode}\n\n"
        "Per-unit statistics:\n"
        "- share: fraction of all eligible weights held by the unit (global sparsity gain = sum(share * added sparsity)).\n"
        "- z_sens: z-score of the unit's Wanda sensitivity (10th percentile of active |W|*||X||). Negative means "
        "below-average sensitivity (safer to prune); positive means riskier.\n"
        "- z_grad: z-score of the mean absolute loss gradient on the unit's active weights. Positive means high loss "
        "impact (riskier).\n"
        "- sparsity: fraction of the unit already pruned.\n\n"
        "Rules:\n"
        f"- additional_sparsity must be in [{ctx['min_inc']}, {ctx['max_inc']}]; a unit's total sparsity may not "
        f"exceed {ctx['max_unit']}.\n"
        "- Use unit names exactly as listed.\n"
        f"- After each step, if perplexity rises by more than {ctx['tau']:.0%} relative to the previous accepted "
        f"model, the step is rolled back.{cap_rule}\n"
        "- Make steady progress toward the target sparsity; do not overshoot it.\n"
        "- Analyse the feedback on your previous decisions and refine your strategy: repeat what worked, become more "
        "conservative after poor outcomes, accelerate when quality stays stable.\n\n"
        'Respond with ONLY a JSON object of the form {"reasoning": "...", "stop_pruning": false, '
        '"layer_decisions": [{"layer": "<unit name>", "additional_sparsity": 0.05}]}.'
    )


def user_prompt(ctx: dict) -> str:
    lines = [
        f"Iteration {ctx['iteration']} (at most {ctx['max_iterations']}).",
        f"Current global sparsity: {ctx['rho']:.4f}. Target sparsity: {ctx['target']:.4f}. "
        f"Remaining gap: {max(0.0, ctx['target'] - ctx['rho']):.4f}.",
        f"Baseline perplexity (original model): {ctx['ppl0']:.4f}. Current perplexity: {ctx['ppl']:.4f} "
        f"({(ctx['ppl'] / ctx['ppl0'] - 1) * 100:+.2f}% vs. baseline).",
    ]
    dv = ctx.get("domain_view")
    if dv is not None:
        lines.append("")
        if dv.get("domains"):
            lines.append(
                f"Estimated accuracy loss per knowledge domain for the current accepted model (vs. original; "
                f"mean [90% interval]); absolute limit at the next step ~{dv['limit_next']:.3f}, imbalance limit "
                f"{dv['limit_rel']:.3f}:")
            for d, s in dv["domains"].items():
                lines.append(f"  {d}: {s['mean']:+.3f} [{s['lo90']:+.3f}, {s['hi90']:+.3f}]")
        else:
            lines.append(f"No domain test has been run yet. Absolute domain-loss limit at the next step "
                         f"~{dv['limit_next']:.3f}; imbalance limit {dv['limit_rel']:.3f}.")
    fb = ctx.get("feedback")
    if fb:
        lines += ["", "Feedback on your previous decision:"]
        lines.append(f"  Iteration {fb['iteration']} reasoning: {fb.get('reasoning', '')[:1200]}")
        dec = ", ".join(f"{u}+{d:.3f}" for u, d in fb.get("decisions", [])[:60]) or "(none)"
        lines.append(f"  Selections: {dec}")
        lines.append(f"  Outcome: {fb['outcome'].upper()}. Sparsity gain: {fb['sparsity_gain']:+.4f}. "
                     f"Perplexity change: {fb['ppl_change_pct']:+.2f}%.")
        if fb.get("domain_note"):
            lines.append(f"  Domain tests: {fb['domain_note']}")
        lines.append(f"  Assessment: {fb['assessment']}")
    hist = ctx.get("history_compact")
    if hist:
        lines += ["", "Earlier iterations (iteration: outcome, sparsity gain, PPL change):"]
        lines += [f"  {h}" for h in hist]
    lines += ["", "Unit profiles (sorted by z_sens, safest first):", "name | share | z_sens | z_grad | sparsity"]
    for p in ctx["profiles"]:
        zs = "nan" if p["z_s"] != p["z_s"] else f"{p['z_s']:+.2f}"
        zg = "nan" if p["z_g"] != p["z_g"] else f"{p['z_g']:+.2f}"
        lines.append(f"{p['name']} | {p['share']:.4f} | {zs} | {zg} | {p['sparsity']:.3f}")
    return "\n".join(lines)


def assessment(outcome: str, gain: float, ppl_pct: float, reasons: list[str]) -> str:
    """Qualitative self-reflection label (paper Sec. 3.4/3.5)."""
    if outcome == "rollback":
        if "no_change" in reasons:
            return ("No change - the requested increments did not change any unit (too small after rounding, or "
                    "units already at their cap/converted). Choose other units or larger increments.")
        if "ppl" in reasons or "nan" in reasons:
            return "Poor - Excessive PPL degradation, consider more conservative approach."
        if "domain_fail" in reasons:
            return "Poor - Knowledge-domain accuracy loss exceeded its limit; avoid units likely tied to the affected domains and use smaller increments."
        if "domain_inconclusive" in reasons:
            return "Uncertain - The testing budget could not confirm that all domains are safe; take a smaller, safer step."
        if "deploy" in reasons:
            return "Poor - Deployment requirement failed (latency/memory); prefer changes that actually speed up inference."
        return "Poor - Step rejected."
    if gain < 0.005:
        return "Too conservative - Minimal sparsity gain; you can prune more aggressively."
    if ppl_pct < 2.0 and gain >= 0.02:
        return "Excellent - High sparsity gain with minimal PPL impact."
    if ppl_pct < 5.0:
        return "Good - Reasonable trade-off between sparsity gain and PPL impact."
    return "Fair - Accepted, but the PPL impact was notable; consider a more conservative approach."


# ============================================================================ parsing / sanitizing
class ParseError(ValueError):
    pass


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_response(text: str) -> dict:
    t = _THINK.sub("", text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ParseError("no JSON object in response")
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("response is not a JSON object")
    decs = obj.get("layer_decisions", [])
    if not isinstance(decs, list):
        raise ParseError("layer_decisions must be a list")
    clean = []
    for d in decs:
        if not isinstance(d, dict) or "layer" not in d:
            raise ParseError("each decision needs 'layer' and 'additional_sparsity'")
        try:
            v = float(d.get("additional_sparsity", 0.0))
        except (TypeError, ValueError) as e:
            raise ParseError("additional_sparsity must be a number") from e
        clean.append({"layer": str(d["layer"]).strip(), "additional_sparsity": v})
    stop = obj.get("stop_pruning", False)
    if isinstance(stop, str):
        stop = stop.strip().lower() in ("true", "yes", "1")
    return {"reasoning": str(obj.get("reasoning", ""))[:4000], "stop_pruning": bool(stop), "layer_decisions": clean}


def sanitize(parsed: dict, state, cfg: dict, target: float) -> tuple[list[tuple[str, float]], list[str]]:
    """Clamp, de-duplicate and scale decisions so they are valid and do not overshoot the target."""
    s = cfg["search"]
    lo, hi = float(s["min_increment"]), float(s["max_increment"])
    tol = float(s.get("target_tolerance", 0.005))
    mode = state.mode
    cap = (float(cfg["pruning"].get("max_unit_sparsity", 0.9)) if mode == "unstructured"
           else float(cfg["pruning"].get("structured", {}).get("max_unit_sparsity", 0.875)))
    notes, merged = [], {}
    for d in parsed.get("layer_decisions", []):
        name = d["layer"]
        if name not in state.by_name:
            notes.append(f"unknown unit '{name}' ignored")
            continue
        merged[name] = merged.get(name, 0.0) + float(d["additional_sparsity"])
    rho = state.global_sparsity()
    room = max(0.0, target + tol - rho)
    out: list[tuple[str, float]] = []
    if mode == "nm":
        n, m = state.nm
        per = 1 - n / m
        gain = 0.0
        for name in merged:  # keep the agent's order
            if state.unit_sparsity(name) > 0:
                notes.append(f"{name} already {n}:{m}; ignored")
                continue
            g = per * state.share(name)
            if gain + g > room + 1e-12 and out:
                notes.append(f"{name} dropped to avoid overshooting the target")
                continue
            out.append((name, per))
            gain += g
        return out, notes
    for name, inc in merged.items():
        inc2 = min(max(inc, lo), hi)
        if inc2 != inc:
            notes.append(f"{name}: increment {inc:.3f} clamped to {inc2:.3f}")
        cur = state.unit_sparsity(name)
        inc2 = min(inc2, cap - cur)
        if inc2 <= 1e-6:
            notes.append(f"{name} already at the per-unit cap")
            continue
        out.append((name, inc2))
    gain = sum(state.share(n_) * d for n_, d in out)
    if gain > room > 0:
        f = room / gain
        out = [(n_, d * f) for n_, d in out]
        notes.append(f"increments scaled by {f:.3f} to avoid overshooting the target")
    return out, notes


# ============================================================================ HTTP clients
class _HTTPError(Exception):
    def __init__(self, status: int | None, msg: str, retryable: bool):
        super().__init__(msg)
        self.status, self.retryable = status, retryable


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise _HTTPError(e.code, f"HTTP {e.code}: {detail}", retryable=e.code in (408, 409, 425, 429) or e.code >= 500)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise _HTTPError(None, f"network error: {e}", retryable=True)
    except json.JSONDecodeError as e:
        raise _HTTPError(None, f"non-JSON response: {e}", retryable=True)


def call_endpoint(ep: dict, model: str, system: str, user: str, temperature: float, max_tokens: int,
                  seed: int | None, timeout: float, use_response_format: bool = True) -> tuple[str, dict]:
    api = ep.get("api", "openai")
    key = os.environ.get(ep["api_key_env"], "") if ep.get("api_key_env") else ""
    if ep.get("api_key_env") and not key:
        raise _HTTPError(None, f"environment variable {ep['api_key_env']} is not set", retryable=False)
    if api == "anthropic":
        url = (ep.get("base_url") or "https://api.anthropic.com/v1").rstrip("/") + "/messages"
        body = {"model": model, "system": system, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": user}]}
        r = _post(url, {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, timeout)
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        return text, r.get("usage", {})
    url = ep["base_url"].rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if seed is not None and ep.get("send_seed", True):
        body["seed"] = int(seed)
    if use_response_format and ep.get("response_format", "json_object"):
        body["response_format"] = {"type": ep.get("response_format", "json_object")}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = _post(url, headers, body, timeout)
    try:
        text = r["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise _HTTPError(None, f"unexpected response shape: {str(r)[:300]}", retryable=True) from e
    return text, r.get("usage", {})


# ============================================================================ controllers
class LLMController:
    kind = "llm"

    def __init__(self, cfg: dict, cache: KVCache, heartbeat=None):
        c = cfg["controller"]
        self.model = c["model"]
        self.temperature = float(c.get("temperature", 0.5))
        self.max_tokens = int(c.get("max_tokens", 4096))
        self.retries = int(c.get("retries", 6))
        self.timeout = float(c.get("timeout_s", 180))
        self.backoff = list(c.get("backoff_s", [5, 15, 45, 120, 300, 600]))
        allow_switch = bool(c.get("allow_model_switch", False))
        eps = c.get("endpoints") or []
        self.endpoints = [e for e in eps if allow_switch or e.get("model", self.model) == self.model]
        if not self.endpoints:
            raise ValueError("controller.endpoints is empty (or no endpoint serves controller.model)")
        self.cache = cache
        self.hb = heartbeat
        self.seed = cfg["run"]["seed"]

    def propose(self, iteration: int, system: str, user: str, attempt: int = 0) -> tuple[dict, dict]:
        key = f"iter{iteration:06d}/a{attempt}"
        prompt_sha = sha256_text(system + "\n\x1e\n" + user)
        hit = self.cache.get(key)
        if hit is not None:
            if hit.get("prompt_sha") != prompt_sha:
                log.warning("iteration %d: prompt differs from the cached one; replaying cached decision for determinism", iteration)
            return hit["parsed"], {**hit["meta"], "replayed": True}
        rng = random.Random(f"{self.seed}:{iteration}:{attempt}")
        last_err = "no attempt"
        dead: set[int] = set()  # endpoints with permanent errors (bad key, 401/403/404) are not retried
        for round_ in range(self.retries):
            if len(dead) == len(self.endpoints):
                break
            for i_ep, ep in enumerate(self.endpoints):
                if i_ep in dead:
                    continue
                model = ep.get("model", self.model)
                check_stop()
                if self.hb:
                    self.hb.beat("controller", f"iter {iteration} {ep.get('name', ep.get('base_url'))} try {round_ + 1}")
                t0 = time.time()
                try:
                    text, usage = self._call(ep, model, system, user, iteration)
                    try:
                        parsed = parse_response(text)
                    except ParseError as pe:
                        fix = (user + "\n\nYour previous reply could not be parsed (" + str(pe) +
                               "). Reply with ONLY the JSON object.")
                        text, usage2 = self._call(ep, model, system, fix, iteration)
                        parsed = parse_response(text)
                        usage = {"first": usage, "repair": usage2}
                except ParseError as pe:
                    last_err = f"{ep.get('name')}: unparseable response: {pe}"
                    log.warning(last_err)
                    continue
                except _HTTPError as he:
                    last_err = f"{ep.get('name')}: {he}"
                    log.warning("controller endpoint %s failed: %s", ep.get("name"), he)
                    if not he.retryable:
                        dead.add(i_ep)
                    continue
                meta = {"endpoint": ep.get("name", ep.get("base_url")), "model": model, "usage": usage,
                        "latency_s": round(time.time() - t0, 3), "replayed": False, "prompt_sha": prompt_sha}
                self.cache.put(key, {"parsed": parsed, "meta": meta, "prompt_sha": prompt_sha,
                                     "raw": text[:20000], "system": system, "user": user})
                return parsed, meta
            if len(dead) == len(self.endpoints) or round_ == self.retries - 1:
                break
            wait = self.backoff[min(round_, len(self.backoff) - 1)] * (0.75 + 0.5 * rng.random())
            log.warning("all controller endpoints failed (round %d/%d); waiting %.0fs", round_ + 1, self.retries, wait)
            end = time.time() + wait
            while time.time() < end:
                check_stop()
                if self.hb:
                    self.hb.beat("controller", f"backoff {int(end - time.time())}s")
                time.sleep(min(2.0, max(0.0, end - time.time())))
        raise ControllerUnavailable(last_err)

    def _call(self, ep, model, system, user, iteration):
        try:
            return call_endpoint(ep, model, system, user, self.temperature, self.max_tokens,
                                 self.seed * 100_003 + iteration, self.timeout)
        except _HTTPError as he:
            if he.status == 400:  # some servers reject response_format/seed: retry once without them
                return call_endpoint({**ep, "send_seed": False}, model, system, user, self.temperature,
                                     self.max_tokens, None, self.timeout, use_response_format=False)
            raise


class HeuristicController:
    """Non-LLM controller for ablations (and optional fallback)."""

    kind = "heuristic"

    def __init__(self, cfg: dict):
        h = cfg["controller"].get("heuristic", {})
        self.g0 = float(h.get("initial_gain", 0.02))
        self.grow = float(h.get("grow", 1.25))
        self.shrink = float(h.get("shrink", 0.5))
        self.gmin = float(h.get("min_gain", 0.004))
        self.gmax = float(h.get("max_gain", 0.05))
        self.inc = float(h.get("unit_increment", 0.05))
        self.tabu = int(h.get("tabu_iterations", 2))
        self.cfg = cfg

    def _gain(self, history: list[dict]) -> float:
        g = self.g0
        for h in history:
            if h["outcome"] == "accept" and h.get("ppl_change_pct", 0) < 2.0:
                g *= self.grow
            elif h["outcome"] == "rollback":
                g *= self.shrink
            g = min(self.gmax, max(self.gmin, g))
        return g

    def propose(self, ctx: dict, history: list[dict]) -> dict:
        gain_goal = min(self._gain(history), max(0.0, ctx["target"] - ctx["rho"]))
        tabu = set()
        for h in history[-self.tabu:]:
            if h["outcome"] == "rollback":
                tabu.update(u for u, _ in h.get("decisions", []))
        cands = [p for p in ctx["profiles"] if p["name"] not in tabu and p["sparsity"] < ctx["max_unit"] - 1e-6]
        if ctx["mode"] == "nm":
            cands = [p for p in cands if p["sparsity"] == 0]

        def risk(p):
            zs = p["z_s"] if p["z_s"] == p["z_s"] else 0.0
            zg = p["z_g"] if p["z_g"] == p["z_g"] else 0.0
            return 0.5 * (zs + zg)

        cands.sort(key=lambda p: (risk(p), p["name"]))
        n, m = ctx["nm"]
        decisions, acc = [], 0.0
        for p in cands:
            if acc >= gain_goal - 1e-9:
                break
            inc = (1 - n / m) if ctx["mode"] == "nm" else min(self.inc, ctx["max_unit"] - p["sparsity"])
            decisions.append({"layer": p["name"], "additional_sparsity": round(inc, 4)})
            acc += p["share"] * inc
        return {"reasoning": f"heuristic: least-sensitive-first, gain goal {gain_goal:.4f}",
                "stop_pruning": not decisions, "layer_decisions": decisions}
