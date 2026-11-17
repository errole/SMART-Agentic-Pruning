"""The iterative agent-guided pruning loop (paper Algorithm 1) with CAP-Prune's gate.

Per iteration t:
  1. profile the accepted model (Wanda sensitivity z-scores; gradient z-scores every 3rd iteration)
  2. ask the controller for layer decisions (LLM with self-reflection feedback, or heuristic)
  3. build the candidate masks deterministically from (accepted masks, decisions, profile)
  4. gate:  (a) PPL rollback rule (paper, tau = 15 %)
            (b) CAP only: budgeted sequential domain test  -> pass / fail / inconclusive
            (c) CAP only, hardware modes: deployment check (latency / memory vs. parent)
  5. accept (write content-addressed masks) or roll back; record feedback; checkpoint.

Crash safety: everything an iteration needs that is expensive or stochastic -
the profile, the controller response, every probe answer, the candidate's
perplexity and benchmark - is persisted or cached *before* it is used, keyed
by iteration / candidate id.  A crash at any instant therefore resumes from the
last checkpoint and replays the interrupted iteration to the identical outcome.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..config import loss_limit_final
from ..eval.mmlu import MMLUScorer
from ..eval.perplexity import perplexity
from ..pruning.profile import (collect_act_sqnorm, collect_grad_importance, load_profile, save_profile,
                               sensitivity, zscores)
from ..pruning.wanda import raise_unit
from ..utils.faults import fault_point
from ..utils.io import append_jsonl, short_hash
from ..utils.log import get_logger
from ..utils.runtime import ControllerUnavailable, check_stop
from .controller import HeuristicController, LLMController, assessment, sanitize, system_prompt, user_prompt
from .testing import FAIL, INCONCLUSIVE, Limits, loss_limit, sequential_test

log = get_logger("search")


def initial_search_state(ppl0: float, bench0: dict | None) -> dict:
    return {
        "iteration": 1,
        "state_id": None,  # None = dense (all-ones masks)
        "rho": 0.0,
        "ppl0": ppl0,
        "ppl": ppl0,
        "history": [],
        "consecutive_rejections": 0,
        "rollbacks": 0,
        "accepts": 0,
        "stop_reason": None,
        "grad_scores": None,
        "grad_iteration": None,
        "domain_view": None,
        "parent_bench": bench0,
        "budget": {"probe_questions": 0, "probe_tests": 0, "early_stops": 0, "controller_calls": 0,
                   "controller_replays": 0, "controller_tokens_in": 0, "controller_tokens_out": 0,
                   "ppl_evals": 0, "deploy_checks": 0, "heuristic_fallbacks": 0},
    }


class Search:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rc = ctx.rc
        self.S = ctx.S["search"]
        s = self.rc["search"]
        self.T = int(s["max_iterations"])
        self.tau = float(s["ppl_rollback_threshold"])
        self.tol = float(s.get("target_tolerance", 0.005))
        self.patience = s.get("patience")
        self.target = float(self.rc["run"]["target"])
        self.policy = self.rc["testing"]["policy"]
        ck = self.rc["controller"]["kind"]
        self.heur = HeuristicController(self.rc)
        self.llm = LLMController(self.rc, ctx.ctrl_cache, ctx.hb) if ck == "llm" else None
        t = self.rc["testing"]
        self.eps_final = loss_limit_final(self.rc)
        self.scorer = None
        if self.policy != "none":
            self.scorer = MMLUScorer(ctx.model, ctx.tok, ctx.data.mmlu_fewshot, int(t.get("shots", 0)),
                                     int(self.rc["model"].get("max_seq_len", 2048)), ctx.S["batch"], "probe_tokens",
                                     int(t.get("max_batch_tokens", 32768)), ctx.hb)

    # ------------------------------------------------------------------ helpers
    def _journal(self, rec: dict) -> None:
        append_jsonl(self.ctx.run_dir / "journal.jsonl", {"ts": time.time(), **rec})

    def _limits(self, rho: float) -> Limits:
        t = self.rc["testing"]
        ll = t["loss_limit"]
        return Limits(eps_abs=loss_limit(rho, self.target, self.eps_final, float(ll.get("floor", 0.02)),
                                         ll.get("schedule", "linear")),
                      eps_rel=float(t.get("imbalance_limit", 0.06)),
                      eps_rel_min=float(t.get("imbalance_min_loss", 0.02)))

    def _done(self) -> str | None:
        S = self.S
        if S["stop_reason"]:
            return S["stop_reason"]
        if S["rho"] >= self.target - self.tol:
            return "target_reached"
        if S["iteration"] > self.T:
            return "max_iterations"
        if self.patience and S["consecutive_rejections"] >= int(self.patience):
            return "patience_exhausted"
        if self.policy != "none":
            t = self.rc["testing"]
            left = int(t["total_budget"]) - S["budget"]["probe_questions"]
            if left < int(t["min_per_domain"]) * len(self.ctx.data.domains):
                return "probe_budget_exhausted"
        return None

    # ------------------------------------------------------------------ profile
    def _profile(self, t: int):
        ctx = self.ctx
        path = ctx.run_dir / "profiles" / f"iter-{t:06d}.safetensors"
        dev = next(ctx.model.parameters()).device
        if path.exists():
            try:
                return load_profile(path, dev)
            except Exception as e:
                log.warning("profile for iteration %d unreadable (%s); recomputing", t, e)
        p = self.rc["profiling"]
        calib = ctx.calib
        rng = np.random.default_rng([self.rc["run"]["seed"], t, 17])
        a_idx = np.sort(rng.choice(calib.shape[0], size=min(int(p["activation_samples"]), calib.shape[0]), replace=False))
        ctx.hb.beat("profile", f"iter {t} activations")
        act = collect_act_sqnorm(ctx.model, ctx.state, calib[a_idx], ctx.S["batch"], ctx.hb,
                                 int(p.get("activation_batch_size", 4)))
        sens = sensitivity(ctx.state, act, float(self.rc["pruning"].get("sensitivity_percentile", 10)))
        every = int(p.get("gradient_every", 3))
        S = self.S
        if S["grad_scores"] is None or (t - 1) % every == 0:
            g_idx = np.sort(rng.choice(calib.shape[0], size=min(int(p["gradient_samples"]), calib.shape[0]), replace=False))
            ctx.hb.beat("profile", f"iter {t} gradients")
            grad = collect_grad_importance(ctx.model, ctx.state, calib[g_idx], ctx.hb)
        else:
            grad = dict(S["grad_scores"])
        save_profile(path, act, sens, grad)  # persist BEFORE use -> identical replay after a crash
        return act, sens, grad

    # ------------------------------------------------------------------ controller context
    def _ctx_for_controller(self, t: int, sens: dict, grad: dict) -> dict:
        st = self.ctx.state
        zs, zg = zscores(sens), zscores(grad)
        profiles = [{"name": u.name, "share": st.share(u.name), "z_s": zs[u.name], "z_g": zg[u.name],
                     "sparsity": st.unit_sparsity(u.name)} for u in st.units]
        profiles.sort(key=lambda p: (p["z_s"] if p["z_s"] == p["z_s"] else 1e9, p["name"]))
        mode = st.mode
        max_unit = (float(self.rc["pruning"].get("max_unit_sparsity", 0.9)) if mode == "unstructured"
                    else float(self.rc["pruning"].get("structured", {}).get("max_unit_sparsity", 0.875)) if mode == "structured"
                    else 1 - st.nm[0] / st.nm[1])
        S = self.S
        hist = S["history"]
        win = int(self.rc["controller"].get("history_window", 1))
        dv = None
        if self.policy != "none":
            nxt = min(self.target, S["rho"] + 0.02)
            lim = self._limits(nxt)
            dv = {"domains": S["domain_view"], "limit_next": lim.eps_abs, "limit_rel": lim.eps_rel}
        return {
            "iteration": t, "max_iterations": self.T, "model_name": self.rc["model"]["name_or_path"],
            "mode": mode, "nm": st.nm, "channel_multiple": self.rc["pruning"].get("structured", {}).get("channel_multiple", 64),
            "target": self.target, "rho": S["rho"], "ppl0": S["ppl0"], "ppl": S["ppl"], "tau": self.tau,
            "min_inc": self.rc["search"]["min_increment"], "max_inc": self.rc["search"]["max_increment"],
            "max_unit": max_unit, "profiles": profiles,
            "feedback": hist[-1]["feedback"] if hist else None,
            "history_compact": [h["compact"] for h in hist[-win:-1]] if win > 1 else None,
            "domain_view": dv,
        }

    def _propose(self, t: int, cctx: dict) -> tuple[dict, dict]:
        b = self.S["budget"]
        if self.llm is None:
            return self.heur.propose(cctx, self.S["history"]), {"source": "heuristic"}
        t0 = time.time()
        try:
            try:
                parsed, meta = self.llm.propose(t, system_prompt(cctx), user_prompt(cctx))
            finally:  # API latency and backoff waits are not GPU time
                b["controller_seconds"] = b.get("controller_seconds", 0.0) + (time.time() - t0)
        except ControllerUnavailable as e:
            if self.rc["controller"].get("on_unavailable", "pause") == "heuristic":
                log.error("controller unavailable (%s); heuristic fallback for iteration %d", e, t)
                b["heuristic_fallbacks"] += 1
                return self.heur.propose(cctx, self.S["history"]), {"source": "heuristic_fallback", "error": str(e)}
            raise
        if meta.get("replayed"):
            b["controller_replays"] += 1
        b["controller_calls"] += 1
        usages = [meta.get("usage") or {}]
        if "first" in usages[0]:  # a JSON-repair re-ask happened: count both calls
            usages = [usages[0].get("first") or {}, usages[0].get("repair") or {}]
        for u in usages:
            b["controller_tokens_in"] += int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
            b["controller_tokens_out"] += int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
        return parsed, {"source": "llm", **{k: meta[k] for k in ("endpoint", "model", "replayed") if k in meta}}

    # ------------------------------------------------------------------ evaluation pieces
    def _ppl(self, cand_id: str) -> float:
        key = f"ppl|{cand_id}"
        hit = self.ctx.eval_cache.get(key)
        if hit is not None:
            return float(hit)
        self.ctx.hb.beat("gate", "perplexity")
        v = perplexity(self.ctx.model, self.ctx.data.ppl_gate, self.ctx.S["batch"], "ppl_gate_bs", 4, self.ctx.hb)
        self.ctx.eval_cache.put(key, v if math.isfinite(v) else "nan")
        return v

    def _probe_eval(self, cand_id: str):
        cache = self.ctx.eval_cache

        def ev(qs: list[dict]) -> dict[str, bool]:
            keys = {q["qid"]: f"probe|{cand_id}|{q['qid']}" for q in qs}
            have = cache.get_many(keys.values())
            todo = [q for q in qs if keys[q["qid"]] not in have]
            if todo:
                check_stop()
                res = self.scorer.score(todo)
                cache.put_many({keys[q]: r["correct"] for q, r in res.items()})
                fault_point("mid_probe")
                have.update({keys[q]: r["correct"] for q, r in res.items()})
            return {q["qid"]: bool(have[keys[q["qid"]]]) for q in qs}

        return ev

    def _deploy(self, cand_id: str) -> tuple[bool, dict]:
        d = self.rc.get("deploy", {})
        if d.get("check", "off") != "per_step" or self.ctx.state.mode == "unstructured":
            return True, {"skipped": True}
        key = f"bench|{cand_id}"
        res = self.ctx.eval_cache.get(key)
        if res is None:
            from ..eval.bench import benchmark
            from ..pruning.materialize import materialize

            q = d.get("quick", {})
            self.ctx.hb.beat("gate", "deployment check")
            m, info = materialize(self.ctx.model, self.ctx.state, self.rc)
            try:
                res = benchmark(m, self.ctx.model.config.vocab_size, [int(q.get("prompt_len", 512))],
                                int(q.get("gen_tokens", 32)), 1, int(q.get("warmup", 2)), int(q.get("repeats", 5)),
                                heartbeat=self.ctx.hb)
                res["materialize"] = info
            finally:
                del m
                from ..utils.gpu import free_memory

                free_memory()
            self.ctx.eval_cache.put(key, res)
        self.S["budget"]["deploy_checks"] += 1
        parent = self.S.get("parent_bench")
        ok, why = True, []
        if parent:
            tol = float(d.get("max_latency_ratio_vs_parent", 1.03))
            r = next(iter(res["by_prompt_len"].values()))
            p = next(iter(parent["by_prompt_len"].values()))
            for metric in d.get("require", ["prefill", "decode"]):
                k = "prefill_ms_median" if metric == "prefill" else "decode_ms_per_token_median"
                if r[k] > p[k] * tol:
                    ok = False
                    why.append(f"{metric} {r[k]:.2f}ms > {p[k]:.2f}ms x {tol}")
            cap = d.get("max_peak_memory_gb")
            if cap and r.get("peak_allocated_bytes", 0) > float(cap) * (1 << 30):
                ok = False
                why.append("peak memory above cap")
        return ok, {"bench": res, "reasons": why}

    # ------------------------------------------------------------------ main loop
    def run(self) -> str:
        ctx, S = self.ctx, self.S
        while True:
            reason = self._done()
            if reason:
                S["stop_reason"] = reason
                ctx.checkpoint()
                log.info("search finished: %s (sparsity %.4f, ppl %.3f, %d accepts, %d rollbacks)",
                         reason, S["rho"], S["ppl"], S["accepts"], S["rollbacks"])
                return reason
            check_stop()
            t = S["iteration"]
            ctx.hb.beat("search", f"iteration {t}")
            fault_point("hang_profile")
            act, sens, grad = self._profile(t)
            fault_point("after_profile")
            S["grad_scores"] = grad
            cctx = self._ctx_for_controller(t, sens, grad)
            check_stop()
            parsed, src = self._propose(t, cctx)
            fault_point("after_controller")
            decisions, notes = sanitize(parsed, ctx.state, self.rc, self.target)
            rec = {"iteration": t, "controller": src, "reasoning": parsed.get("reasoning", "")[:2000],
                   "decisions": [(u, round(d, 5)) for u, d in decisions], "notes": notes}
            if parsed.get("stop_pruning") and src.get("source") != "heuristic_fallback":
                rec.update(outcome="stop")
                self._journal({"type": "iteration", **rec})
                S["stop_reason"] = "agent_stop"
                S["iteration"] = t + 1
                ctx.checkpoint()
                continue
            # ---- build candidate deterministically
            parent_masks = {n: m for n, m in ctx.state.masks.items()}
            changed = []
            for name, inc in decisions:
                new_ratio = ctx.state.unit_sparsity(name) + inc
                if raise_unit(ctx.state, name, new_ratio, act, self.rc):
                    changed.append(name)
            cand_id = short_hash(S["state_id"], t, rec["decisions"])
            rho_c = ctx.state.global_sparsity()
            gain = rho_c - S["rho"]
            if not changed:
                outcome, reasons, ppl_c, test, dep = "noop", ["no_change"], S["ppl"], None, None
            else:
                ctx.state.apply(ctx.dense, only=changed)
                fault_point("after_candidate_apply")
                outcome, reasons, ppl_c, test, dep = self._gate(t, cand_id, rho_c)
            rho_prev, ppl_prev = S["rho"], S["ppl"]
            ppl_pct = (ppl_c / ppl_prev - 1) * 100 if (ppl_c == ppl_c and math.isfinite(ppl_c)) else float("inf")
            if outcome == "accept":
                ctx.save_state_masks(cand_id)
                fault_point("after_masks_saved")
                S.update(state_id=cand_id, rho=rho_c, ppl=ppl_c, accepts=S["accepts"] + 1, consecutive_rejections=0)
                if test is not None:
                    S["domain_view"] = {d: {k: round(v[k], 4) for k in ("mean", "lo90", "hi90")} for d, v in test.domains.items()}
                if dep and dep.get("bench"):
                    S["parent_bench"] = dep["bench"]
            else:
                # roll back: restore parent masks and weights of the changed units
                for n in changed:
                    ctx.state.masks[n] = parent_masks[n]
                if changed:
                    ctx.state.apply(ctx.dense, only=changed)
                S["consecutive_rejections"] += 1
                if outcome == "rollback":
                    S["rollbacks"] += 1
            fb_outcome = "accept" if outcome == "accept" else "rollback"
            domain_note = None
            if test is not None:
                worst = max(test.domains.items(), key=lambda kv: kv[1]["mean"])
                p_max = max(v["p_violate"] for v in test.domains.values())
                domain_note = (f"{test.decision} after {test.used} questions (highest per-domain P(limit violated)={p_max:.2f}); "
                               f"most affected: {worst[0]} {worst[1]['mean']:+.3f}")
            feedback = {"iteration": t, "reasoning": rec["reasoning"], "decisions": rec["decisions"],
                        "outcome": fb_outcome, "sparsity_gain": gain, "ppl_change_pct": ppl_pct if math.isfinite(ppl_pct) else 999.0,
                        "domain_note": domain_note, "assessment": assessment(fb_outcome, gain, ppl_pct, reasons)}
            rec.update(cand_id=cand_id, outcome=outcome, reasons=reasons, rho_before=round(rho_prev, 6),
                       ppl_before=ppl_prev, rho_candidate=round(rho_c, 6), ppl_candidate=ppl_c,
                       ppl_change_pct=feedback["ppl_change_pct"],
                       test=test.to_json() if test is not None else None,
                       deploy={k: v for k, v in (dep or {}).items() if k != "bench"},
                       feedback=feedback,
                       compact=f"{t}: {fb_outcome}, gain {gain:+.4f}, PPL {feedback['ppl_change_pct']:+.2f}%")
            S["history"].append({k: rec[k] for k in ("iteration", "outcome", "reasons", "decisions", "feedback", "compact",
                                                     "ppl_change_pct", "rho_candidate", "cand_id")})
            self._journal({"type": "iteration", **rec})
            log.info("iter %d: %s %s | rho %.4f -> %.4f | ppl %+.2f%%%s", t, outcome.upper(), ",".join(reasons) or "",
                     rho_prev, rho_c, feedback["ppl_change_pct"],
                     f" | test {test.decision} n={test.used}" if test is not None else "")
            S["iteration"] = t + 1
            ctx.checkpoint()

    def _gate(self, t: int, cand_id: str, rho_c: float):
        S = self.S
        S["budget"]["ppl_evals"] += 1
        ppl_c = self._ppl(cand_id)
        if not math.isfinite(ppl_c):
            return "rollback", ["nan"], float("nan"), None, None
        if (ppl_c - S["ppl"]) / S["ppl"] > self.tau:
            return "rollback", ["ppl"], ppl_c, None, None
        test = None
        if self.policy != "none":
            t_cfg = self.rc["testing"]
            left = int(t_cfg["total_budget"]) - S["budget"]["probe_questions"]
            step_budget = min(int(t_cfg["step_budget"]), left)
            test = sequential_test(
                domains=self.ctx.data.domains, pool=self.ctx.data.probe_by_domain,
                dense_correct=self.ctx.dense_probe, evaluate=self._probe_eval(cand_id), policy=self.policy,
                step_budget=step_budget, min_per_domain=int(t_cfg["min_per_domain"]),
                round_size=int(t_cfg["round_size"]), limits=self._limits(rho_c),
                accept_max=float(t_cfg["accept_max_violation"]), reject_min=float(t_cfg["reject_min_violation"]),
                rule=t_cfg.get("decision_rule", "per_domain"), min_fail_n=int(t_cfg.get("min_fail_n", 0)),
                prior=tuple(t_cfg.get("prior", [0.5, 4.0, 0.5])), mc_samples=int(t_cfg.get("mc_samples", 4000)),
                alpha=float(t_cfg.get("alpha", 0.25)), seed=self.rc["run"]["seed"], iteration=t,
                on_round=lambda r: self.ctx.hb.beat("gate", f"test round {r['round']} n={r['used']}"),
            )
            S["budget"]["probe_questions"] += test.used
            S["budget"]["probe_tests"] += 1
            S["budget"]["early_stops"] += int(test.early_stop)
            if test.decision == FAIL:
                return "rollback", ["domain_fail"], ppl_c, test, None
            if test.decision == INCONCLUSIVE:
                return "rollback", ["domain_inconclusive"], ppl_c, test, None
        ok, dep = self._deploy(cand_id)
        if not ok:
            return "rollback", ["deploy"], ppl_c, test, dep
        return "accept", [], ppl_c, test, dep
