"""One experiment run (method x target x seed) as a sequence of resumable stages.

    dense : final_mmlu -> final_fqa -> final_ppl -> benchmark -> summary
    wanda : prune_wanda -> final_* -> benchmark -> summary
    search: search -> final_* -> benchmark -> summary

A stage is skipped when its ``stages/<name>.done`` marker exists, so after a
crash only the unfinished stage resumes.  Long stages resume *inside*
themselves: the search from its last per-iteration checkpoint, final
evaluations from per-question caches.

Exit codes: see ``capprune.utils.runtime``.
"""
from __future__ import annotations

import faulthandler
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from .config import run_config, run_id, runs_dir, data_dir, science_hash
from .data.prepare import PreparedData
from .model import build_units, load_model, load_tokenizer, tokenizer_id
from .pruning.masks import DenseStore, PruneState
from .utils.cache import KVCache
from .utils.checkpoint import CheckpointStore, NoCheckpoint
from .utils.faults import fault_point
from .utils.gpu import GpuClock, free_memory
from .utils.io import DiskSpaceError, append_jsonl, atomic_write_json, read_json, read_jsonl, short_hash
from .utils.log import get_logger, setup_logging
from .utils import runtime as rt

log = get_logger("runner")

FINAL_STAGES = ["final_mmlu", "final_fqa", "final_ppl", "benchmark", "summary"]


class SharedLock(rt.RunLock):
    """Waits (instead of failing) while another run computes a shared reference."""

    def acquire(self) -> None:
        while True:
            try:
                return super().acquire()
            except rt.RunLockedError:
                time.sleep(5)


class RunContext:
    def __init__(self, cfg: dict, method: str, target: float, seed: int, force_resume: bool = False):
        kind = cfg["methods"][method].get("kind", "search")
        if kind == "dense":
            target, seed = 0.0, 0
        self.rc = run_config(cfg, method, target, seed)
        self.kind = kind
        self.id = "dense" if kind == "dense" else run_id(method, target, seed)
        self.root = runs_dir(cfg)
        self.run_dir = self.root / self.id
        self.shared_dir = self.root / "_shared"
        self.force_resume = force_resume

    # ------------------------------------------------------------------ setup
    def open(self) -> None:
        rc = self.rc
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.run_dir / "logs" / "worker.log")
        self._fault_file = open(self.run_dir / "logs" / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=self._fault_file)
        prev_hb = self._read_prev_heartbeat()
        self.hb = rt.Heartbeat(self.run_dir, float(rc.get("reliability", {}).get("heartbeat_s", 5)))
        self.hb.beat("setup", "loading", force=True)
        self._check_run_json()
        rel = rc.get("reliability", {})
        mirror = rel.get("mirror_dir")
        mirror_path = (Path(os.path.expandvars(mirror)) / self.id) if mirror else None
        self.store = CheckpointStore(self.run_dir, int(rel.get("keep_checkpoints", 3)), mirror_path,
                                     float(rel.get("min_free_gb", 2.0)))
        self.eval_cache = KVCache(self.run_dir / "caches" / "eval.sqlite")
        self.ctrl_cache = KVCache(self.run_dir / "caches" / "controller.sqlite")

        torch.manual_seed(rc["run"]["seed"])
        self.tok = load_tokenizer(rc)
        self.data = PreparedData(data_dir(rc), tokenizer_id(rc))
        log.info("loading model %s", rc["model"]["name_or_path"])
        self.model = load_model(rc)
        self.units = build_units(self.model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))
        dense_dev = rel.get("dense_copy_device") or None
        self.dense = DenseStore(self.units, dense_dev)
        nm = rc["pruning"].get("nm", {})
        self.state = PruneState(self.units, rc["pruning"]["mode"], (int(nm.get("n", 2)), int(nm.get("m", 4))))
        n_cal = int(rc["data"]["calibration"]["num_samples"])
        self.calib = self.data.calibration_pool[self.data.calibration_indices(rc["run"]["seed"], n_cal)]

        # ---- checkpoint or fresh start
        # Attempt count and time lost to failures are derived from the append-only journal, not the
        # checkpoint: a crash before the next checkpoint must still be counted.
        journal = self.run_dir / "journal.jsonl"
        starts = [r for r in read_jsonl(journal) if r.get("type") == "start"]
        try:
            self.S, _ = self.store.load_latest()
            resumed = True
        except NoCheckpoint as e:
            resumed = False
            self._no_ckpt_error = e
        lost = 0.0
        if starts and prev_hb and prev_hb.get("ts"):
            ref = max(self.S.get("saved_at", 0.0) if resumed else 0.0, float(starts[-1]["ts"]))
            lost = max(0.0, float(prev_hb["ts"]) - ref)
        self.attempts = len(starts) + 1
        self.lost_seconds = sum(float(s.get("lost_s", 0.0)) for s in starts) + lost
        append_jsonl(journal, {"ts": time.time(), "type": "start", "attempt": self.attempts, "lost_s": lost,
                               "pid": os.getpid(), "resumed_from": self.S["ckpt_index"] if resumed else None})
        if resumed:
            self.S["attempts"], self.S["gpu_seconds_lost"] = self.attempts, self.lost_seconds
            log.info("RESUMING %s from checkpoint #%d (attempt %d; ~%.0fs of work is being redone)", self.id,
                     self.S["ckpt_index"], self.attempts, lost)
        else:
            e = self._no_ckpt_error
            if any((self.run_dir / "ckpt").glob("ckpt-*.json")):
                log.error("ALL checkpoints of %s are unusable (%s). Restarting the run from the beginning; "
                          "cached profiles, controller replies and evaluations make the replay fast and identical.",
                          self.id, e)
            self.S = {"version": 1, "run": rc["run"], "science_hash": science_hash(rc), "ckpt_index": 0,
                      "attempts": self.attempts, "state_id": None, "search": None, "batch": {}, "gpu_seconds": 0.0,
                      "gpu_seconds_lost": self.lost_seconds, "started_at": time.time(), "stages_done": []}
        self.clock = GpuClock(self.S.get("gpu_seconds", 0.0))

        # ---- shared dense references (computed on the still-dense model)
        if self.kind == "search":
            self._shared_refs()

        # ---- restore pruning state
        sid = self.S.get("state_id")
        if sid:
            self.state.load(self.store.mask_path(sid))
            self.state.apply(self.dense)
            fp = self.state.fingerprint()
            if self.S.get("fingerprint") and self.S["fingerprint"] != fp:
                raise RuntimeError(f"restored masks fingerprint {fp} != checkpoint {self.S['fingerprint']}")
            log.info("restored pruning state %s (sparsity %.4f)", sid, self.state.global_sparsity())

    def _read_prev_heartbeat(self) -> dict | None:
        p = self.run_dir / "heartbeat.json"
        try:
            return read_json(p) if p.exists() else None
        except Exception:
            return None

    def _check_run_json(self) -> None:
        p = self.run_dir / "run.json"
        h = science_hash(self.rc)
        if p.exists():
            old = read_json(p)
            if old.get("science_hash") != h and not self.force_resume:
                raise rt.ConfigError(
                    f"{self.id}: the experiment configuration changed since this run started "
                    f"(hash {old.get('science_hash')} -> {h}). Resuming would mix two experiments. "
                    "Use a new runs_dir, delete the run directory, or pass --force-resume.")
            return
        env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        try:
            import transformers

            env["transformers"] = transformers.__version__
        except ImportError:
            pass
        atomic_write_json(p, {"run_id": self.id, "science_hash": h, "config": {k: v for k, v in self.rc.items() if not k.startswith("_")},
                              "environment": env, "created": time.time()})

    def _shared_refs(self) -> None:
        rc = self.rc
        t = rc["testing"]
        man = self.data.manifest["files"]
        mkey = short_hash(rc["model"]["name_or_path"], rc["model"].get("revision"), rc["model"].get("dtype"))
        d = self.shared_dir / mkey
        d.mkdir(parents=True, exist_ok=True)
        # perplexity of the dense model on the pruning-time PPL set
        pkey = short_hash(man["ppl_gate.safetensors"]["sha256"])
        ppath = d / f"ppl0-{pkey}.json"
        if not ppath.exists():
            with SharedLock(d):
                if not ppath.exists():
                    from .eval.perplexity import perplexity

                    self.hb.beat("setup", "dense perplexity")
                    v = perplexity(self.model, self.data.ppl_gate, self.S["batch"], "ppl_gate_bs", 4, self.hb)
                    atomic_write_json(ppath, {"ppl0": v})
        self.ppl0 = float(read_json(ppath)["ppl0"])
        # dense answers to the whole probe pool
        self.dense_probe = {}
        if t["policy"] != "none":
            qkey = short_hash(man["mmlu_probe.jsonl"]["sha256"], man["mmlu_fewshot.jsonl"]["sha256"],
                              t.get("shots", 0), rc["model"].get("max_seq_len", 2048))
            qpath = d / f"dense_probe-{qkey}.json"
            if not qpath.exists():
                with SharedLock(d):
                    if not qpath.exists():
                        from .eval.mmlu import MMLUScorer

                        self.hb.beat("setup", "dense probe answers")
                        sc = MMLUScorer(self.model, self.tok, self.data.mmlu_fewshot, int(t.get("shots", 0)),
                                        int(rc["model"].get("max_seq_len", 2048)), self.S["batch"], "probe_tokens",
                                        int(t.get("max_batch_tokens", 32768)), self.hb)
                        res = sc.score(self.data.mmlu_probe)
                        atomic_write_json(qpath, {q: r["correct"] for q, r in res.items()}, indent=None)
            self.dense_probe = read_json(qpath)
        # dense quick benchmark (deployment-check baseline)
        self.bench0 = None
        dcfg = rc.get("deploy", {})
        if dcfg.get("check", "off") == "per_step" and rc["pruning"]["mode"] != "unstructured":
            q = dcfg.get("quick", {})
            bkey = short_hash(q, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
            bpath = d / f"bench0-{bkey}.json"
            if not bpath.exists():
                with SharedLock(d):
                    if not bpath.exists():
                        from .eval.bench import benchmark

                        self.hb.beat("setup", "dense quick benchmark")
                        b = benchmark(self.model, self.model.config.vocab_size, [int(q.get("prompt_len", 512))],
                                      int(q.get("gen_tokens", 32)), 1, int(q.get("warmup", 2)), int(q.get("repeats", 5)))
                        atomic_write_json(bpath, b)
            self.bench0 = read_json(bpath)

    # ------------------------------------------------------------------ persistence
    def save_state_masks(self, state_id: str) -> None:
        self.hb.beat(None, "saving masks", force=True)
        self.store.save_masks(state_id, lambda p: self.state.save(p), self.state.approx_file_bytes())

    def checkpoint(self) -> None:
        S = self.S
        if S.get("search") is not None:
            S["state_id"] = S["search"]["state_id"]
        S["gpu_seconds"] = self.clock.commit()
        S["fingerprint"] = self.state.fingerprint() if S.get("state_id") else None
        S["ckpt_index"] += 1
        S["saved_at"] = time.time()
        self.store.save(S["ckpt_index"], S)
        self.hb.beat(None, f"checkpoint #{S['ckpt_index']}", force=True)

    def done_stage(self, stage: str, info=None) -> None:
        # Stage completion lives INSIDE the checkpoint, so it can never disagree with the saved state.
        self.S.setdefault("stages_done", []).append(stage)
        self.S.setdefault("stage_info", {})[stage] = info
        self.checkpoint()
        append_jsonl(self.run_dir / "journal.jsonl", {"ts": time.time(), "type": "stage_done", "stage": stage})

    # ------------------------------------------------------------------ stages
    def stages(self) -> list[str]:
        head = {"dense": [], "wanda": ["prune_wanda"], "search": ["search"]}[self.kind]
        fe = self.rc.get("final_eval", {})
        tail = [s for s in FINAL_STAGES if not (s == "final_fqa" and not fe.get("freebaseqa", True))
                and not (s == "final_ppl" and not fe.get("wikitext", True))
                and not (s == "benchmark" and not self.rc.get("benchmark", {}).get("enabled", True))]
        return head + tail

    def run_stages(self) -> None:
        for st in self.stages():
            if st in self.S.get("stages_done", []):
                continue
            rt.check_stop()
            log.info("=== stage %s ===", st)
            self.hb.beat(st, "", force=True)
            getattr(self, f"_stage_{st}")()
        atomic_write_json(self.run_dir / "DONE", {"finished": time.time(), "gpu_seconds": self.S["gpu_seconds"]})

    def _stage_search(self) -> None:
        from .search.loop import Search, initial_search_state

        if self.S.get("search") is None:
            self.S["search"] = initial_search_state(self.ppl0, self.bench0)
            self.checkpoint()
        reason = Search(self).run()
        self.done_stage("search", {"stop_reason": reason})

    def _stage_prune_wanda(self) -> None:
        from .pruning.profile import collect_act_sqnorm, sensitivity, zscores
        from .pruning.wanda import sequential_wanda

        order = None
        if self.state.mode == "nm":
            act = collect_act_sqnorm(self.model, self.state, self.calib, self.S["batch"], self.hb)
            z = zscores(sensitivity(self.state, act))
            order = sorted(z, key=lambda k: (z[k], k))
        bs = int(self.rc["profiling"].get("activation_batch_size", 4))
        sequential_wanda(self.model, self.state, self.dense, self.calib, float(self.rc["run"]["target"]), self.rc,
                         bs, self.hb, order)
        sid = "wanda-" + short_hash(self.id, self.state.fingerprint())
        self.save_state_masks(sid)
        self.S["state_id"] = sid
        self.done_stage("prune_wanda", {"sparsity": self.state.global_sparsity()})

    def _sid(self) -> str:
        return self.S.get("state_id") or "dense"

    def _stage_final_mmlu(self) -> None:
        from .eval.mmlu import MMLUScorer

        fe = self.rc.get("final_eval", {})
        rows = self.data.eval_sets(fe.get("split", "final"))[0]
        sc = MMLUScorer(self.model, self.tok, self.data.mmlu_fewshot, int(fe.get("mmlu_shots", 5)),
                        int(fe.get("max_seq_len", 4096)), self.S["batch"], "final_tokens",
                        int(fe.get("max_batch_tokens", 32768)), self.hb)
        res = self._chunked_cached(f"final_mmlu|{self._sid()}", rows, lambda chunk: {q: r["correct"] for q, r in sc.score(chunk).items()})
        agg = aggregate_mcq(rows, res)
        atomic_write_json(self.run_dir / "final" / "mmlu.json", {"per_question": res, **agg})
        self.done_stage("final_mmlu", {"overall": agg["overall"]})

    def _stage_final_fqa(self) -> None:
        from .eval.freebaseqa import evaluate

        fe = self.rc.get("final_eval", {})
        rows = self.data.eval_sets(fe.get("split", "final"))[1]
        res = self._chunked_cached(
            f"final_fqa|{self._sid()}", rows,
            lambda chunk: {q: r["correct"] for q, r in evaluate(self.model, self.tok, chunk, self.data.fqa_fewshot,
                                                                self.S["batch"], int(fe.get("fqa_max_new_tokens", 24)),
                                                                heartbeat=self.hb).items()})
        em = sum(res.values()) / max(1, len(res))
        atomic_write_json(self.run_dir / "final" / "freebaseqa.json", {"per_question": res, "exact_match": em, "n": len(res)})
        self.done_stage("final_fqa", {"exact_match": em})

    def _stage_final_ppl(self) -> None:
        from .eval.perplexity import perplexity, stream_to_windows

        seq = int(self.rc["data"]["wikitext"].get("seq_len", 2048))
        split = self.rc.get("final_eval", {}).get("split", "final")
        v = perplexity(self.model, stream_to_windows(self.data.eval_sets(split)[2], seq), self.S["batch"], "wikitext_bs", 2, self.hb)
        atomic_write_json(self.run_dir / "final" / "wikitext2.json", {"perplexity": v, "seq_len": seq, "split": split})
        self.done_stage("final_ppl", {"perplexity": v})

    def _stage_benchmark(self) -> None:
        from .eval.bench import benchmark
        from .pruning.materialize import materialize

        b = self.rc.get("benchmark", {})
        if self.kind == "dense":
            m, info = self.model, {"mode": "dense"}
        else:
            m, info = materialize(self.model, self.state, self.rc)
        try:
            res = benchmark(m, self.model.config.vocab_size, [int(x) for x in b.get("prompt_lens", [512, 2048])],
                            int(b.get("gen_tokens", 128)), int(b.get("batch_size", 1)), int(b.get("warmup", 3)),
                            int(b.get("repeats", 10)), heartbeat=self.hb)
        finally:
            if m is not self.model:
                del m
                free_memory()
        res["materialize"] = info
        atomic_write_json(self.run_dir / "final" / "benchmark.json", res)
        self.done_stage("benchmark")

    def _stage_summary(self) -> None:
        f = self.run_dir / "final"

        def opt(name):
            p = f / name
            return read_json(p) if p.exists() else None

        mm, fq, wt, bm = opt("mmlu.json"), opt("freebaseqa.json"), opt("wikitext2.json"), opt("benchmark.json")
        if mm:
            mm = {k: v for k, v in mm.items() if k != "per_question"}
        if fq:
            fq = {k: v for k, v in fq.items() if k != "per_question"}
        srch = self.S.get("search") or {}
        summary = {
            "run_id": self.id, **self.rc["run"], "mode": self.rc["pruning"]["mode"],
            "achieved_sparsity": self.state.global_sparsity(),
            "unit_sparsity": self.state.all_unit_sparsity(),
            "stop_reason": srch.get("stop_reason"),
            "iterations": (srch.get("iteration", 1) - 1) if srch else None,
            "accepts": srch.get("accepts"), "rollbacks": srch.get("rollbacks"),
            "budget": srch.get("budget"),
            "pruning_time_ppl": {"dense": srch.get("ppl0"), "final": srch.get("ppl")} if srch else None,
            "mmlu": mm, "freebaseqa": fq, "wikitext2": wt, "benchmark": bm,
            "wall_seconds": self.clock.total(),
            "controller_wait_seconds": ((srch.get("budget") or {}).get("controller_seconds", 0.0)),
            "gpu_seconds": self.clock.total() - ((srch.get("budget") or {}).get("controller_seconds", 0.0)),
            "gpu_seconds_lost_to_failures": self.lost_seconds,
            "attempts": self.attempts,
        }
        atomic_write_json(f / "summary.json", summary)
        self.done_stage("summary")

    def _chunked_cached(self, prefix: str, rows: list[dict], fn) -> dict[str, bool]:
        chunk = int(self.rc.get("final_eval", {}).get("chunk_size", 512))
        keys = {r["qid"]: f"{prefix}|{r['qid']}" for r in rows}
        have = self.eval_cache.get_many(keys.values())
        todo = [r for r in rows if keys[r["qid"]] not in have]
        if have:
            log.info("%s: %d/%d answers recovered from cache", prefix, len(have), len(rows))
        for i in range(0, len(todo), chunk):
            rt.check_stop()
            part = todo[i : i + chunk]
            res = fn(part)
            self.eval_cache.put_many({keys[q]: bool(v) for q, v in res.items()})
            have.update({keys[q]: bool(v) for q, v in res.items()})
            self.hb.beat(None, f"{prefix.split('|')[0]} {len(have)}/{len(rows)}")
            fault_point("mid_" + prefix.split("|")[0])
            self.S["gpu_seconds"] = self.clock.total()
        return {r["qid"]: bool(have[keys[r["qid"]]]) for r in rows}

    def close(self) -> None:
        for c in (getattr(self, "eval_cache", None), getattr(self, "ctrl_cache", None)):
            if c is not None:
                c.close()


def aggregate_mcq(rows: list[dict], res: dict[str, bool]) -> dict:
    by_d, by_c = {}, {}
    for r in rows:
        c = res[r["qid"]]
        for agg, k in ((by_d, r["domain"]), (by_c, r["category"])):
            a = agg.setdefault(k, [0, 0])
            a[0] += int(c)
            a[1] += 1
    return {
        "overall": sum(res.values()) / max(1, len(res)), "n": len(res),
        "by_domain": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_d.items())},
        "by_category": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_c.items())},
    }


def execute(cfg: dict, method: str, target: float, seed: int, force_resume: bool = False) -> int:
    """Run (or resume) one run. Returns a process exit code."""
    rt.install_signal_handlers()
    ctx = RunContext(cfg, method, target, seed, force_resume)
    lock = rt.RunLock(ctx.run_dir)
    try:
        lock.acquire()
    except rt.RunLockedError as e:
        print(f"[capprune] {e}", file=sys.stderr)
        return rt.EXIT_LOCKED
    try:
        ctx.open()
        ctx.run_stages()
        log.info("run %s complete", ctx.id)
        return rt.EXIT_OK
    except rt.GracefulExit:
        log.warning("stop requested: progress is checkpointed; resume with the same command")
        return rt.EXIT_INTERRUPTED
    except rt.ControllerUnavailable as e:
        log.error("LLM controller unavailable (%s). Run paused; it resumes from the same iteration.", e)
        return rt.EXIT_CONTROLLER_UNAVAILABLE
    except DiskSpaceError as e:
        log.error("%s", e)
        return rt.EXIT_DISK
    except rt.ConfigError as e:
        log.error("configuration error: %s", e)
        return rt.EXIT_CONFIG
    except KeyboardInterrupt:
        log.warning("interrupted")
        return rt.EXIT_INTERRUPTED
    except Exception:
        log.error("run crashed:\n%s", traceback.format_exc())
        return 1
    finally:
        try:
            ctx.close()
        finally:
            lock.release()
