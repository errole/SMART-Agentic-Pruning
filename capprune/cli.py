"""Command-line interface: ``python -m capprune <command> ...``

  prepare    download/tokenize data into disjoint splits (+ pre-download the model)
  run        run or resume ONE run (method x target x seed)
  supervise  run/resume the whole matrix with crash/hang recovery (recommended)
  status     show progress of every run
  verify     integrity-check prepared data and all checkpoints
  report     aggregate finished runs into tables (CSV + Markdown)
  make-tiny  build a tiny random Qwen3 for offline smoke tests

  adaptive-fixture  write the deterministic CPU replay fixture
  adaptive-build    score dense + pruned candidates with the pipeline into a replay cache
  adaptive-replay   run the allocation policies on a replay cache and score them

  gate-design       choose the live capability gate's settings on development data
  live-bench        interleaved dense-vs-pruned latency/memory benchmark of each arm's selected checkpoint
  live-report       compare the arms of the live gate experiment
  gap-diagnostic    2x2 (pruning-time pool vs held-out) x (0-shot vs 5-shot) paired-loss diagnostic of one arm
  controller-check  send one real CAP-Prune prompt to the configured controller and validate the proposal
  fixed-state       compare controllers on frozen checkpoint states: prepare | record-models | query | score | report
  ranker            learned ranking of pruning actions: collect (GPU trajectories) | evaluate (selectors, CPU)
  gradient-live     gradient-only live run analysis: audit-data | register | freeze | diagnose | audit |
                    audit-materialized | report
  gradient-boundary compression boundary, gate-margin calibration, policy comparison: recover | margin-plan | audit |
                    calibrate | compare
  archive-checkpoints  rebuild + verify every accepted checkpoint of finished runs into an archive (runs are only read)
  population-regate retrospective population-level re-gating of finished runs (CPU): register | run
  pool-signals      4-letter log-probabilities on the gate pool and continuous paired signals: score | analyze
  screen-confirm    screen-then-confirm protocol: confirmation sample sizes per category: sizes
  matched-signals   equal-size actions of stall / ranker parents: KL vs. net flip as choice signals: register | score | analyze |
                    compare
  results-index     one index of every experiment: configs, code hashes, commit, reports, headline numbers
  backup            zip the runs root with a per-file SHA-256 manifest and copy it to verified destinations
                    (the live run itself: `supervise` with a `kind: ranked_search` method)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .utils import runtime as rt


def _cfg(args):
    from .config import load_config

    return load_config(args.config, args.set)


def cmd_prepare(args) -> int:
    from .config import data_dir
    from .data.prepare import prepare
    from .model import load_tokenizer, model_path, tokenizer_id
    from .utils.log import setup_logging

    cfg = _cfg(args)
    setup_logging()
    m = cfg["model"]
    if not Path(model_path(cfg)).exists() and not args.skip_model_download:
        from huggingface_hub import snapshot_download

        print(f"Pre-downloading {m['name_or_path']} so runs never wait on the network...")
        snapshot_download(m["name_or_path"], revision=m.get("revision"),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges*", "vocab*"])
    tok = load_tokenizer(cfg)
    man = prepare(cfg, tok, data_dir(cfg), tokenizer_id(cfg))
    print("counts:", man["counts"])
    print("probe questions per domain:", man["probe_domains"])
    print("final questions per domain:", man["final_domains"])
    return 0


def cmd_run(args) -> int:
    from .runner import execute

    try:
        cfg = _cfg(args)
    except rt.ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return rt.EXIT_CONFIG
    return execute(cfg, args.method, float(args.target), int(args.seed), args.force_resume)


def _csv(s, typ=str):
    return [typ(x) for x in s.split(",")] if s else None


def cmd_supervise(args) -> int:
    from .supervisor import Supervisor, build_plan

    cfg = _cfg(args)
    plan = build_plan(cfg, _csv(args.methods), _csv(args.targets, float), _csv(args.seeds, int))
    if args.dry_run:
        for p in plan:
            print(p["id"])
        return 0
    return Supervisor(cfg, args.config, args.set).run(plan)


def cmd_status(args) -> int:
    from .config import runs_dir
    from .supervisor import build_plan
    from .utils.checkpoint import CheckpointStore, NoCheckpoint
    from .utils.io import read_json

    cfg = _cfg(args)
    root = runs_dir(cfg)
    led_p = root / "_matrix" / "ledger.json"
    led = read_json(led_p) if led_p.exists() else {}
    now = time.time()
    print(f"{'run':38} {'status':18} {'stage':12} {'iter':>5} {'sparsity':>9} {'tries':>5} {'heartbeat':>10}")
    for item in build_plan(cfg):
        d = root / item["id"]
        status = "done" if (d / "DONE").exists() else led.get(item["id"], {}).get("status", "pending" if not d.exists() else "started")
        stage, it, rho, tries, hb = "", "", "", led.get(item["id"], {}).get("attempts", ""), ""
        if d.exists():
            try:
                S, _ = CheckpointStore(d).load_latest()
                srch = S.get("search") or {}
                it = str(srch.get("iteration", 1) - 1) if srch else ""  # completed iterations
                rho = f"{srch.get('rho', 0):.4f}" if srch else ""
                done = S.get("stages_done", [])
                stage = done[-1] if done else "search" if srch else ""
            except NoCheckpoint:
                pass
            try:
                h = read_json(d / "heartbeat.json")
                hb = f"{now - h['ts']:.0f}s ago" if status != "done" else ""
                stage = stage if status == "done" else h.get("phase", stage)
            except Exception:
                pass
        print(f"{item['id']:38} {status:18} {stage:12} {it:>5} {rho:>9} {str(tries):>5} {hb:>10}")
    return 0


def cmd_verify(args) -> int:
    from .config import data_dir, runs_dir
    from .data.prepare import verify
    from .model import tokenizer_id
    from .utils.checkpoint import CheckpointStore, NoCheckpoint

    cfg = _cfg(args)
    ok = True
    try:
        verify(data_dir(cfg), tokenizer_id(cfg))
        print("data: OK")
    except Exception as e:
        ok = False
        print(f"data: FAILED - {e}")
    root = runs_dir(cfg)
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.name.startswith("_") or not d.is_dir():
                continue
            try:
                S, _ = CheckpointStore(d).load_latest()
                print(f"{d.name}: OK (checkpoint #{S['ckpt_index']}, state {S.get('state_id')})")
            except NoCheckpoint as e:
                if (d / "ckpt").exists() and any((d / "ckpt").iterdir()):
                    ok = False
                    print(f"{d.name}: NO VALID CHECKPOINT - {e}")
    return 0 if ok else 1


def cmd_report(args) -> int:
    from .report import write_report

    p = write_report(_cfg(args), args.reference, args.bootstrap)
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten to {p.parent}")
    return 0


def cmd_make_tiny(args) -> int:
    from .model import build_tiny_model

    print(build_tiny_model(args.out))
    return 0


def cmd_adaptive_fixture(args) -> int:
    from .adaptive.fixture import write_fixture

    print(f"fixture written to {write_fixture(args.out)}")
    return 0


def cmd_adaptive_build(args) -> int:
    from .adaptive.build import build
    from .adaptive.config import load_adaptive
    from .utils.log import setup_logging

    setup_logging()
    build(load_adaptive(args.config, args.set), _csv(args.only), args.force, args.dry_run)
    return 0


def cmd_adaptive_replay(args) -> int:
    from .adaptive.config import cfg_path, load_adaptive
    from .adaptive.replay import replay_main

    cfg = load_adaptive(args.config, args.set)
    cache = Path(args.cache) if args.cache else cfg_path(cfg, cfg["cache_dir"])
    out = Path(args.out) if args.out else cfg_path(cfg, cfg["out_dir"])
    p = replay_main(cfg, cache, out)
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten to {p.parent}")
    return 0


def cmd_gate_design(args) -> int:
    from .adaptive.config import load_adaptive
    from .adaptive.gate_design import design

    p = design(load_adaptive(args.config, args.set))
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten to {p.parent}")
    return 0


def cmd_live_bench(args) -> int:
    from .live_bench import live_bench
    from .utils.log import setup_logging

    setup_logging()
    p = live_bench(_cfg(args), _csv(args.methods))
    print(f"written to {p}")
    return 0


def cmd_live_report(args) -> int:
    from .live_report import write_live_report

    p = write_live_report(_cfg(args))
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten to {p.parent}")
    return 0


def cmd_gap_diagnostic(args) -> int:
    from .gap_diagnostic import reanalyze, run
    from .utils.log import setup_logging

    setup_logging()
    out = Path(args.out).expanduser()
    p = reanalyze(_cfg(args), out, args.bootstrap) if args.reanalyze else run(_cfg(args), args.arm, out, args.bootstrap)
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten to {p.parent}")
    return 0


def cmd_controller_check(args) -> int:
    import json

    from .controller_check import check

    res = check(_cfg(args), args.method, Path(args.prompts_from).expanduser(), args.iteration, Path(args.out).expanduser())
    print(json.dumps({k: v for k, v in res.items() if k != "raw_response"}, indent=1))
    print("\nraw response (first 1500 chars):\n" + (res.get("raw_response") or "")[:1500])
    return 0 if res.get("ok") else 1


def cmd_fixed_state(args) -> int:
    from . import fixed_state as fs
    from .utils.log import setup_logging

    setup_logging()
    cfg = _cfg(args)
    if args.phase == "prepare":
        p = fs.prepare(cfg)
    elif args.phase == "record-models":
        p = fs.record_models(cfg)
    elif args.phase == "query":
        if not args.model:
            raise SystemExit("--model is required for the query phase")
        p = fs.query(cfg, args.model)
    elif args.phase == "score":
        p = fs.score(cfg)
    else:
        p = fs.report(cfg)
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_ranker(args) -> int:
    from . import ranker
    from .utils.log import setup_logging

    setup_logging()
    cfg = _cfg(args)
    if args.phase == "collect":
        p = ranker.collect(cfg)
    else:
        p = ranker.evaluate(cfg)
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_gradient_live(args) -> int:
    from . import gradient_audit as ga
    from .utils.log import setup_logging

    cfg = _cfg(args)
    setup_logging()
    fn = {"audit-data": ga.audit_data, "register": ga.register, "freeze": ga.freeze, "diagnose": ga.diagnose,
          "audit": ga.audit, "audit-materialized": ga.audit_materialized, "report": ga.report}[args.phase]
    p = fn(cfg)
    if args.phase in ("report", "audit-materialized"):
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_gradient_boundary(args) -> int:
    from . import gradient_boundary as gb
    from .utils.log import setup_logging

    cfg = _cfg(args)
    setup_logging()
    fn = {"recover": gb.recover, "margin-plan": gb.margin_plan, "audit": gb.boundary_audit, "calibrate": gb.calibrate,
          "compare": gb.compare}[args.phase]
    p = fn(cfg)
    if p.suffix == ".md":
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_archive_checkpoints(args) -> int:
    from .archive import archive_all
    from .utils.io import read_json
    from .utils.log import setup_logging

    setup_logging()
    p = archive_all(Path(args.root).expanduser(), Path(args.out).expanduser(), _csv(args.experiments))
    idx = read_json(p)
    for r in idx["runs"]:
        print(f"{r['run_dir']}: {r['status']} ({r.get('states', 0)} states)")
    print(f"\nwritten: {p}")
    return 0 if all(not r["status"].startswith("FAILED") for r in idx["runs"]) else 1


def cmd_results_index(args) -> int:
    from .results_index import build_index

    p = build_index(Path(args.registry), Path(args.root).expanduser())
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_backup(args) -> int:
    from .results_index import backup

    rec = backup(Path(args.root).expanduser(), [Path(d).expanduser() for d in args.dest], Path(args.stage).expanduser())
    for c in rec["copies"]:
        print(f"{c['path']}: verified={c['verified']} (sha256 match {c['sha256_match']}, {c['members']} members)")
    print(f"{rec['files']} files, {rec['bytes_uncompressed'] / 1e6:.0f} MB -> {rec['zip_bytes'] / 1e6:.0f} MB, sha256 {rec['zip_sha256']}")
    return 0 if all(c["verified"] for c in rec["copies"]) else 1


def cmd_population_regate(args) -> int:
    from . import population_regate as pr
    from .utils.log import setup_logging

    setup_logging()
    cfg = _cfg(args)
    p = pr.register(cfg) if args.phase == "register" else pr.run(cfg)
    if p.suffix == ".md":
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_pool_signals(args) -> int:
    from . import pool_signals as ps
    from .utils.log import setup_logging

    setup_logging()
    cfg = _cfg(args)
    p = ps.score(cfg) if args.phase == "score" else ps.analyze(cfg)
    if p.suffix == ".md":
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_matched_signals(args) -> int:
    from . import matched_signals as ms
    from .utils.log import setup_logging

    setup_logging()
    cfg = _cfg(args)
    p = {"register": ms.register, "score": ms.score, "analyze": ms.analyze, "compare": ms.compare}[args.phase](cfg)
    if p.suffix == ".md":
        print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def cmd_screen_confirm(args) -> int:
    from .screen_confirm import write_sizes

    p = write_sizes(_cfg(args))
    print(p.read_text(encoding="utf-8"))
    print(f"\nwritten: {p}")
    return 0


def _utf8_output() -> None:
    """Never let a non-ASCII character (the report prints a Greek Delta) crash a print.

    On Windows a redirected/piped stdout uses the legacy ANSI code page (cp1252),
    which cannot encode it.  Pipes and files get UTF-8 - matching the UTF-8 log
    files the supervisor redirects workers into; an interactive console keeps
    its own encoding and only gains the "replace" safety net.
    """
    for s in (sys.stdout, sys.stderr):
        try:
            if s.isatty():
                s.reconfigure(errors="replace")
            else:
                s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # replaced/closed stream (e.g. under a test harness): leave it alone


def main(argv=None) -> int:
    _utf8_output()
    ap = argparse.ArgumentParser(prog="capprune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable)")

    p = sub.add_parser("prepare"); common(p)
    p.add_argument("--skip-model-download", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run"); common(p)
    p.add_argument("--method", required=True)
    p.add_argument("--target", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-resume", action="store_true", help="resume even if the science config changed")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("supervise"); common(p)
    p.add_argument("--methods"); p.add_argument("--targets"); p.add_argument("--seeds")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_supervise)
    p = sub.add_parser("status"); common(p); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("verify"); common(p); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("report"); common(p)
    p.add_argument("--reference", default="paper_agent")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("make-tiny"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_make_tiny)
    p = sub.add_parser("adaptive-fixture"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_adaptive_fixture)
    p = sub.add_parser("adaptive-build"); common(p)
    p.add_argument("--only", help="comma-separated model names to (re)build, e.g. dense,wsp20")
    p.add_argument("--force", action="store_true", help="rebuild models whose cache file already exists")
    p.add_argument("--dry-run", action="store_true", help="validate config, data and candidate files; load no model")
    p.set_defaults(fn=cmd_adaptive_build)
    p = sub.add_parser("adaptive-replay"); common(p)
    p.add_argument("--cache", help="replay cache directory (default: cache_dir from the config)")
    p.add_argument("--out", help="output directory (default: out_dir from the config)")
    p.set_defaults(fn=cmd_adaptive_replay)
    p = sub.add_parser("gate-design"); common(p); p.set_defaults(fn=cmd_gate_design)
    p = sub.add_parser("live-bench"); common(p)
    p.add_argument("--methods", help="comma-separated arms (default: the matrix)")
    p.set_defaults(fn=cmd_live_bench)
    p = sub.add_parser("live-report"); common(p); p.set_defaults(fn=cmd_live_report)
    p = sub.add_parser("gap-diagnostic"); common(p)
    p.add_argument("--arm", required=True, help="live arm whose selected checkpoint is diagnosed, e.g. greedy_cap")
    p.add_argument("--out", required=True, help="output directory (kept apart from the run artifacts)")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--reanalyze", action="store_true", help="recompute tables from the saved per-item answers (no GPU)")
    p.set_defaults(fn=cmd_gap_diagnostic)
    p = sub.add_parser("controller-check"); common(p)
    p.add_argument("--method", default="llm_cap", help="arm whose controller settings are used")
    p.add_argument("--prompts-from", required=True, help="a run's caches/controller.sqlite (opened read-only)")
    p.add_argument("--iteration", type=int, default=2)
    p.add_argument("--out", required=True, help="where the check's own cache and JSON result go (not a run dir)")
    p.set_defaults(fn=cmd_controller_check)
    p = sub.add_parser("fixed-state"); common(p)
    p.add_argument("phase", choices=["prepare", "record-models", "query", "score", "report"])
    p.add_argument("--model", help="controller model for the query phase, e.g. qwen2.5:14b-instruct")
    p.set_defaults(fn=cmd_fixed_state)
    p = sub.add_parser("ranker"); common(p)
    p.add_argument("phase", choices=["collect", "evaluate"])
    p.set_defaults(fn=cmd_ranker)
    p = sub.add_parser("gradient-live"); common(p)
    p.add_argument("phase", choices=["audit-data", "register", "freeze", "diagnose", "audit", "audit-materialized", "report"])
    p.set_defaults(fn=cmd_gradient_live)
    p = sub.add_parser("archive-checkpoints")
    p.add_argument("--root", default="~/cap-prune-runs", help="runs root: <experiment>/<run>/journal.jsonl")
    p.add_argument("--out", default="~/cap-prune-runs/_archive/checkpoints")
    p.add_argument("--experiments", help="comma-separated experiment folders (default: all)")
    p.set_defaults(fn=cmd_archive_checkpoints)
    p = sub.add_parser("screen-confirm"); common(p)
    p.add_argument("phase", choices=["sizes"])
    p.set_defaults(fn=cmd_screen_confirm)
    p = sub.add_parser("matched-signals"); common(p)
    p.add_argument("phase", choices=["register", "score", "analyze", "compare"])
    p.set_defaults(fn=cmd_matched_signals)
    p = sub.add_parser("pool-signals"); common(p)
    p.add_argument("phase", choices=["score", "analyze"])
    p.set_defaults(fn=cmd_pool_signals)
    p = sub.add_parser("population-regate"); common(p)
    p.add_argument("phase", choices=["register", "run"])
    p.set_defaults(fn=cmd_population_regate)
    p = sub.add_parser("results-index")
    p.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "configs" / "results_registry.yaml"))
    p.add_argument("--root", default="~/cap-prune-runs")
    p.set_defaults(fn=cmd_results_index)
    p = sub.add_parser("backup")
    p.add_argument("--root", default="~/cap-prune-runs")
    p.add_argument("--dest", action="append", required=True, help="destination folder (repeatable); every copy is verified")
    p.add_argument("--stage", required=True, help="where the zip is built (normally the first destination)")
    p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("gradient-boundary"); common(p)
    p.add_argument("phase", choices=["recover", "margin-plan", "audit", "calibrate", "compare"])
    p.set_defaults(fn=cmd_gradient_boundary)
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except rt.ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return rt.EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
