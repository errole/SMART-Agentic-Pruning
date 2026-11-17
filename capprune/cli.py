"""Command-line interface: ``python -m capprune <command> ...``

  prepare    download/tokenize data into disjoint splits (+ pre-download the model)
  run        run or resume ONE run (method x target x seed)
  supervise  run/resume the whole matrix with crash/hang recovery (recommended)
  status     show progress of every run
  verify     integrity-check prepared data and all checkpoints
  report     aggregate finished runs into tables (CSV + Markdown)
  make-tiny  build a tiny random Qwen3 for offline smoke tests
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
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except rt.ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return rt.EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
