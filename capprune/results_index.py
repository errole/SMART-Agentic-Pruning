"""Results index and verified backups of the runs root (``capprune results-index`` / ``capprune backup``).

results-index  One table of every registered experiment (configs/results_registry.yaml): its configs with SHA-256,
               the git commit and whether the working tree was clean, code hashes recorded at registration and the
               files that changed since, report paths, the per-run science hashes and environments, and headline
               numbers read from the experiment's own JSON outputs (never re-computed).  Also lists the checkpoint
               archive, the test records and the backups.
backup         Zips the runs root (every file, including caches and logs), writes a per-file SHA-256 manifest, copies
               the zip to every destination and re-verifies each copy (SHA-256 of the copy + zip integrity test).
               Nothing in the runs root is modified except the backups ledger (<root>/_archive/backups.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import yaml

from .utils.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .utils.log import get_logger

log = get_logger("results")
REPO = Path(__file__).resolve().parents[1]


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env.update(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    return env


def _opt(p: Path):
    try:
        return read_json(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _pts(x):
    return "n/a" if x is None else f"{100 * x:+.2f}"


# ============================================================================ headline extractors
def _search_runs(root: Path, folder: str) -> list[str]:
    base = root / folder
    dense = _opt(base / "dense" / "final" / "summary.json")
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "dense"):
        s = _opt(d / "final" / "summary.json")
        if s is None:
            out.append(f"{d.name}: no final summary (run unfinished)")
            continue
        line = f"{d.name}: MLP/eligible sparsity {100 * s['achieved_sparsity']:.2f} %, stop {s.get('stop_reason')}"
        if dense and s.get("mmlu") and dense.get("mmlu"):
            dl = {c: dense["mmlu"]["by_category"][c]["acc"] - v["acc"] for c, v in s["mmlu"]["by_category"].items()}
            w = max(dl, key=dl.get)
            line += (f"; held-out MMLU (5-shot) worst category {w} {_pts(dl[w])} pts, overall "
                     f"{_pts(dense['mmlu']['overall'] - s['mmlu']['overall'])} pts")
        out.append(line)
    return out


def _ranked_run(root: Path, folder: str) -> list[str]:
    A = root / folder / "_analysis"
    fr, mat, aud = _opt(A / "frozen" / "frozen.json"), _opt(A / "audit_materialized" / "audit_materialized.json"), _opt(A / "audit" / "audit.json")
    out = []
    if fr:
        out.append(f"frozen `{fr['state_id']}`: {fr['accepts']} accepts / {fr['attempts']} proposals, {fr['questions']:,} paired "
                   f"gate questions, stop {fr['stop_reason']}, MLP sparsity {100 * fr['mlp_sparsity']:.2f} %")
    if mat:
        sz = mat["size"]
        out.append(f"materialised: {sz['parameters_removed']:,} params removed ({100 * sz['fraction_of_total_parameters_removed']:.2f} %)")
        for s in mat["sets"].values():
            st = s["stats"]["materialized"]
            c = st["conclusion"]
            cats = [k for k in st if k not in ("overall", "conclusion")]
            mx = max(cats, key=lambda k: st[k]["ub_simultaneous"])
            out.append(f"{s['name']} ({s['label']}): worst {c['worst_category']} {_pts(c['worst_loss'])}, largest simultaneous UB "
                       f"{mx} {_pts(st[mx]['ub_simultaneous'])} -> {c['verdict']}")
    elif aud:
        for s in aud["sets"].values():
            c = s["stats"]["conclusion"]
            out.append(f"{s['name']} ({s['label']}, masked): worst {c['worst_category']} {_pts(c['worst_loss'])} -> {c['verdict']}")
    return out


def _boundary(root: Path, folder: str) -> list[str]:
    B, cal = _opt(root / folder / "boundary.json"), _opt(root / folder / "calibration.json")
    out = []
    for s in (B or {}).get("states", []):
        out.append(f"{s['name']}: {100 * s['size']['fraction_of_total_parameters_removed']:.2f} % params, largest UB "
                   + ", ".join(f"{v['name']} {_pts(v['max_ub'])}" for v in s["sets"].values()) + f", meets audit: {s['meets_audit']}")
    if cal:
        out.append(f"margin calibration: chosen {cal['chosen_margin_pts']} pts (conclusive: {cal['conclusive']})")
    return out


def _compare(root: Path, folder: str) -> list[str]:
    S = _opt(root / folder / "comparison.json")
    out = []
    for k, a in (S or {}).get("arms", {}).items():
        m = a.get("materialized") or {}
        sz = m.get("size") or {}
        verdicts = "; ".join(f"{s['name']}: {s['stats']['materialized']['conclusion']['verdict']}" for s in (m.get("sets") or {}).values())
        out.append(f"{k} ({a['label']}): {100 * sz.get('fraction_of_total_parameters_removed', 0):.2f} % params, "
                   f"{a['cost']['paired_questions']:,} paired questions, {a['cost']['search_s']:.0f} s search; {verdicts}")
    return out


def _ranker(root: Path, folder: str) -> list[str]:
    E = _opt(root / folder / "evaluation.json")
    if not E:
        return []
    import numpy as np

    test = [r for r in E["results"] if r["fold"].startswith("T")]
    sels = list(dict.fromkeys(r["selector"] for r in test))
    return [f"unseen trajectories, mean worst-category loss: " + ", ".join(
        f"{s} {_pts(float(np.mean([r['worst'] for r in test if r['selector'] == s])))}" for s in sels)]


def _fixed_state(root: Path, folder: str) -> list[str]:
    S = _opt(root / folder / "summary.json")
    return [f"{c}: mean worst-category loss {_pts(v['mean_worst_loss'])}, accepts {v['accept']}/{v['valid']}"
            for c, v in (S or {}).get("summary", {}).items()]


def _gap(root: Path, folder: str) -> list[str]:
    S = _opt(root / folder / "summary.json")
    return [f"{k}: overall {_pts(v['overall'])}, worst {v['worst']} {_pts(v[v['worst']])}" for k, v in (S or {}).get("table", {}).items()]


def _adaptive_replay(root: Path, folder: str) -> list[str]:
    S = _opt(root / folder / "replay" / "summary.json")
    return [f"{p}: {v['questions']:,} questions, accept/reject/inconclusive {v['accept']}/{v['reject']}/{v['inconclusive']}, "
            f"false accepts {v['false_accept']}" for p, v in (S or {}).get("per_policy", {}).items()]


def _gate_design(root: Path, folder: str) -> list[str]:
    S = _opt(root / folder / "design" / "summary.json")
    return [f"locked: {S['locked']}"] if S else []


EXTRACTORS = {"search_runs": _search_runs, "ranked_run": _ranked_run, "boundary": _boundary, "compare": _compare,
              "ranker": _ranker, "fixed_state": _fixed_state, "gap": _gap, "adaptive_replay": _adaptive_replay,
              "gate_design": _gate_design}


# ============================================================================ the index
def build_index(registry: Path, root: Path) -> Path:
    from .gradient_audit import _code_hashes

    reg = yaml.safe_load(registry.read_text(encoding="utf-8"))
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    current = _code_hashes()
    exps = []
    for e in reg["experiments"]:
        folder = root / e["folder"]
        rec = {**{k: e.get(k) for k in ("id", "title", "folder", "label", "docs")}, "exists": folder.exists(),
               "configs": [{"path": c, "sha256": sha256_file(REPO / c) if (REPO / c).exists() else None} for c in e["configs"]],
               "reports": [{"path": str(root / r), "exists": (root / r).exists()} for r in e["reports"]]}
        runs = []
        for rj in sorted(folder.glob("*/run.json")) + sorted(folder.glob("run.json")):
            j = _opt(rj)
            if j:
                runs.append({"run": rj.parent.name, "science_hash": j.get("science_hash"), "environment": j.get("environment"),
                             "created": j.get("created")})
        rec["runs"] = runs
        regs = sorted(folder.glob("**/registration.json"))
        if regs:
            r = _opt(regs[0])
            rec["registered_code_sha256_16"] = r.get("code_sha256_16")
            rec["code_files_changed_since_registration"] = sorted(
                f for f, h in (r.get("code_sha256_16") or {}).items() if current.get(f) != h)
            rec["registered_at"] = r.get("registered_at")
        try:
            rec["headline"] = EXTRACTORS[e["extractor"]](root, e["folder"]) if folder.exists() else ["folder missing"]
        except Exception as ex:  # noqa: BLE001
            rec["headline"] = [f"extractor failed: {ex}"]
        exps.append(rec)
    tests = sorted(p for p in (root / "_tests").glob("*") if p.is_dir()) if (root / "_tests").exists() else []
    idx = {"created": time.time(), "root": str(root), "git": {"commit": head, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                                                            "working_tree_clean": status == "", "dirty_files": status.splitlines()[:20]},
           "code_sha256_16_at_index_time": current, "environment": environment(), "experiments": exps,
           "checkpoint_archive": _opt(root / "_archive" / "checkpoints" / "index.json"),
           "tests": [{"dir": str(t), "record": _opt(t / "record.json")} for t in tests],
           "backups": _opt(root / "_archive" / "backups.json")}
    atomic_write_json(root / "results_index.json", idx)
    atomic_write_text(root / "RESULTS_INDEX.md", render(idx))
    return root / "RESULTS_INDEX.md"


def render(idx: dict) -> str:
    g = idx["git"]
    L = ["# CAP-Prune results index", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(idx['created']))} from `{idx['root']}`. Code: commit "
         f"`{g['commit']}` on `{g['branch']}` (working tree clean: {g['working_tree_clean']}). Environment: {idx['environment']}.", "",
         "Code provenance: results produced before the first research commit came from uncommitted code; for runs with a "
         "`registration.json` the SHA-256 (16 hex) of the key source files at registration is listed with the files that have "
         "changed since. Headline numbers are copied from each experiment's own JSON outputs.", "",
         "| experiment | label | configs (sha256/12) | reports | code changed since registration |", "|---|---|---|---|---|"]
    for e in idx["experiments"]:
        cf = "<br>".join(f"`{c['path']}` {str(c['sha256'])[:12]}" for c in e["configs"])
        rp = "<br>".join(f"`{r['path']}`{'' if r['exists'] else ' (MISSING)'}" for r in e["reports"]) or "-"
        ch = (", ".join(e["code_files_changed_since_registration"]) or "none") if "code_files_changed_since_registration" in e else "not registered (pre-commit code)"
        L.append(f"| **{e['id']}**: {e['title']} | {e['label']} | {cf} | {rp} | {ch} |")
    L += ["", "## Headline numbers", ""]
    for e in idx["experiments"]:
        L.append(f"### {e['id']}: {e['title']}")
        L.append(f"Folder `{e['folder']}` ({'present' if e['exists'] else 'MISSING'}); docs `{e['docs']}`; runs: "
                 + (", ".join(f"{r['run']} (science {r['science_hash']})" for r in e["runs"]) or "none") + ".")
        L += [f"* {h}" for h in e["headline"]] + [""]
    ca = idx.get("checkpoint_archive")
    if ca:
        L += ["## Checkpoint archive (every accepted state, rebuilt and verified)", ""]
        L += [f"* `{r['run_dir']}`: {r['status']}, {r.get('states', 0)} states" for r in ca["runs"]] + [""]
    L += ["## Test records", ""]
    for t in idx["tests"]:
        rec = t.get("record") or {}
        L.append(f"* `{t['dir']}`: {rec.get('summary', 'see folder')} (commit `{str(rec.get('commit', 'n/a'))[:12]}`)")
    b = idx.get("backups")
    if b:
        L += ["", "## Backups (verified copies of the runs root)", ""]
        for x in b.get("backups", []):
            L += [f"* {x['created_iso']}: `{x['zip_name']}` ({x['zip_bytes'] / 1e6:.0f} MB, {x['files']} files, sha256 "
                  f"`{x['zip_sha256'][:16]}...`) -> " + "; ".join(f"`{c['path']}` verified={c['verified']}" for c in x["copies"])]
    return "\n".join(L) + "\n"


# ============================================================================ backup
def backup(root: Path, dests: list[Path], stage: Path) -> dict:
    """Zip ``root`` (all files) + manifest, copy to every destination, verify every copy."""
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    commit = git("rev-parse", "--short=12", "HEAD")
    name = f"cap-prune-runs_{stamp}_{commit}.zip"
    stage.mkdir(parents=True, exist_ok=True)
    zp = stage / name
    files = sorted(p for p in root.rglob("*") if p.is_file())
    manifest = []
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            rel = p.relative_to(root).as_posix()
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            manifest.append({"path": rel, "bytes": p.stat().st_size, "sha256": h.hexdigest()})
            z.write(p, rel)
        z.writestr("BACKUP_MANIFEST.json", json.dumps({"root": str(root), "created": stamp, "commit": commit,
                                                      "files": manifest}, indent=1))
    zsha = sha256_file(zp)
    copies = []
    for d in dests:
        d.mkdir(parents=True, exist_ok=True)
        tgt = d / name
        if tgt.resolve() != zp.resolve():  # the staging folder may itself be a destination
            shutil.copy2(zp, tgt)
        (d / (name + ".sha256")).write_text(f"{zsha}  {name}\n", encoding="utf-8")
        (d / (name.replace(".zip", "") + "_manifest.json")).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        ok = sha256_file(tgt) == zsha
        with zipfile.ZipFile(tgt) as z:
            bad = z.testzip()
            n = len(z.namelist())
        copies.append({"path": str(tgt), "verified": ok and bad is None and n == len(files) + 1, "sha256_match": ok,
                       "zip_test_first_bad_member": bad, "members": n})
    if all(zp.resolve() != (d / name).resolve() for d in dests):
        zp.unlink()
    rec = {"created_iso": stamp, "zip_name": name, "zip_sha256": zsha, "zip_bytes": (dests[0] / name).stat().st_size,
           "files": len(files), "bytes_uncompressed": sum(m["bytes"] for m in manifest), "commit": commit, "copies": copies,
           "seconds": time.time() - t0}
    led = root / "_archive" / "backups.json"
    data = _opt(led) or {"backups": []}
    data["backups"].append(rec)
    atomic_write_json(led, data)
    return rec
