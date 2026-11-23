"""Archive every accepted checkpoint of finished runs (``capprune archive-checkpoints``).

A run's checkpoint store keeps the masks of only its last few states (``reliability.keep_checkpoints``), so the
intermediate accepted checkpoints of older runs exist only implicitly: the journal records every accepted action and
the run persisted the profile each action was built from.  This rebuilds the whole accepted chain from the dense
model, exactly as the run built it, and writes every state's mask to an archive OUTSIDE the run directory (the run's
own files are only read).

Verification of every rebuilt state, before it is written:
* ranked_search runs: the journal's trial mask hash (``trial_mask_hash``) and state id must match;
* search runs: the journal's MLP sparsity after the step (``rho_after``) must match;
* both: wherever the run (or a frozen analysis copy) still stores a mask for that state, its content must be
  identical; the last accepted state must equal the run's final checkpoint.
A mismatch aborts that run's archive (nothing partial is presented as verified).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .utils.io import atomic_write_json, sha256_file
from .utils.log import get_logger

log = get_logger("archive")


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


def _stored_masks(run_dir: Path) -> dict[str, Path]:
    """state id -> a stored mask file: the run's own masks plus frozen / boundary copies made by the analyses."""
    out = {p.stem: p for p in (run_dir / "masks").glob("*.safetensors")}
    for p in run_dir.parent.glob("_analysis/frozen/*.safetensors"):
        out.setdefault(p.stem, p)
    return out


def archive_run(run_dir: Path, dest: Path, model_cache: dict) -> dict:
    import torch

    from .fixed_state import _geometry
    from .gradient_live import build_trial
    from .model import build_units, load_model
    from .pruning.masks import DenseStore, PruneState
    from .pruning.profile import load_profile
    from .pruning.wanda import raise_unit
    from .ranker import canon, mask_hash
    from .utils.checkpoint import CheckpointStore
    from .utils.io import read_json

    rj = read_json(run_dir / "run.json")
    rc = dict(rj["config"])
    # a local model path is relative to the run's config file (recorded as config_path since this was added; every
    # config lives in the repo's configs/ folder, so that is the fallback for older runs)
    rc["_config_dir"] = str(Path(rj["config_path"]).parent) if rj.get("config_path") else \
        str(Path(__file__).resolve().parents[1] / "configs")
    kind, mode = rc["run"]["kind"], rc["pruning"]["mode"]
    att = _attempts(run_dir)
    acc = [a for a in att if a.get("gate_decision") == "accept"]
    rec = {"run_dir": str(run_dir), "run_id": run_dir.name, "kind": kind, "mode": mode, "science_hash": rj.get("science_hash"),
           "journal_sha256": sha256_file(run_dir / "journal.jsonl"), "attempts": len(att), "accepted": len(acc), "states": []}
    if not acc:
        return {**rec, "status": "nothing to archive (no accepted state)"}
    key = (rc["model"]["name_or_path"], rc["model"].get("dtype"), mode, tuple(rc["pruning"].get("eligible", [])))
    if key not in model_cache:
        model_cache.clear()
        model = load_model(rc)
        units = build_units(model, mode, rc["pruning"].get("eligible", []))
        model_cache[key] = (model, units, DenseStore(units))
    model, units, dense = model_cache[key]
    PruneState(units, mode).apply(dense)  # the dense weights: the scores of kept channels equal the run's
    dev = next(model.parameters()).device
    stored = _stored_masks(run_dir)
    geo = _geometry(rc, units[0].channels) if mode == "structured" else None
    st = PruneState(units, mode)
    dest.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(acc, 1):
        t = a["iteration"]
        if kind == "ranked_search":
            if a["parent_mask_hash"] != mask_hash(st):
                raise RuntimeError(f"{run_dir.name} attempt {t}: parent does not match the rebuilt chain")
            pt = int(a.get("profile_attempt") or t)
            act, _, _ = load_profile(run_dir / "profiles" / f"iter-{pt:06d}.safetensors", dev)
            action = {u: int(b) for u, b in a["chosen"]["action"].items()}
            if canon(action) != a["chosen"]["action_key"]:
                raise RuntimeError(f"{run_dir.name} attempt {t}: recorded action and key disagree")
            build_trial(st, dense, action, act, rc, geo)
            checks = {"journal_trial_mask_hash": mask_hash(st) == a["trial_mask_hash"]}
        else:
            act, _, _ = load_profile(run_dir / "profiles" / f"iter-{t:06d}.safetensors", dev)
            for name, inc in a["decisions"]:
                raise_unit(st, name, st.unit_sparsity(name) + float(inc), act, rc)
            checks = {"journal_rho_after": abs(st.global_sparsity() - float(a["rho_after"])) < 5e-6}
        sid = a.get("cand_id") or a.get("trial_id")
        if sid in stored:
            ref = PruneState(units, mode)
            ref.load(stored[sid])
            checks["stored_mask_identical"] = all(bool(torch.equal(ref.masks[n], st.masks[n])) for n in st.masks)
        if not all(checks.values()):
            raise RuntimeError(f"{run_dir.name} attempt {t} ({sid}): rebuilt state fails verification {checks}")
        p = dest / f"{i:02d}-attempt{t:03d}-{sid}.safetensors"
        st.save(p)
        rec["states"].append({"step": i, "attempt": t, "state_id": sid, "mlp_sparsity": st.global_sparsity(),
                              "mask_hash": mask_hash(st), "file": p.name, "sha256": sha256_file(p),
                              "source": "replayed + identical to a stored mask" if sid in stored else "replayed from the journal",
                              "checks": checks})
    try:
        S, _ = CheckpointStore(run_dir).load_latest()
        final_ok = S.get("state_id") == rec["states"][-1]["state_id"]
        if final_ok and S.get("state_id") in stored:
            ref = PruneState(units, mode)
            ref.load(stored[S["state_id"]])
            final_ok = mask_hash(ref) == rec["states"][-1]["mask_hash"]
    except Exception as e:  # noqa: BLE001
        final_ok = f"no readable checkpoint: {e}"
    rec["final_checkpoint_matches_last_archived_state"] = final_ok
    if final_ok is not True:
        raise RuntimeError(f"{run_dir.name}: last archived state does not match the run's final checkpoint ({final_ok})")
    atomic_write_json(dest / "manifest.json", rec)
    log.info("%s: %d accepted states archived and verified", run_dir.name, len(rec["states"]))
    return {**rec, "status": "archived"}


def archive_all(root: Path, out: Path, only: list[str] | None = None) -> Path:
    """Every run directory under ``root`` (``<experiment>/<run>/journal.jsonl``); writes ``out/<experiment>/<run>/``."""
    runs = sorted(p.parent for p in root.glob("*/*/journal.jsonl") if not p.parent.parent.name.startswith("_"))
    if only:
        runs = [r for r in runs if r.parent.name in only]
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    index = {"created": time.time(), "root": str(root), "runs": []}
    for r in runs:
        dest = out / r.parent.name / r.name
        tmp = dest.with_name(dest.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            res = archive_run(r, tmp, cache)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(tmp, ignore_errors=True)
            index["runs"].append({"run_dir": str(r), "status": f"FAILED: {e}"})
            log.error("%s: %s", r, e)
            continue
        if res["status"] == "archived":
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        index["runs"].append({k: v for k, v in res.items() if k != "states"} | {"states": len(res.get("states", []))})
    atomic_write_json(out / "index.json", index)
    return out / "index.json"
