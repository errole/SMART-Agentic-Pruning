"""Checkpoint store with verification, rotation, mirroring and fallback.

Layout inside a run directory::

    ckpt/ckpt-000012.json     sealed JSON (payload + sha256) - small, written every iteration
    ckpt/LATEST               name of the newest complete checkpoint
    masks/<state_id>.safetensors  pruning masks of an *accepted* model state (content-addressed)
    masks/<state_id>.sha256       checksum sidecar

Writing order guarantees a crash at any instant leaves a loadable checkpoint:
(1) mask file + sidecar, (2) checkpoint JSON, (3) LATEST pointer.  A pointer
is only ever moved to a fully written, checksummed checkpoint.

On load we try LATEST, then every older checkpoint (newest first), then the
same in the mirror directory.  Each candidate's JSON seal and its mask file's
SHA-256 are verified before it is used.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from .io import (
    DiskSpaceError,
    atomic_write_text,
    copy_file_atomic,
    ensure_free_space,
    finalize_tmp,
    read_sealed_json,
    sha256_file,
    write_sealed_json,
)
from .faults import fault_point
from .log import get_logger

log = get_logger("checkpoint")
_CKPT_RE = re.compile(r"^ckpt-(\d{6})\.json$")


class NoCheckpoint(Exception):
    pass


class CheckpointStore:
    def __init__(self, run_dir: str | os.PathLike, keep: int = 3, mirror_dir: str | os.PathLike | None = None,
                 min_free_gb: float = 2.0, keep_all_masks: bool = False):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "ckpt"
        self.mask_dir = self.run_dir / "masks"
        self.keep = max(1, int(keep))
        # keep_all_masks: never delete the mask file of an accepted state (only checkpoint JSONs rotate), so every
        # intermediate checkpoint of a run stays on disk (structured masks are ~25 KB per state)
        self.keep_all_masks = bool(keep_all_masks)
        self.mirror = Path(mirror_dir) if mirror_dir else None
        self.min_free_gb = min_free_gb
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ masks
    def mask_path(self, state_id: str, root: Path | None = None) -> Path:
        return (root or self.run_dir) / "masks" / f"{state_id}.safetensors"

    def has_valid_masks(self, state_id: str, root: Path | None = None) -> bool:
        p = self.mask_path(state_id, root)
        side = p.with_suffix(".sha256")
        if not p.exists() or not side.exists():
            return False
        try:
            return side.read_text(encoding="utf-8").strip() == sha256_file(p)
        except OSError:
            return False

    def save_masks(self, state_id: str, writer: Callable[[Path], None], approx_bytes: int) -> str:
        """``writer(tmp_path)`` must write the safetensors file. Returns its sha256."""
        p = self.mask_path(state_id)
        if self.has_valid_masks(state_id):
            return p.with_suffix(".sha256").read_text(encoding="utf-8").strip()
        ensure_free_space(self.mask_dir, approx_bytes * 2, self.min_free_gb)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        try:
            writer(tmp)
            finalize_tmp(tmp, p)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        digest = sha256_file(p)
        atomic_write_text(p.with_suffix(".sha256"), digest + "\n")
        self._mirror_file(p)
        self._mirror_file(p.with_suffix(".sha256"))
        return digest

    # ------------------------------------------------------------------ checkpoints
    def _list(self, root: Path) -> list[tuple[int, Path]]:
        d = root / "ckpt"
        if not d.exists():
            return []
        out = []
        for f in d.iterdir():
            m = _CKPT_RE.match(f.name)
            if m:
                out.append((int(m.group(1)), f))
        return sorted(out, reverse=True)

    def save(self, index: int, payload: dict) -> Path:
        """Persist a checkpoint. ``payload['state_id']`` must reference saved masks (or be None)."""
        sid = payload.get("state_id")
        if sid and not self.has_valid_masks(sid):
            raise RuntimeError(f"refusing to checkpoint: masks for state {sid} are missing or corrupt")
        try:
            ensure_free_space(self.ckpt_dir, 1 << 20, self.min_free_gb)
        except DiskSpaceError:
            self._prune(force_keep=1)
            ensure_free_space(self.ckpt_dir, 1 << 20, self.min_free_gb)
        path = self.ckpt_dir / f"ckpt-{index:06d}.json"
        write_sealed_json(path, payload)
        fault_point("before_latest_pointer")
        atomic_write_text(self.ckpt_dir / "LATEST", path.name + "\n")
        self._mirror_file(path)
        self._mirror_file(self.ckpt_dir / "LATEST")
        self._prune()
        self._prune_mirror()
        return path

    def _prune_mirror(self) -> None:
        if not self.mirror or not (self.mirror / "ckpt").exists():
            return
        keep_names = {p.name for _, p in self._list(self.run_dir)}
        for _, p in self._list(self.mirror):
            if p.name not in keep_names:
                try:
                    p.unlink()
                except OSError:
                    pass
        if self.keep_all_masks:
            return
        refs = self._referenced_states(self.run_dir)
        for m in (self.mirror / "masks").glob("*.safetensors") if (self.mirror / "masks").exists() else []:
            if m.stem not in refs:
                for p in (m, m.with_suffix(".sha256")):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def _referenced_states(self, root: Path) -> set[str]:
        refs = set()
        for _, f in self._list(root):  # only retained checkpoints remain on disk
            try:
                p = read_sealed_json(f)
                for key in ("state_id", "best_state_id"):
                    if p.get(key):
                        refs.add(p[key])
                refs.update(p.get("pinned_states", []))
            except Exception:
                continue
        return refs

    def _prune(self, force_keep: int | None = None) -> None:
        """Keep the ``keep`` newest checkpoints PLUS the newest checkpoint of each of the
        last ``keep`` distinct model states, so that a corrupted mask file can always be
        bypassed by falling back to an older accepted state."""
        keep = force_keep or self.keep
        ckpts = self._list(self.run_dir)
        retained = {p for _, p in ckpts[:keep]}
        if force_keep is None:
            states_seen: list[str | None] = []
            for _, p in ckpts:
                try:
                    sid = read_sealed_json(p).get("state_id")
                except Exception:
                    continue
                if sid not in states_seen:
                    states_seen.append(sid)
                    retained.add(p)
                if len(states_seen) >= keep:
                    break
        for _, f in ckpts:
            if f in retained:
                continue
            try:
                f.unlink()
            except OSError:
                pass
        if self.keep_all_masks:
            return
        refs = self._referenced_states(self.run_dir)
        for m in self.mask_dir.glob("*.safetensors"):
            if m.stem not in refs:
                for p in (m, m.with_suffix(".sha256")):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def load_latest(self) -> tuple[dict, Path]:
        """Return ``(payload, root)`` of the newest valid checkpoint, verifying masks.

        ``root`` is the run dir or the mirror dir the checkpoint came from (the
        mask file must be read from the same root).
        """
        roots = [self.run_dir] + ([self.mirror] if self.mirror else [])
        errors = []
        for root in roots:
            # Every checkpoint is sealed (checksummed), so the highest-numbered one that verifies
            # is the newest complete state - even if a crash hit before LATEST was updated.
            candidates = [p for _, p in self._list(root)]
            for c in candidates:
                try:
                    payload = read_sealed_json(c)
                except Exception as e:  # corrupt or partial
                    errors.append(f"{c}: {e}")
                    continue
                sid = payload.get("state_id")
                if sid and not self.has_valid_masks(sid, root):
                    errors.append(f"{c}: masks {sid} missing/corrupt")
                    continue
                if errors:
                    log.warning("Recovered from checkpoint %s after skipping: %s", c, "; ".join(errors))
                if root != self.run_dir:
                    self._restore_from_mirror(payload, c)
                return payload, self.run_dir
        raise NoCheckpoint("; ".join(errors) or "no checkpoints")

    # ------------------------------------------------------------------ mirror
    def _mirror_target(self, p: Path) -> Path | None:
        if not self.mirror:
            return None
        return self.mirror / p.relative_to(self.run_dir)

    def _mirror_file(self, p: Path) -> None:
        tgt = self._mirror_target(p)
        if tgt is None:
            return
        try:
            copy_file_atomic(p, tgt)
        except Exception as e:  # mirror is best-effort; never fail the run because of it
            log.warning("mirror copy of %s failed: %s", p.name, e)

    def _restore_from_mirror(self, payload: dict, ckpt_path: Path) -> None:
        log.warning("Primary checkpoints unusable; restoring from mirror %s", ckpt_path)
        copy_file_atomic(ckpt_path, self.ckpt_dir / ckpt_path.name)
        atomic_write_text(self.ckpt_dir / "LATEST", ckpt_path.name + "\n")
        for sid in {payload.get("state_id"), payload.get("best_state_id"), *payload.get("pinned_states", [])}:
            if sid and not self.has_valid_masks(sid):
                for suffix in (".safetensors", ".sha256"):
                    src = self.mirror / "masks" / f"{sid}{suffix}"
                    if src.exists():
                        copy_file_atomic(src, self.mask_dir / f"{sid}{suffix}")


