"""Unit tests for crash-safety primitives: atomic writes, sealed JSON, journal,
SQLite caches, run lock, and checkpoint fallback/mirroring/retention."""
import json
import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from capprune.utils.cache import KVCache
from capprune.utils.checkpoint import CheckpointStore, NoCheckpoint
from capprune.utils.io import (append_jsonl, atomic_write_json, read_jsonl, read_sealed_json,
                               write_sealed_json)
from capprune.utils.runtime import RunLock, RunLockedError, pid_alive


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"x": 1})
    atomic_write_json(p, {"x": 2})
    assert json.loads(p.read_text()) == {"x": 2}
    assert [f.name for f in tmp_path.iterdir()] == ["a.json"]


def test_sealed_json_detects_corruption(tmp_path):
    p = tmp_path / "s.json"
    write_sealed_json(p, {"iteration": 7})
    assert read_sealed_json(p) == {"iteration": 7}
    obj = json.loads(p.read_text())
    obj["payload"]["iteration"] = 8  # silent bit-flip / manual edit
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        read_sealed_json(p)
    p.write_text('{"payload": {"iter')  # torn write
    with pytest.raises(Exception):
        read_sealed_json(p)


def test_jsonl_tolerates_torn_last_line(tmp_path):
    p = tmp_path / "j.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    with open(p, "a") as f:
        f.write('{"a": 3, "b": ')  # crash mid-append
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]


def test_kvcache_roundtrip_and_corruption_quarantine(tmp_path):
    c = KVCache(tmp_path / "c.sqlite")
    c.put_many({"k1": {"v": 1}, "k2": True})
    assert c.get("k1") == {"v": 1} and c.get_many(["k1", "k2", "k3"]) == {"k1": {"v": 1}, "k2": True}
    c.close()
    (tmp_path / "c.sqlite").write_bytes(b"this is not a database" * 100)
    for extra in ("-wal", "-shm"):
        try:
            os.unlink(str(tmp_path / "c.sqlite") + extra)
        except FileNotFoundError:
            pass
    c2 = KVCache(tmp_path / "c.sqlite")  # quarantines the corrupt file and starts fresh
    assert c2.get("k1") is None
    assert any(".corrupt-" in f.name for f in tmp_path.iterdir())


def test_run_lock_blocks_live_owner_and_takes_over_stale(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RunLockedError):
            other = RunLock(tmp_path)
            # pretend to be a different live process: the lock owner (us) is alive
            import capprune.utils.runtime as rt

            orig = os.getpid
            try:
                rt.os.getpid = lambda: orig() + 1
                other.acquire()
            finally:
                rt.os.getpid = orig
    # stale lock from a dead PID is taken over
    (tmp_path / "LOCK").write_text(json.dumps({"pid": 999999, "host": __import__("socket").gethostname(), "since": 0}))
    assert not pid_alive(999999)
    with RunLock(tmp_path) as lk:
        assert lk.held


def _masks(store, sid, value=1):
    store.save_masks(sid, lambda p: save_file({"m": np.full(8, value, np.uint8)}, str(p)), 64)


def test_checkpoint_fallback_corrupt_latest_and_corrupt_masks(tmp_path):
    st = CheckpointStore(tmp_path, keep=3)
    _masks(st, "A")
    st.save(1, {"state_id": "A", "i": 1})
    _masks(st, "B", 2)
    for i in range(2, 7):  # many rejected steps that all reference state B
        st.save(i, {"state_id": "B", "i": i})
    p, _ = st.load_latest()
    assert p["i"] == 6
    # retention kept the newest checkpoint of the older distinct state A
    assert (tmp_path / "ckpt" / "ckpt-000001.json").exists()
    # 1) corrupt the newest checkpoint file -> previous checkpoint is used
    (tmp_path / "ckpt" / "ckpt-000006.json").write_text("{garbage")
    p, _ = st.load_latest()
    assert p["i"] == 5
    # 2) corrupt the mask file of state B -> every checkpoint of B is skipped, falls back to A
    (tmp_path / "masks" / "B.safetensors").write_bytes(b"\x00" * 10)
    p, _ = st.load_latest()
    assert p["state_id"] == "A" and p["i"] == 1


def test_checkpoint_mirror_restores_when_primary_lost(tmp_path):
    prim, mir = tmp_path / "run", tmp_path / "mirror"
    st = CheckpointStore(prim, keep=2, mirror_dir=mir)
    _masks(st, "S1")
    st.save(1, {"state_id": "S1", "i": 1})
    st.save(2, {"state_id": "S1", "i": 2})
    # simulate losing the primary disk contents
    for d in ("ckpt", "masks"):
        for f in (prim / d).iterdir():
            f.unlink()
    st2 = CheckpointStore(prim, keep=2, mirror_dir=mir)
    p, _ = st2.load_latest()
    assert p["i"] == 2 and st2.has_valid_masks("S1")


def test_no_checkpoint(tmp_path):
    with pytest.raises(NoCheckpoint):
        CheckpointStore(tmp_path).load_latest()


def test_prepare_rebuilds_corrupt_files_instead_of_trusting_them(smoke_assets):
    from capprune.config import data_dir
    from capprune.data.prepare import prepare, verify
    from capprune.model import load_tokenizer, tokenizer_id

    cfg = smoke_assets
    d = data_dir(cfg)
    good = (d / "mmlu_probe.jsonl").read_text()
    (d / "mmlu_probe.jsonl").write_text(good[: len(good) // 2])  # truncated file
    prepare(cfg, load_tokenizer(cfg), d, tokenizer_id(cfg))
    verify(d, tokenizer_id(cfg))
    assert (d / "mmlu_probe.jsonl").read_text() == good  # deterministic rebuild, not a re-checksummed corrupt file
