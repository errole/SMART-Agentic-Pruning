"""Fault injection for recovery testing (inert unless CAPPRUNE_FAULTS is set).

    CAPPRUNE_FAULTS="after_controller@3,mid_final_mmlu@1,hang_profile@2"
    CAPPRUNE_FAULT_STATE=/path/to/faults.json

``name@k`` hard-kills the process (``os._exit(137)``, like SIGKILL or a power
cut) the k-th time - counted across restarts - that execution reaches
``fault_point(name)``.  Each fault fires once.  Names starting with ``hang_``
block forever instead (to exercise the supervisor's hang detection).
"""
from __future__ import annotations

import json
import os
import time


def _spec() -> dict[str, int]:
    out = {}
    for item in os.environ.get("CAPPRUNE_FAULTS", "").split(","):
        if "@" in item:
            n, k = item.strip().split("@")
            out[n] = int(k)
    return out


def fault_point(name: str) -> None:
    if not os.environ.get("CAPPRUNE_FAULTS"):
        return
    want = _spec()
    if name not in want:
        return
    path = os.environ.get("CAPPRUNE_FAULT_STATE")
    st = {"counts": {}, "fired": []}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
    st["counts"][name] = st["counts"].get(name, 0) + 1
    tag = f"{name}@{want[name]}"
    fire = st["counts"][name] == want[name] and tag not in st["fired"]
    if fire:
        st["fired"].append(tag)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)
            f.flush()
            os.fsync(f.fileno())
    if not fire:
        return
    if name.startswith("hang_"):
        while True:
            time.sleep(3600)
    os._exit(137)
