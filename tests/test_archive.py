"""Research-record keeping: every accepted checkpoint is archived (rebuilt + verified), future runs keep every mask,
the results index reads the experiments' own outputs, and backups are verified copies (tiny Qwen3, CPU)."""
from __future__ import annotations

import json
import shutil
import zipfile

import pytest

from conftest import ROOT
from live_smoke import LIVE_SMOKE, make_live_smoke_data
from test_recovery import cli

GL = ROOT / "configs" / "gradient_live_smoke.yaml"


def test_keep_all_masks_keeps_every_state(tmp_path):
    from capprune.utils.checkpoint import CheckpointStore

    def run(keep_all):
        d = tmp_path / ("all" if keep_all else "default")
        st = CheckpointStore(d, keep=1, keep_all_masks=keep_all)
        for i in range(5):
            st.save_masks(f"s{i}", lambda p, i=i: p.write_bytes(bytes([i]) * 10), 10)
            st.save(i + 1, {"state_id": f"s{i}"})
        return sorted(p.stem for p in (d / "masks").glob("*.safetensors"))

    assert run(True) == [f"s{i}" for i in range(5)]
    assert len(run(False)) < 5  # the old behaviour prunes masks of states no retained checkpoint references


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arch")
    make_live_smoke_data(tmp / "data-live")
    sets = ["--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}", "--set", "reliability.restart_backoff_s=[0.2]",
            "--set", "reliability.keep_all_masks=false"]  # like the older runs: only the last states keep their masks
    rc, out = cli(["supervise", "--config", str(GL), "--set", f"experiment.runs_dir={(tmp / 'root' / 'grad').as_posix()}",
                   "--set", "gate.capability.tolerance=0.6", "--set", "ranked_search.max_proposals=5", *sets])
    assert rc == 0, out
    rc, out = cli(["supervise", "--config", str(LIVE_SMOKE), "--methods", "greedy_cap",
                   "--set", f"experiment.runs_dir={(tmp / 'root' / 'live').as_posix()}", *sets])
    assert rc == 0, out
    return tmp


def test_archive_rebuilds_and_verifies_every_accepted_state(runs):
    from capprune.archive import archive_all

    root = runs / "root"
    grad = root / "grad" / "gradient_top1__t00__s0"
    assert len(list((grad / "masks").glob("*.safetensors"))) < 5  # pruned: intermediate states are not stored
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal"))}
    idx = json.loads(archive_all(root, runs / "archive").read_text())
    st = {r["run_dir"].split("\\")[-1].split("/")[-1]: r for r in idx["runs"]}
    g = st["gradient_top1__t00__s0"]
    assert g["status"] == "archived" and g["states"] == g["accepted"] == 5
    m = json.loads((runs / "archive" / "grad" / "gradient_top1__t00__s0" / "manifest.json").read_text())
    assert all(s["checks"]["journal_trial_mask_hash"] for s in m["states"])
    assert sum("stored" in s["source"] for s in m["states"]) >= 1 and m["final_checkpoint_matches_last_archived_state"] is True
    live = st["greedy_cap__t50__s0"]  # search-loop run: verified against the journal's sparsity and the stored masks
    assert live["status"] == "archived" and live["states"] == live["accepted"] >= 1
    after = {p: p.read_bytes() for p in before}
    assert after == before  # the runs are only read


def test_archive_refuses_a_chain_that_does_not_reproduce(runs, tmp_path):
    from capprune.archive import archive_all

    src = runs / "root" / "grad"
    bad = tmp_path / "root" / "grad"
    shutil.copytree(src, bad)
    j = bad / "gradient_top1__t00__s0" / "journal.jsonl"
    lines = []
    for line in j.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "iteration" and r["iteration"] == 2:
            r["trial_mask_hash"] = "0" * 20
        lines.append(json.dumps(r))
    j.write_text("\n".join(lines) + "\n", encoding="utf-8")
    idx = json.loads(archive_all(tmp_path / "root", tmp_path / "archive").read_text())
    assert idx["runs"][0]["status"].startswith("FAILED") and not (tmp_path / "archive" / "grad" / "gradient_top1__t00__s0").exists()


def test_results_index_and_verified_backup(runs, tmp_path):
    from capprune.results_index import backup, build_index

    root = runs / "root"
    reg = tmp_path / "reg.yaml"
    reg.write_text("experiments:\n  - {id: g, title: smoke, folder: grad, configs: [configs/gradient_live_smoke.yaml], "
                   "reports: [], docs: x, extractor: ranked_run, label: smoke}\n"
                   "  - {id: l, title: smoke live, folder: live, configs: [configs/live_smoke.yaml], reports: [], docs: x, "
                   "extractor: search_runs, label: smoke}\n", encoding="utf-8")
    p = build_index(reg, root)
    idx = json.loads((root / "results_index.json").read_text())
    assert [e["id"] for e in idx["experiments"]] == ["g", "l"] and idx["git"]["commit"]
    assert all(c["sha256"] for e in idx["experiments"] for c in e["configs"])
    assert any("greedy_cap" in h for h in idx["experiments"][1]["headline"])
    assert p.read_text(encoding="utf-8").startswith("# CAP-Prune results index")
    rec = backup(root, [tmp_path / "b1", tmp_path / "b2"], tmp_path / "b1")
    assert all(c["verified"] for c in rec["copies"]) and len(rec["copies"]) == 2
    with zipfile.ZipFile(tmp_path / "b2" / rec["zip_name"]) as z:
        man = json.loads(z.read("BACKUP_MANIFEST.json"))
        assert len(man["files"]) == rec["files"] and "RESULTS_INDEX.md" in z.namelist()
    assert json.loads((root / "_archive" / "backups.json").read_text())["backups"][-1]["zip_sha256"] == rec["zip_sha256"]
