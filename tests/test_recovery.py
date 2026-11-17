"""End-to-end recovery tests on the tiny Qwen3 (CPU, ~1-2 min total).

They hard-kill the worker at the most delicate moments, hang it, stop it with
a signal, take the LLM controller offline, and corrupt every checkpoint - and
check that the run always finishes with *exactly* the result of an
uninterrupted run.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import ROOT, SMOKE, endpoint_override, free_port
from mock_controller import serve

FAULTS = ("after_profile@2,after_controller@3,after_candidate_apply@4,mid_probe@3,after_masks_saved@2,"
          "before_latest_pointer@5,mid_final_mmlu@2,mid_final_fqa@1")


def cli(args, env=None, timeout=600):
    e = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    p = subprocess.run([sys.executable, "-m", "capprune", *args], cwd=ROOT, env=e, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def iterations(run_dir: Path) -> list[tuple]:
    last = {}
    for line in (run_dir / "journal.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "iteration":
            last[r["iteration"]] = (r["iteration"], r.get("cand_id"), r["outcome"], tuple(r.get("reasons", [])))
    return [last[k] for k in sorted(last)]


def summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "final" / "summary.json").read_text())


SAME = ["achieved_sparsity", "stop_reason", "iterations", "accepts", "rollbacks", "unit_sparsity"]


def assert_same_result(a: dict, b: dict):
    for k in SAME:
        assert a[k] == b[k], k
    assert a["mmlu"]["overall"] == b["mmlu"]["overall"]
    assert a["mmlu"]["by_domain"] == b["mmlu"]["by_domain"]
    assert a["freebaseqa"]["exact_match"] == b["freebaseqa"]["exact_match"]
    assert abs(a["wikitext2"]["perplexity"] - b["wikitext2"]["perplexity"]) < 1e-6
    assert a["budget"]["probe_questions"] == b["budget"]["probe_questions"]


@pytest.fixture()
def mock(smoke_assets):
    port = free_port()
    srv, state = serve(port)
    yield port, state
    srv.shutdown()


def _common(port, runs):
    return ["--set", endpoint_override(port), "--set", f"experiment.runs_dir={runs}",
            "--set", "reliability.restart_backoff_s=[0.2]"]


def test_hard_kills_at_every_critical_point_resume_identically(mock, tmp_path):
    port, state = mock
    rc, out = cli(["run", "--config", str(SMOKE), "--method", "cap_prune", "--target", "0.2", "--seed", "1",
                   *_common(port, tmp_path / "ref")])
    assert rc == 0, out
    ref_calls = state.calls
    fired = tmp_path / "fired.json"
    rc, out = cli(["supervise", "--config", str(SMOKE), "--methods", "cap_prune", "--targets", "0.2", "--seeds", "1",
                   *_common(port, tmp_path / "faulty")],
                  env={"CAPPRUNE_FAULTS": FAULTS, "CAPPRUNE_FAULT_STATE": str(fired)})
    assert rc == 0, out
    assert len(json.loads(fired.read_text())["fired"]) == len(FAULTS.split(","))  # every fault really fired
    ref, flt = tmp_path / "ref" / "cap_prune__t20__s1", tmp_path / "faulty" / "cap_prune__t20__s1"
    assert iterations(ref) == iterations(flt)
    s_ref, s_flt = summary(ref), summary(flt)
    assert_same_result(s_ref, s_flt)
    assert s_flt["attempts"] == len(FAULTS.split(",")) + 1
    assert state.calls - ref_calls == ref_calls  # crashes never caused an extra (stochastic) LLM query
    ledger = json.loads((tmp_path / "faulty" / "_matrix" / "ledger.json").read_text())
    assert ledger["cap_prune__t20__s1"]["status"] == "done"
    assert ledger["cap_prune__t20__s1"]["exits"].count(137) == len(FAULTS.split(","))


def test_all_checkpoints_corrupt_replays_from_caches(mock, tmp_path):
    port, _ = mock
    fired = tmp_path / "fired.json"
    env = {"CAPPRUNE_FAULTS": "after_controller@5", "CAPPRUNE_FAULT_STATE": str(fired)}
    args = ["run", "--config", str(SMOKE), "--method", "cap_prune", "--target", "0.2", "--seed", "2",
            *_common(port, tmp_path / "r")]
    rc, out = cli(args, env)
    assert rc == 137
    run = tmp_path / "r" / "cap_prune__t20__s2"
    for f in (run / "ckpt").glob("ckpt-*.json"):
        f.write_text("corrupted")
    rc, out = cli(args, env)
    assert rc == 0, out
    assert "ALL checkpoints" in (run / "logs" / "worker.log").read_text()
    rc, out = cli(["run", "--config", str(SMOKE), "--method", "cap_prune", "--target", "0.2", "--seed", "2",
                   *_common(port, tmp_path / "ref")])
    assert rc == 0, out
    assert_same_result(summary(tmp_path / "ref" / "cap_prune__t20__s2"), summary(run))


def test_hang_is_detected_and_worker_restarted(mock, tmp_path):
    port, _ = mock
    fired = tmp_path / "fired.json"
    rc, out = cli(["supervise", "--config", str(SMOKE), "--methods", "paper_agent", "--targets", "0.2", "--seeds", "0",
                   *_common(port, tmp_path / "h"), "--set", "reliability.hang_timeout_s=8",
                   "--set", "reliability.heartbeat_s=1"],
                  env={"CAPPRUNE_FAULTS": "hang_profile@3", "CAPPRUNE_FAULT_STATE": str(fired)})
    assert rc == 0, out
    led = json.loads((tmp_path / "h" / "_matrix" / "ledger.json").read_text())["paper_agent__t20__s0"]
    assert 124 in led["exits"] and led["status"] == "done"


def test_graceful_stop_during_controller_outage_then_resume(smoke_assets, tmp_path):
    port = free_port()  # nothing listening yet: controller is "down"
    args = ["run", "--config", str(SMOKE), "--method", "cap_prune", "--target", "0.2", "--seed", "0",
            *_common(port, tmp_path / "g"), "--set", "controller.retries=50", "--set", "controller.backoff_s=[1]"]
    # Windows has no SIGTERM: Popen.send_signal(SIGTERM) is TerminateProcess (a hard kill, exit code 1).
    # The graceful stop there is CTRL_BREAK_EVENT, which needs the child in its own process group -
    # exactly how the supervisor stops its workers.
    win = os.name == "nt"
    p = subprocess.Popen([sys.executable, "-m", "capprune", *args], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONUNBUFFERED": "1"},
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if win else 0)
    hb = tmp_path / "g" / "cap_prune__t20__s0" / "heartbeat.json"
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            if json.loads(hb.read_text()).get("phase") == "controller":
                break
        except Exception:
            pass
        time.sleep(0.5)
    p.send_signal(signal.CTRL_BREAK_EVENT if win else signal.SIGTERM)
    rc = p.wait(timeout=60)
    assert rc == 75, p.stdout.read()
    srv, _ = serve(port)
    try:
        rc, out = cli(args)
        assert rc == 0, out
    finally:
        srv.shutdown()
    jr = (tmp_path / "g" / "cap_prune__t20__s0" / "journal.jsonl").read_text()
    starts = [json.loads(l) for l in jr.splitlines() if '"type": "start"' in l]
    assert len(starts) == 2 and starts[1]["resumed_from"] is not None


def test_supervisor_waits_out_controller_outage(smoke_assets, tmp_path):
    port = free_port()
    holder = {}
    ledger = tmp_path / "o" / "_matrix" / "ledger.json"

    def controller_comes_back():
        # Bring the controller up only AFTER the worker has paused on the outage (exit 76).  A fixed
        # delay races the worker's start-up: where importing torch + loading the model takes longer
        # than the delay, the controller is already back at the first query and no outage is seen.
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                if 76 in json.loads(ledger.read_text())["cap_prune__t20__s3"]["exits"]:
                    break
            except Exception:
                pass  # ledger not written yet / mid-replace
            time.sleep(0.2)
        holder["srv"] = serve(port)[0]

    threading.Thread(target=controller_comes_back, daemon=True).start()
    try:
        rc, out = cli(["supervise", "--config", str(SMOKE), "--methods", "cap_prune", "--targets", "0.2", "--seeds", "3",
                       *_common(port, tmp_path / "o"), "--set", "reliability.controller_wait_s=1"])
        assert rc == 0, out
        led = json.loads((tmp_path / "o" / "_matrix" / "ledger.json").read_text())["cap_prune__t20__s3"]
        assert 76 in led["exits"] and led["status"] == "done"
    finally:
        time.sleep(0.1)
        if "srv" in holder:
            holder["srv"].shutdown()


def test_config_change_blocks_resume(mock, tmp_path):
    port, _ = mock
    base = ["run", "--config", str(SMOKE), "--method", "paper_agent", "--target", "0.2", "--seed", "0",
            *_common(port, tmp_path / "c")]
    rc, out = cli(base, env={"CAPPRUNE_FAULTS": "after_controller@2", "CAPPRUNE_FAULT_STATE": str(tmp_path / "f.json")})
    assert rc == 137
    rc, out = cli(base + ["--set", "search.ppl_rollback_threshold=0.5"])
    assert rc == 2 and "configuration changed" in out
    rc, out = cli(base + ["--set", "reliability.max_restarts=9"])  # reliability-only change: fine
    assert rc == 0, out


@pytest.mark.parametrize("mode", ["nm", "structured"])
def test_hardware_modes_end_to_end(mock, tmp_path, mode):
    port, _ = mock
    rc, out = cli(["run", "--config", str(SMOKE), "--method", "cap_prune", "--target", "0.2", "--seed", "0",
                   *_common(port, tmp_path / mode), "--set", f"pruning.mode={mode}"])
    assert rc == 0, out
    s = summary(tmp_path / mode / "cap_prune__t20__s0")
    assert s["achieved_sparsity"] > 0.05 and s["budget"]["deploy_checks"] > 0
    if mode == "structured":
        assert s["benchmark"]["materialize"]["channels_removed"] > 0


def test_full_matrix_and_report(mock, tmp_path):
    port, _ = mock
    rc, out = cli(["supervise", "--config", str(SMOKE), *_common(port, tmp_path / "m")])
    assert rc == 0, out
    rc, out = cli(["report", "--config", str(SMOKE), "--set", f"experiment.runs_dir={tmp_path / 'm'}"])
    assert rc == 0, out
    rep = (tmp_path / "m" / "_report" / "report.md").read_text(encoding="utf-8")
    for m in ("wanda", "paper_agent", "cap_prune", "heuristic_cap"):
        assert m in rep
    rc, out = cli(["status", "--config", str(SMOKE), "--set", f"experiment.runs_dir={tmp_path / 'm'}"])
    assert rc == 0 and out.count("done") >= 5
    rc, out = cli(["verify", "--config", str(SMOKE), "--set", f"experiment.runs_dir={tmp_path / 'm'}"])
    assert rc == 0, out


def test_dev_split_keeps_final_sets_untouched(mock, tmp_path):
    port, _ = mock
    over = [*_common(port, tmp_path / "dev"), "--set", "final_eval.split=dev", "--set", "benchmark.enabled=false"]
    for m in ("dense", "cap_prune"):
        rc, out = cli(["run", "--config", str(SMOKE), "--method", m, "--target", "0.2", "--seed", "0", *over])
        assert rc == 0, out
    mm = json.loads((tmp_path / "dev" / "cap_prune__t20__s0" / "final" / "mmlu.json").read_text())
    assert all("/validation/" in q for q in mm["per_question"])  # MMLU validation, never the final test split
    rc, out = cli(["report", "--config", str(SMOKE), *over])
    assert rc == 0 and "DEVELOPMENT split" in out, out


def test_second_supervisor_refuses_to_start(smoke_assets, tmp_path):
    from capprune.utils.runtime import RunLock

    with RunLock(tmp_path / "s" / "_matrix"):  # this test process plays the first supervisor
        rc, out = cli(["supervise", "--config", str(SMOKE), "--methods", "dense",
                       "--set", f"experiment.runs_dir={tmp_path / 's'}"])
    assert rc == 77, out
