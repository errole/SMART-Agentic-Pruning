"""Build a replay cache from the existing Qwen3 evaluation pipeline.

For the dense checkpoint and then every candidate in ``build.candidates``:

  1. produce the candidate with existing code
       kind: wanda   structured Wanda at ``sparsity`` (pruning/wanda.py::sequential_wanda) on the
                     pipeline's calibration windows
       kind: masks   a mask file saved by a CAP-Prune run (``<run>/masks/<state>.safetensors``)
       kind: hf_dir  any already-pruned Hugging Face checkpoint directory (same tokenizer)
  2. materialise it (pruning/materialize.py): structured MLP channels are physically removed,
     so every timing below is of the deployable model
  3. score every search-pool item with eval/mmlu.py::MMLUScorer, *batch by batch in the
     replay's fixed question order and look layout*, timing each batch (CUDA-synchronised);
     look 0 = the minimum sample, then one batch per later look
  4. score the holdout split in one pass (timed separately; never used by the search)
  5. measure prefill/decode latency on the materialised model (eval/bench.py)

and write ``models/<name>.json`` (see cache.py).  A model whose file exists is
skipped, so an interrupted build resumes; delete a file to rebuild that model.
Per model the times are kept apart: setup (prune/load + materialise), each
search-pool batch, holdout, benchmark.
"""
from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path

from ..utils.log import get_logger
from .cache import write_items, write_model
from .config import cfg_path
from .evaluator import Settings, look_sizes, question_order

log = get_logger("adaptive.build")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device):
    """Wall time of fn(), with the GPU queue drained before and after (so it is the GPU time of fn)."""
    _sync(device)
    t = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t


def build(acfg: dict, only: list[str] | None = None, force: bool = False, dry_run: bool = False) -> Path:
    import torch

    from ..config import data_dir, deep_merge, load_config
    from ..data.prepare import PreparedData
    from ..eval.bench import benchmark
    from ..eval.mmlu import MMLUScorer
    from ..model import build_units, load_model, load_tokenizer, tokenizer_id
    from ..pruning.masks import DenseStore, PruneState
    from ..pruning.materialize import materialize, param_bytes
    from ..pruning.wanda import sequential_wanda
    from ..utils.gpu import free_memory

    t_start = time.perf_counter()
    b = acfg["build"]
    s = Settings.from_dict(acfg["evaluation"])
    root = cfg_path(acfg, acfg["cache_dir"])
    pcfg = load_config(cfg_path(acfg, b["pipeline_config"]), b.get("pipeline_overrides") or [])
    mode = pcfg["pruning"]["mode"]
    specs = [{"name": "dense", "kind": "dense"}] + list(b["candidates"])
    for spec in specs:
        if not _NAME.match(spec["name"]) or (spec["name"] == "dense" and spec["kind"] != "dense"):
            raise ValueError(f"invalid candidate name {spec['name']!r} (letters, digits, _ . - ; not 'dense')")
    if only:
        unknown = set(only) - {sp["name"] for sp in specs}
        if unknown:
            raise ValueError(f"--only names not in build.candidates: {sorted(unknown)}")

    for spec in specs:
        if spec["kind"] in ("masks", "hf_dir") and not cfg_path(acfg, spec["path"]).exists():
            raise FileNotFoundError(f"candidate {spec['name']}: {cfg_path(acfg, spec['path'])} does not exist")
        if spec["kind"] == "wanda" and not 0 < float(spec["sparsity"]) < 1:
            raise ValueError(f"candidate {spec['name']}: sparsity must be in (0, 1)")
        if spec["kind"] not in ("dense", "wanda", "masks", "hf_dir"):
            raise ValueError(f"unknown candidate kind {spec['kind']!r} (wanda | masks | hf_dir)")

    # ---- items: search pool and holdout, domain = the configured row field (category = 4 broad MMLU groups)
    hold_split = b.get("holdout_split", "final")  # null = no holdout (e.g. gate development must not touch final data)
    if b.get("pool_split", "probe") == hold_split:
        raise ValueError("pool_split and holdout_split must differ")
    data = PreparedData(data_dir(pcfg), tokenizer_id(pcfg))  # verifies the prepared files' checksums
    splits = {"probe": data.mmlu_probe, "final": data.mmlu_final, "dev": data.mmlu_dev}
    field = b.get("domain_field", "category")
    pool_rows = [r for r in splits[b.get("pool_split", "probe")] if r[field] in s.domains]
    hold_rows = [r for r in splits[hold_split] if r[field] in s.domains] if hold_split else []
    found = {r[field] for r in pool_rows}
    if found != set(s.domains):
        raise ValueError(f"domains {sorted(set(s.domains) - found)} have no items in field '{field}'")
    shots = int(b.get("shots", 0))
    max_len = int(b.get("max_seq_len", pcfg["model"].get("max_seq_len", 2048)))
    max_tok = int(b.get("max_batch_tokens", 32768))

    by_id = {r["qid"]: r for r in pool_rows + hold_rows}
    pool_ids = {d: [r["qid"] for r in pool_rows if r[field] == d] for d in s.domains}
    order = {d: question_order(ids, s.seed, d) for d, ids in pool_ids.items()}
    chunks = {}
    for d in s.domains:  # look layout: [0, m), [m, m+b), ... up to the whole pool
        sizes = look_sizes(len(order[d]), s.min_sample[d], s.batch_size[d], len(order[d]))
        chunks[d] = list(zip([0] + sizes[:-1], sizes))
    layout = {"seed": s.seed, "min_sample": s.min_sample, "batch_size": s.batch_size}
    for d in s.domains:
        log.info("%-16s pool %5d (%d timed batches)  holdout %5d", d, len(order[d]), len(chunks[d]),
                 sum(1 for r in hold_rows if r[field] == d))
    log.info("models: %s", ", ".join(f"{sp['name']} ({sp['kind']}{', ' + str(sp.get('sparsity')) if sp['kind'] == 'wanda' else ''})"
                                    for sp in specs))
    if dry_run:
        log.info("dry run: config, prepared data and candidate files OK; cache would be written to %s", root)
        return root

    root.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(pcfg)
    log.info("loading %s (%s mode)", pcfg["model"]["name_or_path"], mode)
    model = load_model(pcfg)
    units = build_units(model, mode, pcfg["pruning"].get("eligible", []))
    dense_store = DenseStore(units, (pcfg.get("reliability") or {}).get("dense_copy_device") or None)
    batch_store: dict = {}
    counter = MMLUScorer(model, tok, data.mmlu_fewshot, shots, max_len, batch_store, "adaptive", max_tok)
    write_items(root, [{"item_id": r["qid"], "domain": r[field], "split": sp, "tokens": len(counter.encode(r))}
                       for sp, rows in (("pool", pool_rows), ("holdout", hold_rows)) for r in rows])
    del counter

    n_cal = int(pcfg["data"]["calibration"]["num_samples"])
    calib = data.calibration_pool[data.calibration_indices(int(b.get("calibration_seed", 0)), n_cal)]
    measure_latency = b.get("latency", {}) is not None  # latency: null skips the benchmark
    lat = b.get("latency") or {}
    L = int(lat.get("prompt_len", 512))
    built = []
    for order_idx, spec in enumerate(specs):
        name = spec["name"]
        if (only and name not in only) or ((root / "models" / f"{name}.json").exists() and not force):
            continue
        log.info("=== %s (%s) ===", name, spec["kind"])
        state = None
        t0 = time.perf_counter()
        if spec["kind"] == "dense":
            m, info = model, {"kind": "dense"}
        elif spec["kind"] in ("wanda", "masks"):
            PruneState(units, mode).apply(dense_store)  # start from the dense weights
            state = PruneState(units, mode)
            if spec["kind"] == "wanda":
                sequential_wanda(model, state, dense_store, calib, float(spec["sparsity"]), pcfg,
                                 int(pcfg["profiling"].get("activation_batch_size", 4)))
            else:
                state.load(cfg_path(acfg, spec["path"]))
                state.apply(dense_store)
            m, minfo = materialize(model, state, pcfg)
            info = {"kind": spec["kind"], "target_sparsity": spec.get("sparsity"), "mask_path": spec.get("path"),
                    "mode": mode, "achieved_sparsity": state.global_sparsity(), "materialize": minfo}
        else:  # hf_dir
            path = cfg_path(acfg, spec["path"])
            m = load_model(deep_merge(pcfg, {"model": {"name_or_path": str(path), "revision": None}}))
            info = {"kind": "hf_dir", "path": str(path)}
        dev = next(m.parameters()).device
        _sync(dev)
        setup_s = time.perf_counter() - t0
        info["parameters"] = sum(p.numel() for p in m.parameters())
        info["weights_bytes"] = param_bytes(m)

        sc = MMLUScorer(m, tok, data.mmlu_fewshot, shots, max_len, batch_store, "adaptive", max_tok)
        sc.score([by_id[i] for i in order[s.domains[0]][:2]])  # warm-up (kernels, allocator), not timed
        correct: dict[str, int] = {}
        batch_seconds: dict[str, list[float]] = {}
        for d in s.domains:
            batch_seconds[d] = []
            for a, z in chunks[d]:
                res, t = _timed(lambda: sc.score([by_id[i] for i in order[d][a:z]]), dev)
                correct.update({q: int(v["correct"]) for q, v in res.items()})
                batch_seconds[d].append(t)
        res, hold_s = _timed(lambda: sc.score(hold_rows), dev) if hold_rows else ({}, 0.0)
        correct.update({q: int(v["correct"]) for q, v in res.items()})
        latency, bench_s = None, 0.0
        if measure_latency:
            bench, bench_s = _timed(lambda: benchmark(m, model.config.vocab_size, [L], int(lat.get("gen_tokens", 64)), 1,
                                                      int(lat.get("warmup", 3)), int(lat.get("repeats", 10))), dev)
            bl = bench["by_prompt_len"][str(L)]
            latency = {"decode_ms_per_token": bl["decode_ms_per_token_median"],
                       "decode_ms_per_token_iqr": bl["decode_ms_per_token_iqr"],
                       "prefill_ms": bl["prefill_ms_median"], "prompt_len": L,
                       "gen_tokens": int(lat.get("gen_tokens", 64)), "repeats": bl["repeats"],
                       "device": bench["device"], "peak_allocated_bytes": bl.get("peak_allocated_bytes")}
        pool_flat = [i for ids in pool_ids.values() for i in ids]
        info["pool_accuracy"] = sum(correct[i] for i in pool_flat) / len(pool_flat)
        info["holdout_accuracy"] = (sum(correct[r["qid"]] for r in hold_rows) / len(hold_rows)) if hold_rows else None
        rec = {"name": name, "role": "dense" if spec["kind"] == "dense" else "candidate", "order": order_idx - 1,
               "correct": correct,
               "latency": latency,
               "timing": {"layout": layout, "batch_seconds": batch_seconds, "setup_seconds": setup_s,
                          "holdout_seconds": hold_s, "bench_seconds": bench_s,
                          "pool_seconds": sum(sum(v) for v in batch_seconds.values())},
               "info": info}
        write_model(root, rec)
        built.append(name)
        log.info("%s: pool acc %.4f, holdout acc %s, decode %s ms/token, pool scoring %.1fs, setup %.1fs",
                 name, info["pool_accuracy"],
                 "n/a" if info["holdout_accuracy"] is None else f"{info['holdout_accuracy']:.4f}",
                 "n/a" if latency is None else f"{latency['decode_ms_per_token']:.2f}",
                 rec["timing"]["pool_seconds"], setup_s)
        if m is not model:
            del m
            free_memory()
        if state is not None:
            PruneState(units, mode).apply(dense_store)  # restore dense for the next candidate

    prov_path = root / "build.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"invocations": []}
    prov.update({"pipeline_config": str(cfg_path(acfg, b["pipeline_config"])), "pipeline_overrides": b.get("pipeline_overrides"),
                 "model": pcfg["model"]["name_or_path"], "mode": mode, "domain_field": field,
                 "pool_split": b.get("pool_split", "probe"), "holdout_split": b.get("holdout_split", "final"),
                 "shots": shots, "max_seq_len": max_len, "max_batch_tokens": max_tok, "layout": layout,
                 "data_manifest_tokenizer": data.manifest.get("tokenizer")})
    prov["invocations"].append({"built": built, "wall_seconds": time.perf_counter() - t_start, "finished": time.time(),
                                "torch": torch.__version__, "platform": platform.platform(),
                                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                                and pcfg["model"].get("device", "cuda").startswith("cuda") else "cpu"})
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True), encoding="utf-8")
    log.info("cache written to %s (built: %s)", root, ", ".join(built) or "nothing new")
    return root
