"""Data for the live-gate CPU smoke test.

The tiny smoke model is random, so on synthetic questions it is right ~25 % of the
time and pruning breaks and fixes answers in equal measure (paired loss ~0): the
capability gate would never see a real loss.  This prepares synthetic data with
the live-smoke config (larger pools: 80-320 items per category) and replaces the
*pruning-time* (probe) answer key with the dense model's own 0-shot predictions,
scored exactly as the run will score them.  The dense model is then 100 % correct
there, every changed answer counts as "broken", and the size of a proposal
controls the loss - so the scripted mock controller can produce an ACCEPT, a
REJECT and an INCONCLUSIVE on purpose.  Final-audit, development and calibration
data are left as generated.

    python tests/live_smoke.py            # writes .smoke/data-live (needs the tiny model: capprune make-tiny)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIVE_SMOKE = ROOT / "configs" / "live_smoke.yaml"


def make_live_smoke_data(dst: Path | None = None) -> Path:
    from capprune.config import data_dir, load_config
    from capprune.data.prepare import prepare
    from capprune.eval.mmlu import MMLUScorer
    from capprune.model import build_tiny_model, load_model, load_tokenizer, tokenizer_id
    from capprune.utils.io import atomic_write_json, read_json, sha256_file

    cfg = load_config(LIVE_SMOKE)
    build_tiny_model(ROOT / ".smoke" / "tiny-qwen3")
    dst = dst or data_dir(cfg)
    if dst.exists():  # empty it but keep the folder (sync clients such as OneDrive may lock the folder itself)
        for p in dst.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    prepare(cfg, load_tokenizer(cfg), dst, tokenizer_id(cfg))
    rows = [json.loads(l) for l in open(dst / "mmlu_probe.jsonl", encoding="utf-8") if l.strip()]
    fewshot = [json.loads(l) for l in open(dst / "mmlu_fewshot.jsonl", encoding="utf-8") if l.strip()]
    t = cfg["testing"]
    sc = MMLUScorer(load_model(cfg), load_tokenizer(cfg), fewshot, int(t.get("shots", 0)),
                    int(cfg["model"].get("max_seq_len", 2048)), {}, "probe_tokens", int(t.get("max_batch_tokens", 32768)))
    pred = sc.score(rows)
    for r in rows:
        r["answer"] = pred[r["qid"]]["pred"]
    with open(dst / "mmlu_probe.jsonl", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    man = read_json(dst / "manifest.json")
    man["files"]["mmlu_probe.jsonl"] = {"sha256": sha256_file(dst / "mmlu_probe.jsonl"),
                                        "bytes": (dst / "mmlu_probe.jsonl").stat().st_size}
    man["info"]["live_smoke"] = "probe answers replaced by the dense tiny model's 0-shot predictions"
    atomic_write_json(dst / "manifest.json", man)
    return dst


if __name__ == "__main__":
    print(f"live-smoke data written to {make_live_smoke_data()}")
