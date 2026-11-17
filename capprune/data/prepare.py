"""Dataset preparation with strictly disjoint roles.

=================  ===========================================  ==========================
role               source                                        used for
=================  ===========================================  ==========================
calibration        C4 train shard 0 (random 2048-token windows)  Wanda scores, profiling
ppl_gate           C4 validation shard 0                         pruning-time PPL rollback
mmlu_dev           MMLU ``validation`` split                     method development/tuning
mmlu_probe         MMLU ``test``, stratified fraction            pruning-time domain tests
mmlu_final         MMLU ``test``, the remaining questions        final held-out evaluation
mmlu_fewshot       MMLU ``dev`` (5 per subject)                  5-shot exemplars only
fqa_final          FreebaseQA ``eval``                           final factual-QA metric
fqa_dev            FreebaseQA ``dev``                            method development
wikitext2_val      WikiText-2 raw ``validation``                 method development
fqa_fewshot        FreebaseQA ``train`` (first k)                QA exemplars only
wikitext2_test     WikiText-2 raw ``test``                       final perplexity
=================  ===========================================  ==========================

Everything is written atomically into ``data_dir`` together with a manifest of
SHA-256 checksums; ``load_prepared`` verifies those before a run starts, so a
truncated download or a partially written file can never silently enter an
experiment.  ``prepare`` is idempotent: re-running it skips finished parts.
"""
from __future__ import annotations

import json
import random
import string
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..utils.io import atomic_write_json, atomic_write_text, finalize_tmp, read_json, sha256_file
from ..utils.log import get_logger
from .domains import SUBJECT_TO_CATEGORY, domain_of

log = get_logger("data")

FREEBASEQA_URLS = {
    "train": "https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/FreebaseQA-train.json",
    "eval": "https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/FreebaseQA-eval.json",
    "dev": "https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/FreebaseQA-dev.json",
}
# Checksums published in the Hugging Face ``freebase_qa`` dataset_infos.json.
FREEBASEQA_SHA256 = {
    "train": "b9769a0040032e39a62bd4b0b99d7dfa1a3fe29c2108dbf6245f62874d0d4753",
    "eval": "14d29d8180d2eaa44eda444debba08f292f42337e098e5b717455e462d278451",
}

LETTERS = ["A", "B", "C", "D"]


# ============================================================================ helpers
def _save_tokens(path: Path, tensor: torch.Tensor) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    save_file({"tokens": tensor.contiguous()}, str(tmp))
    finalize_tmp(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _download(url: str, dest: Path, sha256: str | None, retries: int = 5) -> Path:
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            tmp = dest.with_name(f".{dest.name}.part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if sha256 and sha256_file(tmp) != sha256:
                raise IOError(f"checksum mismatch for {url}")
            finalize_tmp(tmp, dest)
            return dest
        except Exception as e:  # network hiccup -> retry with backoff
            last = e
            wait = min(60, 2 ** attempt)
            log.warning("download %s failed (%s); retry in %ds", url, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}: {last}")


def _hf_load(name: str, config: str | None, split: str, revision: str | None = None, data_files=None):
    from datasets import load_dataset  # lazy: not needed for synthetic smoke tests

    kw: dict[str, Any] = {"split": split}
    if revision:
        kw["revision"] = revision
    if data_files:
        kw["data_files"] = data_files
    last = None
    for attempt in range(5):
        try:
            return load_dataset(name, config, **kw) if config else load_dataset(name, **kw)
        except Exception as e:
            last = e
            log.warning("load_dataset(%s, %s, %s) failed (%s); retrying", name, config, split, e)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed to load {name}/{config}/{split}: {last}")


# ============================================================================ builders
def _calibration_pool(texts_iter_len, get_text, tokenizer, pool: int, seq_len: int, seed: int) -> torch.Tensor:
    """Wanda-style random windows: random document longer than seq_len, random offset."""
    rng = random.Random(seed)
    out, tries = [], 0
    n = texts_iter_len
    while len(out) < pool:
        tries += 1
        if tries > pool * 200:
            raise RuntimeError("could not find enough long documents for calibration")
        ids = tokenizer(get_text(rng.randrange(n)), add_special_tokens=False)["input_ids"]
        if len(ids) <= seq_len:
            continue
        s = rng.randrange(0, len(ids) - seq_len)
        out.append(ids[s : s + seq_len])
    return torch.tensor(out, dtype=torch.int32)


def _windows_from_stream(text: str, tokenizer, n: int, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    need = n * seq_len
    if len(ids) < need:
        raise RuntimeError(f"text too short for {n}x{seq_len} tokens ({len(ids)})")
    return torch.tensor(ids[:need], dtype=torch.int32).view(n, seq_len)


def _mmlu_rows(ds, split: str) -> list[dict]:
    rows = []
    counters: dict[str, int] = {}
    for r in ds:
        subj = r["subject"]
        i = counters.get(subj, 0)
        counters[subj] = i + 1
        rows.append({
            "qid": f"{subj}/{split}/{i}",
            "subject": subj,
            "question": r["question"].strip(),
            "choices": [str(c).strip() for c in r["choices"]],
            "answer": int(r["answer"]),
        })
    return rows


def _split_probe_final(test_rows: list[dict], probe_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_subj: dict[str, list[dict]] = {}
    for r in test_rows:
        by_subj.setdefault(r["subject"], []).append(r)
    probe, final = [], []
    for subj in sorted(by_subj):
        rows = sorted(by_subj[subj], key=lambda r: r["qid"])
        rng = random.Random(f"{seed}:{subj}")
        rng.shuffle(rows)
        k = int(round(probe_fraction * len(rows)))
        probe += rows[:k]
        final += rows[k:]
    return probe, final


def _annotate(rows: list[dict], custom_domains: dict | None) -> list[dict]:
    for r in rows:
        r["domain"] = domain_of(r["subject"], custom_domains)
        r["category"] = SUBJECT_TO_CATEGORY.get(r["subject"], "Other")
    return rows


def _parse_freebaseqa(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for q in data["Questions"]:
        answers: set[str] = set()
        for parse in q.get("Parses", []):
            for ans in parse.get("Answers", []):
                for name in ans.get("AnswersName", []):
                    if isinstance(name, list):
                        answers.update(str(x) for x in name)
                    else:
                        answers.add(str(name))
        if not answers:
            continue
        out.append({"qid": q["Question-ID"], "question": q["RawQuestion"].strip(), "answers": sorted(answers)})
    return out


# ============================================================================ synthetic (smoke tests)
def _synthetic(cfg: dict, tokenizer, out: Path) -> dict:
    d = cfg["data"]
    seed = int(d.get("seed", 1234))
    rng = random.Random(seed)
    words = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 7))) for _ in range(300)]

    def text(nw: int) -> str:
        return " ".join(rng.choice(words) for _ in range(nw))

    cal = d["calibration"]
    docs = [text(rng.randint(cal["seq_len"], cal["seq_len"] * 2)) for _ in range(200)]
    pool = _calibration_pool(len(docs), lambda i: docs[i], tokenizer, cal["pool_size"], cal["seq_len"], seed)
    _save_tokens(out / "calibration_pool.safetensors", pool)
    pg = d["ppl_gate"]
    _save_tokens(out / "ppl_gate.safetensors",
                 _windows_from_stream(text(pg["num_samples"] * pg["seq_len"] * 2), tokenizer, pg["num_samples"], pg["seq_len"]))
    wt = d["wikitext"]
    _save_tokens(out / "wikitext2_test.safetensors",
                 torch.tensor(tokenizer(text(wt.get("synthetic_tokens", 4000)), add_special_tokens=False)["input_ids"], dtype=torch.int32))
    _save_tokens(out / "wikitext2_val.safetensors",
                 torch.tensor(tokenizer(text(wt.get("synthetic_tokens", 4000)), add_special_tokens=False)["input_ids"], dtype=torch.int32))
    subjects = d.get("synthetic_subjects") or ["high_school_mathematics", "astronomy", "anatomy",
                                                "machine_learning", "prehistory", "philosophy", "sociology", "marketing"]
    per = int(d.get("synthetic_questions_per_subject", 40))

    def mcq(subj, split, i):
        return {"qid": f"{subj}/{split}/{i}", "subject": subj, "question": text(rng.randint(5, 15)) + "?",
                "choices": [text(rng.randint(1, 3)) for _ in range(4)], "answer": rng.randrange(4)}

    test = [mcq(s, "test", i) for s in subjects for i in range(per)]
    probe, final = _split_probe_final(test, d["mmlu"]["probe_fraction"], seed)
    _write_jsonl(out / "mmlu_probe.jsonl", _annotate(probe, d.get("custom_domains")))
    _write_jsonl(out / "mmlu_final.jsonl", _annotate(final, d.get("custom_domains")))
    _write_jsonl(out / "mmlu_dev.jsonl", _annotate([mcq(s, "validation", i) for s in subjects for i in range(8)], d.get("custom_domains")))
    _write_jsonl(out / "mmlu_fewshot.jsonl", _annotate([mcq(s, "dev", i) for s in subjects for i in range(5)], d.get("custom_domains")))
    qa = [{"qid": f"syn-{i}", "question": text(6) + "?", "answers": [rng.choice(words)]} for i in range(60)]
    _write_jsonl(out / "fqa_fewshot.jsonl", qa[:5])
    _write_jsonl(out / "fqa_final.jsonl", qa[5:])
    _write_jsonl(out / "fqa_dev.jsonl", [{"qid": f"syn-dev-{i}", "question": text(6) + "?", "answers": [rng.choice(words)]}
                                         for i in range(40)])
    return {"source": "synthetic", "seed": seed}


# ============================================================================ real data
def _real(cfg: dict, tokenizer, out: Path, raw_dir: Path) -> dict:
    d = cfg["data"]
    seed = int(d.get("seed", 1234))
    info: dict[str, Any] = {"source": "real", "seed": seed}

    cal = d["calibration"]
    if not (out / "calibration_pool.safetensors").exists():
        log.info("Building calibration pool (%d x %d tokens) from C4 train shard...", cal["pool_size"], cal["seq_len"])
        c4 = _hf_load(cal.get("dataset", "allenai/c4"), None, "train", cal.get("revision"),
                      data_files={"train": cal.get("file", "en/c4-train.00000-of-01024.json.gz")})
        pool = _calibration_pool(len(c4), lambda i: c4[i]["text"], tokenizer, cal["pool_size"], cal["seq_len"], seed)
        _save_tokens(out / "calibration_pool.safetensors", pool)
    info["calibration"] = {k: cal.get(k) for k in ("dataset", "file", "revision", "pool_size", "seq_len")}

    pg = d["ppl_gate"]
    if not (out / "ppl_gate.safetensors").exists():
        log.info("Building pruning-time perplexity set from C4 validation shard...")
        c4v = _hf_load(pg.get("dataset", "allenai/c4"), None, "validation", pg.get("revision"),
                       data_files={"validation": pg.get("file", "en/c4-validation.00000-of-00008.json.gz")})
        text = " ".join(c4v[i]["text"] for i in range(min(len(c4v), 3000)))
        _save_tokens(out / "ppl_gate.safetensors", _windows_from_stream(text, tokenizer, pg["num_samples"], pg["seq_len"]))
    info["ppl_gate"] = {k: pg.get(k) for k in ("dataset", "file", "revision", "num_samples", "seq_len")}

    wt = d["wikitext"]
    if not (out / "wikitext2_test.safetensors").exists():
        log.info("Tokenizing WikiText-2 test...")
        w = _hf_load(wt.get("dataset", "Salesforce/wikitext"), wt.get("config", "wikitext-2-raw-v1"), "test", wt.get("revision"))
        ids = tokenizer("\n\n".join(w["text"]), add_special_tokens=False)["input_ids"]
        _save_tokens(out / "wikitext2_test.safetensors", torch.tensor(ids, dtype=torch.int32))
    if not (out / "wikitext2_val.safetensors").exists():
        w = _hf_load(wt.get("dataset", "Salesforce/wikitext"), wt.get("config", "wikitext-2-raw-v1"), "validation", wt.get("revision"))
        ids = tokenizer("\n\n".join(w["text"]), add_special_tokens=False)["input_ids"]
        _save_tokens(out / "wikitext2_val.safetensors", torch.tensor(ids, dtype=torch.int32))
    info["wikitext"] = {k: wt.get(k) for k in ("dataset", "config", "revision")}

    mm = d["mmlu"]
    if not all((out / f).exists() for f in GROUPS[4]):
        log.info("Loading MMLU and building probe/final/dev splits...")
        name, rev = mm.get("dataset", "cais/mmlu"), mm.get("revision")
        test = _mmlu_rows(_hf_load(name, "all", "test", rev), "test")
        val = _mmlu_rows(_hf_load(name, "all", "validation", rev), "validation")
        dev = _mmlu_rows(_hf_load(name, "all", "dev", rev), "dev")
        probe, final = _split_probe_final(test, mm["probe_fraction"], seed)
        cd = d.get("custom_domains")
        _write_jsonl(out / "mmlu_fewshot.jsonl", _annotate(dev, cd))
        _write_jsonl(out / "mmlu_dev.jsonl", _annotate(val, cd))
        _write_jsonl(out / "mmlu_probe.jsonl", _annotate(probe, cd))
        _write_jsonl(out / "mmlu_final.jsonl", _annotate(final, cd))
    info["mmlu"] = {k: mm.get(k) for k in ("dataset", "revision", "probe_fraction")}

    fq = d["freebaseqa"]
    if not all((out / f).exists() for f in GROUPS[5]):
        log.info("Downloading FreebaseQA...")
        split = fq.get("split", "eval")
        ev = _download(FREEBASEQA_URLS[split], raw_dir / f"FreebaseQA-{split}.json", FREEBASEQA_SHA256.get(split))
        tr = _download(FREEBASEQA_URLS["train"], raw_dir / "FreebaseQA-train.json", FREEBASEQA_SHA256["train"])
        final_rows = _parse_freebaseqa(ev)
        if fq.get("max_questions"):
            rows = sorted(final_rows, key=lambda r: r["qid"])
            random.Random(seed).shuffle(rows)
            final_rows = sorted(rows[: int(fq["max_questions"])], key=lambda r: r["qid"])
        _write_jsonl(out / "fqa_final.jsonl", final_rows)
        _write_jsonl(out / "fqa_fewshot.jsonl", _parse_freebaseqa(tr)[: int(fq.get("shots", 5))])
    if not (out / "fqa_dev.jsonl").exists():
        dv = _download(FREEBASEQA_URLS["dev"], raw_dir / "FreebaseQA-dev.json", None)
        _write_jsonl(out / "fqa_dev.jsonl", _parse_freebaseqa(dv))
    info["freebaseqa"] = {k: fq.get(k) for k in ("split", "max_questions", "shots")}
    return info


FILES = ["calibration_pool.safetensors", "ppl_gate.safetensors", "wikitext2_test.safetensors", "wikitext2_val.safetensors",
         "mmlu_probe.jsonl", "mmlu_final.jsonl", "mmlu_dev.jsonl", "mmlu_fewshot.jsonl",
         "fqa_final.jsonl", "fqa_dev.jsonl", "fqa_fewshot.jsonl"]


# Files produced together: if any member is bad, the whole group is rebuilt.
GROUPS = [
    ["calibration_pool.safetensors"], ["ppl_gate.safetensors"], ["wikitext2_test.safetensors"],
    ["wikitext2_val.safetensors"], ["mmlu_probe.jsonl", "mmlu_final.jsonl", "mmlu_dev.jsonl", "mmlu_fewshot.jsonl"],
    ["fqa_final.jsonl", "fqa_fewshot.jsonl"], ["fqa_dev.jsonl"],
]
TOKENIZED = {"calibration_pool.safetensors", "ppl_gate.safetensors", "wikitext2_test.safetensors", "wikitext2_val.safetensors"}


def _discard_bad(out_dir: Path, tokenizer_id: str) -> None:
    """Delete every file that is missing from / disagrees with the manifest (and its group), so that
    a rebuild can never re-checksum a corrupt file as if it were valid."""
    mpath = out_dir / "manifest.json"
    try:
        m = read_json(mpath)
    except Exception:
        m = {"files": {}, "tokenizer": None}
    bad = set()
    for f in FILES:
        meta = m["files"].get(f)
        p = out_dir / f
        if meta is None or not p.exists() or sha256_file(p) != meta["sha256"]:
            bad.add(f)
        elif f in TOKENIZED and m.get("tokenizer") != tokenizer_id:
            bad.add(f)
    for g in GROUPS:
        if bad & set(g):
            for f in g:
                if (out_dir / f).exists():
                    log.warning("discarding %s (missing/corrupt/stale); it will be rebuilt", f)
                    (out_dir / f).unlink()
    if mpath.exists():
        mpath.unlink()


def prepare(cfg: dict, tokenizer, out_dir: Path, tokenizer_id: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        try:
            verify(out_dir, tokenizer_id)
            log.info("Prepared data already present and verified at %s", out_dir)
            return read_json(mpath)
        except Exception as e:
            log.warning("Existing prepared data failed verification (%s); rebuilding the affected parts", e)
            _discard_bad(out_dir, tokenizer_id)
    elif any((out_dir / f).exists() for f in FILES):
        # files without a manifest = an interrupted earlier prepare: trust nothing
        for f in FILES:
            if (out_dir / f).exists():
                (out_dir / f).unlink()
    src = cfg["data"].get("source", "real")
    info = _synthetic(cfg, tokenizer, out_dir) if src == "synthetic" else _real(cfg, tokenizer, out_dir, out_dir / "raw")
    files = {f: {"sha256": sha256_file(out_dir / f), "bytes": (out_dir / f).stat().st_size} for f in FILES}
    probe = _read_jsonl(out_dir / "mmlu_probe.jsonl")
    final = _read_jsonl(out_dir / "mmlu_final.jsonl")
    overlap = {r["qid"] for r in probe} & {r["qid"] for r in final}
    if overlap:
        raise RuntimeError(f"probe/final overlap: {len(overlap)} questions")
    counts = {"mmlu_probe": len(probe), "mmlu_final": len(final)}
    for name in ("mmlu_dev", "fqa_final", "fqa_dev"):
        counts[name] = len(_read_jsonl(out_dir / f"{name}.jsonl"))
    manifest = {"tokenizer": tokenizer_id, "info": info, "files": files, "counts": counts,
                "probe_domains": _domain_counts(probe), "final_domains": _domain_counts(final)}
    atomic_write_json(mpath, manifest)
    log.info("Prepared data: %s", counts)
    return manifest


def _domain_counts(rows):
    c: dict[str, int] = {}
    for r in rows:
        c[r["domain"]] = c.get(r["domain"], 0) + 1
    return dict(sorted(c.items()))


def verify(out_dir: Path, tokenizer_id: str) -> dict:
    m = read_json(out_dir / "manifest.json")
    if m.get("tokenizer") != tokenizer_id:
        raise RuntimeError(f"data was tokenized for {m.get('tokenizer')}, not {tokenizer_id}; re-run prepare")
    for f, meta in m["files"].items():
        p = out_dir / f
        if not p.exists() or sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"prepared file {f} missing or corrupt; re-run `capprune prepare`")
    return m


class PreparedData:
    """Read-only view over a verified ``data_dir``."""

    def __init__(self, out_dir: Path, tokenizer_id: str):
        self.dir = out_dir
        self.manifest = verify(out_dir, tokenizer_id)
        self.calibration_pool = load_file(str(out_dir / "calibration_pool.safetensors"))["tokens"].long()
        self.ppl_gate = load_file(str(out_dir / "ppl_gate.safetensors"))["tokens"].long()
        self.wikitext = load_file(str(out_dir / "wikitext2_test.safetensors"))["tokens"].long()
        self.wikitext_val = load_file(str(out_dir / "wikitext2_val.safetensors"))["tokens"].long()
        self.mmlu_probe = _read_jsonl(out_dir / "mmlu_probe.jsonl")
        self.mmlu_final = _read_jsonl(out_dir / "mmlu_final.jsonl")
        self.mmlu_dev = _read_jsonl(out_dir / "mmlu_dev.jsonl")
        self.mmlu_fewshot = _read_jsonl(out_dir / "mmlu_fewshot.jsonl")
        self.fqa_final = _read_jsonl(out_dir / "fqa_final.jsonl")
        self.fqa_dev = _read_jsonl(out_dir / "fqa_dev.jsonl")
        self.fqa_fewshot = _read_jsonl(out_dir / "fqa_fewshot.jsonl")
        self.probe_by_domain: dict[str, list[dict]] = {}
        for r in self.mmlu_probe:
            self.probe_by_domain.setdefault(r["domain"], []).append(r)
        self.domains = sorted(self.probe_by_domain)

    def eval_sets(self, split: str) -> tuple[list[dict], list[dict], torch.Tensor]:
        """(MMLU rows, FreebaseQA rows, WikiText tokens) for ``final`` (held-out) or ``dev`` (method development)."""
        if split == "dev":
            return self.mmlu_dev, self.fqa_dev, self.wikitext_val
        if split != "final":
            raise ValueError(f"final_eval.split must be 'final' or 'dev', got {split!r}")
        return self.mmlu_final, self.fqa_final, self.wikitext

    def calibration_indices(self, seed: int, n: int) -> list[int]:
        """The run seed selects which calibration windows this run uses (as in Wanda's seeded C4 sampling)."""
        P = self.calibration_pool.shape[0]
        if n > P:
            raise ValueError(f"calibration.num_samples={n} exceeds pool_size={P}")
        rng = np.random.default_rng([int(seed), 7_919])
        return sorted(rng.choice(P, size=n, replace=False).tolist())
