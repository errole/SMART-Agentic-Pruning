"""Closed-book FreebaseQA (the base paper's factual-knowledge metric): greedy few-shot generation, exact match."""
from __future__ import annotations

import re
import string

import torch

from ..utils.gpu import AdaptiveBatcher

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def build_prompt(q: str, fewshot: list[dict]) -> str:
    ex = "".join(f"Q: {e['question']}\nA: {e['answers'][0]}\n\n" for e in fewshot)
    return f"{ex}Q: {q}\nA:"


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], fewshot: list[dict], batch_store: dict, max_new_tokens: int = 24,
             initial_bs: int = 32, heartbeat=None) -> dict[str, dict]:
    from transformers import GenerationConfig

    dev = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gc = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad,
                          eos_token_id=tokenizer.eos_token_id, temperature=None, top_p=None, top_k=None)
    prompts = [(r, tokenizer(build_prompt(r["question"], fewshot), add_special_tokens=False)["input_ids"]) for r in rows]
    prompts.sort(key=lambda x: len(x[1]), reverse=True)
    out: dict[str, dict] = {}

    def run(batch):
        L = max(len(p) for _, p in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, (_, p) in enumerate(batch):
            ids[i, L - len(p) :] = torch.tensor(p)
            att[i, L - len(p) :] = 1
        gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), generation_config=gc)
        res = []
        for (r, _), g in zip(batch, gen[:, L:].tolist()):
            text = tokenizer.decode(g, skip_special_tokens=True).split("\n")[0].strip()
            golds = {normalize(a) for a in r["answers"]}
            res.append((r["qid"], {"pred": text, "correct": normalize(text) in golds}))
        if heartbeat:
            heartbeat.beat(None, f"freebaseqa {len(out) + len(res)}/{len(prompts)}")
        return res

    for qid, rec in AdaptiveBatcher(batch_store, "fqa_bs", initial_bs).run(prompts, run):
        out[qid] = rec
    return out
