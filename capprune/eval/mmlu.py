"""MMLU multiple-choice scoring by next-token logits of " A"/" B"/" C"/" D".

Prompts follow the lm-evaluation-harness format.  Batches are left-padded
with explicit ``position_ids`` so a question's score does not depend on what
it was batched with, and only the last position's logits are materialised.
Batches are formed by a token budget that halves (and is remembered) on CUDA OOM.
"""
from __future__ import annotations

from typing import Callable

import torch

from ..utils.gpu import free_memory, is_oom
from ..utils.log import get_logger

log = get_logger("mmlu")
LETTERS = ["A", "B", "C", "D"]


def format_question(r: dict, with_answer: bool) -> str:
    s = r["question"] + "\n" + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, r["choices"])) + "Answer:"
    if with_answer:
        s += f" {LETTERS[r['answer']]}\n\n"
    return s


def build_prompt(r: dict, fewshot_by_subject: dict[str, list[dict]], shots: int) -> str:
    head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
    ex = fewshot_by_subject.get(r["subject"], [])[:shots]
    return head + "".join(format_question(e, True) for e in ex) + format_question(r, False)


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for L in LETTERS:
        t = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        ids.append(t[-1])
    if len(set(ids)) != 4:
        raise RuntimeError("tokenizer maps answer letters to non-distinct tokens")
    return ids


class MMLUScorer:
    def __init__(self, model, tokenizer, fewshot: list[dict], shots: int, max_len: int,
                 batch_store: dict, key: str, max_batch_tokens: int = 32768, heartbeat=None):
        self.model = model
        self.tok = tokenizer
        self.shots = shots
        self.max_len = max_len
        self.store = batch_store
        self.key = key
        self.budget = int(batch_store.get(key, max_batch_tokens))
        self.hb = heartbeat
        self.letter_ids = letter_token_ids(tokenizer)
        self.fewshot: dict[str, list[dict]] = {}
        for e in fewshot:
            self.fewshot.setdefault(e["subject"], []).append(e)
        self.dev = next(model.parameters()).device
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    def encode(self, r: dict) -> list[int]:
        k = self.shots
        while True:
            ids = self.tok(build_prompt(r, self.fewshot, k), add_special_tokens=False)["input_ids"]
            if len(ids) <= self.max_len or k == 0:
                return ids[-self.max_len :]
            k -= 1  # drop exemplars for very long questions (e.g. professional_law)

    @torch.no_grad()
    def _score_batch(self, seqs: list[list[int]]) -> list[int]:
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s) :] = torch.tensor(s)
            att[i, L - len(s) :] = 1
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        ids, att, pos = ids.to(self.dev), att.to(self.dev), pos.to(self.dev)
        try:
            out = self.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False)
        logits = out.logits[:, -1, :].float()
        return logits[:, self.letter_ids].argmax(-1).tolist()

    def score(self, rows: list[dict], on_progress: Callable[[int, int], None] | None = None) -> dict[str, dict]:
        """Returns ``{qid: {"pred": int, "correct": bool}}``."""
        enc = [(r, self.encode(r)) for r in rows]
        enc.sort(key=lambda x: len(x[1]), reverse=True)  # longest first: OOM shows up immediately
        out: dict[str, dict] = {}
        i = 0
        while i < len(enc):
            L = len(enc[i][1])
            n = max(1, self.budget // max(L, 1))
            batch = enc[i : i + n]
            try:
                preds = self._score_batch([s for _, s in batch])
            except BaseException as e:
                if not is_oom(e) or n == 1:
                    raise
                free_memory()
                self.budget = max(L, self.budget // 2)
                self.store[self.key] = self.budget
                log.warning("OOM scoring MMLU; token budget -> %d", self.budget)
                continue
            for (r, _), p in zip(batch, preds):
                out[r["qid"]] = {"pred": int(p), "correct": bool(p == r["answer"])}
            i += len(batch)
            if self.hb:
                self.hb.beat(None, f"mmlu {i}/{len(enc)}")
            if on_progress:
                on_progress(i, len(enc))
        return out
