"""Target-model loading and the definition of prunable *units*.

* ``unstructured`` and ``nm`` modes: a unit is one linear module of a decoder
  layer (``L12.q_proj`` ... ``L12.down_proj``) - 7 x 28 = 196 units for
  Qwen3-1.7B, matching the base paper's per-linear-layer decisions.
* ``structured`` mode: a unit is one MLP block (``L12.mlp``); pruning removes
  intermediate channels jointly from gate/up/down so the block physically
  shrinks (real speed-up on any GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .utils.log import get_logger

log = get_logger("model")

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


@dataclass
class Unit:
    name: str
    layer: int
    kind: str  # linear kind (q_proj ...) or "mlp"
    linears: dict[str, nn.Linear] = field(repr=False)
    numel: int = 0  # eligible weights in this unit

    @property
    def weight(self) -> torch.Tensor:  # for linear units
        return self.linears["w"].weight

    @property
    def channels(self) -> int:  # for mlp units
        return self.linears["down"].weight.shape[1]


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def device_of(cfg: dict) -> torch.device:
    want = cfg["model"].get("device", "cuda")
    if want.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU (only sensible for smoke tests)")
        return torch.device("cpu")
    return torch.device(want)


def model_path(cfg: dict) -> str:
    """A Hub id stays as is; a local path is resolved relative to the config file."""
    name = str(cfg["model"]["name_or_path"])
    looks_local = name.startswith((".", "/", "~", "\\")) or (len(name) > 1 and name[1] == ":")
    if looks_local and "_config_dir" in cfg:
        from .config import resolve_path

        return str(resolve_path(cfg, name))
    return name


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(model_path(cfg), revision=m.get("revision"), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenizer_id(cfg: dict) -> str:
    m = cfg["model"]
    return f"{m['name_or_path']}@{m.get('revision') or 'main'}"


def load_model(cfg: dict):
    from transformers import AutoModelForCausalLM

    m = cfg["model"]
    dev = device_of(cfg)
    dtype = _dtype(m.get("dtype", "bfloat16"))
    if dev.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    kw = dict(revision=m.get("revision"), attn_implementation=m.get("attn_implementation", "sdpa"),
              low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path(cfg), dtype=dtype, **kw)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(model_path(cfg), torch_dtype=dtype, **kw)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    return model


def decoder_layers(model) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        ok = True
        for a in path.split("."):
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok:
            return obj
    raise RuntimeError("cannot locate decoder layers on this model")


def build_units(model, mode: str, eligible: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for i, layer in enumerate(decoder_layers(model)):
        if mode == "structured":
            mlp = layer.mlp
            lin = {"gate": mlp.gate_proj, "up": mlp.up_proj, "down": mlp.down_proj}
            numel = sum(l.weight.numel() for l in lin.values())
            units.append(Unit(f"L{i}.mlp", i, "mlp", lin, numel))
            continue
        for kind in ATTN + MLP:
            if kind not in eligible:
                continue
            parent = layer.self_attn if kind in ATTN else layer.mlp
            lin = getattr(parent, kind)
            units.append(Unit(f"L{i}.{kind}", i, kind, {"w": lin}, lin.weight.numel()))
    if not units:
        raise RuntimeError("no prunable units found")
    return units


# ---------------------------------------------------------------------------- tiny model for smoke tests
def build_tiny_model(out_dir: str | Path, vocab_words: int = 400, seed: int = 0) -> Path:
    """A randomly initialised miniature Qwen3 plus a small BPE tokenizer, saved locally.

    Exercises the exact Qwen3 code paths (GQA, q/k norms, rotary, gated MLP)
    without any download, so the whole pipeline can be tested on a laptop.
    """
    import random
    import string

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    out = Path(out_dir)
    if (out / "config.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    corpus = [" ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 7)))
                       for _ in range(30)) for _ in range(300)]
    corpus += ["Answer: A B C D ? . , \n Q: A:"] * 20
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|endoftext|>",
                                     pad_token="<|endoftext|>")
    hf_tok.save_pretrained(out)
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, eos_token_id=hf_tok.eos_token_id, pad_token_id=hf_tok.pad_token_id)
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    return out
