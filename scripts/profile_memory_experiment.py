"""Profiling memory on SmolLM3-3B-Base: is the gradient profile identical, and does it fit, with (A) the dense copy
moved to CPU during profiling and (B) additionally activation checkpointing? No benchmark item is scored."""
import json
import os
import time

import torch

from capprune.config import data_dir, load_config, run_config
from capprune.data.prepare import PreparedData
from capprune.gradient_live import load_dense_copy
from capprune.model import build_units, load_model, tokenizer_id
from capprune.pruning.masks import DenseStore, PruneState
from capprune.pruning.profile import collect_act_sqnorm, collect_grad_importance

rc = run_config(load_config("configs/confirmatory_smollm3.yaml"), "cap_top1", 0.0, 0)
data = PreparedData(data_dir(rc), tokenizer_id(rc))
model = load_model(rc)
model.eval()
units = build_units(model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))
dense = DenseStore(units)
dm = load_dense_copy(rc, model, units, dense)
st = PruneState(units, rc["pruning"]["mode"])
import numpy as np

calib = data.calibration_pool[data.calibration_indices(0, int(rc["data"]["calibration"]["num_samples"]))]
pp = rc["profiling"]
prng = np.random.default_rng([int(rc["ranker"].get("profile_seed", 0)), 1, 17])
a_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["activation_samples"]), calib.shape[0]), replace=False))
g_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["gradient_samples"]), calib.shape[0]), replace=False))
out = {}

# activations (forward only) with both copies on the GPU
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
act = collect_act_sqnorm(model, st, calib[a_idx], {}, None, int(pp.get("activation_batch_size", 4)))
torch.cuda.synchronize()
out["activations_both_on_gpu"] = {"s": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
print(out, flush=True)

# A: dense copy on CPU during the gradient pass
dm.to("cpu")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
gA = collect_grad_importance(model, st, calib[g_idx])
torch.cuda.synchronize()
out["A_grad_dense_on_cpu"] = {"s": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
print(out, flush=True)

# B: A + activation checkpointing (train mode; dropout is 0 in this model)
print("attention_dropout", getattr(model.config, "attention_dropout", None), "hidden_dropout", getattr(model.config, "hidden_dropout", None), flush=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
gB = collect_grad_importance(model, st, calib[g_idx])
torch.cuda.synchronize()
model.eval()
model.gradient_checkpointing_disable()
out["B_grad_dense_on_cpu_checkpointing"] = {"s": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
rel = max(abs(gA[k] - gB[k]) / max(abs(gA[k]), 1e-30) for k in gA)
out["A_vs_B_max_relative_diff"] = rel
out["A_vs_B_identical"] = all(gA[k] == gB[k] for k in gA)
order = lambda g: sorted(g, key=g.get)
out["A_vs_B_same_unit_order"] = order(gA) == order(gB)
dm.to(next(model.parameters()).device)
print(out, flush=True)
json.dump(out, open(os.path.expanduser("~/cap-prune-runs/confirmatory/profile_memory_experiment.json"), "w"), indent=1)
