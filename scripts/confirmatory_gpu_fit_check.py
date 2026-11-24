"""GPU fit check with profiling.gradient_checkpointing (no benchmark item scored): load the model and the dense copy,
profile the dense parent exactly as the live loop does; for Granite also verify the profile is identical without
checkpointing (dense copy moved to CPU so that pass fits)."""
import json
import os
import time

import numpy as np
import torch

from capprune.config import data_dir, load_config, run_config
from capprune.data.prepare import PreparedData
from capprune.gradient_live import load_dense_copy, profile_parent
from capprune.model import build_units, load_model, tokenizer_id
from capprune.pruning.masks import DenseStore, PruneState
from capprune.pruning.profile import collect_grad_importance

res = {}
import sys
for m in sys.argv[1:]:
    rc = run_config(load_config(f"configs/confirmatory_{m}.yaml"), "cap_top1", 0.0, 0)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    t0 = time.time()
    model = load_model(rc)
    model.eval()
    units = build_units(model, rc["pruning"]["mode"], rc["pruning"].get("eligible", []))
    dense = DenseStore(units)
    dm = load_dense_copy(rc, model, units, dense)
    st = PruneState(units, rc["pruning"]["mode"])
    calib = data.calibration_pool[data.calibration_indices(0, int(rc["data"]["calibration"]["num_samples"]))]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    act, sens, grad = profile_parent(model, st, calib, rc, {}, None)
    torch.cuda.synchronize()
    r = {"load_s": round(t1 - t0, 1), "profile_s": round(time.time() - t1, 1),
         "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "units": len(units),
         "channels_per_unit": units[0].channels, "total_params": sum(p.numel() for p in model.parameters())}
    if m == "granite" and not os.environ.get("SKIP_IDENTITY"):
        pp = rc["profiling"]
        prng = np.random.default_rng([int(rc["ranker"].get("profile_seed", 0)), 1, 17])
        prng.choice(calib.shape[0], size=min(int(pp["activation_samples"]), calib.shape[0]), replace=False)
        g_idx = np.sort(prng.choice(calib.shape[0], size=min(int(pp["gradient_samples"]), calib.shape[0]), replace=False))
        dm.to("cpu")
        torch.cuda.empty_cache()
        g_plain = collect_grad_importance(model, st, calib[g_idx], checkpointing=False)
        r["profile_identical_without_checkpointing"] = all(g_plain[k] == grad[k] for k in grad)
    res[m] = r
    print(m, r, flush=True)
    del model, dm, dense, act, sens, grad
    torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser(f"~/cap-prune-runs/confirmatory/gpu_fit_check_{sys.argv[1]}.json"), "w"), indent=1)
