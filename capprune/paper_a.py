"""Paper A (workshop) figures and tables, regenerated from run files by one command:

    python -m capprune paper-a --config configs/confirmatory.yaml --out docs/paper/workshop

Reads only files under the runs root:
* ``confirmatory/study/study.json`` (the confirmatory summary: primary endpoint, C1, C4, C6, S1-S4, cost);
* ``exploratory/frontier.json``, ``exploratory/loss_curves.json``, ``exploratory/c1_mechanism.json``
  (``capprune exploratory``; EXPLORATORY).

Writes ``figures/*.png`` + ``*.pdf`` and ``tables/*.md`` under ``--out``. Every table names its source file.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import resolve_path
from .confirmatory import CATS, _git
from .utils.io import read_json

ARMS = ("ppl_top1", "cap_top1", "cap_klprobe")
ARM_LABEL = {"ppl_top1": "(a) perplexity", "cap_top1": "(b) gate", "cap_klprobe": "(c) gate + KL"}
# reference palette (dataviz skill), slots 1-3, validated all-pairs; markers differ too (relief for aqua's contrast)
ARM_COLOR = {"ppl_top1": "#2a78d6", "cap_top1": "#eb6834", "cap_klprobe": "#1baf7a"}
ARM_MARK = {"ppl_top1": "o", "cap_top1": "s", "cap_klprobe": "^"}
TOL_COLOR = {"0.02": "#86b6ef", "0.03": "#3987e5", "0.04": "#1c5cab", "0.05": "#0d366b"}  # one-hue ramp, light -> dark
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "lines.linewidth": 1.4, "pdf.fonttype": 42})
    return plt


def _save(fig, out: Path, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / "figures" / f"{name}.{ext}"
        meta = {"CreationDate": None} if ext == "pdf" else {}  # no timestamp: regeneration is byte-reproducible
        fig.savefig(p, dpi=220, bbox_inches="tight", metadata=meta)
        paths.append(str(p))
    return paths


# ============================================================================ figures
def fig_loss_curves(LC: dict, out: Path) -> list[str]:
    plt = _style()
    from matplotlib.lines import Line2D

    written = []
    for M in LC["models"]:
        dense = None
        seeds = sorted({r["seed"] for r in M["runs"]})
        fig, axes = plt.subplots(len(seeds), len(CATS), figsize=(7.2, 1.85 * len(seeds) + 0.6), sharex=True, sharey=True)
        for i, s in enumerate(seeds):
            for j, c in enumerate(CATS):
                ax = axes[i, j]
                ax.axhline(3, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
                ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
                for arm in ("ppl_top1", "cap_klprobe", "cap_top1"):  # (b) drawn last: the shared prefix shows in its colour
                    r = next(x for x in M["runs"] if x["seed"] == s and x["method"] == arm)
                    pos = np.array([p["position"] for p in r["checkpoints"]])
                    est = np.array([100 * p["categories"][c]["estimate"] for p in r["checkpoints"]])
                    lo = np.array([100 * p["categories"][c]["lower"] for p in r["checkpoints"]])
                    hi = np.array([100 * p["categories"][c]["upper"] for p in r["checkpoints"]])
                    col = ARM_COLOR[arm]
                    ax.fill_between(pos, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
                    ax.plot(pos, est, color=col, lw=1.2 if arm == "ppl_top1" else 1.5, zorder=3)
                    # screen stop: the last frozen checkpoint (hollow marker in the arm's shape)
                    ax.plot(pos[-1], est[-1], ARM_MARK[arm], ms=5, mfc=SURF, mec=col, mew=1.3, zorder=5)
                    # confirmation stop: the first failing checkpoint of the fixed sequence (x)
                    ff = r["confirmation_first_failure"]
                    if ff is not None:
                        k = int(np.flatnonzero(pos == ff)[0])
                        ax.plot(pos[k], est[k], "x", ms=6, mew=1.6, color=INK, zorder=6)
                ax.set_yscale("symlog", linthresh=5, linscale=2.2)
                ax.set_ylim(-3, 38)
                ax.set_yticks([-2, 0, 3, 10, 30])
                ax.set_yticklabels(["−2", "0", "3", "10", "30"])
                if i == 0:
                    ax.set_title(c, color=INK)
                if j == 0:
                    ax.set_ylabel(f"seed {s}\nloss (points)")
                if i == len(seeds) - 1:
                    ax.set_xlabel("checkpoint")
        handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARK[a], mfc=SURF, mec=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARMS]
        handles += [Line2D([], [], color=INK, marker="x", ls="", label="confirmation stops (first failure)"),
                    Line2D([], [], color=INK2, ls=(0, (4, 3)), label="3-point limit")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        step = M["step_parameters"] / 1e6
        fig.suptitle(f"{M['label']}, {M['model_key']}: paired loss per category along each trajectory\n(step = {step:.1f} M "
                     f"parameters; band = one-sided 95 % bounds; hollow marker = screen stop; y linear to ±5, log beyond)",
                     fontsize=8, color=INK, y=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 0.98))
        written += _save(fig, out, f"loss_curves_{M['model_key']}")
        plt.close(fig)
        del dense
    return written


def fig_frontier(F: dict, out: Path) -> list[str]:
    plt = _style()
    fig, axes = plt.subplots(len(F["models"]), len(ARMS), figsize=(7.2, 4.6), sharex=True, sharey=True)
    for i, FM in enumerate(F["models"]):
        dense = FM["dense_parameters"]
        full_n = float(np.mean(list(FM["per_category_full"].values())))
        xs = [float(n) for n in F["sizes"] if n != "full"] + [full_n]
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            for t in F["tolerances"]:
                key = f"{t:.2f}"
                row = FM["by_arm"][arm][key]
                med = [100 * row[str(n)]["median"] / dense for n in F["sizes"]]
                q25 = [100 * row[str(n)]["q25"] / dense for n in F["sizes"]]
                q75 = [100 * row[str(n)]["q75"] / dense for n in F["sizes"]]
                off = 1.0 + 0.035 * (F["tolerances"].index(t) - 1.5)  # small horizontal offsets so the IQR bars do not overlap
                xo = [x * off for x in xs]
                ax.errorbar(xo, med, yerr=[np.subtract(med, q25), np.subtract(q75, med)], color=TOL_COLOR[key], marker="o",
                            ms=3.5, lw=1.4, elinewidth=1.0, capsize=0)
                if j == len(ARMS) - 1:
                    ax.annotate(f"{round(100 * t)} pts", (xo[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=7, color=INK2)
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(["500", "1,000", "2,000", "full"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(ARM_LABEL[arm], color=INK)
            if j == 0:
                ax.set_ylabel(f"{FM['label']}\ncertified (% of parameters)")
            if i == len(F["models"]) - 1:
                ax.set_xlabel("confirmation items per category")
    fig.suptitle("Certification frontier (EXPLORATORY): median and IQR over 50 stratified subsamples of each seed (full set: "
                 "the 3 seeds);\nlines = per-category limit", fontsize=8, color=INK)
    fig.tight_layout()
    w = _save(fig, out, "certification_frontier")
    plt.close(fig)
    return w


def fig_overshoot_cost(S: dict, out: Path) -> list[str]:
    plt = _style()
    fig, axes = plt.subplots(2, len(S["models"]), figsize=(7.2, 4.4), sharey="row")
    for j, M in enumerate(S["models"]):
        dense = M["dense_parameters"]
        sec = {x["run_id"]: x for x in M["per_run_secondary"]}
        for i, (key, lab) in enumerate((("S1", "S1 overshoot\n(% of parameters)"), ("S2", "S2 items per\nconfirmed M parameters"))):
            ax = axes[i, j]
            for k, arm in enumerate(ARMS):
                rows = sorted((x for x in sec.values() if x["method"] == arm), key=lambda x: x["seed"])
                vals = [100 * x["S1"] / dense if key == "S1" else x["S2"]["items_per_confirmed_M"] for x in rows]
                xj = k + np.array([-0.12, 0.0, 0.12])[: len(vals)]
                ax.scatter(xj, vals, s=22, color=ARM_COLOR[arm], marker=ARM_MARK[arm], zorder=3, edgecolors=SURF, linewidths=0.6)
                ax.hlines(np.mean(vals), k - 0.25, k + 0.25, color=INK, lw=1.2, zorder=4)
            ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
            ax.set_xlim(-0.6, len(ARMS) - 0.4)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{M['label']}: {M['model_key']}", color=INK)
            ax.set_ylim(bottom=0)
    fig.suptitle("Overshoot (S1) and evaluation cost per confirmed parameter (S2), per seed (markers) and mean (bar); "
                 "added after the pilot", fontsize=8, color=INK)
    fig.tight_layout()
    w = _save(fig, out, "overshoot_and_cost")
    plt.close(fig)
    return w


# ============================================================================ tables
def _M(v):
    return "–" if v is None else f"{v / 1e6:.1f} M"


def _pt(v):
    return "–" if v is None else f"{100 * v:+.2f}"


def tables(S: dict, F: dict, C1m: dict, src: dict) -> dict[str, str]:
    T = {}
    # primary
    L = ["# Primary endpoint: confirmed parameters removed per model x arm x seed", "", f"Source: `{src['study']}` "
         "(`models[].primary`). Seed 0 of model 1 is the pilot.", "",
         "| model | seed | (a) perplexity | (b) gate | (c) gate + KL | (b) − (a) | (c) − (b) |", "|---|---|---|---|---|---|---|"]
    for M in S["models"]:
        P, dp = M["primary"], M["dense_parameters"]
        for s in sorted(P["ppl_top1"]["per_seed"], key=int):
            L.append(f"| {M['label']} | {s} | " + " | ".join(f"{_M(P[a]['per_seed'][s])} ({100 * P[a]['per_seed'][s] / dp:.2f} %)" for a in ARMS)
                     + f" | {_M(P['b_minus_a']['per_seed'][s])} | {_M(P['c_minus_b']['per_seed'][s])} |")
        L.append(f"| {M['label']} | mean [range] | " + " | ".join(f"{_M(P[a]['mean'])} [{_M(P[a]['min'])}, {_M(P[a]['max'])}]" for a in ARMS)
                 + f" | {_M(P['b_minus_a']['mean'])} | {_M(P['c_minus_b']['mean'])} |")
    T["primary"] = "\n".join(L) + "\n"
    # S1-S4
    L = ["# Secondary endpoints S1–S4 (added after the pilot; model 1 seed 0's values are post hoc)", "",
         f"Source: `{src['study']}` (`models[].secondary`). Mean [range] over seeds; S4 median [IQR] over all draws of an arm's seeds.", ""]
    for M in S["models"]:
        for name, sv in M["secondary"].items():
            if M["model_key"] != "smollm3-3b" and name != "all seeds":
                continue
            L += [f"## {M['label']} ({M['model_key']}), {name}", "", "| endpoint | (a) | (b) | (c) |", "|---|---|---|---|"]
            for key, lab, f in (("S1", "S1 overshoot", _M), ("S2_items_per_M", "S2 items per confirmed M", lambda v: f"{v:,.0f}"),
                                ("S2_gpu_seconds_per_M", "S2 GPU s per confirmed M", lambda v: f"{v:,.1f}"), ("S3", "S3 confirmed at 5 points", _M)):
                X = sv[key]
                L.append(f"| {lab} | " + " | ".join(f"{f(X[a]['mean'])} [{f(X[a]['min'])}, {f(X[a]['max'])}]" for a in ARMS) + " |")
            for n in sorted(sv["S4"]["ppl_top1"], key=lambda n: (n == "full", int(n) if n != "full" else 0)):
                L.append(f"| S4, n = {n} | " + " | ".join(f"{_M(sv['S4'][a][n]['median'])} [{_M(sv['S4'][a][n]['q25'])}, {_M(sv['S4'][a][n]['q75'])}]" for a in ARMS) + " |")
            L.append("")
    T["secondary_S1_S4"] = "\n".join(L) + "\n"
    # C1
    L = ["# C1: pool-to-confirmation gap (confirmation − gate pool estimate at acceptance; tested checkpoints of (b) and (c))", "",
         f"Source: `{src['study']}` (`models[].c1`); C1 mechanism columns (EXPLORATORY): `{src['c1_mechanism']}`.", "",
         "| model | category | gate pool | mean gap | max gap | gate decisions | share using the whole pool | item uses per pool item |",
         "|---|---|---|---|---|---|---|---|"]
    for M, CM in zip(S["models"], C1m["models"]):
        t = M["c1"]["tested_summary"]
        for c in CATS:
            v = CM["categories"][c]
            L.append(f"| {M['label']} | {c} | {v['gate_pool']:,} | {_pt(t['per_category_mean_gap'][c])} | {_pt(t['per_category_max_gap'][c])} | "
                     f"{v['gate_decisions']} | {100 * v['share_of_decisions_using_whole_pool']:.0f} % | {v['item_uses_per_pool_item']:.1f} |")
        L.append(f"| {M['label']} | **worst category** | | **{_pt(t['mean_worst_gap'])}** [{_pt(t['ci95'][0])}, {_pt(t['ci95'][1])}] "
                 f"({t['checkpoints']} checkpoints) | | | | criterion met: {M['c1']['model_criterion_met']} |")
    L += ["", f"C1 supported (both models): **{S['c1_supported']}**. The mechanism columns are 2 models × 4 categories: too few for inference."]
    T["c1"] = "\n".join(L) + "\n"
    # C6
    c6 = S["c6"]
    L = ["# C6: counterfactual log of arm (c), pooled (the rejection is the unit)", "", f"Source: `{src['study']}` (`c6`).", "",
         "| comparison | rejections | better | worse | ties | p (two-sided sign test) |", "|---|---|---|---|---|---|"]
    for name in ("all", "excluding_pilot"):
        for k, lab in (("primary_KL_vs_rank2", "KL vs gradient rank 2 (primary)"), ("secondary_TV_vs_KL", "TV vs KL"),
                       ("secondary_S1c_vs_KL", "S1c vs KL")):
            v = c6[name][k]
            L.append(f"| {lab}, {name.replace('_', ' ')} | {v['rejections']} | {v['better']} | {v['worse']} | {v['ties']} | {v['p_two_sided']:.3f} |")
    L += ["", f"Criterion (majority and p < 0.05): **{c6['criterion_met']}**."]
    T["c6"] = "\n".join(L) + "\n"
    # frontier (exploratory)
    L = ["# Certification frontier (EXPLORATORY)", "", f"Source: `{src['frontier']}`. Certified parameters removed, median [IQR] "
         "over the 50 subsamples of every seed (the full set: one value per seed).", ""]
    for FM in F["models"]:
        L += [f"## {FM['label']} ({FM['model_key']})", "", "| arm | limit | " + " | ".join(f"n = {n}" for n in F["sizes"]) + " |",
              "|---|---|" + "---|" * len(F["sizes"])]
        for arm in ARMS:
            for t in F["tolerances"]:
                row = FM["by_arm"][arm][f"{t:.2f}"]
                L.append(f"| {ARM_LABEL[arm]} | {round(100 * t)} pts | " + " | ".join(
                    f"{_M(row[str(n)]['median'])} [{_M(row[str(n)]['q25'])}, {_M(row[str(n)]['q75'])}]" for n in F["sizes"]) + " |")
        L.append("")
    T["frontier"] = "\n".join(L) + "\n"
    return T



def perplexity_arm_a(cfg: dict, root: Path, LC: dict) -> tuple[str, list[str]]:
    """Section 5.2's figures for arm (a): from each ``ppl_top1`` run's journal (the perplexity screen's own records:
    dense C4 perplexity ``ppl0``, each accepted trial's ``ppl_trial`` and ``relative_increase``), aligned to the frozen
    order by mask hash, and from ``loss_curves.json`` (per-category confirmation estimates)."""
    import json

    curves = {(M["model_key"], r["seed"]): (M, r) for M in LC["models"] for r in M["runs"] if r["method"] == "ppl_top1"}
    rows, inputs = [], []
    for m in cfg["confirmatory"]["study"]["models"]:
        od = resolve_path(cfg, m["out_dir"])
        for b in m["batches"]:
            bdir = od / "batches" / b
            for rec in read_json(bdir / "confirm.json")["runs"]:
                if rec["method"] != "ppl_top1":
                    continue
                rid, seed = rec["run_id"], int(rec["seed"])
                order = read_json(bdir / f"{rid}.order.json")["order"]
                jpath = od / "runs" / rid / "journal.jsonl"
                inputs += [str(jpath), str(bdir / f"{rid}.order.json")]
                steps, rises, ppl0 = {}, [], None
                for line in jpath.read_text(encoding="utf-8").splitlines():
                    j = json.loads(line)
                    g = j.get("gate") or {}
                    if j.get("type") != "iteration" or g.get("kind") != "perplexity":
                        continue
                    ppl0 = g["ppl0"] if ppl0 is None else ppl0
                    if abs(g["ppl0"] - ppl0) > 1e-9:
                        raise SystemExit(f"{jpath}: the dense perplexity changes within the run")
                    rises.append(float(g["relative_increase"]))
                    if g["decision"] == "accept":
                        steps[j["trial_mask_hash"]] = float(g["ppl_trial"])
                ratio = []
                for o in order:
                    if o["mask_hash"] not in steps:
                        raise SystemExit(f"{rid}: frozen checkpoint {o['position']} has no accepted perplexity record")
                    ratio.append(steps[o["mask_hash"]] / ppl0)
                M, r = curves[(read_json(bdir / "report.json")["model_key"], seed)]
                dense = int(rec["dense_parameters"])
                last = r["checkpoints"][-1]
                est = {c: last["categories"][c]["estimate"] for c in CATS}
                worst = max(est, key=est.get)
                claim = r["confirmation_claim_position"] or 0
                rows.append({"label": M["label"], "model": M["model_key"], "seed": seed, "ppl0": ppl0,
                             "accepted": len(steps), "proposals": len(rises), "max_rise": max(rises),
                             "ratio_claim": ratio[claim - 1] if claim else None, "ratio_end": ratio[-1],
                             "claim": claim, "end": r["screen_stop_position"],
                             "end_frac": last["params_removed"] / dense,
                             "overshoot_frac": (last["params_removed"] - (r["checkpoints"][claim - 1]["params_removed"] if claim else 0)) / dense,
                             "est": est, "worst": worst})
    rows.sort(key=lambda x: (x["model"] != "smollm3-3b", x["seed"]))

    def _r3(v):
        return "–" if v is None else f"{v:.3f}"

    L = ["# Arm (a), the perplexity screen: the figures behind section 5.2", "",
         "Sources: each `ppl_top1` run's `journal.jsonl` (dense C4 perplexity `ppl0`, each trial's `ppl_trial` and "
         "`relative_increase`; the threshold is 15 % over the parent) aligned to `batches/<batch>/<run>.order.json` by mask "
         "hash; per-category estimates from `~/cap-prune-runs/exploratory/loss_curves.json`.", "",
         "| model | seed | accepted / proposals | largest one-step rise | dense C4 perplexity | perplexity / dense at the claim | "
         "at the screen's end | claim (checkpoint) | steps from claim to end | removed at the end | overshoot | "
         + " | ".join(f"{c} at the end" for c in CATS) + " | worst |", "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * (len(CATS) + 1)]
    for x in rows:
        L.append(f"| {x['label']} | {x['seed']} | {x['accepted']} / {x['proposals']} | {100 * x['max_rise']:.1f} % | {x['ppl0']:.2f} | "
                 f"{_r3(x['ratio_claim'])} | "
                 f"{x['ratio_end']:.3f} | {x['claim']} | {x['end'] - x['claim']} | {100 * x['end_frac']:.2f} % | {100 * x['overshoot_frac']:.2f} % | "
                 + " | ".join(f"{100 * x['est'][c]:+.2f}" for c in CATS) + f" | {x['worst']} |")

    def rng(vals, f):
        return f"{f(min(vals))} to {f(max(vals))}"
    L += ["", "**Ranges quoted in the draft** (computed here; check against the text):", ""]
    for key, lab in (("smollm3-3b", "SmolLM3"), ("granite-3.3-2b", "Granite")):
        xs = [x for x in rows if x["model"] == key]
        L.append(f"* {lab}: removed at the end {rng([x['end_frac'] for x in xs], lambda v: f'{100 * v:.1f} %')}; per-category "
                 f"loss at the end {rng([v for x in xs for v in x['est'].values()], lambda v: f'{100 * v:+.1f}')}; worst category: "
                 f"{', '.join(sorted({x['worst'] for x in xs}))}; perplexity / dense at the end {rng([x['ratio_end'] for x in xs], lambda v: f'{v:.2f}')}; "
                 f"largest one-step rise {rng([x['max_rise'] for x in xs], lambda v: f'{100 * v:.1f} %')}.")
    L.append(f"* All runs: claims at checkpoints {rng([x['claim'] for x in rows], str)}; {rng([x['end'] - x['claim'] for x in rows], str)} "
             f"steps from claim to the screen's end; overshoot {rng([x['overshoot_frac'] for x in rows], lambda v: f'{100 * v:.1f} %')}; "
             f"worst-category loss at the end {rng([x['est'][x['worst']] for x in rows], lambda v: f'{100 * v:+.1f}')} points; "
             f"perplexity / dense at the claim {rng([x['ratio_claim'] for x in rows if x['ratio_claim']], lambda v: f'{v:.3f}')}.")
    return "\n".join(L) + "\n", inputs


def build(cfg: dict, out: Path) -> Path:
    commit, clean = _git("rev-parse", "HEAD"), _git("status", "--porcelain") == ""  # before anything is written
    root = resolve_path(cfg, "~/cap-prune-runs")
    src = {"study": root / "confirmatory" / "study" / "study.json", "frontier": root / "exploratory" / "frontier.json",
           "loss_curves": root / "exploratory" / "loss_curves.json", "c1_mechanism": root / "exploratory" / "c1_mechanism.json"}
    S, F, LC, C1m = (read_json(src[k]) for k in ("study", "frontier", "loss_curves", "c1_mechanism"))
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = fig_loss_curves(LC, out) + fig_frontier(F, out) + fig_overshoot_cost(S, out)
    rel = {k: "~/cap-prune-runs/" + v.relative_to(root).as_posix() for k, v in src.items()}
    T = tables(S, F, C1m, rel)
    T["perplexity_arm_a"], ppl_inputs = perplexity_arm_a(cfg, root, LC)
    for name, text in T.items():
        (out / "tables" / f"{name}.md").write_text(text, encoding="utf-8")
    manifest = ["# Generated files", "", "Regenerate with:", "", "```", "python -m capprune paper-a --config configs/confirmatory.yaml "
                "--out docs/paper/workshop", "```", "", f"Last generated at commit `{commit[:7]}` "
                f"(clean tree before generating: {clean}), {time.strftime('%Y-%m-%d %H:%M')}.", "",
                "Inputs (run files):", ""] + [f"* `{v}`" for v in rel.values()] + [
                "* the `ppl_top1` runs' `journal.jsonl` and `<run>.order.json` (for `tables/perplexity_arm_a.md`)", "", "Outputs:", ""]
    manifest += [f"* `figures/{Path(p).name}`" for p in figs] + [f"* `tables/{n}.md`" for n in T]
    manifest += ["", "Not generated: `tables/deviations.md` (transcribed from `docs/confirmatory_report.md`) and `draft.md`."]
    (out / "GENERATED.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return out / "GENERATED.md"
