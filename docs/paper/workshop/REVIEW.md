# Review of the workshop draft (2026-11-25)

Reviewed: `draft.md` at `3558664`. Result: `draft_v2.md` in this folder, made from `draft.md` by 29 scripted edits.
`draft.md` is untouched, and no recorded result, table or figure was changed.

**Update, 2026-11-26.**
* v2 was checked against the run files and replaced `draft.md` in `440833e`; `draft_v2.md` no longer exists. The
  checks and the three mismatches are under "Number check of v2" below.
* The items that are done are marked under "Still open".
* §7's "One operator and one ranker" now reports the EXPLORATORY compensation check (⟨comp⟩, merged in PR #9),
  replacing "We did not test them". Its figures were checked against `compensation-feasibility/analysis/analysis.json`;
  two were reworded: the reconstruction-error cut is given over the 13 steps (median ×0.71; ×0.66 over 40 steps),
  and perplexity is lower only from step 9. The 13-step range is named before the refit's effects, which it scopes,
  and the bullet cites ⟨comp⟩ by key.

## What was checked

* **Every number in the draft** against `tables/*.md`, `docs/confirmatory_report.md`, `docs/exploratory_report.md`,
  `docs/confirmatory_study_plan.md` and `docs/bound_development_plan.md`. All of them match.
* **Not checkable from here** (the runs folder is not connected): the admissibility accuracies (54.5 % and 46.6 %), and
  the Qwen sentence in §2.2 (1,200–1,500 items per category). Both cite run files directly.
  * **Checked 2026-11-26.** `confirmatory/<model>/admissibility.json` gives 0.5451 (SmolLM3) and 0.4659 (Granite).
    `population-regate/report.md`, "Required pool sizes", gives 1,200–1,500: n50 at the median disagreement rate and
    a true loss of 0. §2.2 now says it holds at the median rate.
* **References.** Checked today against arXiv or the publisher: Zhang 2608.02940, Felicio 2610.00694, Alvarez
  2408.04759, Zecchin 2409.15844 (now ICML 2025), MMLU-CF 2412.15194 (now ACL 2025), Jazbec 2405.20915, Miller
  2411.00640, Sidik 2003. The Zhang quote is in the paper's §3.4, not its abstract.
  Added from memory and still to check: Berger 1982, Berger & Hsu 1996, Clopper & Pearson 1934, Nam 1997, Tango 1998,
  Hendrycks 2021, Raffel 2020, Frantar & Alistarh 2023.
* **The exact test** (`capprune/bounds.py`). The statistic matches Tango's restricted MLE. An independent size check
  at 1,107 nuisance values off the 400-point grid, at the 3-point limit:

  | n | critical z | largest size on the grid | largest size off the grid |
  |---|---|---|---|
  | 500 | −1.6588 | 0.04988 | 0.04988 |
  | 1,000 | −1.6684 | 0.04840 | 0.04838 |
  | 3,000 | −1.6505 | 0.04983 | **0.05001** |
  | 3,500 | −1.6498 | 0.05000 | 0.04991 |
  | 3,600 | −1.6609 | 0.04942 | 0.04882 |

  * The grid supremum is exceeded off the grid by 0.00001 at n = 3,000. The test is exact only up to grid resolution.
  * The binding nuisance values are at 96–99 % disagreement, far from the data (4–15 %), where the size is at most
    0.0494. The critical value is set by a region the data never visit.
  * Paper A uses this test only in one exploratory paragraph, which v2 now words accordingly. For the journal version,
    restrict the nuisance range (Berger–Boos) or refine the grid, and record it as a dated note in the bound plan (§8).

## Errors fixed in v2

1. "5.1 Why the screens coincide" sat under §4 and collided with "5.1 C1". It is now §4.1, as the text cites it.
2. S1 overshoot cited Figure 3; it is Figure 4.
3. The abstract called the sample-size result exploratory. It is S4, a secondary endpoint added after the pilot.
4. "Three secondary hypotheses" mixed up the hypotheses (C1, C4, C6) with the secondary endpoints (S1–S4).
5. Amendments 1–3 read as three changes to the study plan. Amendments 1 and 2 are the protocol's and predate the plan.
6. "C4 perplexity" (the corpus) collided with hypothesis C4.

## Omissions fixed in v2

1. **The cost result was missing from the abstract and conclusion.** Arm (a) never rejected a step, yet after
   confirmation it certified as much as the capability gate in all six pairs, at the lowest cost (S2). The draft
   presented only the gate's lower overshoot.
2. **Arm (a) is in effect an unscreened trajectory** (30 of 30 accepted; the 15 % threshold is relative to the parent).
   v2 says so, and says the result does not speak to a stricter perplexity rule or to Kodathala & Vunnam's system.
3. **S3 at 5 points:** in Granite seed 2 the gate arm certified 47.2 M less than the perplexity arm.
4. **The gate alone is not a certificate:** its final checkpoint was not certified in 4 of 6 pairs.
5. **C1–C6 were never defined.** v2 has one table in §3.
6. **The guarantee's conditions and scope** (frozen order; independent items; the MMLU and MMLU-CF mixture; 0-shot), the
   source counts, deduplication and the pilot's read of the sealed set.
7. **Why the claims are small:** the weakest category's loss at the first checkpoint past each claim (+2.0 to +4.2).
8. **C6:** rejections cluster within runs, so the sign test's independence is doubtful.
9. **Attribution:** non-inferiority framing; Nam, Tango and Sidik for the exact test; Berger for the IUT.
10. **Limitations:** one operator and one ranker; ranges are over 18 runs.

## Number check of v2 against the run files (2026-11-26)

**Method.** Every changed line of v2 that contains a number was compared with the run files: `study.json`,
`registration.json`, `loss_curves.json`, `frontier.json`, `real_data.json`, the runs' `run.json` configs and the
`ppl_top1` journals. Bibliographic numbers (years, volumes, pages, arXiv ids) were not checked here.

**Three statements did not match. All are fixed in `draft.md`.**
1. **"With 500 confirmation items per category, nothing was certifiable"** (abstract and S4; the S4 sentence was
   already in v1). The median is zero in every run, but 84 of the 900 subsamples certified one or two steps at
   3 points (`study.json`, `per_run_secondary[].S4`). At 2 points none did.
   * The same overstatement is in `docs/confirmatory_report.md`, which now has an erratum, and in
     `docs/exploratory_report.md`, which is corrected.
2. **"+2.0 to +4.2 points"** for the weakest category at the first checkpoint past each claim. It is +2.14 to +4.20
   (`real_data.json`; the low end is Granite seed 2, Other +2.14), so "+2.1 to +4.2".
3. **"Confirmation stops … varies by seed in Granite"** (§6; already in v1). Other ends confirmation in every Granite
   run, with STEM and Social Sciences in some (`loss_curves.json`).

**Also reworded:** §6's 500-item frontier line now says it is a median.

**Everything else matched.** That includes:
* the source and deduplication counts (7,317 / 788 / 4,995; 135, 6, 25,573);
* the 32 × 2,048-token C4 perplexity set;
* the S2 GPU seconds;
* −47.2 / 62.9 / 110.1 M at 5 points;
* the C6 per-run tallies;
* 1.05–1.84×.

**§5.2's arm-(a) figures** are generated in `tables/perplexity_arm_a.md` and match the text: 18.4 / 18.6 %, +9.9 to
+20.7 and +14.3 to +32.1, Social Sciences worst, claims at 2–4, 26–28 steps, 16–17 % overshoot and 18–32 points.

## Still open

**Needs the run files:**
* **Done (2026-11-26):** the two items listed as not checkable above are checked.
* **Done (2026-11-26):** arm (a)'s perplexity against the dense model, at the claims, the first failing checkpoints and
  the endpoints, plus a cumulative 15 % rule, post hoc. It is in §5.2 and `tables/perplexity_arm_a.md`.
* **Done (`440833e`):** the number check on v2 (above), and `draft.md` replaced with it.
* **Done (`440833e`):** a test for the exact test's size off the grid, and the dated note in
  `docs/bound_development_plan.md` §8.

**Needs a venue:**
* v2 is about 5,900 words with tables. A 4-page track needs the main text near 2,500 words, with §3's deviations, the
  cost and time paragraph, Table 2's mechanism columns and §6 moved to an appendix. An 8–9 page track fits as it is.
* Convert the bulleted body to prose and typeset it in the venue's template.
* Check the venue's policies on a later archival version and on LLM use in writing and analysis.

**Needs the team:** authorship and the reframing from the original proposal.
