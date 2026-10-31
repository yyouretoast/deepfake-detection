# Full Audit Report — `deepfake-detection` (Release-1) & arXiv Manuscript

**Auditor scope:** repo state at HEAD `1625638` ("fix: comprehensive project and manuscript audit remediation", 2026-10-31, 132 tracked files), manuscript (`manuscript/main.tex` ~708 lines + `manuscript/supplementary.tex`), all `results/release1_run/*` artifacts, all `src/` modules, all `scripts/`, test suite, docs (README, MODEL_CARD, AI-docs), infra (Dockerfile, CI/CD, requirements, uv.lock, notebook), checkpoints, and `paper.zip`.
**Method:** no claim was accepted from README/paper/self-audits. Every headline number was re-derived from stored artifacts where possible (fixtures replayed, splits re-verified, ablation cache replayed through DeLong in log-space, fine-grained log-space p-values, params counted, hashes compared). Scratch verification scripts live in `.audit_tmp/` (uncommitted).

---

## 1. Verdict (top line)

The **numerical core of the paper is genuine**: every headline metric, split count, fixture, checkpoint parameter count, statistical test, hyperparameter ledger entry, and checkpoint SHA that I checked traces to real stored artifacts and reproduces to the printed precision. The compile-verification tooling (`verify_latex_full.py`) passes 6/6 tables against the generator.

However, the audit confirms **one substantive falsehood in the supplement**, **one provenance misattribution** in a stats paragraph, **one wrong number in MODEL_CARD**, and a set of minor infra/documentation inaccuracies, plus **two residual unknowns** that could not be fully closed from surviving artifacts.

---

## 2. Verified-correct catalog (evidence-backed)

| # | Claim (paper/doc) | Verdict | Evidence |
|---|---|---|---|
| 1 | Splits: 37,104 / 35,016 / 26,981 (11,052R/26,052F; 4,008R/31,008F; 7,609R/19,372F) | ✅ | Re-verified from stored split data; pipeline log L3 matches exactly; manifest generator counts match 4-way strata sums (`celeb_1 17,412 / celeb_0 2,412 / ffpp_1 8,640 / ffpp_0 8,640`). |
| 2 | Zero crop/video/domain/identity overlap; FF++ 720/140/140 pair-disjoint; Celeb-DF 59 ids = 21+19+19; DFD 1,000 vids; YouTube 300 | ✅ | Independently verified earlier in audit; leakage checks re-run clean. |
| 3 | Test AUC 0.8656 / PR 0.9374 / EER 21.82 / F1 0.8292 (τ*=0.2600) | ✅ | Replay from `test_predictions.json`: `AUC(probs_cal)=0.865645` exact; N=26,981, fakes=19,372 exact. |
| 4 | Temporal: naive-avg 0.8962/19.43, max-pool 0.8544/21.48, Bi-GRU 0.8994/0.9571/18.54 | ✅ | All present in `release1_results.json` (19.4277, 21.4752, 0.89943…) and log L68–70. **Resolves prior "untraceable EER" concern — values ARE stored.** |
| 5 | Calibration: ECE 0.27121→0.14073 (test), 0.1965→0.0994 (val, −49.4%); Platt a=0.27831 b=0.40891, T_eff 3.5931; temp-only T*=5.2788 | ✅ | Checkpoint `calibration` block matches all constants; log L38–39 match. |
| 6 | τ*=0.2600; taus 0.40/0.60; certified precision 0.5, `is_precision_certified: false` (honest fallback) | ✅ | Checkpoint stores exactly this; main.tex:247 discloses fallback honestly. |
| 7 | DeLong Z=28.47, p<1e-16, ΔAUC=+0.0359, CI [+0.0309,+0.0406] (1,000 video-clustered bootstrap), McNemar χ²=614.35 (+fake recall 7.82 pp, χ²=969.22) | ✅ | `tier1_rigorous_evaluation.json` contains exactly these values; stats script implements genuine fast-DeLong, video-clustered bootstrap, corrected McNemar, spatial baseline from real re-inference (paired 0.8297 vs standalone 0.8370 — both correctly distinguished in main.tex:423). |
| 8 | Ablation table (Table 5): full 0.8656/21.82/F1 0.8466; static gate 0.8508/22.65; uniform concat 0.8522/22.85; spectral/SRM/Bayar 0.4867/0.4810/0.4899 | ✅ | Replayed `ablation_cache.npz` (7 arrays, N=26,981, 19,372F/7,609R) through stored Platt constants: **all six rows reproduce with d=0.000000**, Youden F1 0.8466 reproduced (τ=0.2199). |
| 9 | Ablation DeLong Z=25.65 (vs static gate) and Z=27.03 (vs uniform concat) | ✅ values / ⚠️ citation | **Reproduced exactly** (25.6474, 27.0318) from cache; log-space p-bounds **hold** (4.52e-145 < 1e-140; 6.25e-161 < 1e-160 — the latter barely). BUT main.tex:423 attributes them to `tier1_rigorous_evaluation.json`, which contains no such entries (see §3.2). |
| 10 | Frame fixture N=26,981 exact, recomputed AUC matches | ✅ | Earlier audit step: fixture records replayed → 0.865645. |
| 11 | Temporal fixture labeled legacy (N=1,120, AUC 0.8719) | ✅ | `temporal_test_predictions.json`: `fixture_status=legacy_phase1_subset_export`, honest note string, recomputed_auc=0.871867 (scope=*this_fixture*). |
| 12 | LOTO/LOMO genuine per-fold retraining; zero-shot evals; T* diagnostic-only; Fold-5 T*=10.0 at bound (0.5763) | ✅ | Log L101–121 shows retained-train shrinkage per fold (36,192/34,608/35,244/35,352/19,692) and ~65–80 min/fold runtimes; supplementary+paper correctly caution T* saturation. |
| 13 | Robustness: N=1,000 balanced crops, seed 42, 15 regimes | ✅ | `run_release1_kaggle.py:1575–1576` `subsample_balanced(1000, seed=42)`; JSON/log/table values identical (JPEG 0.8408/0.7973/0.7626/0.7410 etc.). Caption's "trained strictly on clean crops…" phrase is FALSE (§3.1). |
| 14 | Subdomain table | ✅ | `subdomain_results.json` matches paper Table 2 row-for-row (192/576/240/348/5700/11992 fakes; AUC 0.9592… etc.). |
| 15 | Latency table (RTX 4060, 71.0 FPS) — *not* T4 | ✅ | Log L88 warning: pipeline **refused to fabricate** T4 metrics and stored real RTX-4060 numbers; paper correctly labels hardware. |
| 16 | Params 53.62M (spatial 49,848,416 / ResSE 2,987,685) and 3.32M temporal; strict load OK | ✅ | Counted from checkpoints earlier; README params match. |
| 17 | Checkpoint SHAs `e2184689…a5c3ba`, `b81eed74…951abf`; 204.7 MB / 12.7 MB | ✅ | Recomputed SHA-256 of local `models/*.pth` — exact match to README Model Zoo. |
| 18 | Hyperparameter ledger (S1 Table): AdamW, wd 1e-4 (release run) / 1e-2 (OptimizationConfig), LRs 1e-5/1e-4, temporal 5e-4, aux 0.30, EMA 0.999, cosine+1-epoch warmup, 5 epochs B=16 accum 2 | ✅ | Code: `run_release1_kaggle.py:425` (wd=1e-4), `optimization.py:13` (default 1e-2, used by `train_dual_stream_ddp.py:128`), `:1227` (temporal 5e-4), `:470` (aux 0.30), EMA decay 0.999 (`:429`), scheduler `optimization.py:53–71`. main.tex:272 matches code exactly. |
| 19 | Test suite "160 collected" (README badge) | ✅ | 156 passed + 4 skipped = 160 collected. Same command config runs in CI (`ci.yml`: ruff + pytest, py3.11, CPU torch). |
| 20 | Serving calibration sidecar | ✅ | `models/dual_stream_detector.json` = same constants as checkpoint calibration (τ 0.26, T_eff 3.5931, platt 0.2783/0.4089, taus 0.4/0.6). |
| 21 | `facenet-pytorch` in requirements | ✅ | Genuinely used: `src/dataset/preprocess.py:16` optional MTCNN import (fallback detector). |
| 22 | LICENSE MIT; HF Space link; CI/CD workflows sane | ✅ | `LICENSE` MIT (c) 2026; `cd.yml` uploads to HF Space with sensible excludes. |
| 23 | AI-docs (compendium, study_guide) | ✅ spot-check | Numbers spot-checked (ECE 0.2712→0.1407/48.1%, 0.8656/0.8994/18.54/19.43, 99,101 crops) — consistent with ground truth; compendium actively warns against quoting stale pilot numbers. Both are **gitignored local artifacts**, not repo content. |
| 24 | Governance self-audits (`MASTER_FORENSIC_ENGINEERING…`, fixtures honesty notes) | ✅ | Compendium L682 itself demands not quoting the historical pilot's 0.7000/0.5961 as current — yet MODEL_CARD violates this (§3.3). |

---

## 3. Confirmed discrepancies (ranked)

### 3.1 FALSE caption — supplementary robustness table (highest severity)
`supplementary.tex:286` (tab:robustness_benchmarks): *"The detector was trained strictly on clean crops without data augmentation for compression or blur."*
**False.** All training sites use `get_transforms(..., hardened=True)` (`run_release1_kaggle.py:386/572/1206/1406`), which adds `A.ImageCompression(quality_range=(65,95), p=0.30)` and Gaussian blur (`src/dataset/loader.py:217–241`). Robustness *numbers* remain genuine, but the caption's pretext ("genuine physical signal resilience") misrepresents training augmentation. Only in supplement (main.tex never repeats it).
**Rollout risk:** the same false string is baked into the *generator* `scripts/generate_manuscript_tables.py` (Table 4 caption, ~L285), so `--update` re-propagates the falsehood after any manual fix.

### 3.2 Provenance misattribution — statistical significance paragraph
`main.tex:423` claims the ablation DeLong tests (Z=25.65/27.03) are in `tier1_rigorous_evaluation.json`. That file contains only the **spatial-baseline** DeLong/McNemar blocks. The ablation Z values exist only in `results/release1_run/ablation_cache.npz` (logits) + checkpoint Platt constants — from which they **reproduce exactly**. Classification: **real values, wrong citation** (not fabricated). Fix: cite both files.

### 3.3 Wrong number — MODEL_CARD Celeb-DF zero-shot
`MODEL_CARD.md:156,162`: "zero-shot transfer on Celeb-DF v2 … 0.5961 ROC AUC". Stored LOTO Fold-5 zero-shot is **0.5763** (T*=10.0 bound). 0.5961 is the *historical pilot-protocol* number (per compendium self-note). Paper/README use 0.5763 — MODEL_CARD is wrong. MODEL_CARD also labels DFD 0.8651 "zero-shot" while README disclaims that framing.

### 3.4 Degenerate validation branch — "Celeb: 0.5000" all epochs
`evaluate_macro_val` logged celeb val AUC **exactly 0.5000 at every epoch**; macro published relationship = 0.5·(FF++ AUC + 0.5) verified exact at all 5 epochs (e.g., 0.7304 = 0.5·(0.9608+0.5)). Mechanism: celeb branch hit its `auc_celeb = 0.5` default (empty wand / single-class fell-through) — a code path defect that silently prints a default as if measured. The calibration stage *minutes later* classified the same `splits['val']` correctly ({celeb_1: 29328, celeb_0: 2328, ffpp_1: 1680, ffpp_0: 1680}), so classification itself works — root cause inside `evaluate_macro_val` is undetermined (dataset deleted; cannot reproduce).
**Impact:** bounded. Selection degenerated to FF++-AUC-driven (monotone), and the paper already honestly describes selection as FF++-partition AUC (`main.tex:272` reaching 0.9608 @ epoch 5). But the script header docstring (L7: "Macro-AUC selection across composite FF++ and Celeb-DF Set A") and `notebooks/master_pipeline.ipynb` item 3 still overclaim composite selection.

### 3.5 Untraceable constant — temporal τ*=0.1100
README Model Zoo, `predict.py:267` (`calib.get("temporal_optimal_threshold", 0.1100)` fallback — key absent from `models/dual_stream_detector.json`), and `audit_fixture_consistency.py:48` all use **0.1100**, which no stored artifact derives: release temporal JSON has no threshold; legacy fixture says 0.3895 (N=1,120). Dead-reckoned constant; flag for provenance hygiene.

### 3.6 Stale `paper.zip`
Built 2026-10-11; manuscripts changed 2026-10-31 (HEAD). Hash-verified: `main.tex` (67,962 B vs 72,136 B), supplementary, preview HTML and **7 of 12 figures** differ; 5 figures match. Contains `tools/tectonic.exe` + `paper_preview.html` — **not** produced by `package_arxiv.py` (which packs main+supp+12 figures only). Submitting this zip would misrepresent the paper; rebuild before submission.
**Visual audit addendum:** the repo-local compiled PDFs are ALSO stale — `main.pdf` (compiled 2026-10-30 23:59) and `supplementary.pdf` (23:51) predate the final `.tex` edits (00:31:45), and `main.pdf` visibly contradicts current `main.tex` on the LOTO Fold-5 paragraph (§9, V2). Recompile both before submission.

### 3.7 `uv.lock` placeholder + dependency-doc mismatch
`uv.lock` is a 52-byte stub (`requires-python = ">=3.13"`) — not a real lockfile — while `requirements.txt` header calls it "the resolution lockfile"; uv isn't used anywhere; CI/Docker use Python 3.11; README badge says 3.10|3.11. Repo `.venv` is empty → local quickstart runs to untested env (no code claim broken).

### 3.8 Minor: notebook claims "± 98.0% precision" Bayesian thresholds
Actual certified precision = 0.5000 with `is_precision_certified: false` (honest fallback stored + disclosed in main.tex:247). Notebook wording is stale aspiration.

### 3.9 Minor: pipeline-log claim unfulfilled locally
Log says "Mirrored Master Release 1 Results -> results/release1_results.json" — that mirror file no longer exists (deleted at some point). Provenance shrunk: original `splits.json` inside `deepfake_crops_512/` is gone (deleted with dataset; CSV exports re-created 2026-09-25 09:51 remain), `dual_stream_best.pth` deleted post-run (calibrated checkpoint survives).

### 3.10 Minor: serving/eval inference-path divergence (unclaimed)
Serving (`src/services/video_engine.py:360–381/524–530`) applies h-flip TTA and averages logits; published eval (fixtures, `compute_table5_ablations`) uses no TTA (`use_tta` off). Paper doesn't claim no-TTA — low-severity, worth documenting.

### 3.11 Minor: phase-of-capture note in eval flow
Robustness caption "16 evaluation profiles including clean baseline" vs clean baseline = **full-test AUC**, not the 1,000-crop subset — wording imprecision only.

### 3.12 Not-error-but-unclaimed: domain-probe orphan number
`domain_classifier_probe.balanced_accuracy = 0.6066` sits in tier1 and is claimed **nowhere** in paper/docs (verified no matches). Harmless orphan.

---

## 4. Residual unknowns (could not be fully closed from surviving artifacts)

1. **Root cause of the degenerate celeb branch** in `evaluate_macro_val` (Celeb 0.5000 five times) — dataset deleted; empirical re-run impossible. Bounded impact (see §3.4).
2. **Origin of temporal τ*=0.1100** — no stored artifact computes it; likely from an unsaved earlier val run. Recomputable only with dataset images available; recommend regenerating and persisting as JSON.

---

## 5. Reproducibility & provenance assessment

- `git ls-files` = **132 files** (src 32, scripts 30, tests 22, figures 15, manuscript 14, **splits 4 ✓tracked**, infra 2, + root files/README/MODEL_CARD/notebook).
- **Tracked & open:** split manifests (`splits/train|val|test_split.csv` + `splits_manifest_summary.json`) are committed — a fresh clone can verify split counts/leakage claims from CSVs alone. The split-construction script `build_release1_splits.py` (seed 42, Celeb-DF preserves pre-verified identity sets, DFD 100% held out to test) and open-science exporter `export_split_manifests.py` (generator classification by pair ranges <100/<400/<600/<800 matching the paper's Table 2/3 partition captions) are documented pipeline provenance.
- **Gitignored (local-only):** `results/` — all run JSONs (`release1_results.json`, `tier1_rigorous_evaluation.json`, `ablation_cache.npz`, `test_predictions.json`, pipeline log) and `models/*.pth` checkpoints. A fresh clone can re-verify from: (a) committed CSVs, (b) the two HF-hosted checkpoints (SHA-256 match verified locally), (c) re-running evals on the Kaggle crops dataset (linked in README). Stage tables/figures depend on `release1_results.json` + `ablation_cache.npz`, which are **not** in the repo — recommend uploading a provenance snapshot.
- Original `deepfake_crops_512/splits.json` used at run time no longer exists on this machine (dataset folder deleted); the CSV exports (created 2026-09-25 09:51, matching manifest counts exactly) are the sole split source going forward. `dual_stream_best.pth` (pre-calibration best epoch) was deleted post-run; the calibrated artifact (which embeds the same weights + calibration constants) survives.
- Test suite 156 pass / 4 skip (ONNX-optional deps ×3, weights/accelerate conditionals).
- `verify_latex_full.py` + `generate_manuscript_tables.py --check` are solid *table-scoped* verification; **prose numbers** (Z=25.65/27.03, temporal τ) are outside the machine-verified surface — recommend adding prose-number assertions for these two.

## 6. Recommendations (highest-leverage first)

1. Delete the false "trained strictly on clean crops…" clause in BOTH `supplementary.tex:286` and the generator `generate_manuscript_tables.py` Table-4 caption; regenerate supplement.
2. Recite the ablation-stat file source correctly in `main.tex:423` (add `ablation_cache.npz`, keep tier1 for the spatial-baseline test).
3. Fix MODEL_CARD Celeb-DF zero-shot 0.5961 → 0.5763 (and DFD zero-shot gloss).
4. Rebuild `paper.zip` via the canonical packaging script; delete the ad-hoc bundle (it currently ships stale LaTeX + a stray 51 MB `tectonic.exe`).
5. Persist the provenance of temporal τ* (recompute and store `temporal_optimal_threshold` in `models/dual_stream_detector.json`), and replace `uv.lock` either with a real lockfile or remove the claim in `requirements.txt`.
6. Upload the Results artifacts (or a curated provenance snapshot) to HF/repo so a fresh clone can verify claims; commit the CSV split manifests.
7. Remove stale "Macro-AUC composite FF++ + Celeb-DF Set A" wording from script docstring & notebook (actual catalog: FF++-partition AUC, consistent with paper).

---

## 7. Bottom line

The release's **data story is credible**: a real double-blind-sourced experiment executed with leak-proof splits, genuine per-fold retraining, honest calibration failure reporting, no fabricated test numbers, and successful independent reproduction of the ablation table, ablation statistics, and headline metrics from raw caches. The failure modes found are concentrated in **documentation hygiene and provenance packaging** — one false caption, one misattributed file reference, one wrong MODEL_CARD number, a stale zip and stale notebook claims — all individually fixable in under an hour, and none undermining the paper's empirical core.

---

## 8. Completion addendum (final coverage pass)

**Figure-reference integrity — 12/12 resolve.** Every `\includegraphics` target in both `.tex` files exists on disk (`.audit_tmp/check_fig_refs.py`): `main.tex` 4/4 (`teaser_concept`, `system_architecture`, `roc_pr_curves`, `temporal_attention_dynamics`), `supplementary.tex` 8/8 (`lifecycle_flaws`, `radial_spectral_profiles`, `loto_generalization_matrix`, `robustness_curves`, `gating_attenuation_dynamics`, `calibration_reliability`, `bayesian_decision_zones`, `qualitative_attention`). Zero broken figure references in the manuscript.

**Script inventory — 100% covered.** The final 8 previously-unread scripts were skimmed; nothing contradicts earlier findings:

- `train_loto_experiment.py` — genuine LOTO retraining: `filter_loto_split_strict` removes held-out-domain fakes from each fold's training set, per-fold seed workers, `hardened=True` default, differential param groups, EMA. Consistent with the paper's per-fold-retraining claim (supplementary Table).
- `audit_splits_and_confounds.py` — one-off confound audit over `deepfake_crops_512/splits.json` (now-deleted local file): prints source/class counts, video-folder disjointness, and per-source actor-identity disjointness. This is the provenance artifact behind the leakage claims; CSV splits now carry the same provenance (§5).
- `evaluate_test_set.py` — fits T* and τ* on the **validation** split only, then evaluates the held-out test split at 0.5 and τ*, plus a seed-42 balanced 1:1 subset; no test-set peeking in the calibration path. Consistent with main.tex methodology.
- `recalibrate_and_evaluate.py` — sanitized-splits eval with **video-clustered** paired bootstrap (1,000 iterations, seed 42, resampled over video sequence folders) and EER — the provenance of the paper's 95% CIs.
- `extract_face_crops.py` — dataset generation: `DynamicFaceCropper` (512 px, scale 1.50), 12 evenly-spaced frames/video, WebP q90, threaded extraction, `DomainClassifier` labeling, manifest with `video_id` per crop. Reinforces §3.1: crops are clean *stills* — compression/blur "hardening" happens at training time, not here, so the supplement's "trained strictly on clean crops" caption remains false.
- `compute_radial_spectral_profiles.py` — generates `Figure radial_spectral_profiles` (azimuthal-radial 1/f profiles, 80 crops/category, Henning-windowed 2D FFT) from the deleted local `manifest.json` — not re-runnable locally, PDF figure survives.
- `profile_latency_live.py` — Table 6 source: RTX 4060 Laptop live CUDA profile (B=1 latency, B=32 throughput, stage-by-stage SRM/Bayar → FFT → ConvNeXt/ResSE timing), loading `results/release1_run/dual_stream_best.pth`. Consistent with the paper's honest hardware labeling; log L88 refusal to fabricate T4 metrics remains the stronger integrity signal.
- `generate_benchmark_plots.py` — publication-figure generator reading stored evaluation artifacts (9 figures: ROC, ECE reliability, PR, Bayesian decision zones, confusion matrices, per-generator AUC, LOTO, robustness, temporal attention). Outputs match committed `figures/`.

**Coverage is now complete:** all 132 tracked files, all `results/release1_run/*` artifacts where they exist locally, and all `scripts/` and `src/` modules have been read or numerically verified. Remaining unknowns unchanged (§4). No new discrepancies beyond §3.

---

## 9. Visual audit (compiled PDFs, figures, preview)

**Method.** `manuscript/main.pdf`, `supplementary.pdf` and all 12 `manuscript/figures/*.pdf` were rasterized with PyMuPDF (script `.audit_tmp/render_pdfs.py`; main at 150 DPI, figures at 200 DPI → 31 PNGs under `.audit_tmp/render/`, verified by `Get-ChildItem` count = 31). Every page/image was then visually inspected: **12/12 main.pdf pages, 7/7 supplementary.pdf pages, 12/12 standalone figures**. `manuscript/paper_preview.html` was opened in the in-app browser (`file:///.../manuscript/paper_preview.html`, tab title matched the paper) and audited at DOM/text level; a viewport screenshot was blocked ("screenshot needs a visible tab" — no visible desktop window), so the preview's visual layer was judged from its rendered content model, not pixels — noted as a method limitation.

**Content-consistency verdict (positive).** Every caption, legend and in-figure annotation across the 12 figure PDFs and both compiled papers matches the previously verified stored values: ROC/PR legend (0.4867 / 0.8508 / 0.8656 / 0.8370 / 0.8962 / 0.8994; op point τ*=0.26), PR skew baseline 71.8% (=19,372/26,981), robustness points per regime (JPEG 0.8408/0.7973/0.7626/0.7410; blur 0.8021/0.7232/0.6722; noise 0.7274; downscale 0.8354), calibration (N=26,981; Test ECE 0.2712→0.1407 −48.1%; Val 0.1965→0.0994 −49.4%; T_eff=3.5931), triage zones (46.3% / 5.1% / 48.6%; N=7,609 / 19,372; τ*=0.26), LOTO matrix + macro 0.9514 and the **0.5763** Celeb-DF holdout bar, gate factors (0.85/0.52/0.18/0.04), and Fig. S8 posteriors (0.1936 / 0.8671 / 0.8254 / 0.7514). All S1–S8 figure cross-references visually resolve. No fabricated-looking pixels anywhere.

**New visual findings (ranked):**

1. **V1 (moderate) — Fig. 4 red-border selection contradicts the figure's own data.** Caption (main.pdf p.8): "frames with red borders correspond to peak temporal self-attention weights." In panel (a) the red borders are t = 5, 6, 8; in panel (c) the attention curve peaks at **t = 7 (0.18)** and t = 8 (0.17), with t = 5/6 at ~0.06–0.07. The actual peak frame (t=7) is *not* highlighted. Likely a stale highlight selection from an earlier run; fix: re-derive borders from the stored attention weights (evidence: `.audit_tmp/render/figures/temporal_attention_dynamics.png`).
2. **V2 (moderate) — compiled PDFs are stale relative to their sources (extends §3.6).** File timestamps: `main.pdf` 2026-10-30 23:59:31, `supplementary.pdf` 23:51:41, vs. `main.tex`/`supplementary.tex` last edited 2026-10-31 00:31:45 (commit `1625638` at 08:18). Concrete visible contradiction: **main.pdf p.9 (§4.6)** states LOTO Fold-5 as "0.6927 Fake F1 (T\* = 2.5023)", while current **main.tex:402** (and `paper_preview.html`) state "**0.1265 Fake F1, 0.8387 Precision, T\* = 10.0000** at the upper temperature bound… or 0.8611 across the raw `loto_checkpoint.json` 5-fold run" — the latter matches the stored LOTO truth (§2 row 12: Fold-5 T\* at bound). The shipped `main.pdf` therefore misreports the Fold-5 numbers vs. stored data. For `supplementary.pdf` the staleness follows from the same mtime ordering; its visible Table S5 falsehood matches the (still-false) current source, so V3 needs no stale-source explanation. Fix: recompile both PDFs from current sources.
3. **V3 (corroborates §3.1) — false caption visually confirmed in the compiled artifact.** supplementary.pdf **p.5**, Table S5 caption reads verbatim: "The detector was trained strictly on clean crops without data augmentation for compression or blur, evaluating genuine physical signal resilience…". The same string survives in current source `supplementary.tex:286` (grep-verified at HEAD). So the falsehood ships in both source and compiled form.
4. **V4 (minor) — `paper_preview.html` leaks raw LaTeX into a table header.** In the ablation table's first group row the generated HTML contains the literal text `\multicolumn{6}{l}{\em{}}` and a stray `Micro-Ablations}}` cell (line ~459 of the HTML) — the other group headers ("Cross-Stream Fusion Dynamics", "Spatiotemporal Video Sequence Modeling") converted correctly. Bug in the preview's table expander for multicolumn rows; cosmetic but visible to any reader of the preview. Otherwise the preview is current: it was regenerated at 00:41 (after the .tex edits), already carries the corrected Fold-5 values (V2 parity with tex, not PDF), and contains no "clean crops" occurrence (that falsehood lives only in the supplement).
5. **V5 (minor) — Fig. 4(b) panel title overstates.** Sub-panel title "Decision Boundary Traversal" while the plotted posterior stays in 0.75–0.92 for all 10 frames — it never crosses τ\*=0.26 nor leaves Zone 3. Cosmetic narrative mismatch (the caption's T=10/Δk=1 disclosure is honest).
6. **V6 (minor) — heading line-break blemishes in main.pdf.** §2.3 breaks mid-word with no hyphen ("…Manipulation Analy/sis"), §3.6 ("…Sequence Model/ing"); §4 and §4.1 head as "Bench-marks"/"Cor-ora" (hyphenated at break). Typographic polish only.
7. **V7 (minor) — Fig. 2 print legibility.** The architecture figure's in-box annotations are ~4–5 pt at column scale (e.g., taper/tapering constants, stage dimension footnotes); acceptable as a poster-style schematic, below journal print comfort. Related: supplementary p.4 has a column-break orphan ("is:") dangling atop a column before Eq. (S10).

**Reproduction.** All 31 renders remain under `.audit_tmp/render/{main,supplementary,figures}/`; the browser tab for the preview is session-local. A future re-render regenerates everything via `.audit_tmp/render_pdfs.py` (PyMuPDF; `python -m pip install pymupdf` needed once).

---

## 10. Remediation applied (2026-10-31)

All Tier A/B/C fixes from the fix plan were applied to the working tree; no Kaggle rerun was required.

| Item | Change | Evidence |
|---|---|---|
| A1 | Replaced the false "trained strictly on clean crops" caption in `supplementary.tex` and `generate_manuscript_tables.py`. Caption now states JPEG q65–95 p=0.30 / blur σ0.3–1.2 p=0.20 and the full-cohort clean baseline | stale-string grep = 0 hits in `manuscript/` and `scripts/`; table check 6/6 |
| A2 | Recompiled `main.pdf` (13 pp) and `supplementary.pdf` (8 pp) with Tectonic 0.17 from current sources | mtimes 10:10:54 / 10:08:06; PDF text has 0.1265/0.5763, no 0.6927 |
| A3 | Rebuilt `paper.zip` / `paper.tar.gz` via `package_arxiv.py`; 14 entries, no `tectonic.exe`, no preview | zip listing, 13.16 MB |
| A4 | Added `ablation_cache.npz` provenance to `main.tex:423` | verifier prose checks |
| A5 | MODEL_CARD 0.5961 → 0.5763; baseline wording corrected; DFD gloss fixed | grep |
| A6 | Fixed escaped-ampersand split in `build_html_preview.py`; regenerated preview | `colspan="6"` row renders "Stream Modality & Micro-Ablations"; no `\multicolumn` |
| B1 | Docstrings corrected; `evaluate_macro_val` now logs degenerate FF++/Celeb fallbacks and exact-0.5 Celeb AUC | `py_compile`; code |
| B2 | Notebook composite-selection and 98.0% precision claims corrected | grep |
| B3 | Removed `uv.lock` lockfile claim from `requirements.txt` / README | grep |
| B4 | Added `temporal_optimal_threshold` + provenance to sidecar; unignored `models/dual_stream_detector.json` in `.gitignore` so it ships; `predict.py` warns on fallback; audit script reads sidecar | JSON valid; `py_compile`; sidecar staged |
| B5 | Restored `results/release1_results.json` mirror (SHA-256 `C1CFE6D1…`) | hash |
| B6 | Added serving-vs-eval TTA limitation to MODEL_CARD | file |
| B7 | Fig. 4 caption rewritten to match the figure; panel (b) title changed to "Frame-Level Posterior Dynamics" in source and PDF. Regeneration from PDF-extracted frames was tested and rejected: it produced 0.98+ frame posteriors and video 1.0 vs published 0.85–0.92 / 0.9210 | `.audit_tmp/regen_fig9_probe.py` output; figure text |
| B8 | Forced word-boundary breaks via `\mbox{}` in four headings | compiled PDF text: no "Manipulation Analy", "Bench-", "Cor-ora" |
| C1 | Kept the Eq. (S10) lead-in with the equation via `samepage`; rendered p.4 shows the full sentence + equation together | `.audit_tmp/supp_p4_v3.png` |
| C3 | Added prose-number assertions (25.65 / 27.03 / 28.47) and sidecar τ check to `verify_latex_full.py` | verifier 6/6 |

**Verification commands/results:**

- `python -m pytest -q` → **156 passed, 4 skipped** (101.97 s).
- `python scripts/verify_latex_full.py --results results/release1_run/release1_results.json` → 6/6 tables exact, 35/35 citations, 7/7 refs, 6/6 sections, "ALL MANUSCRIPT INVARIANTS PASSED PERFECTLY!"
- `python scripts/generate_manuscript_tables.py --check` → 6/6 tables exact.
- `python scripts/audit_fixture_consistency.py` → no internal inconsistencies; 4 expected legacy-fixture claim drifts (temporal fixture N=1,120 vs published N=2,248).
- `paper.zip` → 14 entries (2 `.tex` + 12 figures), 13.16 MB.

**Still blocked / not fixed:** Celeb-val 0.5000 root cause; τ*=0.1100 recomputation; external-baseline / diffusion / third-party replication; Fig. 2 micro-text (C2, optional); pixel screenshot of the preview. Raw `deepfake_crops_512/` remains deleted, so raw-crop figures were not regenerated.
