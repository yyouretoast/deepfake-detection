# Code Quality Audit — Refactoring, Hygiene & AI-Slop Review

**Scope:** full repo at HEAD `e5dd250` (134 tracked files): `src/` (32 files, 3,578 lines), `scripts/` (30 files, 9,045 lines), `tests/` (22 files, 1,682 lines), `app.py` + `predict.py` (974 lines), docs, infra, notebook, local working tree.
**Method:** every file family read or skimmed directly; lint re-run with ruff 0.17.3 (CI version class); full pytest suite executed twice (before/after fixes); checkbox claims re-computed from artifacts (`audit_fixture_consistency.py`, direct fixture recomputation); GitHub Actions API checked for live CI status.
**Companion:** `AUDIT_REPORT.md` (head repo) covers numerical/paper provenance at an earlier commit; this report covers code quality and repo hygiene only.

---

## 1. Verdict

The codebase is in **genuinely good shape — this is not an AI-slop codebase.** Docstrings are dense, exception tuples are precise, comments explain *why* (the SNR border-margin invariant, the Bayar center re-centering, sequential-vs-seek video decoding). No emoji spam, no hype adverbs, no fabricated claims, no verbose dead scaffolding. The prior audit remediations are confirmed in-tree (`MODEL_CARD` uses 0.5763; the false "trained strictly on clean crops" caption is gone from `supplementary.tex`).

However, the audit found and fixed **one real CI-breaking defect** plus several hygiene/refactoring items:

| # | Finding | Severity | Disposition |
|---|---|---|---|
| 1 | **CI is red on main** — `ruff check scripts` fails with 3 × RUF046 (verified against GitHub Actions run #362 on HEAD: `conclusion: "failure"`) | High | ✅ **Fixed in this audit** (2 safe autofixes + 1 justified `noqa`) |
| 2 | `patch_pathlib_mkdir()` globally monkey-patches `pathlib.Path.mkdir` on a bogus pretext ("Fixes Windows Python 3.10 issue" — none exists), invoked at import-time in 2 scripts and exported by `src.utils` | Medium | ✅ **Removed** (file deleted, 3 call sites cleaned) |
| 3 | Dead sandbox hack in `run_release1_kaggle.py`: `pathlib._NormalAccessor.mkdir = ...` — `Path._NormalAccessor` was **removed in Python 3.12**, so this is a silently-skipped no-op (AttributeError swallowed) | Low | ✅ **Removed** |
| 4 | Stale README paragraph: "Zero-GPU Reproduction" says the single-frame fixture is a legacy subset (N=13,444, AUC 0.8248, T=4.288). Verified: `test_predictions.json` stores `n_samples=26981, roc_auc=0.865645, temperature=3.593101` — it **is** the full Release-1 export; only the temporal fixture is legacy (N=1,120, AUC 0.8719, τ=0.3895) | Medium | ✅ **Corrected** with recomputed numbers |
| 5 | Release-1 calibration constants duplicated in ≥6 files at **inconsistent precision** (0.2783 vs 0.27831112352514586 vs 0.278311): `predict.py`, `video_engine.py`+, `checkpoint.py` (T/τ only), `compute_table5_ablations.py`, `evaluate_statistical_significance_and_domain_probe.py`, `audit_fixture_consistency.py` | Medium | 📋 Recommendation R1 |
| 6 | Two divergent triage classifiers: `predict.py: classify_zone()` ("Zone 1: Authentic Clearance…") vs `checkpoint.py: classify_three_zone()` ("Confirmed Authentic…") — app/CLI JSON reports use different zone vocabularies | Medium | 📋 Recommendation R2 |
| 7 | Two divergent video inference paths: CLI `predict_video` (strided frames, no TTA, mean-aggregation) vs serving `video_engine.process_video_frames` (uniform seek, h-flip TTA, Platt; see AUDIT_REPORT §3.10) | Medium | 📋 Recommendation R3 |
| 8 | `predict.py` loads state dict with `strict=False` and **no missing-key guard**, while `video_engine.load_prediction_engine` raises on missing critical keys — one silent-partial-load CLI path | Low | 📋 Recommendation R4 |
| 9 | `run_release1_kaggle.py` is a 1,751-line monolith containing its own copies of training/eval/calibration logic | Medium | 📋 Recommendation R5 (phased plan below) |
| 10 | `hybrid_detector.py`: `extract_features()` re-implements the ~35-line spatial+spectral+gate encoding pipeline already in `forward()` | Low | 📋 Recommendation R6 |
| 11 | `video_engine.py`: duplicated frame-read loops (lines ~313–327 vs ~338–351) and calls the private `cropper._center_crop()` from outside the class | Low | 📋 Recommendation R7 |
| 12 | Repo hygiene: `.audit_tmp/` ≈75 MB untracked junk (incl. 51 MB `tectonic.exe`, page render PNGs) not covered by `.gitignore`; `paper.tar.gz` (13 MB) untracked stale duplicate of gitignored `paper.zip`; 52-byte `uv.lock` stub; root `__pycache__/`; git repo = 90.57 MB all loose objects (never gc'd) | Low–Med | 📋 Recommendation R8 |
| 13 | README hard-links `results/release1_run/*.json` fixtures, but `results/` is gitignored — links are dead for fresh clones | Low | 📋 Recommendation R9 |

**AI-slop verdict: effectively clean.** Verified explicitly: no emoji decor in `src/`/scripts (grep), no TODO/FIXME/HACK debris in tracked code (only a benign comment in untracked `.audit_tmp/`), no fabricated-looking claims (spot-recomputed; see §3), documentation that *warns against* over-claiming, and a test suite that predicates against leakage. The only slop-flavored artifact was the fake-compatibility-comment monkey-patch (#2/#3), now removed.

---

## 2. Evidence for each fixed item

### 2.1 CI-breaking lint (finding #1)

Local repro before fix (`python -m ruff check scripts`, ruff 0.17.3):

```text
RUF046 [unsafe fix] scripts\generate_publication_figures.py:701: sel = [vals[int(round(i))] for i in indices]
RUF046 [*]          scripts\run_release1_kaggle.py:351:     int(len(np.unique(targets_arr[ffpp_mask]))) ...
RUF046 [*]          scripts\run_release1_kaggle.py:368:     int(len(np.unique(targets_arr[celeb_mask]))) ...
Found 3 errors.
```

GitHub Actions on the same commit:

```json
{"name":"CI Test Suite","head_sha":"e5dd25041e...","run_number":362,"status":"completed","conclusion":"failure"}
```

Fixes applied:
- `run_release1_kaggle.py:351,368` — removed redundant `int()` around `len(np.unique(...))` (already `int`).
- `generate_publication_figures.py:701` — kept `int(round(i))` plus `# noqa: RUF046` and a justification comment: `np.linspace` yields non-integer indices; the round-then-index semantics must not change (ruff's own fixer labels it unsafe).

Post-fix: `python -m ruff check src scripts tests app.py predict.py` → **All checks passed!**

### 2.2 stdlib monkey-patch removal (findings #2, #3)

`patch_pathlib_mkdir()` (deleted `src/utils/fs.py`) replaced the standard-library `pathlib.Path.mkdir` implementation globally, at import time, under the comment *"Fixes Windows Python 3.10 pathlib.Path.mkdir issue with exist_ok"* — no such defect exists; the patch adds no behavior and silently alters stdlib semantics for every consumer. Removed from:

- `scripts/evaluate_statistical_significance_and_domain_probe.py` (import + call)
- `scripts/generate_publication_figures.py` (import + call)
- `src/utils/__init__.py` (import + `__all__` export)
- `src/utils/fs.py` — deleted

Also removed the dead `pathlib._NormalAccessor.mkdir = lambda ...` sandbox hack in `run_release1_kaggle.py` (and its now-unused `import pathlib`), because `_NormalAccessor` no longer exists on Python ≥ 3.12 — the assignment raises `AttributeError`, which was swallowed by `except Exception: pass`.

Net diff: **5 files changed, 6 insertions(+), 37 deletions(-)**, plus `README.md` clarification.

### 2.3 README fixture-paragraph correction (finding #4)

Recomputation performed during this audit:

```text
test_predictions.json   -> stored n_samples=26981, roc_auc=0.865645, temperature=3.59310...
                           recomputed N=26981, fakes=19372, AUC=0.865645  (matches headline)
temporal_test_predictions.json -> fixture_status=legacy_phase1_subset_export,
                           stored AUC 0.87187, naive 0.86331, τ 0.38955, N(labels)=1120
scripts/audit_fixture_consistency.py -> 4 CLAIM-DRIFT warnings, all temporal, no single-frame drift
```

The old README text claimed the single-frame fixture was a legacy subset — contradicting both its own "Known Limitations" §3 and `AUDIT_REPORT.md` row 10. Rewritten to state: single-frame fixture = full Release-1 export (verified); temporal fixture = legacy subset (drift disclosed, `--strict` flag documented).

---

## 3. Checks that verified CLEAN

- **Lint:** ruff green across the whole CI scope after fixes; pyproject `ignore` list retains a justification comment per rule.
- **Tests:** `python -m pytest tests -q` → **156 passed, 4 skipped (160 collected)**, run twice (before and after edits). README badge "160 collected" verified.
- **Fixture ↔ claims:** `audit_fixture_consistency.py` = no internal inconsistencies; claim drift confined to the (disclosed) legacy temporal fixture.
- **Unicode/"mojibake":** suspected `τ→I,`/`α→I?` corruption in `visualization.py`/`loader.py` was **disproved** — files contain correct Unicode (`U+03C4`, `U+03B1`, `U+2014`, `U+2265`, `U+2264`); the garbling was Windows-console cp1252 rendering only.
- **Docs:** `MODEL_CARD` Celeb-DF zero-shot = 0.5763 ✅ (pilot 0.5961 correctly demoted to historical note); false "clean crops" robustness caption absent from `supplementary.tex` (grep = 0 hits); README carries candid negative-framing notes throughout (Known Limitations §1–7, skew caveats, zero-shot disclaimers).
- **Notebook:** `notebooks/master_pipeline.ipynb` = 6 cells, 78 source lines, 0 cached outputs — a thin Kaggle runner, no stale claims.
- **Quality highlights worth keeping:** `RealFFT2DModule` masked-atan2 autograd; Bayar kernel float32 re-centering with written invariant; thread-local YuNet + 3-tier detector fallback; PIL-over-cv2 decoding rationale; guaranteed Grad-CAM hook cleanup; loss-masked corrupt-sample handling; EMA apply/restore with backup.
- **Infra:** Dockerfile (non-root user, healthcheck, precise comment about headless OpenCV), CI (ruff+pytest, py3.11), CD (HF Space upload with explicit excludes) — all sane.

---

## 4. Recommendations (ranked, with concrete approach)

**R1 — Single source of truth for Release-1 calibration constants.**
Create `src/utils/calibration_release1.py` (or extend `checkpoint.py`) with `PLATT_A = 0.27831112352514586`, `PLATT_B = 0.408914...`, `TAU_STAR = 0.2600`, `T_EFF = 3.5931`, `TAU_REAL/TAU_FAKE = 0.40/0.60` plus a docstring linking the sidecar `models/dual_stream_detector.json`. Replace the 6 hard-coded sites. This removes the inconsistent-precision drift class entirely (files listed in §1 #5).

**R2 — One triage classifier.** Delete `predict.py:classify_zone()` and route through `checkpoint.py:classify_three_zone()` (keep the CLI's zone wording as a thin adapter if needed for report compatibility; tests hit both paths, so the diff is test-compatible).

**R3 — One video inference engine.** Make `predict.py: predict_video()` a wrapper over a plain-args call into `video_engine.process_video_frames()`, exposing aggregation policy as CLI flags. Kills the CLI/serving TTA-aggregation divergence (already cross-documented).

**R4 — Missing-key guard parity for the CLI.** Mirror `video_engine`'s critical-key check (spatial_backbone / freq tower / gate_fc / classifier) inside `predict.py:load_detector()` so a truncated checkpoint fails loudly instead of silently degrading.

**R5 — Decompose `run_release1_kaggle.py` (1,751 lines).** Phased, no-behavior-change plan:
1. Extract stage functions into `scripts/release1/` modules: `training.py`, `calibration.py`, `test_eval.py`, `temporal.py`, `loto.py`, `robustness.py`, `latency.py` (they are already linear stages; move bodies, keep orchestration in runner).
2. Shared tail-common helpers (`subsample_balanced`, `compute_eer`, bootstrap) go into `src/evaluation/` and are imported, not copied.
3. Gate: tests/test_scripts.py keeps importing the same entry symbols; nothing else changes.

**R6 — Factor the encoding pipeline.** Add `HybridDeepfakeDetector._encode(x) -> (f_s, f_f, gate, aux)` used by both `forward()` and `extract_features()`; forward consumes it to assemble the fused cat vector, features consume it to return the soft-mixed embedding (~35 duplicated lines removed; state dict unaffected).

**R7 — video_engine cleanup.** Extract a `_read_target_frames(cap, frame_indices, extractor)` helper for the two read loops; either rename `_center_crop` → `center_crop` (public fallback) or add a public `crop_face(fallback="center")` parameter.

**R8 — Repo hygiene sweep.**
- `.audit_tmp/`: keep the `check_*.py` provenance scripts (worth promoting to `scripts/`) or delete; delete the 75 MB of render PNGs + `tectonic.exe`; add `.audit_tmp/` to `.gitignore` if kept.
- Delete `paper.tar.gz` (untracked; the real bundle must be rebuilt via `scripts/package_arxiv.py` — the gitignored `paper.zip` is also stale per AUDIT_REPORT §3.6).
- Delete or build the 52-byte `uv.lock` stub; delete stale root `__pycache__/`, `.cache/`, `.mpl_cache/` (all gitignored).
- Run `git gc` once (all 9,073 objects are loose; ~90.57 MB → expect several-fold compaction with 10–19 MB figure/weights history blobs).

**R9 — README dead links.** Either commit the two prediction fixtures to the repo (they are ~0.9 MB / 0.04 MB — well within budget) or render their self-declared metadata into an `etc/` snapshot; then README + provenance story works from a fresh clone (carries over AUDIT_REPORT recommendation #6, still open).

---

## 5. Changes made in this audit (complete list)

```text
 M README.md                                            (fixture-parity paragraph rewritten)
 M scripts/evaluate_statistical_significance_and_domain_probe.py  (fs patch removed)
 M scripts/generate_publication_figures.py              (fs patch removed; justified noqa)
 M scripts/run_release1_kaggle.py                       (dead pathlib hack + import removed; RUF046 x2)
 D src/utils/fs.py                                      (stdlib monkey-patch module deleted)
 D src/utils/__init__.py patched export                 (import + __all__ entry removed)
 5 files changed, 6 insertions(+), 37 deletions(-)      (+ README edit; pytest unaffected)
```

Verification gates passed after the changes: `ruff check src scripts tests app.py predict.py` → 0 findings; `pytest tests -q` → 156 passed, 4 skipped.

---

## 6. Decision points for the owner (not blocking)

1. Whether to adopt R1–R7 as one remediation PR or split (recommend PR 1 = R1–R4 small code hygiene; PR 2 = R5 monolith decomposition; PR 3 = R6–R7).
2. Whether the `.audit_tmp/check_*.py` verification scripts should be promoted into `scripts/` (they are genuine provenance tooling) or dropped.
3. Committing the prediction fixtures (R9) versus shipping them via Hugging Face only.
