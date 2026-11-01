"""Canonical Release-1 calibration constants (single source of truth).

These constants were fitted on the Release-1 validation cohort by
``scripts/run_release1_kaggle.py`` (stage: ``calibrate_validation_split``) and are
embedded inside ``dual_stream_calibrated.pth`` plus the serving sidecar
``models/dual_stream_detector.json``. Full-precision values below match the
``calibration`` block of ``results/release1_results.json`` exactly; the serving
sidecar intentionally stores rounded display values.

Every consumer that needs Release-1 calibration numbers (serving, CLI, evaluation
scripts, fixture-audit claims) must import them from here instead of re-typing
literals, so full-precision training artifacts and serving code never drift apart.
"""

# Full-precision fitted constants (== dual_stream_calibrated.pth calibration block).
PLATT_SCALE_A: float = 0.27831112352514586
PLATT_BIAS_B: float = 0.4089141487626174
EFFECTIVE_TEMPERATURE: float = 3.5931010853385756
OPTIMAL_THRESHOLD: float = 0.26
TAU_REAL: float = 0.40
TAU_FAKE: float = 0.60

# Legacy temporal-head operating threshold. Provenance unverified: a legacy unsaved
# validation run, not recomputable from stored artifacts (see AUDIT_REPORT.md §3.5).
TEMPORAL_OPTIMAL_THRESHOLD: float = 0.11

# Rounded display precision (matches the sidecar JSON and published tables).
DISPLAY_PLATT_SCALE_A: float = 0.2783
DISPLAY_PLATT_BIAS_B: float = 0.4089
DISPLAY_EFFECTIVE_TEMPERATURE: float = 3.5931

__all__ = [
    "DISPLAY_EFFECTIVE_TEMPERATURE",
    "DISPLAY_PLATT_BIAS_B",
    "DISPLAY_PLATT_SCALE_A",
    "EFFECTIVE_TEMPERATURE",
    "OPTIMAL_THRESHOLD",
    "PLATT_BIAS_B",
    "PLATT_SCALE_A",
    "TAU_FAKE",
    "TAU_REAL",
    "TEMPORAL_OPTIMAL_THRESHOLD",
]
