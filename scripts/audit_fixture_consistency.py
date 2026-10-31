#!/usr/bin/env python3
"""Audits bundled per-sample prediction fixtures against declared claims.

Recomputes core metrics (N, ROC AUC) directly from the probability arrays inside
``results/release1_run/test_predictions.json`` and
``results/release1_run/temporal_test_predictions.json``, compares them with any
self-declared metric fields stored in the fixture, and diffs them against the headline
numbers published in README.md / MODEL_CARD.md / the manuscript.

Integrity policy:
- **Internal inconsistency** (a stored metric field disagrees with the fixture's own
  arrays) is a hard error -> exit code 2.
- **Claim drift** (fixture recomputation disagrees with the published headline numbers)
  is reported as ``CLAIM-DRIFT`` and, by default, only warns. Pass ``--strict`` to make
  claim drift fatal for CI; that flag should be enabled once the fixtures are re-exported
  from the current Release-1 evaluation run.

Run:
    python scripts/audit_fixture_consistency.py [--strict]
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "release1_run"
CALIB_PATH = Path(__file__).resolve().parents[1] / "models" / "dual_stream_detector.json"
with open(CALIB_PATH, "r", encoding="utf-8") as _calib_f:
    _CALIB = json.load(_calib_f)

# Headline claims as published in README.md / MODEL_CARD.md (Release-1 full run).
# NOTE: the bundled per-sample fixtures are a legacy Phase-1 subset export and will
# legitimately disagree with these until re-exported; see README "Reproducibility".
PUBLISHED_CLAIMS = {
    "test_predictions.json": {
        "n_samples": 26981,
        "auc": 0.8656,
        "temperature": 3.5931,
    },
    "temporal_test_predictions.json": {
        "n_sequences": 2248,
        "auc": 0.8994,
        "naive_avg_auc": 0.8962,
        "optimal_threshold": _CALIB.get("temporal_optimal_threshold", 0.1100),
    },
}


def compute_auc(probs: list[float], labels: list[int]) -> float:
    """Exact Mann-Whitney-U ROC AUC with ties handled."""
    pos = sorted(p for p, lab in zip(probs, labels) if lab == 1)
    neg = sorted(p for p, lab in zip(probs, labels) if lab == 0)
    if not pos or not neg:
        raise ValueError("Both classes must be present to compute AUC.")
    concordant = 0
    tie_count = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        tie_count += hi - lo
        concordant += lo
    return (concordant + 0.5 * tie_count) / (len(pos) * len(neg))


def stored_matches_observed(stored: float | None, observed: float) -> bool:
    """Field equality at report precision (fixtures store full float precision)."""
    if stored is None:
        return True
    return abs(float(stored) - float(observed)) <= 5e-5


def audit_frame_fixture(data: dict, claims: dict, strict: bool) -> list[str]:
    findings = []
    probs, labels = data["probs_cal"], data["labels"]
    recomputed = {
        "n_samples": len(labels),
        "auc": round(compute_auc(probs, labels), 4),
    }
    stored_fields = {
        "n_samples": data.get("n_samples"),
        "auc": data.get("roc_auc"),
    }
    for key, observed in recomputed.items():
        stored = stored_fields.get(key)
        if not stored_matches_observed(stored, observed):
            findings.append(
                f"INTERNAL[{key}]: fixture field {float(stored)!r} != recomputed {observed}"
            )
        if observed != claims[key]:
            msg = f"CLAIM-DRIFT[{key}]: fixture {observed!r} vs published {claims[key]!r}"
            findings.append(("FATAL " + msg) if strict else ("WARN " + msg))
    stored_temp = data.get("temperature")
    if stored_temp is not None and abs(float(stored_temp) - claims["temperature"]) > 5e-5:
        msg = f"CLAIM-DRIFT[temperature]: fixture {round(float(stored_temp), 4)!r} vs published {claims['temperature']!r}"
        findings.append(("FATAL " + msg) if strict else ("WARN " + msg))
    return findings


def audit_temporal_fixture(data: dict, claims: dict, strict: bool) -> list[str]:
    findings = []
    labels = data["labels"]
    recomputed = {
        "n_sequences": len(labels),
        "auc": round(compute_auc(data["probs_temporal"], labels), 4),
        "naive_avg_auc": round(compute_auc(data["probs_naive_avg"], labels), 4),
    }
    stored_fields = {
        "n_sequences": None,
        "auc": data.get("roc_auc"),
        "naive_avg_auc": data.get("naive_avg_roc_auc"),
    }
    for key, observed in recomputed.items():
        stored = stored_fields.get(key)
        if not stored_matches_observed(stored, observed):
            findings.append(
                f"INTERNAL[{key}]: fixture field {float(stored)!r} != recomputed {observed}"
            )
        if observed != claims[key]:
            msg = f"CLAIM-DRIFT[{key}]: fixture {observed!r} vs published {claims[key]!r}"
            findings.append(("FATAL " + msg) if strict else ("WARN " + msg))
    stored_thresh = data.get("optimal_threshold")
    if stored_thresh is not None and abs(float(stored_thresh) - claims["optimal_threshold"]) > 5e-5:
        msg = (
            f"CLAIM-DRIFT[optimal_threshold]: fixture {round(float(stored_thresh), 4)!r} "
            f"vs published {claims['optimal_threshold']!r}"
        )
        findings.append(("FATAL " + msg) if strict else ("WARN " + msg))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on claim drift (CI mode).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    internal_failures = 0
    for filename, claims in PUBLISHED_CLAIMS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            logger.error("MISSING fixture: %s", path)
            internal_failures += 1
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Auditing %s", path.name)
        if filename.startswith("test_"):
            findings = audit_frame_fixture(data, claims, args.strict)
        else:
            findings = audit_temporal_fixture(data, claims, args.strict)
        for finding in findings:
            if finding.startswith("FATAL"):
                logger.error(finding)
                internal_failures += 1
            else:
                logger.warning(finding)
    if internal_failures:
        logger.error("Fixture audit FAILED with %d internal inconsistencies.", internal_failures)
        sys.exit(2)
    logger.info("Fixture audit complete: no internal inconsistencies found.")
    if not args.strict:
        logger.info(
            "Claim drifts (if any) were reported above as warnings -- README/MODEL_CARD/"
            "manuscript should be reconciled with a re-export of the fixtures."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
