"""Verification script for Release 1 hardening logic:
1. 4-way stratified sampler stratum assignment
2. Platt scaling with sample weights
3. HybridDeepfakeDetector spatial-only mode (use_fft_branch=False)
4. Domain classifier LOTO matching
"""

import os
import pathlib
import sys

# Compatibility guard for pathlib under sandbox environment
try:
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777, **kwargs: os.mkdir(path, mode, **kwargs)
except Exception:
    pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO_ROOT, ".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch

from kaggle_pipeline.run_release1_kaggle import get_sample_stratum
from src.evaluation.metrics import fit_platt_scaling
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.dataset.domains import DomainClassifier, ManipulationDomain


def test_stratum_assignment():
    assert get_sample_stratum("real/001/frame.webp", 0) == "ffpp_0"
    assert get_sample_stratum("fake/000_003/frame.webp", 1) == "ffpp_1"
    assert get_sample_stratum("real/id0_0000/frame.webp", 0) == "celeb_0"
    assert get_sample_stratum("fake/id0_id1/frame.webp", 1) == "celeb_1"
    assert get_sample_stratum("real/00000/frame.webp", 0) == "dfd_0"  # 5-digit folder
    assert get_sample_stratum("fake/01_02__meeting/frame.webp", 1) == "dfd_1"
    print("PASS: test_stratum_assignment")


def test_platt_sample_weights():
    # Realistic overlapping logits and targets
    np.random.seed(42)
    reals = np.random.normal(loc=-0.8, scale=1.0, size=200)
    fakes = np.random.normal(loc=0.8, scale=1.0, size=200)
    logits = np.concatenate([reals, fakes])
    targets = np.concatenate([np.zeros(200), np.ones(200)])
    weights = np.ones(400) / 400.0

    a, b, eff_t = fit_platt_scaling(logits, targets, sample_weights=weights)
    # Verify calibrated parameters are strictly inside interior (0.101, 9.99)
    assert 0.101 < a < 9.99, f"a={a} hit boundary! Expected interior convergence."
    assert -9.99 < b < 9.99, f"b={b} hit boundary! Expected interior convergence."
    assert 0.101 < eff_t < 9.99, f"eff_t={eff_t} hit boundary!"

    # Boundary guard test: verify that over-separated synthetic logits hit upper bound (a=10.0)
    np.random.seed(42)
    bound_reals = np.random.normal(loc=-2.0, scale=1.0, size=100)
    bound_fakes = np.random.normal(loc=2.0, scale=1.0, size=100)
    bound_logits = np.concatenate([bound_reals, bound_fakes])
    bound_targets = np.concatenate([np.zeros(100), np.ones(100)])
    a_bnd, b_bnd, eff_t_bnd = fit_platt_scaling(bound_logits, bound_targets)
    is_at_bound = (a_bnd <= 0.101 or a_bnd >= 9.99 or abs(b_bnd) >= 9.99 or eff_t_bnd <= 0.101 or eff_t_bnd >= 9.99)
    assert is_at_bound, f"Expected boundary guard condition to trigger on separable logits (got a={a_bnd}, eff_t={eff_t_bnd})!"

    # Inverted logits hit lower bound (a=0.1000, eff_t=10.0000)
    inv_reals = np.random.normal(loc=2.0, scale=1.0, size=100)
    inv_fakes = np.random.normal(loc=-2.0, scale=1.0, size=100)
    inv_logits = np.concatenate([inv_reals, inv_fakes])
    inv_targets = np.concatenate([np.zeros(100), np.ones(100)])
    a_inv, b_inv, eff_t_inv = fit_platt_scaling(inv_logits, inv_targets)
    is_at_inv_bound = (a_inv <= 0.101 or a_inv >= 9.99 or abs(b_inv) >= 9.99 or eff_t_inv <= 0.101 or eff_t_inv >= 9.99)
    assert is_at_inv_bound, f"Expected boundary guard condition to trigger on inverted logits (got a={a_inv}, eff_t={eff_t_inv})!"

    print(f"PASS: test_platt_sample_weights (healthy interior: a={a:.4f}, b={b:.4f}, eff_t={eff_t:.4f}; upper-bound guard: a={a_bnd:.4f}; lower-bound guard: a={a_inv:.4f})")


def test_spatial_only_backbone():
    model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1), f"Expected shape (2, 1), got {out.shape}"
    assert not torch.isnan(out).any()
    print("PASS: test_spatial_only_backbone (forward pass ok)")


def test_domain_classifier_loto():
    assert DomainClassifier.matches_holdout("fake/050_060/f0.png", "deepfakes") is True
    assert DomainClassifier.matches_holdout("fake/250_260/f0.png", "face2face") is True
    assert DomainClassifier.matches_holdout("fake/450_460/f0.png", "faceswap") is True
    assert DomainClassifier.matches_holdout("fake/650_660/f0.png", "neuraltextures") is True
    assert DomainClassifier.matches_holdout("fake/id0_0000.png", "celeb") is True
    assert DomainClassifier.matches_holdout("real/050/f0.png", "deepfakes") is False
    print("PASS: test_domain_classifier_loto")


if __name__ == "__main__":
    test_stratum_assignment()
    test_platt_sample_weights()
    test_spatial_only_backbone()
    test_domain_classifier_loto()
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
