"""Turnkey Release 1 Pipeline Runner for Kaggle Dual/Single Tesla T4 GPU.

Executes the complete Release 1 deepfake detection protocol:
1. Integrity audit of splits (0 identity leakage, 0 DFD in train/val).
2. Dual-stream training on clean 37,104 train split (5 epochs, cosine schedule with 1-epoch warmup,
   differential LRs, FP16 mixed precision with FP32-isolated FFT/atan2).
3. Macro-AUC checkpoint selection on validation (0.5 * FF++ AUC + 0.5 * Celeb-DF AUC).
   The Celeb-DF branch can fall back to 0.5 when its validation stratum is degenerate
   (empty or single-class); this is logged loudly, and in the Release-1 run it made
   selection effectively FF++-AUC-driven (see AUDIT_REPORT.md section 3.4).
4. Validation calibration: Platt temperature scaling T* in [0.1, 10.0], Youden's J threshold tau*,
   and Bayesian dual thresholds (tau_real, tau_fake) with certified precision >= 0.980.
5. Held-out test evaluation (26,981 crops):
   - Overall metrics and trivial baseline row (always-fake).
   - Per-source evaluation against own authentic baseline (FF++, Celeb-DF, DFD) with 1,000-sample
     video-level clustered bootstrap 95% confidence intervals.
   - Fine-grained subdomain breakdown.
6. Spatiotemporal Bi-GRU video sequence modeling with velocity deltas (Delta e_t) and dual-path aggregation.
7. 5-fold Leave-One-Target-Out (LOTO) cross-generator domain generalization benchmark.
8. Robustness stress testing (JPEG, blur, noise, downscaling).
9. Single-T4 CUDA-synchronized latency profiling (B=1 and B=32).
10. Structured provenance JSON export to release1_results.json and LaTeX verification.

Decomposition note (audit R5): every pipeline stage now lives in ``scripts/release1/``
(common, integrity, training, calibration, test_eval, temporal, loto, robustness,
latency). This module stays a thin orchestrator: it re-runs ``main()`` and re-exports
every stage symbol plus the src-layer classes that library consumers (tests,
compute_table5_ablations, run_lomo_canonical) import directly from here, so their
import paths are unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import cv2
import torch

from scripts.release1.calibration import (
    calibrate_validation_split,
)
from scripts.release1.common import (
    compute_sha256,
    get_sample_stratum,
    seed_everything,
    seed_worker,
    setup_logging,
    subsample_balanced,
    subsample_stratified_test,
    subsample_stratified_val,
)
from scripts.release1.integrity import (
    verify_dataset_integrity,
)
from scripts.release1.latency import (
    run_latency_profiling,
)
from scripts.release1.loto import (
    run_loto_experiment,
)
from scripts.release1.robustness import (
    run_robustness_stress_test,
)
from scripts.release1.temporal import (
    extract_clip_embeddings,
    train_and_eval_temporal,
)
from scripts.release1.test_eval import (
    compute_video_clustered_bootstrap_auc,
    evaluate_held_out_test,
)
from scripts.release1.training import (
    evaluate_macro_val,
    train_dual_stream_backbone,
    train_spatial_convnext_baseline,
)
from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier, ManipulationDomain
from src.dataset.loader import (
    SequenceVideoDataset,
    extract_identities,
    get_transforms,
    group_video_sequences,
)
from src.dataset.resolver import DatasetResolver, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import (
    compute_ece,
    compute_eer,
    find_optimal_threshold,
    fit_platt_scaling,
    fit_temperature_log,
)
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import create_scheduler, get_differential_param_groups
from src.utils.checkpoint import clean_state_dict, compute_dual_thresholds_certified

# Keep matplotlib cache inside the repo clone (Kaggle/sandbox HOME dirs can be read-only)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("release1_runner")

# Disable OpenCV multithreading to eliminate fork deadlocks in Linux containers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

__all__ = [
    "BiGRUTemporalDetector",
    "DatasetResolver",
    "DomainClassifier",
    "ExponentialMovingAverage",
    "FaceCropDataset",
    "HybridDeepfakeDetector",
    "ManipulationDomain",
    "ModelEvaluator",
    "SequenceVideoDataset",
    "calibrate_validation_split",
    "clean_state_dict",
    "compute_dual_thresholds_certified",
    "compute_ece",
    "compute_eer",
    "compute_sha256",
    "compute_video_clustered_bootstrap_auc",
    "create_scheduler",
    "evaluate_held_out_test",
    "evaluate_macro_val",
    "extract_clip_embeddings",
    "extract_identities",
    "find_optimal_threshold",
    "fit_platt_scaling",
    "fit_temperature_log",
    "get_differential_param_groups",
    "get_sample_stratum",
    "get_transforms",
    "group_video_sequences",
    "main",
    "resolve_splits_path",
    "run_latency_profiling",
    "run_loto_experiment",
    "run_robustness_stress_test",
    "seed_everything",
    "seed_worker",
    "setup_logging",
    "subsample_balanced",
    "subsample_stratified_test",
    "subsample_stratified_val",
    "train_and_eval_temporal",
    "train_dual_stream_backbone",
    "train_spatial_convnext_baseline",
    "verify_dataset_integrity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Turnkey Release 1 Deepfake Pipeline on Kaggle.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing splits.json and crops.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save weights, logs, and results JSON.")
    parser.add_argument("--epochs", type=int, default=5, help="Backbone training epochs (default: 5).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per process (default: 16).")
    parser.add_argument("--skip_loto", action="store_true", help="Skip 5-fold LOTO experiment.")
    parser.add_argument("--skip_robustness", action="store_true", help="Skip robustness stress-tests.")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test (~200 crops, 1 epoch) across all pipeline stages.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of all stages, bypassing cached checkpoints.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "backbone", "calibrate", "test", "temporal", "loto", "robustness", "latency"])
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    elif os.path.exists("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.abspath("./results_release1")
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Execution Device: %s (Count: %d)", device, torch.cuda.device_count())

    if args.smoke_test:
        logger.info("=== RUNNING IN SMOKE TEST MODE (Rapid Verification) ===")
        backbone_epochs = 1
        batch_size = min(args.batch_size, 8)
    else:
        backbone_epochs = args.epochs
        batch_size = args.batch_size

    data_root = DatasetResolver.find_dataset_root(args.data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    existing_results: dict[str, Any] = {}
    final_json_path = os.path.join(out_dir, "release1_results.json")
    repo_results_path = os.path.join(REPO_ROOT, "results", "release1_run", "release1_results.json")
    for path_cand in [final_json_path, repo_results_path, os.path.join(REPO_ROOT, "results", "release1_results.json")]:
        if os.path.exists(path_cand):
            try:
                with open(path_cand, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    break
            except Exception:
                pass

    do_all = args.stage == "all"
    do_backbone = do_all or (args.stage == "backbone")
    do_calibrate = do_all or (args.stage == "calibrate")
    do_test = do_all or (args.stage == "test")
    do_temporal = do_all or (args.stage == "temporal")
    do_loto = (do_all or (args.stage == "loto")) and not args.skip_loto
    do_robustness = (do_all or (args.stage == "robustness")) and not args.skip_robustness
    do_latency = do_all or (args.stage == "latency")

    # 1. Integrity check
    ledger = verify_dataset_integrity(splits_path)

    # 2. Train backbone
    backbone_weights = os.path.join(out_dir, "dual_stream_best.pth")
    spatial_weights = os.path.join(out_dir, "spatial_convnext_best.pth")
    if do_backbone:
        backbone_weights = train_dual_stream_backbone(
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            epochs=backbone_epochs,
            batch_size=batch_size,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )
        spatial_weights = train_spatial_convnext_baseline(
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            epochs=backbone_epochs,
            batch_size=batch_size,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 3. Calibrate on val
    calibrated_weights = os.path.join(out_dir, "dual_stream_calibrated.pth")
    calib_meta = existing_results.get("calibration", {})
    if do_calibrate:
        calibrated_weights, calib_meta = calibrate_validation_split(
            backbone_path=backbone_weights,
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )
    elif os.path.exists(calibrated_weights):
        ckpt_temp = torch.load(calibrated_weights, map_location=device, weights_only=False)
        calib_meta = ckpt_temp.get("calibration", calib_meta)

    # 4. Evaluate Test Split with Clustered Bootstrap
    test_eval_summary = existing_results.get("test_frame_evaluation", {})
    if do_test:
        test_eval_summary = evaluate_held_out_test(
            calibrated_path=calibrated_weights,
            data_root=data_root,
            splits_path=splits_path,
            calib_meta=calib_meta,
            output_dir=out_dir,
            spatial_checkpoint_path=spatial_weights,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 5. Temporal Video Head
    temporal_summary = existing_results.get("temporal_video_evaluation", {})
    if do_temporal:
        temporal_summary = train_and_eval_temporal(
            backbone_path=calibrated_weights,
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            epochs=1 if args.smoke_test else 5,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 6. LOTO Folds (Optional)
    loto_summary = existing_results.get("loto_cross_generator_benchmark", {})
    if do_loto:
        loto_summary = run_loto_experiment(
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            epochs=1 if args.smoke_test else 3,
            batch_size=batch_size,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 7. Robustness Stress Tests (Optional)
    robustness_summary = existing_results.get("robustness_stress_tests", {})
    if do_robustness:
        robustness_summary = run_robustness_stress_test(
            calibrated_path=calibrated_weights,
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 8. Latency Benchmark
    latency_summary = existing_results.get("latency_profiling", {})
    if do_latency:
        latency_summary = run_latency_profiling(
            calibrated_weights,
            output_dir=out_dir,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )

    # 9. Structured Provenance JSON Assembly
    final_results = {
        "provenance": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "splits_sha256": ledger.get("splits_sha256"),
            "spatial_weights_sha256": compute_sha256(spatial_weights) if os.path.exists(spatial_weights) else None,
            "backbone_weights_sha256": compute_sha256(backbone_weights) if os.path.exists(backbone_weights) else None,
            "calibrated_weights_sha256": compute_sha256(calibrated_weights) if os.path.exists(calibrated_weights) else None,
        },
        "dataset_ledger": ledger,
        "calibration": calib_meta,
        "test_frame_evaluation": test_eval_summary,
        "temporal_video_evaluation": temporal_summary,
        "loto_cross_generator_benchmark": loto_summary,
        "robustness_stress_tests": robustness_summary,
        "latency_profiling": latency_summary,
    }

    if "ablation_study" in existing_results:
        final_results["ablation_study"] = existing_results["ablation_study"]
    elif os.path.exists(os.path.join(out_dir, "ablation_study_results.json")):
        try:
            with open(os.path.join(out_dir, "ablation_study_results.json"), "r", encoding="utf-8") as f:
                final_results["ablation_study"] = json.load(f)
        except Exception:
            pass

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Saved Master Release 1 Results -> %s", final_json_path)

    # Also write to repo results/ directory if available
    repo_results_dir = os.path.join(REPO_ROOT, "results", "release1_run")
    if os.path.exists(repo_results_dir):
        repo_results_file = os.path.join(repo_results_dir, "release1_results.json")
        with open(repo_results_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
    os.makedirs(os.path.dirname(repo_results_path), exist_ok=True)
    with open(repo_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Mirrored Master Release 1 Results -> %s", repo_results_path)

    # 10. Run LaTeX Verifier
    verifier_path = os.path.join(REPO_ROOT, "scripts", "verify_latex_full.py")
    tex_path = os.path.join(REPO_ROOT, "manuscript", "main.tex")
    if os.path.exists(verifier_path) and os.path.exists(tex_path):
        import subprocess
        logger.info("Executing LaTeX Manuscript Invariant & Ledger Verification...")
        subprocess.run([sys.executable, verifier_path, tex_path, "--results", final_json_path], check=False)

    logger.info("=== TURNKEY RELEASE 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
