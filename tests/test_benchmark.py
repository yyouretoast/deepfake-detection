import pytest
import numpy as np
import torch
from benchmark import benchmark_inference, run_fast_benchmark, run_paper_benchmark

def test_benchmark_inference_execution():
    # Run 2 iterations with batch_size=1
    try:
        results = benchmark_inference(iterations=2, batch_size=1, img_size=256)
        assert "pytorch_ms" in results
    except Exception as e:
        pytest.fail(f"benchmark_inference raised unexpected exception: {e}")

def test_run_fast_benchmark():
    try:
        metrics = run_fast_benchmark(num_videos=4, batch_size=2, img_size=256)
        assert "val_auc" in metrics
        assert "val_acc" in metrics
        assert "macro_f1" in metrics
        assert "eer" in metrics
    except Exception as e:
        pytest.fail(f"run_fast_benchmark raised unexpected exception: {e}")

def test_run_paper_benchmark():
    try:
        results = run_paper_benchmark(batch_size=2, img_size=256, fold_samples=4, celeb_samples=4)
        assert "loto_results" in results
        assert len(results["loto_results"]) == 5
        assert "celeb_df_auc" in results
        assert "loto_avg_auc" in results
    except Exception as e:
        pytest.fail(f"run_paper_benchmark raised unexpected exception: {e}")
