import pytest
import numpy as np
import torch
from benchmark import benchmark_inference

def test_benchmark_inference_execution():
    # Run 2 iterations with batch_size=1
    try:
        benchmark_inference(iterations=2, batch_size=1, img_size=256)
    except Exception as e:
        pytest.fail(f"benchmark_inference raised unexpected exception: {e}")
