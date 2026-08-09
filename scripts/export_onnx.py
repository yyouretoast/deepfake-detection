"""Legacy redirect script for backward compatibility."""

from scripts.benchmark_latency import benchmark_inference

if __name__ == "__main__":
    benchmark_inference()

