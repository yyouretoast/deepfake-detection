"""
Legacy redirect script for backward compatibility.
Redirects to scripts/benchmark_latency.py.
"""

from scripts.benchmark_latency import benchmark_inference

if __name__ == '__main__':
    benchmark_inference()
