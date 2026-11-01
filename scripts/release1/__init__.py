"""Release-1 pipeline stages (decomposed from scripts/run_release1_kaggle.py, audit R5).

Inserts the repo root on ``sys.path`` before submodule imports resolve ``from src...``.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
