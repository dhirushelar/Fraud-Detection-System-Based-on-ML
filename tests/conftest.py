"""
Fraud Detection System (FDS) - Pytest Global Configuration & Fixtures
Author: Alpha
"""

import os
import sys
import pytest

# Ensure repository root is on sys.path for all test runs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
