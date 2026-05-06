"""
tests/conftest.py
=================
Pytest configuration for the Customer Churn Prediction System test suite.
Adds the project root and src/ directory to sys.path so imports work
correctly whether tests are run from the project root or the tests/ folder.
"""
import os
import sys

# Project root (one level up from tests/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")

for path in [ROOT, SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)
