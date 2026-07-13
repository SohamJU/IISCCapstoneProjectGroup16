import os
import sys

# Ensure the project root is on sys.path so `src` is importable in tests
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
