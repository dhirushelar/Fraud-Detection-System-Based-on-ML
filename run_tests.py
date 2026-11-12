"""
Fraud Detection System (FDS) - Test Suite Runner
Author: Alpha
"""

import os
import sys
import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/"]
    exit_code = pytest.main(args)
    sys.stdout.flush()
    sys.stderr.flush()
    # Use os._exit to bypass Python 3.14 alpha/beta C-extension DLL unloading issue on Windows
    os._exit(exit_code)
