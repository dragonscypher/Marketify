"""Colab runtime detection and GPU/RAM info.

Exit nonzero if MARKETIFY_REQUIRE_COLAB=1 and not running in Colab.
"""
from __future__ import annotations

import os
import platform
import sys


def is_colab() -> bool:
    return os.environ.get("COLAB_RELEASE_TAG") is not None


def print_info() -> None:
    print(f"executable: {sys.executable}")
    print(f"platform:   {platform.platform()}")
    print(f"COLAB_TAG:  {os.environ.get('COLAB_RELEASE_TAG', 'NOT SET')}")
    print(f"cwd:        {os.getcwd()}")

    if is_colab():
        print("runtime:    Google Colab ✓")
    else:
        print("runtime:    NOT Colab")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU:        {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"GPU RAM:    {mem:.1f} GB")
        else:
            print("GPU:        none / CPU only")
    except ImportError:
        print("GPU:        torch not installed")

    # System RAM
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {ram:.1f} GB")
    except ImportError:
        print("System RAM: psutil not installed")


def main() -> int:
    print_info()
    require = os.environ.get("MARKETIFY_REQUIRE_COLAB", "0") == "1"
    if require and not is_colab():
        print("\nFATAL: MARKETIFY_REQUIRE_COLAB=1 but Colab runtime not connected.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
