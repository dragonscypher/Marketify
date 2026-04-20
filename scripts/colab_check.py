"""Colab runtime detection and GPU/RAM info.

Detects Colab by COLAB_RELEASE_TAG env var (set by real Colab runtime).
Fallback: checks for /content directory.
NEVER fake COLAB_RELEASE_TAG — it must come from real Colab kernel.

Exit nonzero if MARKETIFY_REQUIRE_COLAB=1 and not running in Colab.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def is_colab() -> bool:
    """True only if running inside real Google Colab runtime."""
    if os.environ.get("COLAB_RELEASE_TAG") is not None:
        return True
    # Fallback: /content exists only on Colab VMs
    return sys.platform == "linux" and Path("/content").is_dir()


def get_runtime_info() -> dict:
    """Collect runtime info dict."""
    info = {
        "executable": sys.executable,
        "platform": platform.platform(),
        "colab_tag": os.environ.get("COLAB_RELEASE_TAG", "NOT SET"),
        "cwd": os.getcwd(),
        "is_colab": is_colab(),
        "gpu": "unknown",
        "gpu_ram_gb": None,
        "system_ram_gb": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_ram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        else:
            info["gpu"] = "none / CPU only"
    except ImportError:
        info["gpu"] = "torch not installed"

    try:
        import psutil
        info["system_ram_gb"] = round(
            psutil.virtual_memory().total / (1024**3), 1
        )
    except ImportError:
        pass
    return info


def print_info() -> None:
    info = get_runtime_info()
    print(f"executable: {info['executable']}")
    print(f"platform:   {info['platform']}")
    print(f"COLAB_TAG:  {info['colab_tag']}")
    print(f"cwd:        {info['cwd']}")

    if info["is_colab"]:
        print("runtime:    Google Colab ✓")
    else:
        print("runtime:    NOT Colab ✗")

    print(f"GPU:        {info['gpu']}")
    if info["gpu_ram_gb"] is not None:
        print(f"GPU RAM:    {info['gpu_ram_gb']} GB")
    if info["system_ram_gb"] is not None:
        print(f"System RAM: {info['system_ram_gb']} GB")
    else:
        print("System RAM: psutil not installed")


def main() -> int:
    print_info()
    require = os.environ.get("MARKETIFY_REQUIRE_COLAB", "0") == "1"
    if require and not is_colab():
        print("\nFATAL: MARKETIFY_REQUIRE_COLAB=1 but not in Colab.")
        print("Do NOT fake COLAB_RELEASE_TAG. Switch to real Colab kernel.")
        return 1
    if not is_colab():
        print("\nWARN: Not Colab. Heavy work (training/backtests) must run in Colab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
