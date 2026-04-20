"""Bootstrap Marketify inside Google Colab.

Must run FROM Colab. Requires /content directory.
Clones or pulls repo, installs deps, runs check + full pipeline.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/dragonscypher/Marketify.git"
CONTENT = Path("/content")
PROJECT = CONTENT / "Marketify"


def require_colab_env() -> None:
    """Hard fail if /content missing — means not Colab."""
    if not CONTENT.is_dir():
        print("FATAL: /content not found. This script must run inside Colab.")
        print("Do NOT fake COLAB_RELEASE_TAG or create /content locally.")
        sys.exit(1)

    tag = os.environ.get("COLAB_RELEASE_TAG")
    if tag:
        print(f"Colab runtime confirmed: {tag}")
    else:
        print("WARN: COLAB_RELEASE_TAG not set but /content exists.")
        print("Proceeding — may be Colab with env quirk.")


def clone_or_pull() -> None:
    """Clone repo or pull latest if already cloned."""
    if (PROJECT / ".git").is_dir():
        print(f"Repo exists at {PROJECT}. Pulling latest...")
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(PROJECT), check=True,
        )
    else:
        print(f"Cloning {REPO_URL} → {PROJECT}")
        subprocess.run(
            ["git", "clone", REPO_URL, str(PROJECT)],
            check=True,
        )
    os.chdir(str(PROJECT))
    print(f"cwd: {os.getcwd()}")


def install_deps() -> None:
    req = PROJECT / "requirements.txt"
    if not req.exists():
        print("WARN: requirements.txt not found. Skipping pip install.")
        return
    print("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        check=True,
    )


def run_check() -> None:
    print("\n=== Colab Check ===")
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "colab_check.py")],
        check=True,
    )


def run_pipeline() -> None:
    print("\n=== Running Full Pipeline ===")
    env = os.environ.copy()
    env["MARKETIFY_REQUIRE_COLAB"] = "1"
    result = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "colab_run_all.py")],
        env=env,
    )
    if result.returncode != 0:
        print(f"Pipeline exited with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> int:
    require_colab_env()
    clone_or_pull()
    install_deps()
    run_check()
    run_pipeline()
    print("\n=== Bootstrap complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
