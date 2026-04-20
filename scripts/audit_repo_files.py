"""Audit repo files — verify all expected files exist on disk.

Run this after cloning to confirm .gitignore didn't exclude nested dirs.
Exit 0 if all present, 1 if any missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    # Package root
    "marketify/__init__.py",
    "marketify/config.py",
    "marketify/onboarding.py",
    # data
    "marketify/data/__init__.py",
    "marketify/data/market_data.py",
    "marketify/data/news_data.py",
    # broker
    "marketify/broker/__init__.py",
    "marketify/broker/base.py",
    "marketify/broker/paper.py",
    "marketify/broker/ibkr.py",
    "marketify/broker/alpaca.py",
    # features
    "marketify/features/__init__.py",
    "marketify/features/technical.py",
    "marketify/features/sentiment.py",
    # models
    "marketify/models/__init__.py",
    "marketify/models/xgb_model.py",
    "marketify/models/ensemble.py",
    "marketify/models/drift.py",
    "marketify/models/train_pipeline.py",
    # risk
    "marketify/risk/__init__.py",
    "marketify/risk/risk_engine.py",
    # state
    "marketify/state/__init__.py",
    "marketify/state/store.py",
    # backtest
    "marketify/backtest/__init__.py",
    "marketify/backtest/benchmark.py",
    "marketify/backtest/simulator.py",
    # scripts
    "scripts/colab_check.py",
    "scripts/colab_run_all.py",
    "scripts/colab_bootstrap.py",
    "scripts/validate_marketify.py",
    "scripts/audit_repo_files.py",
    # project config
    "pyproject.toml",
    "requirements.txt",
    "app.py",
]


def main() -> int:
    missing: list[str] = []
    for rel in REQUIRED:
        full = ROOT / rel
        status = "OK" if full.exists() else "MISSING"
        if status == "MISSING":
            missing.append(rel)
        print(f"  [{status:>7s}] {rel}")

    print(f"\nChecked {len(REQUIRED)} files.  Missing: {len(missing)}")
    if missing:
        print("\nMISSING FILES:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("All files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
