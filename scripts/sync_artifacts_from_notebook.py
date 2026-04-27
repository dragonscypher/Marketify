"""Extract Colab artifact bundle from notebook output into local repo.

This is a deterministic local-sync bridge for ignored artifacts.
It does not train locally and does not contact live brokers.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "MARKETIFY_ARTIFACT_BUNDLE_B64_BEGIN"
END = "MARKETIFY_ARTIFACT_BUNDLE_B64_END"
SHA = "MARKETIFY_ARTIFACT_BUNDLE_SHA256="


def _extract_text_outputs(notebook: Path) -> list[str]:
    data = json.loads(notebook.read_text(encoding="utf-8"))
    texts: list[str] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if "text" in output:
                text = output["text"]
                texts.append("".join(text) if isinstance(text, list) else str(text))
            data_payload = output.get("data", {})
            text_plain = data_payload.get("text/plain")
            if text_plain:
                texts.append("".join(text_plain) if isinstance(text_plain, list) else str(text_plain))
    return texts


def _find_bundle_payload(texts: list[str]) -> tuple[str, str | None]:
    for text in reversed(texts):
        if BEGIN not in text or END not in text:
            continue
        before, rest = text.split(BEGIN, 1)
        payload, after = rest.split(END, 1)
        expected_sha = None
        for line in (before + after).splitlines():
            line = line.strip()
            if line.startswith(SHA):
                expected_sha = line.split("=", 1)[1].strip()
        b64 = "".join(payload.split())
        if b64:
            return b64, expected_sha
    raise RuntimeError("artifact bundle markers not found in notebook outputs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Colab artifact bundle from notebook output")
    parser.add_argument("--notebook", default="colab_probe.ipynb", help="notebook containing bundle output")
    parser.add_argument("--output-zip", default="reports/colab_artifact_bundle.zip")
    args = parser.parse_args(argv)

    notebook = (ROOT / args.notebook).resolve()
    output_zip = (ROOT / args.output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    b64, expected_sha = _find_bundle_payload(_extract_text_outputs(notebook))
    raw = base64.b64decode(b64)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"bundle sha256 mismatch: expected {expected_sha}, got {actual_sha}")
    output_zip.write_bytes(raw)

    with zipfile.ZipFile(output_zip) as zf:
        unsafe = [name for name in zf.namelist() if Path(name).is_absolute() or ".." in Path(name).parts]
        if unsafe:
            raise RuntimeError(f"unsafe paths in bundle: {unsafe}")
        zf.extractall(ROOT)

    print(f"SYNC_RESULT=PASS")
    print(f"bundle_zip={output_zip}")
    print(f"bundle_sha256={actual_sha}")
    print(f"extracted_root={ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
