#!/usr/bin/env python3
"""Verify pyproject.toml, __init__.py, and CHANGELOG.md agree on the version.

Run manually with: python scripts/check_version_sync.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "src" / "robometric_frame" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"


def extract(pattern: str, path: Path) -> str:
    match = re.search(pattern, path.read_text())
    if not match:
        raise SystemExit(f"error: could not find version in {path.relative_to(ROOT)}")
    return match.group(1)


def main() -> int:
    pyproject_version = extract(r'(?m)^version\s*=\s*"([^"]+)"', PYPROJECT)
    init_version = extract(r'__version__\s*=\s*"([^"]+)"', INIT_PY)
    changelog_version = extract(r"(?m)^## \[([^\]]+)\]", CHANGELOG)

    versions = {
        "pyproject.toml": pyproject_version,
        "src/robometric_frame/__init__.py": init_version,
        "CHANGELOG.md (top entry)": changelog_version,
    }

    if len(set(versions.values())) > 1:
        print("error: version mismatch across files:")
        for name, version in versions.items():
            print(f"  {name}: {version}")
        return 1

    print(f"OK: version {pyproject_version} matches across all files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
