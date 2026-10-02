#!/usr/bin/env python3
"""
Package Blender Add-on into distributable ZIP archive with SHA256 verification.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = REPO_ROOT / "blender_addon"
RELEASE_DIR = REPO_ROOT / "release"
ADDON_ZIP_NAME = "retopo_flow_blender_addon_v1.0.0.zip"
ADDON_ZIP_PATH = RELEASE_DIR / ADDON_ZIP_NAME


def validate_addon():
    init_py = ADDON_DIR / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"Missing {init_py}")
    
    with open(init_py, "r", encoding="utf-8") as f:
        content = f.read()
    if "bl_info" not in content:
        raise ValueError("Missing 'bl_info' dictionary in __init__.py")
    print(f"✓ Validated blender_addon/__init__.py (bl_info present, size {len(content)} bytes)")


def package_zip():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if ADDON_ZIP_PATH.exists():
        ADDON_ZIP_PATH.unlink()

    print(f"Packaging {ADDON_DIR} -> {ADDON_ZIP_PATH}...")
    with zipfile.ZipFile(ADDON_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in ADDON_DIR.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".pyc") and "__pycache__" not in file_path.parts:
                arcname = file_path.relative_to(REPO_ROOT)
                zf.write(file_path, arcname=arcname)
                print(f"  + Added: {arcname}")

    file_size_kb = ADDON_ZIP_PATH.stat().st_size / 1024.0

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(ADDON_ZIP_PATH, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    checksums_path = RELEASE_DIR / "SHA256SUMS.txt"
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {ADDON_ZIP_NAME}\n")

    print("=" * 70)
    print(f"SUCCESS: Created {ADDON_ZIP_PATH.name} ({file_size_kb:.1f} KB)")
    print(f"SHA256:  {digest}")
    print(f"Saved:   {checksums_path.relative_to(REPO_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    validate_addon()
    package_zip()
