#!/usr/bin/env python3
"""Prepare RICE1 for UCL-Dehaze (unpaired folder layout).

Expected:
  RICE1/
    train/{hazy,clean}/
    test/{hazy,clean}/

Output (default ./datasets/rice1):
  trainA  trainB  testA  testB

Usage:
  python prepare_rice1.py --src /data/RICE1 --dst ./datasets/rice1
  python prepare_rice1.py --src /data/RICE1 --dst ./datasets/rice1 --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SPLITS = {
    "train": ("trainA", "trainB", "hazy", "clean"),
    "test": ("testA", "testB", "hazy", "clean"),
}


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file())


def prepare(src: Path, dst: Path, copy: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"RICE1 root not found: {src}")

    for _, (a_name, b_name, _, _) in SPLITS.items():
        (dst / a_name).mkdir(parents=True, exist_ok=True)
        (dst / b_name).mkdir(parents=True, exist_ok=True)

    for split, (a_name, b_name, hazy_name, clear_name) in SPLITS.items():
        hazy_dir = src / split / hazy_name
        clear_dir = src / split / clear_name
        if not hazy_dir.is_dir() or not clear_dir.is_dir():
            raise FileNotFoundError(f"Missing {hazy_dir} or {clear_dir}")

        for img in collect_images(hazy_dir):
            link_or_copy(img, dst / a_name / img.name, copy)
        for img in collect_images(clear_dir):
            link_or_copy(img, dst / b_name / img.name, copy)

        print(f"[ok] {split}: hazy={len(collect_images(hazy_dir))} "
              f"clean={len(collect_images(clear_dir))} -> {a_name}/{b_name}")

    print(f"\nOutput: {dst.resolve()}")
    print("Test with:")
    print(
        f"  python test.py --dataroot {dst} --name rrshid_ucl "
        f"--preprocess none --load_size 512 --crop_size 512 --num_test 50"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RICE1 for UCL-Dehaze")
    parser.add_argument("--src", type=Path, required=True, help="Path to RICE1 root")
    parser.add_argument("--dst", type=Path, default=Path("./datasets/rice1"))
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()
    prepare(args.src, args.dst, args.copy)


if __name__ == "__main__":
    main()
