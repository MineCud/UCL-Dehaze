#!/usr/bin/env python3
"""Merge RRSHID thin/moderate/thick into UCL-Dehaze unpaired folders.

Expected RRSHID layout:
  RRSHID/
    moderate_fog/{train,test,val}/{hazy,clear|GT}/
    thick_fog/{train,test,val}/{hazy,clear|GT}/
    thin_fog/{train,test,val}/{hazy,gt|GT}/

Output (default ./datasets/rrshid_all):
  trainA  trainB  testA  testB  valA  valB

Filenames are prefixed with the fog level to avoid collisions, e.g. moderate_10.png.

Usage (Docker / Linux):
  python prepare_rrshid.py --src /data/RRSHID --dst ./datasets/rrshid_all
  python prepare_rrshid.py --src /data/RRSHID --dst ./datasets/rrshid_all --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SPLITS = {
    "train": ("trainA", "trainB"),
    "test": ("testA", "testB"),
    "val": ("valA", "valB"),
}

# (fog_level_dir, clear_folder_candidates)
FOG_LEVELS = [
    ("thin_fog", ("gt", "GT", "clear")),
    ("moderate_fog", ("clear", "GT", "gt")),
    ("thick_fog", ("clear", "GT", "gt")),
]


def find_clear_dir(split_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        p = split_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No clear/gt folder under {split_dir}. Tried: {candidates}"
    )


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
        raise FileNotFoundError(f"RRSHID root not found: {src}")

    for _, (a_name, b_name) in SPLITS.items():
        (dst / a_name).mkdir(parents=True, exist_ok=True)
        (dst / b_name).mkdir(parents=True, exist_ok=True)

    counts = {k: {"A": 0, "B": 0} for k in SPLITS}

    for fog_name, clear_candidates in FOG_LEVELS:
        fog_root = src / fog_name
        if not fog_root.is_dir():
            raise FileNotFoundError(f"Missing fog level: {fog_root}")

        for split, (a_name, b_name) in SPLITS.items():
            split_dir = fog_root / split
            hazy_dir = split_dir / "hazy"
            clear_dir = find_clear_dir(split_dir, clear_candidates)

            prefix = fog_name.replace("_fog", "")  # thin / moderate / thick

            for img in collect_images(hazy_dir):
                out = dst / a_name / f"{prefix}_{img.name}"
                link_or_copy(img, out, copy)
                counts[split]["A"] += 1

            for img in collect_images(clear_dir):
                out = dst / b_name / f"{prefix}_{img.name}"
                link_or_copy(img, out, copy)
                counts[split]["B"] += 1

            print(f"[ok] {fog_name}/{split}: hazy={len(collect_images(hazy_dir))} "
                  f"clear={len(collect_images(clear_dir))} -> {a_name}/{b_name}")

    print("\nMerged totals:")
    for split, (a_name, b_name) in SPLITS.items():
        print(f"  {a_name}: {counts[split]['A']}  |  {b_name}: {counts[split]['B']}")
    print(f"\nOutput: {dst.resolve()}")
    print("Train with:")
    print(
        f"  python train.py --dataroot {dst} --name rrshid_ucl "
        f"--preprocess none --load_size 256 --crop_size 256 --batch_size 1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RRSHID for UCL-Dehaze")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to RRSHID root (contains thin_fog/moderate_fog/thick_fog)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("./datasets/rrshid_all"),
        help="Output dataroot for UCL-Dehaze",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (use if symlink fails)",
    )
    args = parser.parse_args()
    prepare(args.src, args.dst, args.copy)


if __name__ == "__main__":
    main()
