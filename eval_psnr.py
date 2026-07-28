#!/usr/bin/env python3
"""Run UCL-Dehaze on paired hazy/GT folders, save dehazed images, report PSNR/SSIM.

Example (RICE1 after symlink prep, or raw folders):
  python eval_psnr.py \\
    --name rice1_ucl \\
    --hazy /workspace/datasets/Rice1/RICE1/test/hazy \\
    --gt   /workspace/datasets/Rice1/RICE1/test/clean \\
    --epoch latest \\
    --gpu_ids 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import create_model
from options.test_options import TestOptions
from util.util import tensor2im, save_image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())


def to_tensor(path: Path, size: int | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transforms = []
    if size is not None:
        transforms.append(T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC))
    transforms.extend([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return T.Compose(transforms)(img).unsqueeze(0)


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-10:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    """Simple channel-wise average SSIM (no scipy dependency beyond numpy)."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    scores = []
    for c in range(3):
        scores.append(_ssim_gray(pred[:, :, c], gt[:, :, c]))
    return float(np.mean(scores))


def _ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    return float(
        ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2))
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    )


def build_opt(args: argparse.Namespace):
    # Minimal options via TestOptions defaults + overrides
    cmd = (
        f"--dataroot {args.hazy} --name {args.name} --model UCL "
        f"--phase test --eval --preprocess none "
        f"--load_size {args.load_size} --crop_size {args.crop_size} "
        f"--gpu_ids {args.gpu_ids} --epoch {args.epoch} "
        f"--checkpoints_dir {args.checkpoints_dir} --results_dir {args.results_dir} "
        f"--bottleneck {args.bottleneck} --use_diff_prior false --use_phys_loss false"
    )
    opt = TestOptions(cmd_line=cmd).parse()
    opt.isTrain = False
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    return opt


def main() -> None:
    parser = argparse.ArgumentParser(description="Dehaze + PSNR/SSIM on paired test set")
    parser.add_argument("--name", type=str, required=True, help="experiment name under checkpoints/")
    parser.add_argument("--hazy", type=Path, required=True, help="folder of hazy images")
    parser.add_argument("--gt", type=Path, required=True, help="folder of clean GT (same filenames)")
    parser.add_argument("--epoch", type=str, default="latest")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--bottleneck", type=str, default="mamba", choices=["sc", "mamba"])
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--out_dir", type=Path, default=None, help="save dehazed images here")
    args = parser.parse_args()

    if not args.hazy.is_dir() or not args.gt.is_dir():
        raise FileNotFoundError(f"Need folders: {args.hazy} and {args.gt}")

    out_dir = args.out_dir or Path(args.results_dir) / args.name / f"eval_{args.epoch}" / "dehazed"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = build_opt(args)
    size = args.crop_size
    # Dummy batch to init G (test path only needs netG)
    hazy_list = list_images(args.hazy)
    if not hazy_list:
        raise RuntimeError(f"No images in {args.hazy}")

    model = create_model(opt)
    dummy = {
        "A": to_tensor(hazy_list[0], size),
        "B": to_tensor(hazy_list[0], size),
        "A_paths": [str(hazy_list[0])],
        "B_paths": [str(hazy_list[0])],
    }
    model.data_dependent_initialize(dummy)
    model.setup(opt)
    model.parallelize()
    model.eval()

    psnrs, ssims = [], []
    missing = 0

    with torch.no_grad():
        for i, hazy_path in enumerate(hazy_list):
            gt_path = args.gt / hazy_path.name
            if not gt_path.is_file():
                print(f"[skip] no GT for {hazy_path.name}")
                missing += 1
                continue

            data = {
                "A": to_tensor(hazy_path, size),
                "B": to_tensor(gt_path, size),
                "A_paths": [str(hazy_path)],
                "B_paths": [str(gt_path)],
            }
            model.set_input(data)
            model.test()
            fake = tensor2im(model.fake_B)
            gt = tensor2im(model.real_B)

            save_image(fake, str(out_dir / hazy_path.name))

            p = psnr(fake, gt)
            s = ssim_rgb(fake, gt)
            psnrs.append(p)
            ssims.append(s)

            if i % 10 == 0:
                print(f"[{i+1}/{len(hazy_list)}] {hazy_path.name}  PSNR={p:.3f}  SSIM={s:.4f}")

    if not psnrs:
        raise RuntimeError("No paired images evaluated.")

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    summary = out_dir.parent / "metrics.txt"
    text = (
        f"images={len(psnrs)}  missing_gt={missing}\n"
        f"PSNR={mean_psnr:.4f} dB\n"
        f"SSIM={mean_ssim:.4f}\n"
    )
    summary.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Dehazed images: {out_dir.resolve()}")
    print(f"Metrics file:   {summary.resolve()}")


if __name__ == "__main__":
    main()
