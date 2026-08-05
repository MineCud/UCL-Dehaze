#!/usr/bin/env python3
"""Visualize VGG16 features used by UCL-Dehaze perceptual (self-contrastive) loss.

Important distinction
---------------------
- PatchNCE contrastive loss uses **netG encoder** layers (default 0,5,9,13,17), NOT VGG.
- VGG is only used in ``perceptual_loss``: features at VGG16 indices 9, 16, 30.

VGG16.features index map (selected):
  9  -> after relu of conv3_3 region (mid-low / texture-ish)
 16  -> after relu near conv4_* (mid-level)
 30  -> near conv5 / deep semantic

Usage (Docker / server):
  python visualize_vgg_features.py \\
    --img /path/to/hazy.png \\
    --out_dir ./results/vgg_vis/hazy \\
    --gpu_ids 0

  # Also compare hazy / dehazed / clear side-by-side activations
  python visualize_vgg_features.py \\
    --img /path/to/hazy.png --clear /path/to/clean.png --dehazed /path/to/out.png \\
    --out_dir ./results/vgg_vis/triple --gpu_ids 0

  # Batch: all hazy + GT images in RSID folders
  python visualize_vgg_features.py --batch \\
    --rsid_root /data/RSID \\
    --out_dir ./results/vgg_vis/rsid_all \\
    --mean_only --gpu_ids 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision import models


# Same selection as models/base_model.py::VGGNet
VGG_SELECT = ["9", "16", "30"]
VGG_NAMES = {
    "9": "feat1 (idx=9, pool2 / 128ch)",
    "16": "feat2 (idx=16, pool3 / 256ch)",
    "30": "feat3 (idx=30, pool5 / 512ch)",
}


class VGGFeat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.select = VGG_SELECT
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        feats = {}
        h = x
        for name, layer in self.vgg._modules.items():
            h = layer(h)
            if name in self.select:
                feats[name] = h
        return feats


def load_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return tf(img).unsqueeze(0).to(device)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """[-1,1] CHW or BCHW -> PIL."""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().cpu()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def feat_to_heatmaps(feat: torch.Tensor, n_show: int = 16) -> list[Image.Image]:
    """Convert (1,C,H,W) to list of grayscale heatmaps (mean + top-energy channels)."""
    f = feat[0].float().cpu()  # C,H,W
    energy = f.abs().mean(dim=(1, 2))
    order = torch.argsort(energy, descending=True)

    maps = []
    # channel-mean activation
    mean_map = f.mean(dim=0)
    maps.append(("mean", _normalize_map(mean_map)))
    # top channels
    for i in range(min(n_show, f.shape[0])):
        idx = int(order[i].item())
        maps.append((f"ch{idx}", _normalize_map(f[idx])))
    return maps


def _normalize_map(m: torch.Tensor) -> Image.Image:
    m = m.detach().float().cpu()
    m = m - m.min()
    if float(m.max()) > 1e-8:
        m = m / m.max()
    arr = (m.numpy() * 255).astype(np.uint8)
    gray = Image.fromarray(arr, mode="L")
    return gray.convert("RGB")


def colorize(gray_rgb: Image.Image) -> Image.Image:
    """Cheap blue-cyan-yellow-red colormap for readability."""
    arr = np.asarray(gray_rgb.convert("L"), dtype=np.float32) / 255.0
    r = np.clip(1.5 * arr - 0.2, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(arr - 0.5) * 2), 0, 1)
    b = np.clip(1.2 * (1 - arr), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def make_grid(images: list[Image.Image], labels: list[str], cell: int = 128) -> Image.Image:
    n = len(images)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + 18)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        thumb = im.resize((cell, cell), Image.BILINEAR)
        canvas.paste(thumb, (c * cell, r * (cell + 18)))
        draw.text((c * cell + 4, r * (cell + 18) + cell + 2), lab, fill=(220, 220, 220))
    return canvas


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())



def save_layer_vis(
    feat: torch.Tensor, out_dir: Path, tag: str, n_show: int, size: int, mean_only: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    # upsample mean map to input size for overlay feel
    feat = feat.detach()
    mean = feat.mean(dim=1, keepdim=True)
    mean_up = F.interpolate(mean, size=(size, size), mode="bilinear", align_corners=False)
    heat = colorize(_normalize_map(mean_up[0, 0].cpu()))
    heat.save(out_dir / f"{tag}_mean_heatmap.png")
    if mean_only:
        return

    maps = feat_to_heatmaps(feat, n_show=n_show)
    imgs, labs = [], []
    for name, im in maps:
        imgs.append(colorize(im))
        labs.append(name)
    grid = make_grid(imgs, labs, cell=128)
    grid.save(out_dir / f"{tag}_channels.png")


def save_compare_mean(all_feats: dict, tags: list[str], out_path: Path, size: int):
    """Side-by-side mean heatmaps for each VGG layer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for lid in VGG_SELECT:
        row = []
        for tag in tags:
            mean = all_feats[tag][lid].mean(dim=1, keepdim=True)
            mean_up = F.interpolate(
                mean, size=(size, size), mode="bilinear", align_corners=False
            )
            row.append(colorize(_normalize_map(mean_up[0, 0].cpu())))
        grid = make_grid(row, [f"{t}-vgg{lid}" for t in tags], cell=size)
        grid.save(out_path.parent / f"{out_path.stem}_vgg{lid}_mean.png")


def process_one_image(
    net: VGGFeat,
    path: Path,
    out_dir: Path,
    tag: str,
    device: torch.device,
    load_size: int,
    n_show: int,
    mean_only: bool,
) -> dict:
    x = load_tensor(path, load_size, device)
    tensor_to_pil(x).save(out_dir / f"{tag}_rgb.png")
    feats = net(x)
    for lid, feat in feats.items():
        save_layer_vis(feat, out_dir, f"vgg{lid}", n_show, load_size, mean_only=mean_only)
    return feats


def batch_from_folders(
    net: VGGFeat,
    hazy_dir: Path,
    gt_dir: Path | None,
    out_root: Path,
    device: torch.device,
    load_size: int,
    n_show: int,
    mean_only: bool,
    compare_pairs: bool,
):
    hazy_images = list_images(hazy_dir)
    if not hazy_images:
        print(f"[skip] no images in {hazy_dir}")
        return
    print(f"[batch] {hazy_dir} -> {len(hazy_images)} images")
    for i, hazy_path in enumerate(hazy_images, 1):
        stem = hazy_path.stem
        hazy_out = out_root / "hazy" / stem
        hazy_feats = process_one_image(
            net, hazy_path, hazy_out, "input", device, load_size, n_show, mean_only
        )
        gt_path = None
        if gt_dir is not None:
            for cand in (gt_dir / hazy_path.name, gt_dir / f"{stem}{hazy_path.suffix}"):
                if cand.is_file():
                    gt_path = cand
                    break
        if gt_path is not None:
            gt_out = out_root / "GT" / stem
            gt_feats = process_one_image(
                net, gt_path, gt_out, "clear", device, load_size, n_show, mean_only
            )
            if compare_pairs:
                save_compare_mean(
                    {"hazy": hazy_feats, "GT": gt_feats},
                    ["hazy", "GT"],
                    out_root / "compare" / stem / "pair",
                    load_size,
                )
        if i % 20 == 0 or i == len(hazy_images):
            print(f"  {hazy_dir.name}: {i}/{len(hazy_images)}")


def batch_rsid_all(
    rsid_root: Path,
    out_root: Path,
    device: torch.device,
    load_size: int,
    n_show: int,
    mean_only: bool,
    compare_pairs: bool,
    net: VGGFeat,
):
    for split in ("train", "test"):
        hazy_dir = rsid_root / split / "hazy"
        gt_dir = rsid_root / split / "GT"
        if not hazy_dir.is_dir():
            print(f"[skip] missing {hazy_dir}")
            continue
        if not gt_dir.is_dir():
            gt_dir = None
            print(f"[warn] no GT for {split}, only hazy heatmaps")
        batch_from_folders(
            net,
            hazy_dir,
            gt_dir,
            out_root / split,
            device,
            load_size,
            n_show,
            mean_only,
            compare_pairs,
        )


def print_vgg_structure():
    vgg = models.vgg16(pretrained=True).features
    print("VGG16.features layers (selected marked with *):")
    for name, layer in vgg._modules.items():
        mark = " *" if name in VGG_SELECT else ""
        print(f"  [{name:>2}] {layer}{mark}")
    print("\nUCL perceptual_loss uses these three feature maps:")
    for k in VGG_SELECT:
        print(f"  - {VGG_NAMES[k]}")
    print(
        "\nPatchNCE (main contrastive) does NOT use VGG; "
        "it uses netG encoder layers --nce_layers (default 0,5,9,13,17)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default=None, help="main image (hazy or any)")
    ap.add_argument("--clear", type=str, default=None, help="optional clear GT")
    ap.add_argument("--dehazed", type=str, default=None, help="optional dehazed result")
    ap.add_argument("--out_dir", type=str, default="./results/vgg_vis")
    ap.add_argument("--load_size", type=int, default=256)
    ap.add_argument("--n_show", type=int, default=16, help="top channels to show per layer")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--list_layers", action="store_true", help="print VGG layer list and exit")
    ap.add_argument("--batch", action="store_true", help="batch mode for folders")
    ap.add_argument("--rsid_root", type=str, default=None, help="RSID root with train/test/hazy,GT")
    ap.add_argument("--hazy_dir", type=str, default=None, help="batch: hazy image folder")
    ap.add_argument("--gt_dir", type=str, default=None, help="batch: GT image folder")
    ap.add_argument("--mean_only", action="store_true", help="batch: save mean heatmaps only (faster)")
    ap.add_argument("--compare_pairs", action="store_true", help="batch: save hazy vs GT compare grids")
    args = ap.parse_args()

    if args.list_layers:
        print_vgg_structure()
        return

    gpu = args.gpu_ids.split(",")[0].strip()
    device = torch.device(f"cuda:{gpu}" if gpu != "-1" and torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    net = VGGFeat().to(device)

    if args.batch:
        if args.rsid_root:
            batch_rsid_all(
                Path(args.rsid_root),
                out_root,
                device,
                args.load_size,
                args.n_show,
                args.mean_only,
                args.compare_pairs,
                net,
            )
        elif args.hazy_dir:
            gt_dir = Path(args.gt_dir) if args.gt_dir else None
            batch_from_folders(
                net,
                Path(args.hazy_dir),
                gt_dir,
                out_root,
                device,
                args.load_size,
                args.n_show,
                args.mean_only,
                args.compare_pairs,
            )
        else:
            ap.error("batch mode needs --rsid_root or --hazy_dir")
        print(f"\nSaved batch visualizations to: {out_root.resolve()}")
        return

    if not args.img:
        ap.error("--img is required unless --list_layers or --batch")

    print_vgg_structure()

    items = [("input", Path(args.img))]
    if args.dehazed:
        items.append(("dehazed", Path(args.dehazed)))
    if args.clear:
        items.append(("clear", Path(args.clear)))

    all_feats = {}
    for tag, path in items:
        x = load_tensor(path, args.load_size, device)
        tensor_to_pil(x).save(out_root / f"{tag}_rgb.png")
        feats = net(x)
        all_feats[tag] = feats
        for lid, feat in feats.items():
            c, h, w = feat.shape[1:]
            print(f"[{tag}] {VGG_NAMES[lid]}: shape=({c},{h},{w})")
            save_layer_vis(
                feat, out_root / tag, f"vgg{lid}", args.n_show, args.load_size, mean_only=False
            )

    # Side-by-side mean heatmaps across images for each VGG layer
    if len(items) > 1:
        for lid in VGG_SELECT:
            row = []
            labs = []
            for tag, _ in items:
                mean = all_feats[tag][lid].mean(dim=1, keepdim=True)
                mean_up = F.interpolate(
                    mean, size=(args.load_size, args.load_size), mode="bilinear", align_corners=False
                )
                row.append(colorize(_normalize_map(mean_up[0, 0].cpu())))
                labs.append(f"{tag}\n{VGG_NAMES[lid]}")
            # also paste RGB of first image as reference strip
            grid = make_grid(row, [f"{t}-vgg{lid}" for t, _ in items], cell=args.load_size)
            grid.save(out_root / f"compare_vgg{lid}_mean.png")

        # Approximate perceptual ratios used in loss (for curiosity)
        if "input" in all_feats and "dehazed" in all_feats and "clear" in all_feats:
            print("\nPerceptual-loss style ratios (smaller means dehazed closer to clear than to hazy):")
            for i, lid in enumerate(VGG_SELECT, 1):
                fx = all_feats["input"][lid]
                fy = all_feats["dehazed"][lid]
                fz = all_feats["clear"][lid]
                num = F.l1_loss(fz, fy)
                den = F.l1_loss(fx, fy).clamp_min(1e-8)
                print(f"  m{i} (vgg{lid}): L1(clear,dehazed)/L1(hazy,dehazed) = {float(num/den):.4f}")

    print(f"\nSaved visualizations to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
