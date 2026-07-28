"""Frozen diffusion prior teacher for UCL-Dehaze (training-only, ablatable).

Provides a clear-domain manifold prior via a real diffusion backbone
(diffusers). Inference never requires this module.

Modes
-----
- feature : align intermediate UNet / encoder features of fake vs clean
- latent  : align VAE latents of fake vs clean
- score   : align noise predictions at a fixed diffusion timestep

Disable with --use_diff_prior false (default) so original UCL is unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_diffusers():
    try:
        import torch
        # Older PyTorch (no torch.xpu) breaks newer diffusers imports.
        if not hasattr(torch, "xpu"):
            import types

            torch.xpu = types.SimpleNamespace(
                is_available=lambda: False,
                empty_cache=lambda: None,
                device_count=lambda: 0,
            )

        import diffusers  # noqa: F401
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "Diffusion prior requires optional deps. Install with:\n"
            "  pip install 'diffusers==0.27.2' 'transformers==4.38.2' "
            "'huggingface_hub==0.21.4' accelerate safetensors\n"
            f"Original error: {e}"
        ) from e
    return AutoencoderKL, UNet2DConditionModel, DDPMScheduler, CLIPTextModel, CLIPTokenizer


class DiffusionPriorTeacher(nn.Module):
    """Frozen real diffusion teacher used only during training."""

    def __init__(
        self,
        name_or_path: str = "runwayml/stable-diffusion-v1-5",
        mode: str = "feature",
        device: Union[torch.device, str] = "cuda",
        dtype: torch.dtype = torch.float16,
        score_timestep: int = 200,
        feature_block: str = "mid",
    ):
        super().__init__()
        mode = mode.lower()
        if mode not in ("feature", "latent", "score"):
            raise ValueError(f"Unknown diff_prior_mode: {mode}")
        self.mode = mode
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self.score_timestep = int(score_timestep)
        self.feature_block = feature_block

        AutoencoderKL, UNet2DConditionModel, DDPMScheduler, CLIPTextModel, CLIPTokenizer = _require_diffusers()

        print(f"[DiffusionPrior] loading teacher from: {name_or_path}  mode={mode}")
        self.vae = AutoencoderKL.from_pretrained(name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.dtype)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

        self.unet = None
        self.scheduler = None
        self.text_encoder = None
        self.null_prompt_embeds = None

        if mode in ("feature", "score"):
            self.unet = UNet2DConditionModel.from_pretrained(name_or_path, subfolder="unet")
            self.unet.requires_grad_(False)
            self.unet.eval()
            self.unet.to(self.device, dtype=self.dtype)

            self.scheduler = DDPMScheduler.from_pretrained(name_or_path, subfolder="scheduler")

            # Empty prompt conditioning (unconditional clear-manifold prior)
            tokenizer = CLIPTokenizer.from_pretrained(name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(name_or_path, subfolder="text_encoder")
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder.to(self.device, dtype=self.dtype)

            tokens = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                self.null_prompt_embeds = self.text_encoder(
                    tokens.input_ids.to(self.device)
                )[0].to(dtype=self.dtype)

        self._feat_cache = {}
        if mode == "feature":
            self._register_feature_hooks()

        # Never train teacher params
        for p in self.parameters():
            p.requires_grad_(False)

    def _register_feature_hooks(self):
        """Capture mid-block / selected up-block activations."""

        def _make_hook(name):
            def hook(_module, _inp, out):
                # out may be Tensor or tuple
                feat = out[0] if isinstance(out, (tuple, list)) else out
                self._feat_cache[name] = feat

            return hook

        # Prefer mid_block; fall back to first up_block if missing
        if hasattr(self.unet, "mid_block") and self.unet.mid_block is not None:
            self.unet.mid_block.register_forward_hook(_make_hook("mid"))
        if hasattr(self.unet, "up_blocks") and len(self.unet.up_blocks) > 0:
            self.unet.up_blocks[0].register_forward_hook(_make_hook("up0"))

    def train(self, mode: bool = True):
        # Always keep teacher in eval
        return super().train(False)

    def _to_teacher_input(self, x: torch.Tensor) -> torch.Tensor:
        """UCL tensors are [-1,1]; match spatial size expected by VAE (multiple of 8)."""
        x = x.to(device=self.device, dtype=self.dtype)
        h, w = x.shape[-2:]
        nh = max(8, (h // 8) * 8)
        nw = max(8, (w // 8) * 8)
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return x

    def encode_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        x = self._to_teacher_input(x)
        posterior = self.vae.encode(x).latent_dist
        if sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z * self.scaling_factor

    def _unet_forward(self, latents: torch.Tensor, timesteps: torch.Tensor):
        b = latents.shape[0]
        encoder_hidden_states = self.null_prompt_embeds.expand(b, -1, -1)
        return self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run VAE->latent and UNet once; return hooked features (grads w.r.t. x kept)."""
        self._feat_cache = {}
        latents = self.encode_latent(x, sample=False)
        b = latents.shape[0]
        # Fixed small noise level for stable feature extraction
        t = torch.full((b,), self.score_timestep, device=latents.device, dtype=torch.long)
        # Add mild noise so UNet activations are informative but not destroyed
        noise = torch.randn_like(latents)
        noisy = self.scheduler.add_noise(latents, noise, t)
        _ = self._unet_forward(noisy, t)

        feats = []
        for key in ("mid", "up0"):
            if key in self._feat_cache:
                feats.append(self._feat_cache[key])
        if not feats:
            # Fallback: use latent itself as feature
            feats = [latents]
        return feats

    def score_predict(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.encode_latent(x, sample=False)
        b = latents.shape[0]
        t = torch.full((b,), self.score_timestep, device=latents.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = self.scheduler.add_noise(latents, noise, t)
        return self._unet_forward(noisy, t)

    def compute_prior_loss(
        self,
        fake: torch.Tensor,
        clean: torch.Tensor,
        hazy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Align fake toward clean clear-manifold prior. Teacher weights frozen."""
        if self.mode == "latent":
            z_fake = self.encode_latent(fake, sample=False)
            with torch.no_grad():
                z_clean = self.encode_latent(clean, sample=False)
            return F.l1_loss(z_fake, z_clean)

        if self.mode == "score":
            # Same noise for paired comparison
            fake_in = self._to_teacher_input(fake)
            clean_in = self._to_teacher_input(clean)
            z_fake = self.encode_latent(fake_in, sample=False)
            with torch.no_grad():
                z_clean = self.encode_latent(clean_in, sample=False)

            b = z_fake.shape[0]
            t = torch.full((b,), self.score_timestep, device=z_fake.device, dtype=torch.long)
            noise = torch.randn_like(z_fake)
            noisy_fake = self.scheduler.add_noise(z_fake, noise, t)
            with torch.no_grad():
                noisy_clean = self.scheduler.add_noise(z_clean, noise, t)
                eps_clean = self._unet_forward(noisy_clean, t)

            eps_fake = self._unet_forward(noisy_fake, t)
            return F.mse_loss(eps_fake, eps_clean)

        # feature (default)
        feats_fake = self.extract_features(fake)
        with torch.no_grad():
            feats_clean = self.extract_features(clean)
        loss = 0.0
        for ff, fc in zip(feats_fake, feats_clean):
            # Spatial size may differ slightly; interpolate if needed
            if ff.shape[-2:] != fc.shape[-2:]:
                fc = F.interpolate(fc, size=ff.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + F.l1_loss(ff, fc.detach())
        return loss / max(len(feats_fake), 1)


def build_diffusion_prior(opt, device) -> Optional[DiffusionPriorTeacher]:
    """Factory: return teacher if enabled, else None. Lazy-imports diffusers."""
    use = getattr(opt, "use_diff_prior", False)
    if not use:
        return None
    # Inference should not load the teacher unless explicitly requested
    if not opt.isTrain and not getattr(opt, "diff_prior_infer", False):
        return None

    name = getattr(opt, "diff_teacher_name_or_path", "runwayml/stable-diffusion-v1-5")
    mode = getattr(opt, "diff_prior_mode", "feature")
    dtype_name = getattr(opt, "diff_prior_dtype", "fp16")
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    t_step = getattr(opt, "diff_score_timestep", 200)

    teacher = DiffusionPriorTeacher(
        name_or_path=name,
        mode=mode,
        device=device,
        dtype=dtype,
        score_timestep=t_step,
    )
    return teacher


def physics_consistency_loss(
    hazy: torch.Tensor,
    dehazed: torch.Tensor,
    omega: float = 0.95,
    t_min: float = 0.1,
) -> torch.Tensor:
    """Lightweight atmospheric scattering consistency (no extra network).

    I ≈ J * t + A * (1 - t)
    Estimate A from brightest pixels in hazy; t from dark-channel heuristic.
    Inputs are UCL tensors in [-1, 1].
    """
    # Map from [-1,1] to [0,1]
    I = ((hazy + 1.0) * 0.5).clamp(0, 1)
    J = ((dehazed + 1.0) * 0.5).clamp(0, 1)

    b, c, h, w = I.shape
    flat = I.reshape(b, c, -1)
    lum = I.mean(dim=1)  # B,H,W
    k = max(1, int(0.001 * h * w))
    _, idx = torch.topk(lum.reshape(b, -1), k=k, dim=1)
    A_list = []
    for i in range(b):
        pix = flat[i, :, idx[i]]  # C,k
        A_list.append(pix.mean(dim=1))
    A = torch.stack(A_list, dim=0).view(b, c, 1, 1).clamp(0.0, 1.0)

    # t = 1 - omega * dark_channel(I / A)
    ratio = (I / (A + 1e-6)).clamp(0, 1)
    dark = ratio.min(dim=1, keepdim=True).values
    dark = -F.max_pool2d(-dark, kernel_size=5, stride=1, padding=2)
    t = (1.0 - omega * dark).clamp(t_min, 1.0)

    recon = J * t + A * (1.0 - t)
    return F.l1_loss(recon, I)
