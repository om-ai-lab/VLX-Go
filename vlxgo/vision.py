"""Online DINOv3/SigLIP patch extraction and grid average pooling for vlxgo."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipImageProcessor, SiglipVisionModel

from .attention import from_pretrained_with_attention

logger = logging.getLogger("vlxgo.vision")


def _sqrt_int(value: int) -> int:
    size = math.isqrt(value)
    if size * size != value:
        raise ValueError(f"Expected a square patch grid, got {value} tokens")
    return size


def grid_pool_tokens(
    patch_tokens: torch.Tensor, height: int, width: int, out_tokens: int
) -> torch.Tensor:
    """Average-pool [B,H*W,C] patch tokens to a square output grid."""
    batch, patches, channels = patch_tokens.shape
    if patches != height * width:
        raise ValueError("Patch count does not match the grid dimensions")
    size = _sqrt_int(out_tokens)
    grid = patch_tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)
    return F.adaptive_avg_pool2d(grid, (size, size)).flatten(2).transpose(1, 2).contiguous()


def adapt_siglip_grid(
    tokens: torch.Tensor, grid_hw: Optional[tuple[int, int]] = None,
    out_hw: tuple[int, int] = (24, 24),
) -> torch.Tensor:
    """Align SigLIP patches with the DINOv3 grid before concatenation."""
    batch, patches, channels = tokens.shape
    height, width = grid_hw or (_sqrt_int(patches), _sqrt_int(patches))
    if height * width != patches:
        raise ValueError("SigLIP patch count does not match its grid")
    grid = tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)
    return F.adaptive_avg_pool2d(grid, out_hw).flatten(2).transpose(1, 2).contiguous()


@dataclass
class VisionCacheConfig:
    dino_model_name: str
    siglip_model_name: str
    image_size: int = 384
    device: str = "cpu"
    local_files_only: bool = True


class VisionFeatureCacher(nn.Module):
    """Frozen vision towers using the export's 384 x 384 preprocessing."""

    def __init__(self, cfg: VisionCacheConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        options = {"local_files_only": cfg.local_files_only}
        logger.info("Loading DINOv3 from %s", cfg.dino_model_name)
        self.dino_proc = AutoImageProcessor.from_pretrained(cfg.dino_model_name, **options)
        self.dino = from_pretrained_with_attention(
            AutoModel, cfg.dino_model_name, model_label="DINOv3",
            flash_torch_dtype=torch.float16, target_device=self.device, **options,
        ).eval().to(self.device)
        self.dino_regs = int(getattr(self.dino.config, "num_register_tokens", 0) or 0)
        self.dino_hidden = self.dino.config.hidden_size

        logger.info("Loading SigLIP from %s", cfg.siglip_model_name)
        self.siglip_proc = SiglipImageProcessor.from_pretrained(cfg.siglip_model_name, **options)
        self.siglip = from_pretrained_with_attention(
            SiglipVisionModel, cfg.siglip_model_name, model_label="SigLIP",
            flash_torch_dtype=torch.float16, target_device=self.device, **options,
        ).eval().to(self.device)
        self.siglip_hidden = self.siglip.config.hidden_size
        self.requires_grad_(False)

    def _preprocess_pils(self, images: list[Image.Image], which: str) -> dict[str, torch.Tensor]:
        size = self.cfg.image_size
        images = [image.resize((size, size), Image.Resampling.BICUBIC) for image in images]
        processor, model = (
            (self.dino_proc, self.dino) if which == "dino" else (self.siglip_proc, self.siglip)
        )
        inputs = processor(images=images, return_tensors="pt", size={"height": size, "width": size})
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)
        return inputs

    @torch.inference_mode()
    def encode_dino(self, images: list[Image.Image]) -> tuple[torch.Tensor, int, int]:
        tokens = self.dino(**self._preprocess_pils(images, "dino")).last_hidden_state
        tokens = tokens[:, 1 + self.dino_regs:, :]  # Exclude CLS and register tokens.
        size = _sqrt_int(tokens.size(1))
        return tokens, size, size

    @torch.inference_mode()
    def encode_siglip(self, images: list[Image.Image], out_hw: tuple[int, int]) -> torch.Tensor:
        tokens = self.siglip(**self._preprocess_pils(images, "siglip")).last_hidden_state
        count = tokens.size(1)
        if math.isqrt(count) ** 2 != count and math.isqrt(count - 1) ** 2 == count - 1:
            tokens = tokens[:, 1:, :]  # Accept encoders that include one CLS token.
        size = _sqrt_int(tokens.size(1))
        return adapt_siglip_grid(tokens, grid_hw=(size, size), out_hw=out_hw)
