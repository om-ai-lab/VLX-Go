"""Load the self-contained vlxgo Hugging Face planner and its vision encoders."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .attention import configure_model_attention
from .vision import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens

logger = logging.getLogger("vlxgo.agent")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "resources" / "vlxgo"


def local_files_only() -> bool:
    """Keep loading offline unless the user explicitly enables Hub access."""
    return os.environ.get("HF_LOCAL_FILES_ONLY", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


class VlxgoAgent:
    """Text-conditioned waypoint planner with DINOv3 and SigLIP features."""

    def __init__(
        self, model_name_or_path: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        source = (
            model_name_or_path
            or os.environ.get("HF_MODEL_DIR")
            or os.environ.get("HF_MODEL_ID")
            or str(DEFAULT_MODEL_DIR)
        )
        if source.startswith(("/", ".", "~")):
            source = str(Path(source).expanduser().resolve())
            if not Path(source).is_dir():
                raise ValueError(f"Hugging Face model directory does not exist: {source}")
        elif Path(source).is_dir():
            source = str(Path(source).resolve())

        self.planner_device = torch.device(
            device or os.environ.get("VLXGO_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        load_options = {
            "local_files_only": local_files_only(),
            "revision": os.environ.get("HF_MODEL_REVISION", "main"),
        }
        logger.info("Loading vlxgo planner from %s on %s", source, self.planner_device)
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, **load_options)
        if config.model_type != "vlxgo":
            raise ValueError(f"Expected a vlxgo export, got model_type={config.model_type!r}")
        if (
            config.use_indicator or config.embodiment not in {"wheeled", "car"}
            or config.use_angle_tvi
        ):
            raise ValueError("VLX-Go requires an [x, y, yaw] planner without angle TVI")
        if config.vision_feat_dim != 1536:
            raise ValueError("The configured DINOv3/SigLIP encoders require vision_feat_dim=1536")
        self.history = int(config.history)
        if self.history < 0 or self.history >= config.max_time or config.n_waypoints < 1:
            raise ValueError("Invalid history, max_time or n_waypoints in model config")

        # Preserve the export's mixed precision; a global dtype would cast the planner.
        model, loading_info = AutoModel.from_pretrained(
            source, config=config, trust_remote_code=True,
            output_loading_info=True, **load_options,
        )
        problems = {
            key: loading_info[key]
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(key)
        }
        if problems:
            raise RuntimeError(f"The vlxgo checkpoint did not load completely: {problems}")
        model.tokenizer = AutoTokenizer.from_pretrained(
            source, fix_mistral_regex=False, **load_options
        )
        self.planner_model = model.to(self.planner_device).eval()
        configure_model_attention(self.planner_model.llm, model_label="vlxgo language backbone")

        resources = Path(__file__).resolve().parents[1] / "resources"
        self.vision_encoder = VisionFeatureCacher(VisionCacheConfig(
            dino_model_name=os.environ.get(
                "DINOV3_MODEL_PATH", str(resources / "dinov3-vits16-pretrain-lvd1689m")
            ),
            siglip_model_name=os.environ.get(
                "SIGLIP_MODEL_PATH", str(resources / "siglip-so400m-patch14-384")
            ),
            device=str(self.planner_device), local_files_only=local_files_only(),
        ))
        feature_dim = self.vision_encoder.dino_hidden + self.vision_encoder.siglip_hidden
        if feature_dim != config.vision_feat_dim:
            raise ValueError(f"Encoder feature dimension {feature_dim} does not match the planner")
        logger.info("vlxgo planner and vision encoders are ready")

    @torch.inference_mode()
    def encode_frame_tokens(self, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 coarse [4,C] and fine [64,C] tokens for one RGB frame."""
        image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        dino, height, width = self.vision_encoder.encode_dino([image])
        siglip = self.vision_encoder.encode_siglip([image], out_hw=(height, width))
        # Match the training/export preprocessing: concatenate, then pool each grid.
        features = torch.cat([dino, siglip], dim=-1)
        coarse = grid_pool_tokens(features, height, width, out_tokens=4)[0].float()
        fine = grid_pool_tokens(features, height, width, out_tokens=64)[0].float()
        return coarse, fine
