"""Check local model and simulation assets before startup without loading weights or creating a simulator."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .constants import HABITAT_ROOT, REPO_ROOT


def check_model_assets():
    resources = REPO_ROOT / "resources"
    sources = {
        "planner": os.environ.get("HF_MODEL_DIR") or os.environ.get("HF_MODEL_ID") or str(resources / "vlxgo"),
        "DINOv3": os.environ.get("DINOV3_MODEL_PATH", str(resources / "dinov3-vits16-pretrain-lvd1689m")),
        "SigLIP": os.environ.get("SIGLIP_MODEL_PATH", str(resources / "siglip-so400m-patch14-384")),
    }
    errors = []
    for label, value in sources.items():
        directory = Path(value).expanduser()
        if not directory.exists() and not value.startswith(("/", ".", "~")):
            # Let Transformers resolve Hub IDs from the local cache or explicitly allowed downloads.
            print(f"[assets] {label}: Hub ID {value}; cache availability checked during model loading")
            continue
        required = [directory / "config.json"]
        if label != "planner":
            required.append(directory / "preprocessor_config.json")
        errors.extend(str(path) for path in required if not path.is_file())
        if not directory.is_dir():
            continue
        weight_names = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                        "pytorch_model.bin.index.json")
        weights = [directory / name for name in weight_names if (directory / name).is_file()]
        if not weights:
            errors.append(f"{directory}: no Hugging Face weights or shard index")
        for weight in weights:
            if weight.name.endswith(".index.json"):
                mapping = json.loads(weight.read_text())["weight_map"]
                shards = [directory / name for name in set(mapping.values())]
            else:
                shards = [weight]
            for shard in shards:
                if not shard.is_file() or shard.stat().st_size < 256:
                    errors.append(f"{shard}: missing weights or download placeholder")
        if label == "planner" and (directory / "config.json").is_file():
            config = json.loads((directory / "config.json").read_text())
            if config.get("model_type") != "vlxgo":
                errors.append(f"{directory}: expected model_type=vlxgo")
            for key in ("AutoConfig", "AutoModel"):
                target = config.get("auto_map", {}).get(key)
                if not isinstance(target, str) or not (directory / (target.split(".")[0] + ".py")).is_file():
                    errors.append(f"{directory}: missing custom module for {key}")
            if not (directory / "tokenizer_config.json").is_file():
                errors.append(f"{directory}: tokenizer_config.json is required")
        print(f"[assets] {label}: {value}")
    if errors:
        raise FileNotFoundError("Incomplete model assets:\n" + "\n".join(errors))
    return sources


def check_simulation_assets(config, episodes):
    paths = {HABITAT_ROOT / "habitat/tasks/track/humanoid_infos.json"}
    simulator = config.habitat.simulator
    # TrackEnv overrides the YAML value with episode.scene_dataset_config before creating the simulator.
    # Bundled annotations use default, so they do not require downloading the unused HSSD dataset.
    if not episodes and simulator.scene_dataset != "default":
        paths.add(Path(simulator.scene_dataset))
    for agent in simulator.agents.values():
        paths.add(Path(agent.articulated_agent_urdf))
        if agent.motion_data_path:
            paths.add(Path(agent.motion_data_path))
    scenes, avatars = set(), set()
    for episode in episodes:
        paths.add(Path(episode.scene_id))
        scenes.add(episode.scene_id)
        if episode.scene_dataset_config and episode.scene_dataset_config != "default":
            paths.add(Path(episode.scene_dataset_config))
        avatars.add(episode.info["main_humanoid_name"])
        avatars.update(episode.info["extra_humanoid_names"])
    for avatar in avatars:
        base = Path("data/humanoids/humanoid_data") / avatar
        paths.update((base / f"{avatar}.urdf", base / f"{avatar}_motion_data_smplx.pkl"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} simulation assets (first 30 shown):\n"
            + "\n".join(missing[:30]) + "\nSee evaluation/README.md for the required data layout."
        )
    print(f"[assets] {len(episodes)} episodes, {len(scenes)} scenes, {len(avatars)} avatars; paths exist")
