"""Run one EVT-Bench split, or only check configuration and asset paths."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys

from .constants import CONFIG_ROOT, HABITAT_ROOT, REPO_ROOT, TASKS, TRACKVLA_REVISION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--task", choices=TASKS)
    config.add_argument("--exp-config", type=Path)
    parser.add_argument("--run-type", choices=["eval"], default="eval")
    parser.add_argument("--split-num", type=int, default=200)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=REPO_ROOT / "results/single_split")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--check-assets", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Hydra configuration overrides")
    args = parser.parse_args()
    if args.split_num <= 0 or not 0 <= args.split_id <= args.split_num:
        parser.error("--split-id must be in [0, --split-num], including the remainder split")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    save_path = args.save_path.expanduser().resolve()
    exp_config = (args.exp_config.expanduser().resolve() if args.exp_config
                  else CONFIG_ROOT / f"track_infer_{args.task}.yaml")
    for name in ("HF_MODEL_DIR", "DINOV3_MODEL_PATH", "SIGLIP_MODEL_PATH"):
        value = os.environ.get(name)
        if value and (Path(value).expanduser().exists() or value.startswith(("/", ".", "~"))):
            os.environ[name] = str(Path(value).expanduser().resolve())
    if os.environ.get("CKPT"):
        raise ValueError("Unset CKPT; select a complete vlxgo Hugging Face export with HF_MODEL_DIR")
    os.environ.setdefault("VLXGO_ATTN_IMPL", "sdpa")
    os.chdir(REPO_ROOT)
    # Use the bundled TrackEnv without depending on another checkout or a system installation.
    sys.path.insert(0, str(HABITAT_ROOT))
    import habitat
    import numpy as np
    from habitat.datasets import make_dataset
    from omegaconf import OmegaConf
    from . import evt_bench  # noqa: F401 - Registers task actions, sensors, and metrics on import.
    from .assets import check_model_assets, check_simulation_assets
    from .resume import pending_episodes
    from .splits import split_episodes

    config = habitat.get_config(str(exp_config), args.opts)
    with habitat.config.read_write(config):
        for agent in config.habitat.simulator.agents.values():
            for sensor in agent.sim_sensors.values():
                if "height" in sensor and "width" in sensor:
                    sensor.height = sensor.width = 384
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    selected = split_episodes(dataset.episodes, args.split_num, args.split_id,
                              config.habitat.simulator.seed)
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    if args.check_assets:
        check_model_assets()
        # Check the entire task for a full run, or only selected episodes for a bounded run.
        check_simulation_assets(config, selected if args.max_episodes else dataset.episodes)
        print(f"[eval] Habitat source: {habitat.__file__}; camera: 384x384; no rollout started")
        return 0
    pending = pending_episodes(selected, save_path) if args.resume else selected
    print(f"[eval] split={args.split_id} total={len(selected)} skipped={len(selected)-len(pending)} "
          f"pending={len(pending)} camera=384x384")
    if not pending:
        return 0
    sources = check_model_assets()
    check_simulation_assets(config, pending)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / f"config_split{args.split_id:03d}_info.json").write_text(json.dumps({
        "trackvla_revision": TRACKVLA_REVISION,
        "models": sources,
        "split_num": args.split_num, "split_id": args.split_id,
        "source_episodes": len(dataset.episodes), "selected_episodes": len(selected),
        "pending_episodes": len(pending), "max_episodes": args.max_episodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": OmegaConf.to_container(config, resolve=True),
    }, indent=2) + "\n")
    dataset_split = copy.copy(dataset)
    dataset_split.episodes = pending
    from .agent import evaluate_agent
    evaluate_agent(config, dataset_split, str(save_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        raise SystemExit(1)
