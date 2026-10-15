# Copyright 2026 The OpenBMB Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""RobotTrack f666593 dataset splitting, including the remainder split."""

from typing import Any, Dict, List, Sequence

import numpy as np


def split_episodes(episodes: Sequence[Any], split_count: int, split_id: int, seed: int) -> List[Any]:
    if split_count <= 0:
        raise ValueError("split_count must be positive")
    if split_id < 0 or split_id > split_count:
        raise ValueError("split_id must be in [0, split_count], including remainder")
    if len(episodes) < split_count:
        raise ValueError("the dataset has fewer episodes than requested splits")

    split_size = len(episodes) // split_count
    selected = np.random.RandomState(seed).choice(
        len(episodes), split_size * split_count, replace=False
    ).tolist()
    by_scene: Dict[str, List[int]] = {}
    for index in selected:
        by_scene.setdefault(episodes[index].scene_id, []).append(index)
    ordered = [index for indices in by_scene.values() for index in indices]
    if split_id == split_count:
        used = set(ordered)
        return [episode for index, episode in enumerate(episodes) if index not in used]
    start = split_id * split_size
    return [episodes[index] for index in ordered[start:start + split_size]]
