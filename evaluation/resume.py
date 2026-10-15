"""Resume evaluation per episode using existing result paths and complete final JSON files."""

import json
import math
import os
from pathlib import Path
import tempfile


def episode_result_path(save_path, episode):
    """Match the existing evaluation writer's scene_key/episode_id.json path."""
    scene_key = os.path.splitext(os.path.basename(episode.scene_id))[0].split(".")[0]
    return Path(save_path) / scene_key / f"{episode.episode_id}.json"


def _finite_number(value, *, allow_bool=False):
    if isinstance(value, bool) and not allow_bool:
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def is_complete_result(result):
    """Count failed rollouts as complete; reject only incomplete results or invalid metrics."""
    if not isinstance(result, dict) or not isinstance(result.get("finish"), bool):
        return False
    if result.get("status") not in ("Normal", "Lost", "Collision"):
        return False
    for key in ("success", "collision"):
        value = result.get(key)
        if not _finite_number(value, allow_bool=True) or value not in (0, 1):
            return False
    for key in ("following_step", "total_step"):
        value = result.get(key)
        if not _finite_number(value) or int(value) != value:
            return False
    following, total = result["following_step"], result["total_step"]
    if not 0 <= following <= total or total <= 0:
        return False
    rate = result.get("following_rate")
    return (
        _finite_number(rate)
        and 0 <= rate <= 1
        and math.isclose(rate, following / total, rel_tol=1e-9, abs_tol=1e-12)
    )


def is_episode_complete(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return is_complete_result(json.load(stream))
    except (OSError, ValueError, UnicodeError):
        return False


def pending_episodes(episodes, save_path):
    """Preserve input order; callers must split the dataset before filtering completed episodes."""
    return [
        episode for episode in episodes
        if not is_episode_complete(episode_result_path(save_path, episode))
    ]


def atomic_write_result(path, result):
    """Write in the same directory before replacing the final file; scoring ignores the .tmp suffix."""
    path = Path(path)
    if not is_complete_result(result):
        raise ValueError(f"Cannot publish invalid episode result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
