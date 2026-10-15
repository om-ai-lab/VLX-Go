"""Pinned evaluation sources and repository paths; scoring is independent of Habitat and the model."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HABITAT_ROOT = REPO_ROOT / "third_party" / "habitat-lab"
CONFIG_ROOT = HABITAT_ROOT / "habitat/config/benchmark/nav/track"
TRACKVLA_REVISION = "c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd"
TASKS = ("stt", "dt", "at")
