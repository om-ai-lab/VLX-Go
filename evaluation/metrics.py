#!/usr/bin/env python3
"""TrackVLA EVT-Bench SR, step-weighted TR, and CR aggregation."""

import argparse
import json
import math
from pathlib import Path

from .constants import TRACKVLA_REVISION

DEFAULT_REFERENCE_STEPS = Path(__file__).resolve().parent / "assets/track_episode_steps.json"


class MetricsError(ValueError):
    """The inputs cannot produce a valid benchmark score."""


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MetricsError(f"Cannot read {path}: {exc}") from exc


def _number(record, key, path, *, integer=False, binary=False):
    value = record.get(key) if isinstance(record, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricsError(f"{path}: {key} must be finite")
    if value < 0 or (integer and int(value) != value) or (binary and value not in (0, 1)):
        raise MetricsError(f"{path}: invalid {key}={value}")
    return value


def load_reference_steps(path=DEFAULT_REFERENCE_STEPS):
    path = Path(path)
    if not path.exists():
        raise MetricsError(f"Reference step table not found: {path}")
    if path.is_dir():
        files = sorted(path.glob("*/*.json"))
        if not files:
            raise MetricsError(f"Reference step directory is empty: {path}")
        return {str(p.relative_to(path)): int(_number(_read_json(p), "total_step", p, integer=True))
                for p in files}
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("trackvla_revision") != TRACKVLA_REVISION:
        raise MetricsError(f"{path}: reference revision must be {TRACKVLA_REVISION}")
    steps = data.get("episodes")
    if not isinstance(steps, dict) or not steps:
        raise MetricsError(f"{path}: missing reference episodes")
    return {key: int(_number({"total_step": value}, "total_step", f"{path}:{key}", integer=True))
            for key, value in steps.items()}


def load_results(result_dir, reference_steps=DEFAULT_REFERENCE_STEPS):
    """Match official analyze_results.py on valid episode records.

    Reference keys are scene-directory/episode-filename, shared by all tasks.
    Like upstream, a missing individual reference uses the observed step count.
    A missing entire reference table is an error, preventing accidental raw TR.
    """
    root = Path(result_dir)
    if not root.is_dir():
        raise MetricsError(f"Result directory not found: {root}")
    references = load_reference_steps(reference_steps)
    totals = dict(episodes=0, success_count=0, collision_count=0,
                  following_steps=0, actual_steps=0, adjusted_steps=0,
                  reference_matched=0, reference_missing=0)
    seen = set()
    for path in sorted(root.rglob("*.json")):
        if "info" in path.name:
            continue
        record = _read_json(path)
        if not isinstance(record, dict):
            raise MetricsError(f"{path}: expected an episode object")
        key = f"{path.parent.name}/{path.name}"
        if key in seen:
            raise MetricsError(f"Duplicate scene/episode {key}; score each task separately")
        seen.add(key)
        success = _number(record, "success", path, binary=True)
        collision = _number(record, "collision", path, binary=True)
        following = _number(record, "following_step", path, integer=True)
        total = _number(record, "total_step", path, integer=True)
        if total <= 0 or following > total:
            raise MetricsError(f"{path}: require 0 <= following_step <= total_step and total_step > 0")
        reference = references.get(key, total)
        totals["episodes"] += 1
        totals["success_count"] += int(bool(success))
        totals["collision_count"] += int(collision)
        totals["following_steps"] += int(following)
        totals["actual_steps"] += int(total)
        totals["adjusted_steps"] += int(max(reference, total))
        totals["reference_matched" if key in references else "reference_missing"] += 1
    if not totals["episodes"]:
        raise MetricsError(f"No episode result JSONs in {root}")
    return totals


def calculate_metrics(totals):
    return {
        "SR": (totals["success_count"] / totals["episodes"]) * 100.0,
        "TR": (totals["following_steps"] / totals["adjusted_steps"]) * 100.0,
        "CR": (totals["collision_count"] / totals["episodes"]) * 100.0,
    }


def format_output(metrics, totals):
    return "\n".join((
        "TrackVLA EVT-Bench Evaluation Metrics",
        f"Episodes evaluated: {totals['episodes']}",
        f"SR: {metrics['SR']:.4f}%  TR: {metrics['TR']:.4f}%  CR: {metrics['CR']:.4f}%",
        f"TR steps: {totals['following_steps']} / {totals['adjusted_steps']} "
        f"(observed: {totals['actual_steps']})",
        f"Reference matches: {totals['reference_matched']}; "
        f"missing references using observed steps: {totals['reference_missing']}",
        f"Reference revision: {TRACKVLA_REVISION}",
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", "--eval_dir", "--path", dest="result_dir", required=True)
    parser.add_argument("--reference-steps", type=Path, default=DEFAULT_REFERENCE_STEPS,
                        help="Pinned JSON table or official track_episode_step directory")
    parser.add_argument("--output", type=Path, help="Optional text report")
    parser.add_argument("--json-output", type=Path, help="Optional *_info.json report")
    args = parser.parse_args(argv)
    if args.json_output and not args.json_output.name.endswith("_info.json"):
        parser.error("--json-output must end with _info.json so scorers exclude it")
    try:
        totals = load_results(args.result_dir, args.reference_steps)
        metrics = calculate_metrics(totals)
        report = format_output(metrics, totals)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps({
                "trackvla_revision": TRACKVLA_REVISION,
                "reference_steps": str(args.reference_steps.resolve()),
                "metrics_percent": metrics, "totals": totals,
            }, indent=2) + "\n")
        print(f"Evaluation Results: {Path(args.result_dir).resolve()}\n{report}")
    except (MetricsError, OSError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())