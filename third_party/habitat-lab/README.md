# Habitat-Lab for VLX-Go evaluation

This directory contains the Habitat-Lab 0.3.1 source used by the VLX-Go tracking
evaluator, including the required TrackEnv and tracking-task adaptations. It is
installed locally; an unmodified upstream package is not interchangeable with it.
No Habitat-Sim binaries, model weights, robot assets, scene datasets or training
outputs are included.

Use the [English evaluation guide](../../evaluation/README.md) or
[Chinese evaluation guide](../../evaluation/README_zh.md) for installation and data preparation.

Upstream Habitat-Lab: <https://github.com/facebookresearch/habitat-lab/tree/v0.3.1>.
Tracking benchmark: <https://github.com/wsakobe/TrackVLA/tree/c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd>.

Original Habitat-Lab files retain their [MIT license](LICENSE). TrackVLA-derived
tracking additions retain the upstream CC BY-NC-SA 4.0 terms.

The VLX-Go integration keeps the existing simulator behavior and stores
`humanoid_infos.json` beside `habitat/tasks/track/track_task.py`, resolved relative
to that module instead of the caller's working directory. Packaging includes this
metadata, configuration files and runtime visualization assets. Caches, external
data and installation artifacts are excluded.
