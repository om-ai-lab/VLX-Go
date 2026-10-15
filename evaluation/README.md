# EVT-Bench evaluation

English | [Chinese](README_zh.md)

## 1. Create the evaluation environment

Use Linux with a working NVIDIA driver for Habitat rendering. Install a matching CUDA build of PyTorch for your driver.

```bash
conda env create -f evaluation/environment.yml
conda activate vlxgo-eval
python -m pip install -r evaluation/requirements.txt
python -m pip install --no-deps -e third_party/habitat-lab
```

## 2. Prepare the model

The model repository [omlab/VLX-Go-1.0](https://huggingface.co/omlab/VLX-Go-1.0).

Official vision encoder downloads:

- DINOv3: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
- SigLIP: [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384).

```bash
hf download omlab/VLX-Go-1.0 --local-dir resources/vlxgo
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir resources/dinov3-vits16-pretrain-lvd1689m
hf download google/siglip-so400m-patch14-384 \
  --local-dir resources/siglip-so400m-patch14-384
```

Request DINOv3 access on its model page and, once approved, run `hf auth login` before downloading. Download the full VLX-Go snapshot, including weights/shards, tokenizer, configuration and custom Python modules. See the [model layout](../deployment/README.md#2-prepare-model-assets).

To use existing directories:

```bash
export HF_MODEL_DIR=/path/to/vlxgo
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m
export SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384
```

Without overrides, the three directories under `resources/` are used. Loading defaults to local files only. `HF_MODEL_ID`, `HF_MODEL_REVISION` and `HF_LOCAL_FILES_ONLY=0` allow explicit Hub loading; unset `HF_MODEL_DIR` when selecting `HF_MODEL_ID`. Custom model code is loaded with `trust_remote_code=True`. The snapshot already contains its language backbone. Its mixed precision is preserved. Evaluation defaults to SDPA; `VLXGO_ATTN_IMPL` can override it.

## 3. Prepare simulation assets

Download HM3D validation scenes and Matterport3D according to the official [Habitat-Sim dataset instructions](https://github.com/facebookresearch/habitat-sim/blob/v0.3.1/DATASETS.md). Preserve complete scene folders, including their meshes and associated files. Download the humanoid archive using the [TrackVLA asset instructions](https://github.com/wsakobe/TrackVLA/tree/c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd#-installation) or its [humanoids.zip link](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view). Extract it under `data/` so that the layout below is obtained.

```text
data/
├── datasets/track/                  # included annotations
│   ├── STT/val/val.json.gz
│   ├── DT/val/val.json.gz
│   └── AT/val/val.json.gz
├── scene_datasets/                 # external downloads
│   ├── hm3d/val/<scene>/<scene>.basis.glb
│   └── mp3d/<scene>/<scene>.glb
├── humanoids/humanoid_data/<avatar>/
│   ├── <avatar>.urdf
│   ├── <avatar>_motion_data_smplx.pkl
│   └── ...                         # meshes and textures
└── robots/hab_spot_arm/
    ├── urdf/hab_spot_arm.urdf
    └── ...                         # meshes and other referenced files
```

You may link existing complete asset directories to `data/scene_datasets`, `data/humanoids` and `data/robots`.

## 4. Run a single episode first

```bash
EVAL_OUT_DIR=results/smoke GPU_IDS=0 \
  bash eval.sh --task stt --max-episodes 1 --no-save-video
```

## 5. Run complete evaluations

```bash
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task stt
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task dt
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task at
```

For all three tasks and multiple GPUs:

```bash
EVAL_OUT_DIR=results/vlxgo \
  bash eval.sh --task all --gpu-ids 0,1 --jobs-per-gpu 1 --no-save-video
```

| Setting / flag | Default | Meaning |
| --- | --- | --- |
| `TASKS` / `--task` | `stt dt at` | Task(s), or `all` |
| `GPU_IDS` / `--gpu-ids` | unset | Physical GPU IDs, comma- or space-separated |
| `NUM_PARALLEL` / `--num-parallel` | `1` | Without an explicit list, select GPUs `0..N-1` |
| `JOBS_PER_GPU` / `--jobs-per-gpu` | `1` | Maximum workers per selected GPU |
| `EVAL_OUT_DIR` | repository `results/` | Parent output directory |
| `SAVE_SUFFIX` | unset | Optional suffix: `<task>_<suffix>` |
| `RESUME` / `--resume` / `--no-resume` | `1` | Skip valid final episode results |
| `SAVE_VIDEO` | `1` | Save first-person and available side-view videos |
| `SAVE_THIRD_PERSON` | `1` | Save third-person videos |
| `--no-save-video` | off | Disable both video groups |
| `PYTHON_BIN` | `python` | Interpreter used by `eval.sh` |
| `--max-episodes N` | unset | Bounded run of up to N episodes per task |
| `--check-assets` | off | Check prerequisites without a rollout |

## 6. Resume and inspect outputs

Run the same command again to resume. Valid completed episodes, including failed tracking or collision outcomes, are skipped. Interrupted episodes restart from the beginning. Missing videos do not invalidate a completed result. To rerun:

```bash
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task stt --no-resume
```

Use a **new output directory when changing checkpoint, task configuration or episode subset**. Resume does not compare model/configuration fingerprints, and independent launchers must not write to the same task output simultaneously.

```text
results/vlxgo/stt/
├── logs/split_000.log
├── config_split000_info.json
└── <scene>/
    ├── <episode>.json              # final metrics; published atomically
    ├── <episode>_info.json         # step records
    ├── <episode>.mp4               # optional first-person video with trajectory
    └── <episode>_third_person.mp4  # optional third-person video
```

The final JSON contains `finish`, `status`, `success`, `collision`, `following_step`, `total_step`, `following_rate`, and the episode instruction when present.

## 7. Calculate SR / TR / CR

Score each task separately; do not pass the parent directory containing all tasks.

```bash
python -m evaluation.metrics --result-dir results/vlxgo/stt \
  --json-output results/vlxgo/stt/metrics_info.json
python -m evaluation.metrics --result-dir results/vlxgo/dt
python -m evaluation.metrics --result-dir results/vlxgo/at
```
