# EVT-Bench 评测

[English](README.md) | 中文

## 1. 创建评测环境

Habitat 渲染需要 Linux 和可用的 NVIDIA 驱动。请根据驱动安装对应 CUDA 版本的 PyTorch。

```bash
conda env create -f evaluation/environment.yml
conda activate vlxgo-eval
python -m pip install -r evaluation/requirements.txt
python -m pip install --no-deps -e third_party/habitat-lab
```

## 2. 准备模型

模型仓库 [omlab/VLX-Go-1.0](https://huggingface.co/omlab/VLX-Go-1.0)。

视觉编码器的官方下载地址：

- DINOv3：[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)。
- SigLIP：[google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)。

```bash
hf download omlab/VLX-Go-1.0 --local-dir resources/vlxgo
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir resources/dinov3-vits16-pretrain-lvd1689m
hf download google/siglip-so400m-patch14-384 \
  --local-dir resources/siglip-so400m-patch14-384
```

下载 DINOv3 前，先在模型页面申请访问，获批后执行 `hf auth login`。VLX-Go 必须下载完整 snapshot，包括权重或分片、tokenizer、配置和自定义 Python 模块，目录示例见[部署指南](../deployment/README_zh.md#2-准备模型)。

使用其他位置的模型时：

```bash
export HF_MODEL_DIR=/path/to/vlxgo
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m
export SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384
```

不设置时使用仓库 `resources/` 下的三个目录。默认只读取本地文件。如需 Hub 加载，可指定 `HF_MODEL_ID`、`HF_MODEL_REVISION` 和 `HF_LOCAL_FILES_ONLY=0`；选择 `HF_MODEL_ID` 前应取消 `HF_MODEL_DIR`。模型使用 `trust_remote_code=True` 加载自定义代码。snapshot 已包含语言 backbone，运行时保留混合精度。评测默认使用 SDPA，可用 `VLXGO_ATTN_IMPL` 覆盖。

## 3. 准备仿真资产

按官方 [Habitat-Sim 数据说明](https://github.com/facebookresearch/habitat-sim/blob/v0.3.1/DATASETS.md)下载 HM3D 验证场景和 Matterport3D，保留完整场景目录及关联文件。按照 [TrackVLA 资产说明](https://github.com/wsakobe/TrackVLA/tree/c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd#-installation)下载 humanoid 数据，也可使用其 [humanoids.zip 链接](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view)，解压到 `data/`，形成下方目录结构。

```text
data/
├── datasets/track/                  # 随仓库提供
│   ├── STT/val/val.json.gz
│   ├── DT/val/val.json.gz
│   └── AT/val/val.json.gz
├── scene_datasets/                 # 单独下载
│   ├── hm3d/val/<scene>/<scene>.basis.glb
│   └── mp3d/<scene>/<scene>.glb
├── humanoids/humanoid_data/<avatar>/
│   ├── <avatar>.urdf
│   ├── <avatar>_motion_data_smplx.pkl
│   └── ...                         # 网格和纹理
└── robots/hab_spot_arm/
    ├── urdf/hab_spot_arm.urdf
    └── ...                         # 网格及其他关联文件
```

已有完整资产可通过软链接放到 `data/scene_datasets`、`data/humanoids` 和 `data/robots`。

## 4. 先运行单个 episode

```bash
EVAL_OUT_DIR=results/smoke GPU_IDS=0 \
  bash eval.sh --task stt --max-episodes 1 --no-save-video
```

## 5. 运行完整评测

```bash
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task stt
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task dt
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task at
```

多 GPU 运行全部任务：

```bash
EVAL_OUT_DIR=results/vlxgo \
  bash eval.sh --task all --gpu-ids 0,1 --jobs-per-gpu 1 --no-save-video
```

| 配置／参数 | 默认值 | 作用 |
| --- | --- | --- |
| `TASKS` / `--task` | `stt dt at` | 单任务、任务列表或 `all` |
| `GPU_IDS` / `--gpu-ids` | 未设置 | 物理 GPU 编号，逗号或空格分隔 |
| `NUM_PARALLEL` / `--num-parallel` | `1` | 未指定列表时使用 GPU `0..N-1` |
| `JOBS_PER_GPU` / `--jobs-per-gpu` | `1` | 每张 GPU 的 worker 数上限 |
| `EVAL_OUT_DIR` | 仓库下 `results/` | 结果父目录 |
| `SAVE_SUFFIX` | 未设置 | 可选任务目录后缀，形成 `<task>_<suffix>` |
| `RESUME` / `--resume` / `--no-resume` | `1` | 跳过已有有效最终结果 |
| `SAVE_VIDEO` | `1` | 保存第一视角及已配置侧视角视频 |
| `SAVE_THIRD_PERSON` | `1` | 保存第三视角视频 |
| `--no-save-video` | 不启用 | 同时关闭两组视频 |
| `PYTHON_BIN` | `python` | `eval.sh` 使用的解释器 |
| `--max-episodes N` | 未设置 | 每个任务最多运行 N 个 episode |
| `--check-assets` | 不启用 | 只检查前置条件，不运行 rollout |

## 6. 续评与结果文件

再次执行同一命令即可续评。有效完成结果会被跳过，包括丢失目标或碰撞结束的 episode。中断的 episode 从头运行；视频缺失不影响有效最终 JSON 的完成判断。强制重跑：

```bash
EVAL_OUT_DIR=results/vlxgo GPU_IDS=0 bash eval.sh --task stt --no-resume
```

**更换 checkpoint、任务配置或 episode 子集时，请使用新输出目录。** 续评不比较模型或配置指纹，也不要让两个独立启动器同时写入同一个任务目录。

```text
results/vlxgo/stt/
├── logs/split_000.log
├── config_split000_info.json
└── <scene>/
    ├── <episode>.json              # 原子写入的最终指标
    ├── <episode>_info.json         # 每步记录
    ├── <episode>.mp4               # 可选第一视角及轨迹叠加
    └── <episode>_third_person.mp4  # 可选第三视角
```

最终 JSON 包含 `finish`、`status`、`success`、`collision`、`following_step`、`total_step`、`following_rate`，以及存在时的 episode 指令。

## 7. 计算 SR／TR／CR

每个任务单独评分，不要传入同时包含三个任务的父目录：

```bash
python -m evaluation.metrics --result-dir results/vlxgo/stt \
  --json-output results/vlxgo/stt/metrics_info.json
python -m evaluation.metrics --result-dir results/vlxgo/dt
python -m evaluation.metrics --result-dir results/vlxgo/at
```
