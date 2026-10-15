# VLX-Go 实机推理部署

[English](README.md) | 中文

## 1. 安装环境

服务推荐使用 Python 3.10 或更新版本，先按硬件安装匹配的 PyTorch 和 torchvision，再安装服务依赖。也支持 CPU 推理，不需要 Habitat。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/requirements.txt
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

DINOv3 需先在模型页面申请访问，获批后通过 `hf auth login` 登录再下载。

```text
resources/
├── vlxgo/
│   ├── config.json
│   ├── model.safetensors
│   ├── configuration_vlxgo.py
│   ├── modeling_vlxgo.py
│   ├── history_tokens.py
│   ├── multimodal_inputs.py
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── dinov3-vits16-pretrain-lvd1689m/
└── siglip-so400m-patch14-384/
```

请下载**完整 snapshot**，包括可能存在的其他 tokenizer 文件和权重分片。模型通过 `AutoModel.from_pretrained(..., trust_remote_code=True)` 加载，请使用已审阅的模型代码。导出已包含微调后的语言 backbone、projector 和 waypoint head。不支持训练 `.pt` 文件。加载保留语言 backbone 的 BF16 和规划组件的 FP32 混合精度。

## 3. 启动服务

```bash
bash deployment/run_engine_server.sh
```

使用其他位置的模型：

```bash
HF_MODEL_DIR=/path/to/vlxgo \
DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m \
SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384 \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/run_engine_server.sh
```

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `HF_MODEL_DIR` | 仓库下 `resources/vlxgo` | 完整模型目录，优先于 `HF_MODEL_ID` |
| `HF_MODEL_ID` | 未设置 | 备选 Hugging Face 仓库名 |
| `HF_MODEL_REVISION` | `main` | 模型和 tokenizer 版本 |
| `HF_LOCAL_FILES_ONLY` | `1` | 显式设为 `0` 才允许访问 Hub |
| `DINOV3_MODEL_PATH` | `resources/dinov3-vits16-pretrain-lvd1689m` | DINOv3 目录或 Hub ID |
| `SIGLIP_MODEL_PATH` | `resources/siglip-so400m-patch14-384` | SigLIP 目录或 Hub ID |
| `PYTHON_BIN` | `python` | Python 解释器 |
| `HOST` / `PORT` | `0.0.0.0` / `8233` | 监听地址和端口 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU |
| `VLXGO_DEVICE` | 有 CUDA 时使用 CUDA，否则 CPU | 如 `cuda:0` 或 `cpu` |
| `VLXGO_ATTN_IMPL` | `auto` | `auto`、`sdpa`、`eager`、`flash_attention_2` 或 `none` |
| `VLXGO_ATTN_STRICT` | 未设置 | `1` 表示 attention 出错时不回退 |
| `VLXGO_ALLOW_TF32` | `1` | 在支持的 CUDA 硬件上启用 TF32 |
| `VLXGO_MATMUL_PRECISION` | `high` | PyTorch float32 矩阵乘精度 |

## 4. HTTP API

模型及两个编码器加载成功后，服务开始接受请求：

```bash
curl --fail http://127.0.0.1:8233/engine/v1/health/ping
```

健康响应为 `{"is_alive": true}`。每个 episode 使用独立 `id`，按观察顺序发送图片。默认指令为 `follow the person`。`image` 支持图片字节的 base64、data URL 或服务端可见路径。

以下示例发送真实图片，无需将图片目录挂载到容器：

```bash
python - /path/to/current-rgb.jpg 8233 <<'PY'
import base64
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

image_path, port = sys.argv[1:]
body = {
    "id": "episode_001",
    "image": base64.b64encode(Path(image_path).read_bytes()).decode("ascii"),
    "instruction": "Follow the person",
}
request = Request(
    f"http://127.0.0.1:{port}/engine/act",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

返回字段：

- `waypoints`：当前导出的 8 个 `[x, y, yaw]` 局部 waypoint，位置单位为米，角度单位为弧度。
- `action`：第二个 waypoint 除以 `0.1` 秒，得到 `[vx, vy, yaw_rate]`；若模型只有一个 waypoint 则使用第一个。
- `history_length`：本次成功请求后缓存的帧数，上限由模型的 history 配置决定。
- `latency_ms`、`lock_wait_ms`、`compute_ms`：耗时；`latency_ms` 不包含图片解码。
- `id` 和 `model_mode`：会话 ID 与 `"text"`。

RGB 输入缩放到 384×384；当前帧产生 64 个 fine token，每个历史帧产生 4 个 coarse token。31 个历史槽只包含过去帧，缺失槽补零并使用 mask 排除。只有预测成功且结果有限时，当前帧才进入历史。请求共用模型锁，不同 session 的历史互相独立。

episode 结束或复用 ID 前清空历史：

```bash
curl --fail -X POST http://127.0.0.1:8233/engine/clear_cache \
  -H 'Content-Type: application/json' -d '{"id":"episode_001"}'
```

返回的 `cleared` 表示该会话是否存在。无效图片返回 400，schema 错误返回 422，推理失败返回 500。对接机器人时按底层控制器的坐标约定和执行接口转换动作；这里提供推理服务，不包含机器人 SDK 驱动。

## 5. Docker 部署

安装 Docker Engine、Compose v2 和 NVIDIA Container Toolkit，确保主机驱动与基础镜像兼容。

```bash
cd deployment
cp .env.example .env
```

编辑 `.env`，填写三个已存在的模型目录。相对路径以 `deployment/` 为基准，默认使用 `../resources/...`。

```dotenv
VLXGO_MODEL_DIR=/path/to/vlxgo
DINOV3_MODEL_DIR=/path/to/dinov3-vits16-pretrain-lvd1689m
SIGLIP_MODEL_DIR=/path/to/siglip-so400m-patch14-384
GPU_ID=0
PORT=8233
VLXGO_ATTN_IMPL=sdpa
```

三个模型分别只读挂载；源目录缺失时直接失败，不会创建空目录。`GPU_ID` 可填写主机 GPU 编号或 UUID，容器内使用 `cuda:0`。`PORT` 只调整主机端口，容器内服务及健康检查保持 8233。Shell 环境变量优先于 `.env`。

默认基础镜像为 `nvcr.io/nvidia/pytorch:25.06-py3`，可通过 `BASE_IMAGE` 切换适配硬件的镜像。镜像必须自带兼容的 PyTorch 和 torchvision，依赖安装会保留其版本。镜像不包含模型权重、评测数据和 Habitat；推理离线运行，自定义代码缓存写入 `/tmp/vlxgo-hf-cache`。

```bash
docker compose config --quiet
docker compose build vlxgo-server
docker compose up -d vlxgo-server
docker compose ps
docker compose logs -f vlxgo-server
```

等待服务 healthy 后，再发送上面的 HTTP 请求。`config --quiet` 只检查配置语法与变量替换，不代表镜像构建、GPU 或推理已经验证。

```bash
docker compose logs --tail=100 vlxgo-server
docker compose stop vlxgo-server
docker compose down
```

修改代码或依赖后，用 `docker compose up -d --build vlxgo-server` 重建并启动。

## 常见问题

- 模型文件缺失或加载不完整：重新下载完整 HF snapshot，包括分片、tokenizer 和自定义 Python 模块。
- DINOv3 下载被拒绝：先在模型页面申请访问，获批后登录。
- GPU 不可用：检查主机驱动及容器 GPU 分配；可通过 `--device cpu` 运行 CPU 推理。
- FlashAttention 导入或加载失败：使用 `--attn-impl sdpa`，核对 PyTorch 环境。
- Docker 中找不到图片路径：发送 base64，或显式挂载对应图片目录。
- 历史帧不符合预期：使用新的 session ID，或调用 `/engine/clear_cache`。
