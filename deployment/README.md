# VLX-Go deployment

English | [Chinese](README_zh.md)

## 1. Install

Use Python 3.10 or later for the server and a PyTorch build suitable for your device. Install PyTorch/torchvision for your CUDA version first when necessary. CPU inference is also supported. Habitat is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/requirements.txt
```

## 2. Prepare model assets

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

DINOv3 requires requesting access on its model page. Once approved, authenticate with `hf auth login` before downloading.

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

Download the **entire snapshot**, including any additional tokenizer files or weight shards. The loader uses `AutoModel.from_pretrained(..., trust_remote_code=True)`; use model code you have reviewed. The planner snapshot already includes the fine-tuned language backbone, projector and waypoint head. Training `.pt` files are not supported. The export's BF16 language backbone and FP32 planner components retain their original dtypes.

## 3. Start

```bash
bash deployment/run_engine_server.sh
```

For assets stored elsewhere:

```bash
HF_MODEL_DIR=/path/to/vlxgo \
DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m \
SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384 \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/run_engine_server.sh
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `HF_MODEL_DIR` | `resources/vlxgo` under the repository | Complete planner snapshot; takes precedence over `HF_MODEL_ID` |
| `HF_MODEL_ID` | unset | Alternative Hugging Face repository ID |
| `HF_MODEL_REVISION` | `main` | Planner/tokenizer revision |
| `HF_LOCAL_FILES_ONLY` | `1` | Set `0` to explicitly allow Hub access |
| `DINOV3_MODEL_PATH` | `resources/dinov3-vits16-pretrain-lvd1689m` | DINOv3 directory or Hub ID |
| `SIGLIP_MODEL_PATH` | `resources/siglip-so400m-patch14-384` | SigLIP directory or Hub ID |
| `PYTHON_BIN` | `python` | Active environment's interpreter |
| `HOST` / `PORT` | `0.0.0.0` / `8233` | Bind address and port |
| `CUDA_VISIBLE_DEVICES` | `0` | Visible GPUs |
| `VLXGO_DEVICE` | CUDA when available, otherwise CPU | Torch device, e.g. `cuda:0` or `cpu` |
| `VLXGO_ATTN_IMPL` | `auto` | `auto`, `sdpa`, `eager`, `flash_attention_2`, or `none` |
| `VLXGO_ATTN_STRICT` | unset | `1` disables attention fallback |
| `VLXGO_ALLOW_TF32` | `1` | Enable TF32 on supported CUDA hardware |
| `VLXGO_MATMUL_PRECISION` | `high` | PyTorch float32 matmul precision |

## 4. Use the HTTP API

The server starts accepting requests after the planner and both encoders load.

```bash
curl --fail http://127.0.0.1:8233/engine/v1/health/ping
```

The health response is `{"is_alive": true}`. Send observations in order, using one session `id` per episode. The default instruction is `follow the person`. `image` accepts base64-encoded image bytes, a data URL, or a server-visible path.

This example sends an actual image without mounting its directory into a container:

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

The response includes:

- `waypoints`: eight `[x, y, yaw]` waypoints for the released model; positions are in meters and yaw is in radians, in the model's local motion frame.
- `action`: the second waypoint divided by `0.1` seconds, returning `[vx, vy, yaw_rate]`. A single-waypoint model uses its first waypoint.
- `history_length`: number of cached past observations after this successful request, capped at the model's configured history length.
- `latency_ms`, `lock_wait_ms`, `compute_ms`: timings; image decoding is excluded from `latency_ms`.
- `id` and `model_mode`: the session ID and `"text"`.

RGB inputs are resized to 384×384. The current frame contributes 64 fine tokens; each historical frame contributes four coarse tokens. Only previous frames enter the 31-slot history. Missing slots are zero-filled and masked. The current frame is appended only after a finite prediction succeeds. Requests share a model lock; histories remain independent by session ID.

Clear history when an episode ends or before reusing its ID:

```bash
curl --fail -X POST http://127.0.0.1:8233/engine/clear_cache \
  -H 'Content-Type: application/json' -d '{"id":"episode_001"}'
```

The result includes `cleared`, indicating whether the session existed. Invalid images return 400, schema errors return 422, and inference failures return 500. Connect the returned motion command to your robot controller's coordinate and execution interface; this repository supplies the inference service, not a robot SDK driver.

## 5. Docker

Install Docker Engine, Compose v2 and NVIDIA Container Toolkit. Use an NVIDIA driver and PyTorch base image compatible with the target hardware.

```bash
cd deployment
cp .env.example .env
```

Edit `.env` to point to three existing model directories. Relative paths are resolved from `deployment/`; the defaults are `../resources/...`.

```dotenv
VLXGO_MODEL_DIR=/path/to/vlxgo
DINOV3_MODEL_DIR=/path/to/dinov3-vits16-pretrain-lvd1689m
SIGLIP_MODEL_DIR=/path/to/siglip-so400m-patch14-384
GPU_ID=0
PORT=8233
VLXGO_ATTN_IMPL=sdpa
```

All model mounts are read-only. Missing source directories fail instead of being created. `GPU_ID` selects a host GPU ID or UUID; the container sees it as `cuda:0`. `PORT` changes the host port; the internal service and health check stay on 8233. Shell environment variables take precedence over `.env` values.

The default `BASE_IMAGE` is `nvcr.io/nvidia/pytorch:25.06-py3`. Override it for your platform using the same Dockerfile. The image must contain compatible PyTorch and torchvision; installation preserves those versions. Weights, benchmark assets and Habitat are excluded from the image. Inference is offline and custom-code caches are writable under `/tmp/vlxgo-hf-cache`.

```bash
docker compose config --quiet
docker compose build vlxgo-server
docker compose up -d vlxgo-server
docker compose ps
docker compose logs -f vlxgo-server
```

Wait for the service to become healthy, then use the HTTP examples above. `config --quiet` validates substitutions and syntax only; it does not establish that the image can build, the GPU works, or inference has succeeded.

```bash
docker compose logs --tail=100 vlxgo-server
docker compose stop vlxgo-server
docker compose down
```

After code or dependency changes, rebuild and restart with `docker compose up -d --build vlxgo-server`.

## Troubleshooting

- Missing files or incomplete model loading: download the entire HF snapshot, including shards, tokenizer and custom Python modules.
- DINOv3 download denied: request access on the model page and, once approved, authenticate.
- No GPU: check the host driver and container GPU allocation; use `--device cpu` for a CPU inference run.
- FlashAttention import or loading errors: use `--attn-impl sdpa` with a compatible PyTorch installation.
- Image path not found in Docker: send base64 image bytes or mount the image path.
- Unexpected history: use a fresh session ID or call `/engine/clear_cache`.
