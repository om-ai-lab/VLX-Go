#!/usr/bin/env python3
"""HTTP inference service for the text-conditioned vlxgo waypoint model."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from vlxgo.attention import configure_torch_inference_backends

from vlxgo.runtime import EngineRuntime

logger = logging.getLogger("vlxgo.server")


class Heartbeat(BaseModel):
    is_alive: bool


class QueryBody(BaseModel):
    id: str = Field(default="default", description="Session ID used for the history cache.")
    image: str
    instruction: Optional[str] = None


class ClearCacheBody(BaseModel):
    id: str


class ActResponse(BaseModel):
    """Inference timing covers lock waiting and computation, excluding image decoding."""
    id: str
    action: list[float]
    waypoints: list[list[float]]
    history_length: int
    latency_ms: float
    lock_wait_ms: float = 0.0
    compute_ms: float = 0.0
    model_mode: str


class ClearCacheResponse(BaseModel):
    id: str
    cleared: bool




def _read_image_bytes(payload: str) -> bytes:
    """Accept base64, a data URL, or an image path visible to the server."""
    if not payload:
        raise ValueError("image payload is empty")
    if payload.startswith("data:image") and "," in payload:
        payload = payload.split(",", 1)[1]
    if len(payload) < 4096:
        try:
            path = Path(payload)
            if path.is_file():
                return path.read_bytes()
        except (OSError, ValueError):
            pass  # Base64 text can exceed the filesystem's filename length limit.
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image must be base64, a data URL, or an existing local image path") from exc


def _decode_rgb_image(payload: str) -> np.ndarray:
    raw = _read_image_bytes(payload)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"failed to decode image: {exc}") from exc




def create_app(runtime: EngineRuntime) -> FastAPI:
    app = FastAPI(title="vlxgo", description="Text-conditioned waypoint inference", version="0.1.0")

    @app.get("/engine/v1/health/ping", response_model=Heartbeat)
    def ping() -> Heartbeat:
        return Heartbeat(is_alive=runtime.is_alive())

    def run_act(body: QueryBody) -> ActResponse:
        try:
            rgb = _decode_rgb_image(body.image)
            return ActResponse(**runtime.infer(body.id, rgb, body.instruction))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failed for session %s", body.id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/engine/act", response_model=ActResponse)
    def act(body: QueryBody) -> ActResponse:
        return run_act(body)

    @app.post("/engine/clear_cache", response_model=ClearCacheResponse)
    def clear_cache(body: ClearCacheBody) -> ClearCacheResponse:
        return ClearCacheResponse(id=body.id, cleared=runtime.clear_cache(body.id))

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vlxgo Hugging Face HTTP inference server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8233")))
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--model-dir", help="Hugging Face export directory or Hub model ID")
    sources.add_argument("--model-id", dest="model_dir", help="Hugging Face Hub model ID")
    parser.add_argument("--device", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--attn-impl", choices=("auto", "eager", "sdpa", "flash_attention_2", "none"))
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    if args.device:
        os.environ["VLXGO_DEVICE"] = args.device
    if args.attn_impl:
        os.environ["VLXGO_ATTN_IMPL"] = args.attn_impl
    configure_torch_inference_backends()
    runtime = EngineRuntime(args.model_dir)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
