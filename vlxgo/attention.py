from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Optional

import torch


logger = logging.getLogger("vlxgo.attention")

_FLASH_ATTN_IMPORT_ERROR: Optional[str] = None


def configure_torch_inference_backends() -> None:
    """Enable CUDA backend options that improve inference throughput."""

    if not torch.cuda.is_available():
        return

    if os.environ.get("VLXGO_ALLOW_TF32", "1").strip().lower() not in {"0", "false", "no", "off"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    precision = os.environ.get("VLXGO_MATMUL_PRECISION", "high").strip()
    if precision and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(precision)
        except ValueError:
            logger.warning("Invalid VLXGO_MATMUL_PRECISION=%s", precision)


def flash_attn_available() -> bool:
    global _FLASH_ATTN_IMPORT_ERROR
    try:
        importlib.import_module("flash_attn")
        _FLASH_ATTN_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _FLASH_ATTN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_attn_impl(value: Optional[str]) -> str:
    if value is None:
        return "auto"
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "flash": "flash_attention_2",
        "flash2": "flash_attention_2",
        "flash_attn": "flash_attention_2",
        "flash_attn_2": "flash_attention_2",
        "fa2": "flash_attention_2",
        "default": "none",
        "disabled": "none",
        "off": "none",
    }
    return aliases.get(normalized, normalized)


def resolve_transformers_attn_impl(
    model_label: str, *, device: Optional[torch.device] = None
) -> Optional[str]:
    requested = _normalize_attn_impl(
        os.environ.get("VLXGO_ATTN_IMPL") or os.environ.get("ATTN_IMPLEMENTATION")
    )

    use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
    if requested in {"", "auto"}:
        if use_cuda and flash_attn_available():
            return "flash_attention_2"
        if use_cuda:
            return "sdpa"
        return None

    if requested in {"none", "eager", "sdpa"}:
        return None if requested == "none" else requested

    if requested == "flash_attention_2":
        if use_cuda and flash_attn_available():
            return "flash_attention_2"
        message = (
            f"{model_label}: flash_attention_2 requires CUDA and flash_attn on the selected device"
        )
        if _FLASH_ATTN_IMPORT_ERROR:
            message = f"{message} ({_FLASH_ATTN_IMPORT_ERROR})"
        if _env_truthy("VLXGO_ATTN_STRICT"):
            raise RuntimeError(message)
        logger.warning("%s; falling back to sdpa/default attention", message)
        return "sdpa" if use_cuda else None

    logger.warning("%s: unsupported VLXGO_ATTN_IMPL=%s; using default attention", model_label, requested)
    return None


def configure_model_attention(model: Any, *, model_label: str) -> None:
    """Set attention on the loaded language backbone without casting its weights."""
    attn_impl = resolve_transformers_attn_impl(model_label, device=next(model.parameters()).device)
    if attn_impl is None:
        return
    try:
        model.set_attn_implementation(attn_impl)
        _announce_attention(model_label, attn_impl)
    except Exception as exc:
        if _env_truthy("VLXGO_ATTN_STRICT"):
            raise
        logger.warning("%s: attention %s unavailable (%s); using sdpa", model_label, attn_impl, exc)
        model.set_attn_implementation("sdpa")


def from_pretrained_with_attention(
    model_cls: Any,
    model_name_or_path: str,
    *,
    model_label: str,
    flash_torch_dtype: Optional[torch.dtype] = None,
    target_device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Any:
    configure_torch_inference_backends()

    attn_impl = resolve_transformers_attn_impl(model_label, device=target_device)
    load_kwargs = dict(kwargs)
    _prefer_dtype_kwarg(load_kwargs)
    if (
        attn_impl == "flash_attention_2"
        and flash_torch_dtype is not None
        and "torch_dtype" not in load_kwargs
        and "dtype" not in load_kwargs
    ):
        load_kwargs["dtype"] = flash_torch_dtype

    if not attn_impl:
        return _call_from_pretrained(model_cls, model_name_or_path, load_kwargs)

    try:
        _announce_attention(model_label, attn_impl)
        return _call_from_pretrained(
            model_cls,
            model_name_or_path,
            {**load_kwargs, "attn_implementation": attn_impl},
        )
    except Exception as exc:
        if _env_truthy("VLXGO_ATTN_STRICT"):
            raise
        if not _is_attention_load_error(exc):
            raise
        logger.warning(
            "%s: attn_implementation=%s failed (%s: %s); retrying with default attention",
            model_label,
            attn_impl,
            type(exc).__name__,
            exc,
        )
        fallback_kwargs = dict(kwargs)
        _prefer_dtype_kwarg(fallback_kwargs)
        return _call_from_pretrained(model_cls, model_name_or_path, fallback_kwargs)


def _announce_attention(model_label: str, attn_impl: str) -> None:
    message = f"[Attention] {model_label}: using attn_implementation={attn_impl}"
    logger.info(message)
    if os.environ.get("VLXGO_ATTENTION_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}:
        print(message)


def _prefer_dtype_kwarg(load_kwargs: dict[str, Any]) -> None:
    if "torch_dtype" in load_kwargs and "dtype" not in load_kwargs:
        load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")


def _call_from_pretrained(model_cls: Any, model_name_or_path: str, load_kwargs: dict[str, Any]) -> Any:
    try:
        return model_cls.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in load_kwargs or not _is_dtype_kwarg_error(exc):
            raise
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        return model_cls.from_pretrained(model_name_or_path, **retry_kwargs)


def _is_dtype_kwarg_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "unexpected keyword" in text and "dtype" in text


def _is_attention_load_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "attn_implementation",
        "flash_attention",
        "flash attention",
        "flash_attn",
        "sdpa",
        "scaled_dot_product_attention",
    )
    return any(marker in text for marker in markers)
