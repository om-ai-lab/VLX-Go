"""Shared temporal inference for robot deployment and Habitat, independent of HTTP and the simulator."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .agent import VlxgoAgent

logger = logging.getLogger("vlxgo.runtime")
DEFAULT_INSTRUCTION = "follow the person"

@dataclass
class SessionState:
    history_size: int
    coarse_hist_tokens: deque[torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self.coarse_hist_tokens = deque(maxlen=self.history_size)


class EngineRuntime:
    """Serialize model access and maintain independent histories per session."""

    def __init__(self, model_name_or_path: Optional[str] = None, *, agent=None) -> None:
        self._lock = threading.RLock()
        self._agent = agent if agent is not None else VlxgoAgent(model_name_or_path)
        self._sessions: dict[str, SessionState] = {}

    def is_alive(self) -> bool:
        return self._agent.planner_model is not None

    def clear_cache(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def infer(
        self, session_id: str, current_rgb: np.ndarray, instruction: Optional[str] = None
    ) -> dict[str, object]:
        start = time.perf_counter()
        with self._lock:
            lock_wait_ms = (time.perf_counter() - start) * 1000.0
            compute_start = time.perf_counter()
            session = self._sessions.setdefault(session_id, SessionState(self._agent.history))
            waypoints = self._predict_waypoints(session, current_rgb, instruction)
            action = self._waypoints_to_action(waypoints)
            history_length = len(session.coarse_hist_tokens)
            if self._agent.planner_device.type == "cuda":
                torch.cuda.synchronize(self._agent.planner_device)
            compute_ms = (time.perf_counter() - compute_start) * 1000.0
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Inference session=%s compute_ms=%.2f latency_ms=%.2f", session_id, compute_ms, latency_ms)
        return {
            "id": session_id, "action": action, "waypoints": waypoints.tolist(),
            "history_length": history_length, "latency_ms": latency_ms,
            "lock_wait_ms": lock_wait_ms, "compute_ms": compute_ms, "model_mode": "text",
        }

    def _predict_waypoints(
        self, session: SessionState, current_rgb: np.ndarray, instruction: Optional[str]
    ) -> np.ndarray:
        coarse, fine = self._agent.encode_frame_tokens(current_rgb)
        history_tokens, history_indices, valid_mask = self._build_history_inputs(session, coarse)
        fine_batch = fine.to(self._agent.planner_device).unsqueeze(0)
        fine_indices = torch.full(
            (1, fine.size(0)), self._agent.history, dtype=torch.long,
            device=self._agent.planner_device,
        )
        with torch.inference_mode():
            prediction = self._agent.planner_model(
                history_tokens, history_indices, fine_batch, fine_indices,
                [instruction or DEFAULT_INSTRUCTION], history_valid_mask=valid_mask,
            )
        waypoints = prediction.detach().float().cpu().numpy()
        expected = (1, self._agent.planner_model.config.n_waypoints, 3)
        if waypoints.shape != expected or not np.isfinite(waypoints).all():
            raise RuntimeError(f"Invalid vlxgo waypoint output: expected finite values with shape {expected}")
        # Current coarse features become history only after a successful prediction.
        session.coarse_hist_tokens.append(coarse.detach().cpu())
        return waypoints[0]

    def _build_history_inputs(
        self, session: SessionState, current_coarse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-align past frames and mask zero-filled missing slots."""
        length = self._agent.history
        reference = current_coarse.to(self._agent.planner_device)
        frames = list(session.coarse_hist_tokens)[-length:] if length else []
        missing = length - len(frames)
        packed = reference.new_zeros((length, *reference.shape))
        if frames:
            packed[missing:] = torch.stack([frame.to(reference) for frame in frames])
        slots = torch.arange(length, device=self._agent.planner_device)
        return (
            packed.flatten(0, 1).unsqueeze(0),
            slots.repeat_interleave(reference.size(0)).unsqueeze(0),
            (slots >= missing).unsqueeze(0),
        )

    @staticmethod
    def _waypoints_to_action(waypoints: np.ndarray) -> list[float]:
        """Convert the second [x,y,yaw] waypoint to a velocity over 0.1 seconds."""
        index = 1 if waypoints.shape[0] > 1 else 0
        return (waypoints[index] / 0.1).tolist()
