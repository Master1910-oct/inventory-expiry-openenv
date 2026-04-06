"""
Inventory & Expiry Management Environment — Python Client
=========================================================
Thin HTTP client that wraps the FastAPI server.
Mirrors the OpenEnv EnvClient interface: reset(), step(), state(), close().
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from models import InventoryAction, InventoryObservation, InventoryState


@dataclass
class StepResult:
    observation: InventoryObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class InventoryEnv:
    """
    Synchronous HTTP client for the Inventory & Expiry Management environment.

    Usage
    -----
    env = InventoryEnv(base_url="http://localhost:7860")
    result = env.reset(task_name="easy_expiry_check")
    result = env.step(InventoryAction(action_type="list_items"))
    state  = env.state()
    env.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ):
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ── core API ──────────────────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> StepResult:
        payload = {} if task_name is None else {"task_name": task_name}
        resp = self._client.post(f"{self._base}/reset", json=payload)
        resp.raise_for_status()
        return self._parse(resp.json())

    def step(self, action: InventoryAction) -> StepResult:
        resp = self._client.post(f"{self._base}/step", json=action.dict(exclude_none=True))
        resp.raise_for_status()
        return self._parse(resp.json())

    def state(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self._base}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse(payload: Dict) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = InventoryObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )


# ---------------------------------------------------------------------------
# Async variant (for use in async RL training loops)
# ---------------------------------------------------------------------------

class AsyncInventoryEnv:
    """Async HTTP client."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(self, task_name: Optional[str] = None) -> StepResult:
        payload = {} if task_name is None else {"task_name": task_name}
        resp = await self._client.post(f"{self._base}/reset", json=payload)
        resp.raise_for_status()
        return InventoryEnv._parse(resp.json())

    async def step(self, action: InventoryAction) -> StepResult:
        resp = await self._client.post(f"{self._base}/step", json=action.dict(exclude_none=True))
        resp.raise_for_status()
        return InventoryEnv._parse(resp.json())

    async def state(self) -> Dict[str, Any]:
        resp = await self._client.get(f"{self._base}/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
