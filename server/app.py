"""
Inventory & Expiry Management Environment — FastAPI Server
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import InventoryAction, InventoryObservation
from server.inventory_environment import InventoryEnvironment
from tasks import TASKS, get_task_names

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Module-level — Pydantic v2 cannot resolve ForwardRef when class is inside a function
class ResetRequest(BaseModel):
    task_name: Optional[str] = None


def _obs_dict(obs: InventoryObservation) -> Dict[str, Any]:
    try:
        return obs.model_dump()   # Pydantic v2
    except AttributeError:
        return obs.dict()         # Pydantic v1


def create_app() -> FastAPI:
    app = FastAPI(
        title="Inventory & Expiry Management OpenEnv",
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    env = InventoryEnvironment(task_name=os.getenv("INVENTORY_TASK", "easy_expiry_check"))

    @app.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        log.exception("Error on %s", request.url)
        return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        return {"environment": "Inventory & Expiry Management", "version": "1.0.0",
                "tasks": get_task_names(), "endpoints": ["/reset", "/step", "/state", "/metadata", "/health"]}

    @app.post("/reset")
    def reset(req: ResetRequest):
        try:
            obs = env.reset(task_name=req.task_name)
            return {"observation": _obs_dict(obs), "reward": obs.reward,
                    "done": obs.done, "info": {"episode_id": env.state.episode_id}}
        except Exception as exc:
            log.exception("Error in /reset")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/step")
    def step(action: InventoryAction):
        try:
            obs = env.step(action)
            return {"observation": _obs_dict(obs), "reward": obs.reward, "done": obs.done,
                    "info": {"step": env.state.step_count,
                             "episode_reward": obs.episode_reward_so_far, "error": obs.error}}
        except Exception as exc:
            log.exception("Error in /step")
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/state")
    def state():
        s = env.state
        return {"episode_id": s.episode_id, "step_count": s.step_count, "task_name": s.task_name,
                "task_completed": s.task_completed, "score": s.score, "sim_date": s.sim_date,
                "inventory_size": len(s.inventory), "disposed_count": len(s.disposed_log),
                "orders_count": len(s.orders), "flagged_batches": s.flagged_batches,
                "required_actions_done": s.required_actions_done}

    @app.get("/metadata")
    def metadata():
        import models as _m
        return {"name": "inventory-expiry-env", "version": "1.0.0",
                "tasks": [{"name": n, "difficulty": t["difficulty"],
                           "description": t["description"], "max_steps": t["max_steps"]}
                          for n, t in TASKS.items()],
                "action_types": [a.value for a in _m.ActionType],
                "reward_range": [-0.5, 3.5]}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)