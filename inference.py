"""
Inventory & Expiry Management Environment — Baseline Inference Script
=====================================================================
Runs an LLM agent against all 3 tasks and reports reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    export API_BASE_URL=https://api.openai.com/v1   # or HF router
    export MODEL_NAME=gpt-4o-mini
    python inference.py

STDOUT format follows the OpenEnv logging spec:
    [START] task=<task> env=inventory-expiry-env model=<model>
    [STEP]  step=<n> action=<action_str> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "MISSING_KEY"
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "600"))

TASKS = ["easy_expiry_check", "medium_restock_transfer", "hard_full_audit"]
BENCHMARK = "inventory-expiry-env"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rw = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rw}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert AI inventory manager operating a warehouse management system.
You interact with the Inventory & Expiry Management environment through a structured API.

Each turn you MUST output a single JSON object — no prose, no markdown fences — representing
the action you want to take. The JSON must have at least "action_type" and any required fields.

Available action_types and their required/optional fields:
- "list_items": filter_location (str?), filter_expiring_within_days (int?), filter_low_stock (bool?)
- "get_item": sku (str, required)
- "restock": sku (str), quantity (int ≥1)
- "remove": sku (str), quantity (int? — omit to remove all)
- "transfer": sku (str), quantity (int), from_location (str), to_location (str)
- "update_price": sku (str), new_price (float >0)
- "flag_expiry": sku (str? or null), batch_id (str?)
- "generate_report": report_type ("expiry_risk"|"waste_summary"|"low_stock"|"full_audit"), days_threshold (int?)
- "place_order": sku (str), order_quantity (int), supplier (str?)
- "done": (no extra fields) — call this ONLY when all objectives are complete

Strategy hints:
- Always start with list_items to understand inventory.
- Items with days_until_expiry < 0 are EXPIRED — remove them.
- Items with days_until_expiry ≤ 7 are near-expiry — flag them.
- Items below reorder_point need restocking or orders.
- Call done when you believe all objectives are complete.

Output ONLY valid JSON. Example:
{"action_type": "list_items"}
{"action_type": "remove", "sku": "SKU-0001"}
{"action_type": "done"}
""").strip()


def build_user_prompt(
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> str:
    hist_block = "\n".join(history[-6:]) if history else "None yet."
    items_summary = ""
    if obs.get("items"):
        items_summary = "\nCurrent items returned:\n" + json.dumps(obs["items"][:8], indent=2)
        if len(obs["items"]) > 8:
            items_summary += f"\n... and {len(obs['items']) - 8} more items."
    report_summary = ""
    if obs.get("report"):
        report_summary = f"\nReport: {obs['report']['summary']}"

    return textwrap.dedent(f"""
    Step {step}
    ─────────────────────────────────────────────────────────
    Last message: {obs.get('message', '')[:500]}
    Last reward: {obs.get('reward', 0.0):.3f}
    Done: {obs.get('done', False)}
    Error: {obs.get('error') or 'none'}
    Stats:
      Total units in stock: {obs.get('total_items_in_stock', '?')}
      Expired SKUs: {obs.get('expired_items_count', '?')}
      Near-expiry SKUs: {obs.get('near_expiry_items_count', '?')}
      Waste units: {obs.get('total_waste_units', '?')}
      Orders placed: {obs.get('total_orders_placed', '?')}
      Episode reward so far: {obs.get('episode_reward_so_far', 0.0):.3f}
    {items_summary}
    {report_summary}
    ─────────────────────────────────────────────────────────
    Recent history:
    {hist_block}
    ─────────────────────────────────────────────────────────
    What is your next action? Output JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        # Strip accidental markdown fences
        raw = raw.strip("`").removeprefix("json").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}. Raw: {raw!r}", flush=True)
        return {"action_type": "list_items"}
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"action_type": "list_items"}


def run_task(client: OpenAI, http: httpx.Client, task_name: str) -> float:
    log_start(task=task_name, model=MODEL_NAME)

    # Reset
    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    obs = payload["observation"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    max_steps = {"easy_expiry_check": 15, "medium_restock_transfer": 25, "hard_full_audit": 40}.get(task_name, 30)

    try:
        for step in range(1, max_steps + 1):
            if obs.get("done"):
                break

            action_dict = get_agent_action(client, step, obs, history)
            action_str = json.dumps(action_dict)

            try:
                step_resp = http.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
                step_resp.raise_for_status()
                step_payload = step_resp.json()
            except Exception as e:
                log_step(step, action_str, 0.0, False, str(e))
                history.append(f"Step {step}: {action_str} → ERROR: {e}")
                continue

            obs = step_payload["observation"]
            reward = step_payload.get("reward", 0.0)
            done = step_payload.get("done", False)
            error = obs.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error)
            history.append(
                f"Step {step}: {action_str} → reward={reward:.3f} | {obs.get('message', '')[:120]}"
            )

            if done:
                break

        # Retrieve final score from state
        state_resp = http.get(f"{ENV_BASE_URL}/state", timeout=10)
        if state_resp.status_code == 200:
            state = state_resp.json()
            score = state.get("score", 0.0)
        else:
            score = 0.0

        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Run error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http   = httpx.Client(timeout=60)

    scores: Dict[str, float] = {}
    print(f"\n{'='*60}", flush=True)
    print(f" Inventory & Expiry Management — Baseline Inference", flush=True)
    print(f" Model: {MODEL_NAME}  |  Tasks: {len(TASKS)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for task in TASKS:
        print(f"\n{'─'*60}", flush=True)
        print(f" Running task: {task}", flush=True)
        print(f"{'─'*60}", flush=True)
        score = run_task(client, http, task)
        scores[task] = score

    http.close()

    print(f"\n{'='*60}", flush=True)
    print(f" FINAL BASELINE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:<30}  [{bar}]  {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score: {avg:.3f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
