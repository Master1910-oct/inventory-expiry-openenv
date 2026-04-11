"""
Inventory & Expiry Management Environment — Baseline Inference Script
=====================================================================
Runs an OpenAI-compatible LLM agent against all 3 inventory management
tasks and reports reproducible baseline scores.

Mandatory environment variables:
    HF_TOKEN       Hugging Face / API key (required)
    API_BASE_URL   LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier   (default: gpt-4o-mini)

Optional:
    ENV_BASE_URL   Running environment server  (default: http://localhost:7860)
    TEMPERATURE    Sampling temperature        (default: 0.2)
    MAX_TOKENS     Max tokens per LLM call     (default: 800)

Usage:
    export HF_TOKEN=hf_...
    python inference.py

STDOUT format (OpenEnv spec — must match exactly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Rules:
    - One [START] per episode, immediately before the first step.
    - One [STEP] per step, immediately after env.step() returns.
    - One [END] after the episode ends — ALWAYS emitted, even on exception.
    - reward and score formatted to exactly 2 decimal places.
    - done and success are lowercase: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line, no embedded newlines.
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI, RateLimitError, APIStatusError

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

# HF_TOKEN is the primary API key per submission spec.
# Falls back to OPENAI_API_KEY for local development convenience.
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "MISSING_KEY"

# Default to HF inference router as required by submission spec.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS",    "800"))

# Retry settings for transient LLM errors
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds

# Keep history window small to avoid prompt bloat over long episodes
_HISTORY_WINDOW = 8

ALL_TASKS = ["easy_expiry_check", "medium_restock_transfer", "hard_full_audit"]

TASK_MAX_STEPS: Dict[str, int] = {
    "easy_expiry_check":       15,
    "medium_restock_transfer": 25,
    "hard_full_audit":         40,
}

BENCHMARK = "inventory-expiry-env"

# ---------------------------------------------------------------------------
# Logging — strict OpenEnv spec compliance
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    """Emit [START] line. One per episode, before first step."""
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line. One per step, immediately after env.step() returns.

    - action: raw string (no extra quoting)
    - reward: 2 decimal places
    - done: lowercase true/false
    - error: raw message string, or null
    """
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line. Always emitted, even on exception.

    - success: lowercase true/false
    - score: 2 decimal places
    - rewards: comma-separated, each 2 decimal places
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
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

    Available action_types and required/optional fields:
      - "list_items":      filter_location (str?), filter_expiring_within_days (int?), filter_low_stock (bool?)
      - "get_item":        sku (str, required)
      - "restock":         sku (str), quantity (int >=1)
      - "remove":          sku (str), quantity (int? — omit to remove all units)
      - "transfer":        sku (str), quantity (int), from_location (str), to_location (str)
      - "update_price":    sku (str), new_price (float >0)
      - "flag_expiry":     sku (str), batch_id (str?)
      - "generate_report": report_type ("expiry_risk"|"waste_summary"|"low_stock"|"full_audit"), days_threshold (int?)
      - "place_order":     sku (str), order_quantity (int), supplier (str?)
      - "done":            (no extra fields) — call ONLY when ALL objectives are complete

    Strategy:
      1. Start every episode with list_items to see all inventory.
      2. Items with days_until_expiry < 0 are EXPIRED — flag_expiry then remove them.
      3. Items with 0 <= days_until_expiry <= 7 are near-expiry — flag_expiry them.
      4. Items with quantity < reorder_point are low-stock — restock or place_order.
      5. Call done when you are confident all objectives are complete.

    Output ONLY valid JSON. No prose. No markdown. Examples:
      {"action_type": "list_items"}
      {"action_type": "flag_expiry", "sku": "SKU-0001", "batch_id": "B001"}
      {"action_type": "remove", "sku": "SKU-0001"}
      {"action_type": "update_price", "sku": "SKU-0003", "new_price": 0.76}
      {"action_type": "restock", "sku": "SKU-0009", "quantity": 50}
      {"action_type": "generate_report", "report_type": "expiry_risk", "days_threshold": 7}
      {"action_type": "done"}
""").strip()


# ---------------------------------------------------------------------------
# JSON extraction — handles markdown fences and stray prose
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from raw LLM output.

    Handles:
      - Clean JSON response
      - JSON wrapped in ```json ... ``` fences
      - Short prose preamble before the JSON object
    Falls back to list_items on any failure.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Fast path: entire cleaned string is valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Slow path: find first {...} block
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"[DEBUG] JSON parse failed, falling back. Raw: {raw!r[:200]}", flush=True)
    return {"action_type": "list_items"}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(step: int, obs: Dict[str, Any], history: List[str]) -> str:
    hist_block = "\n".join(history[-_HISTORY_WINDOW:]) if history else "None yet."

    items_block = ""
    if obs.get("items"):
        shown = obs["items"][:10]
        items_block = "\nItems returned:\n" + json.dumps(shown, indent=2)
        overflow = len(obs["items"]) - 10
        if overflow > 0:
            items_block += f"\n... and {overflow} more items not shown."

    report_block = ""
    if obs.get("report"):
        report_block = f"\nReport summary: {obs['report'].get('summary', '')}"

    return textwrap.dedent(f"""
        Step {step}
        Last message : {obs.get('message', '')[:400]}
        Last reward  : {obs.get('reward', 0.0):.2f}
        Error        : {obs.get('error') or 'none'}

        Inventory stats:
          expired_items_count    = {obs.get('expired_items_count', '?')}
          near_expiry_items_count= {obs.get('near_expiry_items_count', '?')}
          total_items_in_stock   = {obs.get('total_items_in_stock', '?')}
          total_waste_units      = {obs.get('total_waste_units', '?')}
          total_orders_placed    = {obs.get('total_orders_placed', '?')}
          episode_reward_so_far  = {obs.get('episode_reward_so_far', 0.0):.2f}
        {items_block}
        {report_block}

        Recent actions (last {_HISTORY_WINDOW}):
        {hist_block}

        Output your next action as JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# LLM call with retry logic
# ---------------------------------------------------------------------------

def _get_agent_action(
    client: OpenAI,
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and return a parsed action dict.

    Retries up to _MAX_RETRIES times on rate-limit or 5xx errors.
    Returns list_items as a safe fallback on all failures.
    """
    prompt = _build_user_prompt(step, obs, history)
    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
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
            return _extract_json(raw)

        except RateLimitError as exc:
            last_exc = exc
            wait = _RETRY_DELAY * attempt
            print(f"[DEBUG] Rate limit (attempt {attempt}/{_MAX_RETRIES}), sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                print(f"[DEBUG] Server error {exc.status_code} (attempt {attempt}/{_MAX_RETRIES}), retrying", flush=True)
                time.sleep(_RETRY_DELAY)
            else:
                print(f"[DEBUG] Non-retryable API error {exc.status_code}: {exc}", flush=True)
                break

        except Exception as exc:
            print(f"[DEBUG] LLM call error: {exc}", flush=True)
            break

    print(f"[DEBUG] All retries exhausted. Last error: {last_exc}", flush=True)
    return {"action_type": "list_items"}


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, http: httpx.Client, task_name: str) -> float:
    """Run one complete episode for task_name. Returns final score in [0, 1]."""
    log_start(task=task_name, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    obs:         Dict[str, Any] = {}

    try:
        # ── Reset environment ────────────────────────────────────────────
        try:
            reset_resp = http.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name},
                timeout=30,
            )
            reset_resp.raise_for_status()
            obs = reset_resp.json().get("observation", {})
        except Exception as exc:
            print(f"[DEBUG] Reset failed for {task_name!r}: {exc}", flush=True)
            # Cannot run — emit [END] immediately via finally
            return 0.0

        history: List[str] = []
        max_steps = TASK_MAX_STEPS.get(task_name, 30)

        # ── Main step loop ───────────────────────────────────────────────
        for step in range(1, max_steps + 1):
            if obs.get("done"):
                break

            action_dict = _get_agent_action(client, step, obs, history)
            action_str  = json.dumps(action_dict, separators=(",", ":"))

            try:
                step_resp = http.post(
                    f"{ENV_BASE_URL}/step",
                    json=action_dict,
                    timeout=30,
                )
                step_resp.raise_for_status()
                payload = step_resp.json()
            except Exception as exc:
                # Log the step with zero reward and carry on
                log_step(step, action_str, 0.0, False, str(exc))
                history.append(f"Step {step}: {action_str} -> ERROR: {exc}")
                continue

            obs         = payload.get("observation", {})
            reward      = float(payload.get("reward", 0.0))
            done        = bool(payload.get("done", False))
            error       = obs.get("error") or None
            steps_taken = step

            rewards.append(reward)
            log_step(step, action_str, reward, done, error)
            history.append(
                f"Step {step}: {action_str} -> reward={reward:.2f} | {obs.get('message', '')[:100]}"
            )

            if done:
                break

        # ── Fetch final score ────────────────────────────────────────────
        try:
            state_resp = http.get(f"{ENV_BASE_URL}/state", timeout=10)
            if state_resp.status_code == 200:
                score = float(state_resp.json().get("score", 0.0))
        except Exception as exc:
            print(f"[DEBUG] State fetch failed: {exc}", flush=True)

        success = score >= 0.5

    finally:
        # [END] is ALWAYS emitted — spec requirement
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http   = httpx.Client(timeout=60)

    print(f"\n{'='*60}", flush=True)
    print(f"  Inventory & Expiry Management — Baseline Inference",  flush=True)
    print(f"  Model      : {MODEL_NAME}",                           flush=True)
    print(f"  API base   : {API_BASE_URL}",                         flush=True)
    print(f"  Env server : {ENV_BASE_URL}",                         flush=True)
    print(f"{'='*60}\n",                                             flush=True)

    scores: Dict[str, float] = {}

    try:
        for task in ALL_TASKS:
            print(f"\n{'─'*60}", flush=True)
            print(f"  Task: {task}", flush=True)
            print(f"{'─'*60}",       flush=True)
            scores[task] = run_task(client, http, task)
    finally:
        http.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS",  flush=True)
    print(f"{'='*60}",   flush=True)
    for task, s in scores.items():
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {task:<30} [{bar}] {s:.2f}", flush=True)
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average : {avg:.2f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
