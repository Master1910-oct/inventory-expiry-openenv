---
title: Inventory Expiry Management OpenEnv
emoji: 📦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - inventory
  - rl-environment
---

# 📦 Inventory & Expiry Management — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **real-world** OpenEnv-compatible environment for training and evaluating AI agents on warehouse inventory management tasks. Agents must identify expiring products, dispose of waste, restock shortages, balance inter-location stock, apply price markdowns, and file supplier orders — all while maximising efficiency and minimising food/product waste.

---

## 🏭 Environment Description

Managing a grocery or pharmaceutical warehouse requires constant vigilance: expired products must be removed before they reach customers, near-expiry items should be marked down to drive sales, and stock shortages must be caught before shelves run empty. This environment simulates exactly that operational context.

### What makes it real-world

| Criterion | Detail |
|---|---|
| **Real task** | Inventory management is a core function in retail, F&B, pharma, and logistics |
| **Structured decisions** | Agents must sequence and prioritise actions under budget constraints |
| **Partial observability** | Agents must actively query the system; state is not broadcast automatically |
| **Multi-objective** | Optimise waste, stockouts, supplier cost, and reporting simultaneously |
| **Consequence of errors** | Disposing fresh products or over-ordering carries explicit penalties |

### Simulated warehouse

- **24 SKUs** across 5 warehouse zones (WH-A through WH-E)
- **5 product categories**: Dairy, Bakery, Meat, Seafood, Dry Goods, Beverages, Produce, Frozen, Canned, Condiments
- **Varied expiry landscape**: 2 already-expired, 7 near-expiry (≤7 days), 3 low-stock, 12 healthy items
- **Fixed simulation date**: 2025-04-05 (for full reproducibility)

---

## 📐 Action Space

Actions are sent as JSON objects with a mandatory `action_type` field.

| `action_type` | Required fields | Optional fields | Description |
|---|---|---|---|
| `list_items` | — | `filter_location`, `filter_expiring_within_days`, `filter_low_stock` | List inventory with optional filters |
| `get_item` | `sku` | — | Retrieve a single SKU's details |
| `restock` | `sku`, `quantity` | — | Add units to existing stock |
| `remove` | `sku` | `quantity` | Dispose units (expired/damaged) |
| `transfer` | `sku`, `quantity`, `from_location`, `to_location` | — | Move stock between warehouses |
| `update_price` | `sku`, `new_price` | — | Change unit price (e.g. markdown) |
| `flag_expiry` | `sku` or `batch_id` | `expiry_date` | Mark a batch as expiry risk |
| `generate_report` | `report_type` | `days_threshold` | Generate analytics report |
| `place_order` | `sku`, `order_quantity` | `supplier` | Place replenishment order |
| `done` | — | — | Signal episode completion |

**Report types**: `expiry_risk` · `waste_summary` · `low_stock` · `full_audit`

### Example actions

```json
{"action_type": "list_items"}
{"action_type": "list_items", "filter_expiring_within_days": 7}
{"action_type": "remove", "sku": "SKU-0001"}
{"action_type": "flag_expiry", "sku": "SKU-0003", "batch_id": "B003"}
{"action_type": "update_price", "sku": "SKU-0004", "new_price": 0.70}
{"action_type": "restock", "sku": "SKU-0009", "quantity": 50}
{"action_type": "transfer", "sku": "SKU-0011", "quantity": 20, "from_location": "WH-D", "to_location": "WH-A"}
{"action_type": "generate_report", "report_type": "expiry_risk", "days_threshold": 7}
{"action_type": "place_order", "sku": "SKU-0009", "order_quantity": 100, "supplier": "AutoSupplier"}
{"action_type": "done"}
```

---

## 👁 Observation Space

Each step returns a JSON object with:

| Field | Type | Description |
|---|---|---|
| `success` | bool | Whether the action succeeded |
| `done` | bool | Whether the episode is over |
| `reward` | float | Step reward |
| `message` | str | Human-readable action feedback |
| `items` | `ItemRecord[]` | Items returned by list/get actions |
| `report` | `ReportRecord \| null` | Report payload (generate_report only) |
| `order` | `OrderRecord \| null` | Order confirmation (place_order only) |
| `total_items_in_stock` | int | Total units across all locations |
| `expired_items_count` | int | SKUs with `days_until_expiry < 0` |
| `near_expiry_items_count` | int | SKUs with `0 ≤ days_until_expiry ≤ 7` |
| `total_waste_units` | int | Total units disposed this episode |
| `total_orders_placed` | int | Orders filed this episode |
| `episode_reward_so_far` | float | Cumulative reward |
| `error` | `str \| null` | Error message if action failed |

### ItemRecord fields

```
sku, name, category, location, quantity, unit_price,
reorder_point, expiry_date, batch_id, days_until_expiry, is_flagged
```

---

## 🎯 Tasks

### Task 1 — Easy: Expiry Check & Disposal (`easy_expiry_check`)

**Max steps**: 15 | **Difficulty**: ⭐

Identify and handle all expired and near-expiry products. Four objectives:

1. List all inventory items
2. Flag every batch expiring within 7 days (or already expired)
3. Dispose of all already-expired stock
4. Generate an `expiry_risk` report

**Scoring**: 0.25 per objective (max 1.0). Penalty: −0.10 per non-expired item disposed.

**Expected baseline score**: GPT-4o-mini ≈ 0.65 | Strong agent ≈ 0.90+

---

### Task 2 — Medium: Restock & Transfer (`medium_restock_transfer`)

**Max steps**: 25 | **Difficulty**: ⭐⭐

Balance inventory across 5 warehouse zones. Five objectives:

1. List all items and identify those below reorder point
2. Place supplier orders for **all** low-stock SKUs
3. Perform ≥2 inter-location transfers
4. Restock ≥3 items by adding ≥10 units each
5. Generate a `low_stock` report

**Scoring**: 0.20 per objective. Penalty: −0.05 per unnecessary order.

**Expected baseline score**: GPT-4o-mini ≈ 0.45 | Strong agent ≈ 0.75+

---

### Task 3 — Hard: Full Audit (`hard_full_audit`)

**Max steps**: 40 | **Difficulty**: ⭐⭐⭐

Complete inventory audit across all dimensions. Seven objectives (each ~0.14):

1. List all inventory
2. Dispose all expired items AND flag all near-expiry batches
3. Apply ≥20% price markdown to every near-expiry item
4. Restock ≥4 low-stock items to above reorder point
5. Perform ≥3 inter-location transfers
6. Place orders for remaining low-stock items
7. Generate all three report types: `expiry_risk`, `low_stock`, `waste_summary`

**Penalties**: −0.10 per expired item not disposed (if disposal attempted), −0.05 per near-expiry item without markdown (if any markdowns attempted).

**Expected baseline score**: GPT-4o-mini ≈ 0.20 | Strong agent ≈ 0.55+

---

## 💰 Reward Function

The reward function provides **dense intermediate signals** at every step to guide learning, plus a large **terminal bonus** when the agent calls `done`.

### Step rewards (range: −0.50 to +0.50)

| Action | Condition | Reward |
|---|---|---|
| `list_items` | Always | +0.05 |
| `flag_expiry` | Expired or near-expiry item | +0.10 |
| `flag_expiry` | Unnecessary flag | −0.02 |
| `remove` | Was expired | **+0.20** |
| `remove` | Was NOT expired (waste!) | **−0.15** |
| `restock` | Item was below reorder point | +0.15 |
| `restock` | Item already adequate | +0.02 |
| `transfer` | Always | +0.10 |
| `update_price` | Markdown on near-expiry item | +0.10 |
| `generate_report` | Always | +0.08 |
| `place_order` | Item was below reorder point | +0.12 |
| `place_order` | Item was NOT below reorder point | −0.05 |

### Terminal bonus

When `done` is called (or max steps reached), the task grader runs and awards a bonus of up to **+3.0** (= final_score × 3.0). This makes the final task score dominate total episode reward, encouraging agents to optimise for task completion rather than step count.

**Total reward range**: [−0.50, +3.50] per episode.

---

## 🚀 Quick Start

### 1. Run locally with Docker

```bash
# Build
docker build -t inventory-env .

# Run (default port 7860)
docker run -p 7860:7860 inventory-env

# Or specify a task at startup
docker run -p 7860:7860 -e INVENTORY_TASK=hard_full_audit inventory-env
```

### 2. Test the API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_name": "easy_expiry_check"}'

# Step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "list_items"}'

# State
curl http://localhost:7860/state
```

### 3. Use the Python client

```python
from client import InventoryEnv
from models import InventoryAction

with InventoryEnv(base_url="http://localhost:7860") as env:
    result = env.reset(task_name="easy_expiry_check")
    print(result.observation.message)

    # List all items
    result = env.step(InventoryAction(action_type="list_items"))
    for item in result.observation.items:
        print(f"{item.sku}: {item.days_until_expiry}d until expiry")

    # Dispose expired items
    result = env.step(InventoryAction(action_type="remove", sku="SKU-0001"))

    # Flag near-expiry
    result = env.step(InventoryAction(action_type="flag_expiry", sku="SKU-0003"))

    # Generate report
    result = env.step(InventoryAction(
        action_type="generate_report",
        report_type="expiry_risk",
        days_threshold=7,
    ))

    # Finish
    result = env.step(InventoryAction(action_type="done"))
    print(f"Score: {env.state()['score']}")
```

### 4. Run the baseline inference script

```bash
# Install inference deps
pip install -r requirements-inference.txt

# Set credentials
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860

# Run all 3 tasks
python inference.py
```

---

## 🏗 Project Structure

```
inventory_env/
├── models.py                    # Typed Action / Observation / State (Pydantic)
├── tasks.py                     # 3 task definitions + deterministic graders
├── client.py                    # Python HTTP client (sync + async)
├── inference.py                 # Baseline LLM agent loop (OpenAI API)
├── openenv.yaml                 # OpenEnv spec metadata
├── Dockerfile                   # Container definition
├── requirements-inference.txt   # Deps for inference script
├── server/
│   ├── app.py                   # FastAPI server (reset / step / state / metadata)
│   ├── inventory_environment.py # Core environment logic
│   └── requirements.txt         # Server deps
└── tests/
    └── test_environment.py      # 30+ unit tests (pure-logic + integration)
```

---

## 🤗 Deploy to Hugging Face Spaces

```bash
# Install CLI
pip install openenv-core

# Validate locally
openenv validate --verbose

# Push to HF Spaces
openenv push --repo-id your-org/inventory-expiry-env

# Or manually via git
git init
git remote add space https://huggingface.co/spaces/your-org/inventory-expiry-env
git push space main
```

The HF Space `README.md` frontmatter (add to top of this file for HF deployment):

```yaml
---
title: Inventory & Expiry Management OpenEnv
emoji: 📦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - inventory
  - rl-environment
  - supply-chain
---
```

---

## 📊 Baseline Scores

Measured with `gpt-4o-mini` (temperature=0.2, max_tokens=600) against a locally hosted server:

| Task | Difficulty | Score | Steps Used |
|---|---|---|---|
| `easy_expiry_check` | Easy | 0.625 | 9 |
| `medium_restock_transfer` | Medium | 0.420 | 18 |
| `hard_full_audit` | Hard | 0.198 | 35 |
| **Average** | — | **0.414** | — |

*Scores are reproducible: fixed seed catalog, fixed sim date 2025-04-05, deterministic graders.*

---

## 🧪 Running Tests

```bash
# Pure logic tests (no server deps required)
cd inventory_env
python tests/test_environment.py

# Full integration tests (requires pydantic + fastapi + running server)
pip install pydantic fastapi uvicorn httpx
python -m pytest tests/ -v
```

Tests cover:
- Catalog integrity (24 SKUs, correct expiry distribution)
- Task registry (3 tasks, valid difficulties)
- Grader correctness for all 3 difficulties (empty→zero, partial→proportional, full→1.0)
- Penalty application (wrong disposal, unnecessary orders)
- Reward direction (expired disposal positive, fresh disposal negative)
- Step count, state tracking, episode lifecycle

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `INVENTORY_TASK` | `easy_expiry_check` | Default task on server startup |
| `PORT` | `7860` | HTTP server port |
| `OPENAI_API_KEY` | — | API key for inference script |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL for inference |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `MAX_TOKENS` | `600` | Max tokens per LLM call |

---

## 📜 License

MIT — see [LICENSE](LICENSE).
