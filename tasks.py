"""
Inventory & Expiry Management Environment — Task Definitions & Graders
======================================================================
Defines 3 tasks (easy → medium → hard) with deterministic graders
that score agent performance in [0.0, 1.0].
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import InventoryState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2025, 4, 5)   # Fixed sim date for reproducibility


def _days_until(iso: str, today: date = _TODAY) -> int:
    d = date.fromisoformat(iso)
    return (d - today).days


def _make_sku(n: int) -> str:
    return f"SKU-{n:04d}"


# ---------------------------------------------------------------------------
# Seed inventories  (each task gets its own snapshot)
# ---------------------------------------------------------------------------

def _base_catalog() -> List[Dict]:
    """24 product catalog used across all tasks (varied expiry scenarios)."""
    products = [
        # Already expired
        dict(sku=_make_sku(1),  name="Whole Milk 1L",         category="Dairy",    location="WH-A", qty=40,  price=1.20, reorder=20, expiry=(_TODAY - timedelta(days=3)).isoformat(),  batch="B001"),
        dict(sku=_make_sku(2),  name="Greek Yogurt 500g",     category="Dairy",    location="WH-A", qty=15,  price=2.50, reorder=10, expiry=(_TODAY - timedelta(days=1)).isoformat(),  batch="B002"),
        # Expiring within 3 days (critical)
        dict(sku=_make_sku(3),  name="Fresh Cream 250ml",     category="Dairy",    location="WH-A", qty=30,  price=1.80, reorder=15, expiry=(_TODAY + timedelta(days=2)).isoformat(),  batch="B003"),
        dict(sku=_make_sku(4),  name="Sliced Bread 400g",     category="Bakery",   location="WH-B", qty=60,  price=0.95, reorder=30, expiry=(_TODAY + timedelta(days=1)).isoformat(),  batch="B004"),
        # Expiring within 7 days (near-expiry)
        dict(sku=_make_sku(5),  name="Orange Juice 1L",       category="Beverages",location="WH-B", qty=25,  price=3.10, reorder=12, expiry=(_TODAY + timedelta(days=5)).isoformat(),  batch="B005"),
        dict(sku=_make_sku(6),  name="Cheddar Cheese 250g",   category="Dairy",    location="WH-A", qty=18,  price=4.20, reorder=10, expiry=(_TODAY + timedelta(days=6)).isoformat(),  batch="B006"),
        # Expiring within 14 days
        dict(sku=_make_sku(7),  name="Chicken Breast 1kg",    category="Meat",     location="WH-C", qty=50,  price=7.80, reorder=20, expiry=(_TODAY + timedelta(days=10)).isoformat(), batch="B007"),
        dict(sku=_make_sku(8),  name="Salmon Fillet 500g",    category="Seafood",  location="WH-C", qty=22,  price=9.50, reorder=10, expiry=(_TODAY + timedelta(days=12)).isoformat(), batch="B008"),
        # Long shelf life but low stock
        dict(sku=_make_sku(9),  name="Pasta 500g",            category="Dry Goods",location="WH-D", qty=5,   price=1.10, reorder=30, expiry=(_TODAY + timedelta(days=365)).isoformat(),batch="B009"),
        dict(sku=_make_sku(10), name="Canned Tomatoes 400g",  category="Canned",   location="WH-D", qty=8,   price=0.89, reorder=40, expiry=(_TODAY + timedelta(days=730)).isoformat(),batch="B010"),
        # Healthy stock, good expiry
        dict(sku=_make_sku(11), name="Olive Oil 750ml",       category="Oils",     location="WH-D", qty=80,  price=6.50, reorder=20, expiry=(_TODAY + timedelta(days=540)).isoformat(),batch="B011"),
        dict(sku=_make_sku(12), name="Brown Rice 1kg",        category="Dry Goods",location="WH-D", qty=100, price=2.20, reorder=25, expiry=(_TODAY + timedelta(days=400)).isoformat(),batch="B012"),
        dict(sku=_make_sku(13), name="Frozen Peas 1kg",       category="Frozen",   location="WH-E", qty=45,  price=1.90, reorder=20, expiry=(_TODAY + timedelta(days=180)).isoformat(),batch="B013"),
        dict(sku=_make_sku(14), name="Tomato Sauce 500ml",    category="Canned",   location="WH-D", qty=35,  price=2.10, reorder=15, expiry=(_TODAY + timedelta(days=300)).isoformat(),batch="B014"),
        # Expiring soon — needs order
        dict(sku=_make_sku(15), name="Butter 250g",           category="Dairy",    location="WH-A", qty=10,  price=3.00, reorder=15, expiry=(_TODAY + timedelta(days=4)).isoformat(),  batch="B015"),
        dict(sku=_make_sku(16), name="Baby Spinach 150g",     category="Produce",  location="WH-B", qty=20,  price=2.80, reorder=10, expiry=(_TODAY + timedelta(days=3)).isoformat(),  batch="B016"),
        # Normal items
        dict(sku=_make_sku(17), name="Eggs 12-pack",          category="Dairy",    location="WH-A", qty=70,  price=3.50, reorder=30, expiry=(_TODAY + timedelta(days=21)).isoformat(), batch="B017"),
        dict(sku=_make_sku(18), name="Whole Wheat Flour 1kg", category="Dry Goods",location="WH-D", qty=60,  price=1.60, reorder=25, expiry=(_TODAY + timedelta(days=200)).isoformat(),batch="B018"),
        dict(sku=_make_sku(19), name="Plain Yogurt 1kg",      category="Dairy",    location="WH-A", qty=28,  price=3.20, reorder=10, expiry=(_TODAY + timedelta(days=14)).isoformat(), batch="B019"),
        dict(sku=_make_sku(20), name="Sparkling Water 1.5L",  category="Beverages",location="WH-B", qty=90,  price=0.80, reorder=40, expiry=(_TODAY + timedelta(days=548)).isoformat(),batch="B020"),
        dict(sku=_make_sku(21), name="Dark Chocolate 100g",   category="Confect.", location="WH-D", qty=55,  price=2.40, reorder=20, expiry=(_TODAY + timedelta(days=270)).isoformat(),batch="B021"),
        dict(sku=_make_sku(22), name="Almond Milk 1L",        category="Beverages",location="WH-B", qty=33,  price=2.90, reorder=15, expiry=(_TODAY + timedelta(days=30)).isoformat(), batch="B022"),
        dict(sku=_make_sku(23), name="Honey 500g",            category="Condiments",location="WH-D",qty=42,  price=5.50, reorder=10, expiry=(_TODAY + timedelta(days=730)).isoformat(),batch="B023"),
        dict(sku=_make_sku(24), name="Sourdough Loaf",        category="Bakery",   location="WH-B", qty=12,  price=4.00, reorder=8,  expiry=(_TODAY + timedelta(days=2)).isoformat(),  batch="B024"),
    ]
    return products


def _catalog_to_inventory(products: List[Dict]) -> Dict[str, Dict]:
    inventory: Dict[str, Dict] = {}
    for p in products:
        inventory[p["sku"]] = {
            "sku":          p["sku"],
            "name":         p["name"],
            "category":     p["category"],
            "location":     p["location"],
            "quantity":     p["qty"],
            "unit_price":   p["price"],
            "reorder_point":p["reorder"],
            "expiry_date":  p.get("expiry"),
            "batch_id":     p.get("batch"),
            "is_flagged":   False,
        }
    return inventory


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {}


def register_task(name: str, difficulty: str, description: str, instructions: str, max_steps: int):
    def decorator(fn):
        TASKS[name] = dict(
            name=name,
            difficulty=difficulty,
            description=description,
            instructions=instructions,
            max_steps=max_steps,
            seed_fn=fn,
        )
        return fn
    return decorator


# ============================================================
# TASK 1 — EASY: Expiry Check & Disposal
# ============================================================

@register_task(
    name="easy_expiry_check",
    difficulty="easy",
    description=(
        "Identify all expired and near-expiry (≤7 days) items in the warehouse, "
        "flag each batch, remove (dispose) the already-expired units, "
        "and generate an expiry-risk report."
    ),
    instructions=(
        "You manage a grocery warehouse. Today is 2025-04-05.\n"
        "Objectives (in any order):\n"
        "1. List all inventory items.\n"
        "2. Flag every item whose batch is already expired OR expiring within 7 days.\n"
        "3. Remove (dispose) all units of items that are already EXPIRED (days_until_expiry < 0).\n"
        "4. Generate an expiry_risk report.\n"
        "5. Call DONE when finished.\n"
        "Score: 0.25 per objective completed (4 objectives = 1.0 max).\n"
        "Penalty: -0.1 for each disposal action on a non-expired item."
    ),
    max_steps=15,
)
def easy_seed(state: InventoryState) -> InventoryState:
    state.inventory = _catalog_to_inventory(_base_catalog())
    state.sim_date = _TODAY.isoformat()
    return state


def easy_grader(state: InventoryState) -> Tuple[float, str]:
    score = 0.0
    notes = []

    # Obj 1: Did the agent list items?
    if "listed_items" in state.required_actions_done:
        score += 0.25
        notes.append("✓ Listed inventory")
    else:
        notes.append("✗ Never listed inventory")

    # Obj 2: Did the agent flag all expired/near-expiry batches?
    expected_flags = set()
    for item in state.inventory.values():
        if item.get("expiry_date"):
            d = _days_until(item["expiry_date"])
            if d <= 7:
                expected_flags.add(item["batch_id"])

    # Also include items that have been disposed (their batch_id logged)
    for rec in state.disposed_log:
        expected_flags.discard(rec.get("batch_id"))  # removed items can't be reflagged

    flagged = set(state.flagged_batches)
    if expected_flags and expected_flags.issubset(flagged):
        score += 0.25
        notes.append("✓ Flagged all critical/expired batches")
    elif flagged:
        partial = len(expected_flags & flagged) / max(len(expected_flags), 1)
        score += 0.25 * partial * 0.5  # partial credit
        notes.append(f"△ Partially flagged ({len(flagged & expected_flags)}/{len(expected_flags)})")
    else:
        notes.append("✗ No batches flagged")

    # Obj 3: Disposed all expired items
    # An expired item is "handled" if (a) it was logged as disposed with was_expired=True,
    # OR (b) it's still in inventory but quantity == 0 and was expired.
    all_expired_skus = set()
    for sku, item in state.inventory.items():
        if item.get("expiry_date") and _days_until(item["expiry_date"]) < 0:
            all_expired_skus.add(sku)

    disposed_expired_skus = {r["sku"] for r in state.disposed_log if r.get("was_expired", False)}

    # SKUs that are expired AND still have stock remaining (not cleared)
    still_have_expired_stock = {
        sku for sku in all_expired_skus
        if state.inventory[sku]["quantity"] > 0 and sku not in disposed_expired_skus
    }

    if not still_have_expired_stock:
        score += 0.25
        notes.append("✓ Disposed all expired items")
    elif disposed_expired_skus:
        handled = all_expired_skus - still_have_expired_stock
        partial = len(handled) / max(len(all_expired_skus), 1)
        score += 0.25 * partial
        notes.append(f"△ Disposed {len(handled)} of {len(all_expired_skus)} expired SKUs")
    else:
        notes.append("✗ No expired items disposed")

    # Obj 4: Generated expiry-risk report
    if "generated_expiry_risk_report" in state.required_actions_done:
        score += 0.25
        notes.append("✓ Generated expiry-risk report")
    else:
        notes.append("✗ No expiry-risk report generated")

    # Penalties: disposals of non-expired items
    bad_removals = sum(1 for r in state.disposed_log if not r.get("was_expired", False))
    penalty = bad_removals * 0.10
    score = max(0.0, score - penalty)
    if penalty > 0:
        notes.append(f"✗ Penalty: disposed {bad_removals} non-expired item(s) (-{penalty:.2f})")

    score = min(1.0, max(0.0, score))
    return score, " | ".join(notes)


# ============================================================
# TASK 2 — MEDIUM: Multi-Location Restock & Transfer
# ============================================================

@register_task(
    name="medium_restock_transfer",
    difficulty="medium",
    description=(
        "Assess stock levels across 5 warehouse locations, transfer overstocked items "
        "to under-stocked locations, restock items below their reorder point, "
        "place supplier orders for critical shortages, and generate a low-stock report."
    ),
    instructions=(
        "You manage a 5-warehouse distribution centre. Today is 2025-04-05.\n"
        "Objectives:\n"
        "1. List all items and identify those below their reorder_point.\n"
        "2. Place replenishment orders for ALL items below reorder_point (supplier='AutoSupplier').\n"
        "3. Transfer at least 2 items between locations to balance stock "
        "   (e.g., move overstock from WH-D to WH-A).\n"
        "4. Restock (RESTOCK action) at least 3 items by adding ≥10 units each.\n"
        "5. Generate a low_stock report.\n"
        "Score: 0.20 per objective. Penalty: -0.05 per order for a non-low-stock item."
    ),
    max_steps=25,
)
def medium_seed(state: InventoryState) -> InventoryState:
    state.inventory = _catalog_to_inventory(_base_catalog())
    state.sim_date = _TODAY.isoformat()
    return state


def medium_grader(state: InventoryState) -> Tuple[float, str]:
    score = 0.0
    notes = []

    # Obj 1: Listed + identified low-stock items
    if "listed_items" in state.required_actions_done:
        score += 0.20
        notes.append("✓ Listed inventory")
    else:
        notes.append("✗ Never listed inventory")

    # Obj 2: Orders placed for all low-stock items
    low_stock_skus = {
        sku for sku, item in state.inventory.items()
        if item["quantity"] < item["reorder_point"]
    }
    ordered_skus = {o["sku"] for o in state.orders}
    if low_stock_skus and low_stock_skus.issubset(ordered_skus):
        score += 0.20
        notes.append(f"✓ Ordered all {len(low_stock_skus)} low-stock SKUs")
    elif ordered_skus & low_stock_skus:
        partial = len(ordered_skus & low_stock_skus) / max(len(low_stock_skus), 1)
        score += 0.20 * partial
        notes.append(f"△ Ordered {len(ordered_skus & low_stock_skus)}/{len(low_stock_skus)} low-stock SKUs")
    else:
        notes.append("✗ No low-stock orders placed")

    # Obj 3: At least 2 transfers
    transfers = [a for a in state.required_actions_done if a.startswith("transfer:")]
    if len(transfers) >= 2:
        score += 0.20
        notes.append(f"✓ {len(transfers)} transfers performed")
    elif transfers:
        score += 0.10
        notes.append(f"△ Only {len(transfers)} transfer(s) (need 2)")
    else:
        notes.append("✗ No transfers performed")

    # Obj 4: At least 3 restock actions with ≥10 units
    restocks = [a for a in state.required_actions_done if a.startswith("restock:")]
    if len(restocks) >= 3:
        score += 0.20
        notes.append(f"✓ {len(restocks)} restock actions")
    elif restocks:
        score += 0.20 * len(restocks) / 3
        notes.append(f"△ Only {len(restocks)}/3 restock actions")
    else:
        notes.append("✗ No restock actions")

    # Obj 5: Low-stock report
    if "generated_low_stock_report" in state.required_actions_done:
        score += 0.20
        notes.append("✓ Generated low-stock report")
    else:
        notes.append("✗ No low-stock report")

    # Penalty: ordering non-low-stock items
    non_low_orders = len(ordered_skus - low_stock_skus)
    penalty = non_low_orders * 0.05
    score = max(0.0, score - penalty)
    if penalty > 0:
        notes.append(f"✗ Penalty: {non_low_orders} unnecessary orders (-{penalty:.2f})")

    score = min(1.0, max(0.0, score))
    return score, " | ".join(notes)


# ============================================================
# TASK 3 — HARD: Full Inventory Audit & Optimisation
# ============================================================

@register_task(
    name="hard_full_audit",
    difficulty="hard",
    description=(
        "Conduct a complete inventory audit: identify and handle expired & near-expiry items, "
        "balance cross-location stock, replenish critical shortages, adjust prices for "
        "near-expiry items (markdown), generate three distinct reports, and produce a "
        "comprehensive remediation plan before calling DONE."
    ),
    instructions=(
        "Full audit mode. Today is 2025-04-05.\n"
        "Objectives (each worth ~0.14):\n"
        "1. List all inventory items.\n"
        "2. Dispose ALL expired items AND flag ALL near-expiry batches (≤7 days).\n"
        "3. Apply a price markdown (≥20% reduction) to every near-expiry item (≤7 days).\n"
        "4. Restock at least 4 items below their reorder point (add ≥ reorder_point units each).\n"
        "5. Transfer items to balance stock: perform ≥3 transfers across different locations.\n"
        "6. Place orders for remaining low-stock items after restocking.\n"
        "7. Generate ALL THREE report types: expiry_risk, low_stock, waste_summary.\n"
        "Penalty: -0.05 per near-expiry item NOT marked down; -0.10 per expired item NOT disposed."
    ),
    max_steps=40,
)
def hard_seed(state: InventoryState) -> InventoryState:
    state.inventory = _catalog_to_inventory(_base_catalog())
    state.sim_date = _TODAY.isoformat()
    return state


def hard_grader(state: InventoryState) -> Tuple[float, str]:
    score = 0.0
    weight = 1.0 / 7.0
    notes = []

    catalog = _catalog_to_inventory(_base_catalog())

    # Obj 1: Listed items
    if "listed_items" in state.required_actions_done:
        score += weight
        notes.append("✓ Listed inventory")
    else:
        notes.append("✗ No list action")

    # Obj 2: Expired disposed + near-expiry flagged
    expired_skus = {
        sku for sku, item in catalog.items()
        if item.get("expiry_date") and _days_until(item["expiry_date"]) < 0
    }
    near_expiry_batches = {
        item["batch_id"] for item in catalog.values()
        if item.get("expiry_date") and 0 <= _days_until(item["expiry_date"]) <= 7
    }

    disposed_expired = {r["sku"] for r in state.disposed_log if r.get("was_expired", False)}
    still_expired_in_stock = expired_skus - disposed_expired
    flagged = set(state.flagged_batches)
    missing_flags = near_expiry_batches - flagged

    obj2 = 0.0
    if not still_expired_in_stock:
        obj2 += 0.5
    else:
        obj2 += 0.5 * (len(disposed_expired) / max(len(expired_skus), 1))
    if not missing_flags:
        obj2 += 0.5
    else:
        obj2 += 0.5 * (len(flagged & near_expiry_batches) / max(len(near_expiry_batches), 1))
    score += weight * obj2
    notes.append(f"{'✓' if obj2 == 1.0 else '△'} Expiry disposal/flagging ({obj2:.2f})")

    # Obj 3: Markdowns on near-expiry items (≤7 days)
    near_expiry_skus = {
        sku for sku, item in catalog.items()
        if item.get("expiry_date") and 0 <= _days_until(item["expiry_date"]) <= 7
    }
    marked_down = set()
    for sku in near_expiry_skus:
        orig_price = catalog[sku]["unit_price"]
        curr = state.inventory.get(sku, {})
        if curr and curr.get("unit_price", orig_price) <= orig_price * 0.80:
            marked_down.add(sku)
    md_ratio = len(marked_down) / max(len(near_expiry_skus), 1)
    score += weight * md_ratio
    notes.append(f"{'✓' if md_ratio == 1.0 else '△'} Price markdowns ({len(marked_down)}/{len(near_expiry_skus)})")

    # Obj 4: Restock ≥4 low-stock items
    low_stock_skus = {sku for sku, item in catalog.items() if item["quantity"] < item["reorder_point"]}
    restocked_skus = {a.split(":")[1] for a in state.required_actions_done if a.startswith("restock:")}
    restocked_low = restocked_skus & low_stock_skus
    rs_ratio = min(len(restocked_low) / 4, 1.0)
    score += weight * rs_ratio
    notes.append(f"{'✓' if rs_ratio == 1.0 else '△'} Restocked {len(restocked_low)}/4 low-stock items")

    # Obj 5: ≥3 transfers
    transfers = [a for a in state.required_actions_done if a.startswith("transfer:")]
    tr_ratio = min(len(transfers) / 3, 1.0)
    score += weight * tr_ratio
    notes.append(f"{'✓' if tr_ratio == 1.0 else '△'} {len(transfers)}/3 transfers")

    # Obj 6: Orders for remaining low-stock after restock
    ordered_skus = {o["sku"] for o in state.orders}
    # Low-stock items that weren't restocked enough
    still_low = {
        sku for sku in low_stock_skus
        if state.inventory.get(sku, {}).get("quantity", 0) < catalog[sku]["reorder_point"]
    }
    if not still_low:
        score += weight
        notes.append("✓ All remaining low-stock items ordered")
    elif still_low.issubset(ordered_skus):
        score += weight
        notes.append("✓ Ordered all remaining low-stock items")
    elif ordered_skus & still_low:
        partial = len(ordered_skus & still_low) / max(len(still_low), 1)
        score += weight * partial
        notes.append(f"△ Orders for {len(ordered_skus & still_low)}/{len(still_low)} remaining low-stock")
    else:
        notes.append("✗ No orders for remaining low-stock")

    # Obj 7: All 3 report types generated
    reports_needed = {"generated_expiry_risk_report", "generated_low_stock_report", "generated_waste_summary_report"}
    reports_done = reports_needed & set(state.required_actions_done)
    rp_ratio = len(reports_done) / 3
    score += weight * rp_ratio
    notes.append(f"{'✓' if rp_ratio == 1.0 else '△'} Reports: {len(reports_done)}/3")

    # Penalties — only applied if the agent attempted the objective but left items behind
    expired_not_disposed = len(still_expired_in_stock)
    not_marked_down = len(near_expiry_skus - marked_down)
    disposed_any = len(state.disposed_log) > 0
    attempted_markdown = bool(marked_down)
    penalty = 0.0
    if disposed_any and expired_not_disposed:
        penalty += expired_not_disposed * 0.10
    if attempted_markdown and not_marked_down:
        penalty += not_marked_down * 0.05
    score = max(0.0, score - penalty)
    if penalty > 0:
        notes.append(f"✗ Penalties: {expired_not_disposed} undisposed expired, {not_marked_down} no-markdown (-{penalty:.2f})")

    score = min(1.0, max(0.0, score))
    return score, " | ".join(notes)


# ---------------------------------------------------------------------------
# Unified grader router
# ---------------------------------------------------------------------------

GRADERS = {
    "easy_expiry_check":     easy_grader,
    "medium_restock_transfer": medium_grader,
    "hard_full_audit":       hard_grader,
}


def get_task_names() -> List[str]:
    return list(TASKS.keys())


def seed_state(state: InventoryState) -> InventoryState:
    task = TASKS.get(state.task_name)
    if task is None:
        raise ValueError(f"Unknown task: {state.task_name!r}. Available: {get_task_names()}")
    return task["seed_fn"](state)


def grade(state: InventoryState) -> Tuple[float, str]:
    grader = GRADERS.get(state.task_name)
    if grader is None:
        raise ValueError(f"No grader for task: {state.task_name!r}")
    return grader(state)
