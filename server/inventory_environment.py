"""
Inventory & Expiry Management Environment — Core Logic
=======================================================
Implements reset(), step(), state() following the OpenEnv interface.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root (parent of server/) is on sys.path — works on all OS
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import copy
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType,
    InventoryAction,
    InventoryObservation,
    InventoryState,
    ItemRecord,
    OrderRecord,
    ReportRecord,
    ReportType,
)
from tasks import TASKS, grade, get_task_names, seed_state

_TODAY = date(2025, 4, 5)
_NEAR_EXPIRY_DAYS = 7
_CRITICAL_EXPIRY_DAYS = 3


def _days_until(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    return (date.fromisoformat(iso) - _TODAY).days


def _item_to_record(item: Dict) -> ItemRecord:
    d = _days_until(item.get("expiry_date"))
    return ItemRecord(
        sku=item["sku"],
        name=item["name"],
        category=item["category"],
        location=item["location"],
        quantity=item["quantity"],
        unit_price=item["unit_price"],
        reorder_point=item["reorder_point"],
        expiry_date=item.get("expiry_date"),
        batch_id=item.get("batch_id"),
        days_until_expiry=d,
        is_flagged=item.get("is_flagged", False),
    )


def _compute_stats(state: InventoryState) -> Dict[str, int]:
    total = expired = near_expiry = 0
    for item in state.inventory.values():
        total += item["quantity"]
        d = _days_until(item.get("expiry_date"))
        if d is not None:
            if d < 0:
                expired += 1
            elif d <= _NEAR_EXPIRY_DAYS:
                near_expiry += 1
    waste = sum(r.get("quantity", 0) for r in state.disposed_log)
    return dict(
        total_items_in_stock=total,
        expired_items_count=expired,
        near_expiry_items_count=near_expiry,
        total_waste_units=waste,
        total_orders_placed=len(state.orders),
    )


def _step_reward(action_type: ActionType, context: Dict) -> float:
    r = 0.0
    if action_type == ActionType.LIST_ITEMS:
        r = 0.05
    elif action_type == ActionType.FLAG_EXPIRY:
        r = 0.10 if (context.get("was_expired") or context.get("near_expiry")) else -0.02
    elif action_type == ActionType.REMOVE:
        r = 0.20 if context.get("was_expired") else -0.15
    elif action_type == ActionType.RESTOCK:
        r = 0.15 if context.get("was_below_reorder") else 0.02
    elif action_type == ActionType.TRANSFER:
        r = 0.10
    elif action_type == ActionType.UPDATE_PRICE:
        r = 0.10 if context.get("is_markdown") else 0.00
    elif action_type == ActionType.GENERATE_REPORT:
        r = 0.08
    elif action_type == ActionType.PLACE_ORDER:
        r = 0.12 if context.get("was_below_reorder") else -0.05
    elif action_type == ActionType.DONE:
        r = 0.0
    return r


class InventoryEnvironment:
    """OpenEnv-compatible Inventory & Expiry Management environment."""

    def __init__(self, task_name: str = "easy_expiry_check"):
        if task_name not in get_task_names():
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {get_task_names()}")
        self._task_name = task_name
        self._state = InventoryState()
        self._episode_reward = 0.0

    def reset(self, task_name: Optional[str] = None) -> InventoryObservation:
        if task_name:
            self._task_name = task_name
        self._state = InventoryState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
        )
        self._state = seed_state(self._state)
        self._episode_reward = 0.0
        task_info = TASKS[self._task_name]
        stats = _compute_stats(self._state)
        return InventoryObservation(
            success=True, done=False, reward=0.0,
            message=(
                f"Episode started. Task: [{self._task_name}] ({task_info['difficulty'].upper()})\n"
                f"{task_info['instructions']}\n\n"
                f"Inventory loaded: {len(self._state.inventory)} SKUs across 5 warehouses.\n"
                f"Sim date: {self._state.sim_date}"
            ),
            items=[], episode_reward_so_far=0.0, **stats,
        )

    @property
    def state(self) -> InventoryState:
        return self._state

    def step(self, action: InventoryAction) -> InventoryObservation:
        s = self._state
        s.step_count += 1
        max_steps = TASKS[self._task_name]["max_steps"]

        if s.task_completed:
            return self._make_obs(False, True, 0.0, "Episode already completed.", stats=_compute_stats(s))

        act = ActionType(action.action_type)
        reward = 0.0
        message = ""
        items_out: List[ItemRecord] = []
        report_out: Optional[ReportRecord] = None
        order_out: Optional[OrderRecord] = None
        error_out: Optional[str] = None
        done = False
        ctx: Dict[str, Any] = {}

        try:
            if act == ActionType.LIST_ITEMS:
                reward, message, items_out, ctx = self._handle_list(action)
            elif act == ActionType.GET_ITEM:
                reward, message, items_out, ctx = self._handle_get(action)
            elif act == ActionType.RESTOCK:
                reward, message, ctx = self._handle_restock(action)
            elif act == ActionType.REMOVE:
                reward, message, ctx = self._handle_remove(action)
            elif act == ActionType.TRANSFER:
                reward, message, ctx = self._handle_transfer(action)
            elif act == ActionType.UPDATE_PRICE:
                reward, message, ctx = self._handle_update_price(action)
            elif act == ActionType.FLAG_EXPIRY:
                reward, message, ctx = self._handle_flag(action)
            elif act == ActionType.GENERATE_REPORT:
                reward, message, report_out, ctx = self._handle_report(action)
            elif act == ActionType.PLACE_ORDER:
                reward, message, order_out, ctx = self._handle_order(action)
            elif act == ActionType.DONE:
                reward, message, done, ctx = self._handle_done()
            else:
                error_out = f"Unknown action type: {act}"
                reward = -0.05
        except Exception as exc:
            error_out = str(exc)
            reward = -0.05
            message = f"Error executing action: {exc}"

        step_r = _step_reward(act, ctx)
        total_step_reward = reward + step_r
        self._episode_reward += total_step_reward

        if s.step_count >= max_steps and not done:
            done = True
            final_score, grade_notes = grade(s)
            bonus = final_score * 2.0
            total_step_reward += bonus
            self._episode_reward += bonus
            s.task_completed = True
            s.score = final_score
            message += f"\n\nMax steps reached. Final score: {final_score:.3f}\n{grade_notes}"

        if done and not s.task_completed:
            s.task_completed = True

        stats = _compute_stats(s)
        return InventoryObservation(
            success=(error_out is None), done=done,
            reward=round(total_step_reward, 4), message=message,
            items=items_out, report=report_out, order=order_out,
            episode_reward_so_far=round(self._episode_reward, 4),
            error=error_out, **stats,
        )

    def _handle_list(self, action):
        s = self._state
        if "listed_items" not in s.required_actions_done:
            s.required_actions_done.append("listed_items")
        items = list(s.inventory.values())
        if action.filter_location:
            items = [i for i in items if i["location"] == action.filter_location]
        if action.filter_expiring_within_days is not None:
            items = [i for i in items if i.get("expiry_date") and
                     0 <= _days_until(i["expiry_date"]) <= action.filter_expiring_within_days]
        if action.filter_low_stock:
            items = [i for i in items if i["quantity"] < i["reorder_point"]]
        records = [_item_to_record(i) for i in items]
        msg = f"Listed {len(records)} item(s)."
        return 0.0, msg, records, {}

    def _handle_get(self, action):
        sku = action.sku
        if not sku or sku not in self._state.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        return 0.0, f"Retrieved item {sku}.", [_item_to_record(self._state.inventory[sku])], {}

    def _handle_restock(self, action):
        s = self._state
        sku, qty = action.sku, action.quantity
        if not sku or sku not in s.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        if not qty or qty < 1:
            raise ValueError("quantity must be >= 1.")
        item = s.inventory[sku]
        was_below = item["quantity"] < item["reorder_point"]
        item["quantity"] += qty
        key = f"restock:{sku}"
        if key not in s.required_actions_done:
            s.required_actions_done.append(key)
        msg = f"Restocked {sku} (+{qty}). New qty: {item['quantity']}. {'Below reorder point.' if was_below else 'Stock adequate.'}"
        return 0.0, msg, {"was_below_reorder": was_below}

    def _handle_remove(self, action):
        s = self._state
        sku = action.sku
        if not sku or sku not in s.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        item = s.inventory[sku]
        qty = min(action.quantity or item["quantity"], item["quantity"])
        d = _days_until(item.get("expiry_date"))
        was_expired = d is not None and d < 0
        s.disposed_log.append({"sku": sku, "name": item["name"], "quantity": qty,
                                "batch_id": item.get("batch_id"), "was_expired": was_expired,
                                "removed_at": datetime.utcnow().isoformat()})
        item["quantity"] -= qty
        msg = f"Removed {qty} unit(s) of {sku}. {'Reason: EXPIRED.' if was_expired else 'Reason: manual removal (item was NOT expired).'}"
        return 0.0, msg, {"was_expired": was_expired}

    def _handle_transfer(self, action):
        s = self._state
        sku, qty = action.sku, action.quantity
        src, dst = action.from_location, action.to_location
        if not sku or sku not in s.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        if not qty or qty < 1:
            raise ValueError("quantity must be >= 1.")
        item = s.inventory[sku]
        if item["location"] != src:
            raise ValueError(f"{sku} is in {item['location']}, not {src}.")
        if item["quantity"] < qty:
            raise ValueError(f"Insufficient stock: have {item['quantity']}, need {qty}.")
        item["quantity"] -= qty
        dst_sku = f"{sku}-{dst}"
        if dst_sku in s.inventory:
            s.inventory[dst_sku]["quantity"] += qty
        else:
            new_item = copy.deepcopy(item)
            new_item["location"] = dst
            new_item["quantity"] = qty
            new_item["sku"] = dst_sku
            s.inventory[dst_sku] = new_item
        s.required_actions_done.append(f"transfer:{sku}:{src}->{dst}")
        return 0.0, f"Transferred {qty} x {sku} from {src} to {dst}.", {}

    def _handle_update_price(self, action):
        s = self._state
        sku, new_price = action.sku, action.new_price
        if not sku or sku not in s.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        if not new_price or new_price <= 0:
            raise ValueError("new_price must be > 0.")
        item = s.inventory[sku]
        old_price = item["unit_price"]
        item["unit_price"] = new_price
        pct = (old_price - new_price) / old_price * 100
        is_markdown = new_price < old_price
        d = _days_until(item.get("expiry_date"))
        near = d is not None and 0 <= d <= _NEAR_EXPIRY_DAYS
        msg = f"Price {sku}: ${old_price:.2f} -> ${new_price:.2f} ({'markdown' if is_markdown else 'markup'} {abs(pct):.1f}%)."
        return 0.0, msg, {"is_markdown": is_markdown and near}

    def _handle_flag(self, action):
        s = self._state
        sku, batch = action.sku, action.batch_id
        item = None
        if sku and sku in s.inventory:
            item = s.inventory[sku]
            batch = batch or item.get("batch_id")
        elif batch:
            for it in s.inventory.values():
                if it.get("batch_id") == batch:
                    item = it
                    break
        if item is None:
            raise ValueError(f"Cannot find item by SKU={sku!r} or batch={batch!r}.")
        item["is_flagged"] = True
        if batch and batch not in s.flagged_batches:
            s.flagged_batches.append(batch)
        d = _days_until(item.get("expiry_date"))
        was_expired = d is not None and d < 0
        near_expiry = d is not None and 0 <= d <= _NEAR_EXPIRY_DAYS
        msg = f"Flagged batch {batch} ({item['name']}). {'EXPIRED.' if was_expired else f'Expiring in {d} days.' if d is not None else 'No expiry date.'}"
        return 0.0, msg, {"was_expired": was_expired, "near_expiry": near_expiry}

    def _handle_report(self, action):
        s = self._state
        rt = action.report_type
        threshold = action.days_threshold or _NEAR_EXPIRY_DAYS

        if rt in (ReportType.EXPIRY_RISK, "expiry_risk"):
            items_data = sorted(
                [{"sku": i["sku"], "name": i["name"], "days_until_expiry": _days_until(i.get("expiry_date")),
                  "quantity": i["quantity"], "location": i["location"], "batch_id": i.get("batch_id")}
                 for i in s.inventory.values()
                 if i.get("expiry_date") and _days_until(i["expiry_date"]) is not None and _days_until(i["expiry_date"]) <= threshold],
                key=lambda x: x["days_until_expiry"]
            )
            summary = f"Expiry Risk ({threshold}d): {len(items_data)} at risk. Expired: {sum(1 for x in items_data if x['days_until_expiry'] < 0)}."
            key = "generated_expiry_risk_report"
        elif rt in (ReportType.WASTE_SUMMARY, "waste_summary"):
            items_data = s.disposed_log[:]
            summary = f"Waste Summary: {len(s.disposed_log)} events, {sum(r.get('quantity',0) for r in s.disposed_log)} units."
            key = "generated_waste_summary_report"
        elif rt in (ReportType.LOW_STOCK, "low_stock"):
            items_data = sorted(
                [{"sku": i["sku"], "name": i["name"], "quantity": i["quantity"],
                  "reorder_point": i["reorder_point"], "deficit": i["reorder_point"] - i["quantity"]}
                 for i in s.inventory.values() if i["quantity"] < i["reorder_point"]],
                key=lambda x: x["deficit"], reverse=True
            )
            summary = f"Low Stock: {len(items_data)} SKUs below reorder. Deficit: {sum(x['deficit'] for x in items_data)} units."
            key = "generated_low_stock_report"
        elif rt in (ReportType.FULL_AUDIT, "full_audit"):
            stats = _compute_stats(s)
            items_data = [{"sku": i["sku"], "qty": i["quantity"]} for i in s.inventory.values()]
            summary = f"Full Audit: {len(s.inventory)} SKUs, {stats['expired_items_count']} expired."
            key = "generated_full_audit_report"
        else:
            raise ValueError(f"Unknown report_type: {rt!r}")

        if key not in s.required_actions_done:
            s.required_actions_done.append(key)
        report = ReportRecord(report_type=str(rt), generated_at=datetime.utcnow().isoformat(),
                              summary=summary, items=items_data)
        return 0.0, summary, report, {}

    def _handle_order(self, action):
        s = self._state
        sku = action.sku
        qty = action.order_quantity or action.quantity or 50
        supplier = action.supplier or "DefaultSupplier"
        if not sku or sku not in s.inventory:
            raise ValueError(f"SKU '{sku}' not found.")
        item = s.inventory[sku]
        was_below = item["quantity"] < item["reorder_point"]
        order_id = str(uuid.uuid4())[:8].upper()
        order_rec = {"order_id": order_id, "sku": sku, "quantity": qty, "supplier": supplier,
                     "status": "confirmed", "placed_at": datetime.utcnow().isoformat()}
        s.orders.append(order_rec)
        msg = f"Order #{order_id}: {qty} x {sku} from {supplier}. {'Low stock triggered.' if was_below else 'Stock was adequate.'}"
        return 0.0, msg, OrderRecord(**order_rec), {"was_below_reorder": was_below}

    def _handle_done(self):
        s = self._state
        final_score, grade_notes = grade(s)
        s.task_completed = True
        s.score = final_score
        bonus = final_score * 3.0
        msg = f"Agent called DONE.\nFinal score: {final_score:.3f} / 1.0\n{grade_notes}\nSteps: {s.step_count}"
        return bonus, msg, True, {}

    def _make_obs(self, success, done, reward, message, stats=None):
        return InventoryObservation(
            success=success, done=done, reward=reward, message=message,
            episode_reward_so_far=round(self._episode_reward, 4), **(stats or {}),
        )
