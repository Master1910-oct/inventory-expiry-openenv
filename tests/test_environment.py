"""
Inventory & Expiry Management Environment — Test Suite
======================================================
Run with: python -m pytest tests/ -v
or:        python tests/test_environment.py
"""

from __future__ import annotations

import sys
import os
import unittest
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── lightweight stubs so tests run without pydantic installed ──────────────

try:
    from models import InventoryAction, InventoryObservation, InventoryState, ActionType, ReportType
    from tasks import (
        TASKS, get_task_names, seed_state, grade,
        easy_grader, medium_grader, hard_grader,
        _base_catalog, _catalog_to_inventory, _days_until, _TODAY,
    )
    from server.inventory_environment import InventoryEnvironment
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


# ──────────────────────────────────────────────────────────────────────────
# Pure-logic tests (no pydantic, no FastAPI needed)
# ──────────────────────────────────────────────────────────────────────────

class TestCatalogAndTasks(unittest.TestCase):
    """Test seed data integrity."""

    def test_catalog_has_24_products(self):
        cat = _base_catalog()
        self.assertEqual(len(cat), 24)

    def test_all_skus_unique(self):
        cat = _base_catalog()
        skus = [p["sku"] for p in cat]
        self.assertEqual(len(skus), len(set(skus)))

    def test_catalog_has_expired_items(self):
        cat = _base_catalog()
        expired = [p for p in cat if _days_until(p.get("expiry")) < 0]
        self.assertGreaterEqual(len(expired), 2, "Need at least 2 expired items for easy task")

    def test_catalog_has_near_expiry_items(self):
        cat = _base_catalog()
        near = [p for p in cat if 0 <= _days_until(p.get("expiry")) <= 7]
        self.assertGreaterEqual(len(near), 4)

    def test_catalog_has_low_stock_items(self):
        cat = _base_catalog()
        low = [p for p in cat if p["qty"] < p["reorder"]]
        self.assertGreaterEqual(len(low), 2)

    def test_catalog_to_inventory_structure(self):
        inv = _catalog_to_inventory(_base_catalog())
        for sku, item in inv.items():
            self.assertIn("sku", item)
            self.assertIn("quantity", item)
            self.assertIn("unit_price", item)
            self.assertIn("reorder_point", item)
            self.assertEqual(sku, item["sku"])

    def test_task_names(self):
        names = get_task_names()
        self.assertIn("easy_expiry_check", names)
        self.assertIn("medium_restock_transfer", names)
        self.assertIn("hard_full_audit", names)

    def test_task_difficulties(self):
        for name, task in TASKS.items():
            self.assertIn(task["difficulty"], ["easy", "medium", "hard"])
            self.assertGreater(task["max_steps"], 0)


class TestGraderEasy(unittest.TestCase):
    """Verify easy task grader produces correct scores."""

    def _make_state(self) -> "InventoryState":
        from dataclasses import dataclass, field as dc_field
        # Use the real InventoryState
        s = InventoryState()
        s.task_name = "easy_expiry_check"
        s = seed_state(s)
        return s

    def test_zero_score_on_empty_state(self):
        s = self._make_state()
        score, notes = easy_grader(s)
        self.assertAlmostEqual(score, 0.0, places=2)
        self.assertIn("✗", notes)

    def test_full_score_on_perfect_completion(self):
        s = self._make_state()

        # Mark all objectives done
        s.required_actions_done.append("listed_items")
        s.required_actions_done.append("generated_expiry_risk_report")

        # Flag all near-expiry / expired batches
        for item in s.inventory.values():
            if item.get("expiry_date") and _days_until(item["expiry_date"]) <= 7:
                bid = item["batch_id"]
                if bid not in s.flagged_batches:
                    s.flagged_batches.append(bid)

        # Dispose all expired items properly
        for sku, item in list(s.inventory.items()):
            if item.get("expiry_date") and _days_until(item["expiry_date"]) < 0:
                s.disposed_log.append({
                    "sku": sku,
                    "quantity": item["quantity"],
                    "batch_id": item.get("batch_id"),
                    "was_expired": True,
                })
                item["quantity"] = 0

        score, notes = easy_grader(s)
        self.assertGreaterEqual(score, 0.75, f"Expected ≥ 0.75, got {score:.3f} | {notes}")

    def test_penalty_for_removing_non_expired(self):
        s = self._make_state()
        s.required_actions_done.append("listed_items")
        s.required_actions_done.append("generated_expiry_risk_report")

        # Dispose a non-expired item
        for sku, item in s.inventory.items():
            if item.get("expiry_date") and _days_until(item["expiry_date"]) > 0:
                s.disposed_log.append({
                    "sku": sku, "quantity": 1,
                    "batch_id": item.get("batch_id"),
                    "was_expired": False,
                })
                break

        score, notes = easy_grader(s)
        self.assertIn("Penalty", notes)


class TestGraderMedium(unittest.TestCase):
    def _make_state(self) -> "InventoryState":
        s = InventoryState()
        s.task_name = "medium_restock_transfer"
        s = seed_state(s)
        return s

    def test_zero_score_on_empty(self):
        s = self._make_state()
        score, _ = medium_grader(s)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_partial_score_list_only(self):
        s = self._make_state()
        s.required_actions_done.append("listed_items")
        score, notes = medium_grader(s)
        self.assertAlmostEqual(score, 0.20, places=2)
        self.assertIn("✓ Listed inventory", notes)

    def test_transfers_counted(self):
        s = self._make_state()
        s.required_actions_done += [
            "listed_items",
            "transfer:SKU-0011:WH-D->WH-A",
            "transfer:SKU-0012:WH-D->WH-B",
        ]
        score, notes = medium_grader(s)
        self.assertIn("✓", notes)
        self.assertGreaterEqual(score, 0.40)


class TestGraderHard(unittest.TestCase):
    def _make_state(self) -> "InventoryState":
        s = InventoryState()
        s.task_name = "hard_full_audit"
        s = seed_state(s)
        return s

    def test_all_three_reports_needed(self):
        s = self._make_state()
        s.required_actions_done += [
            "listed_items",
            "generated_expiry_risk_report",
            "generated_low_stock_report",
        ]
        score1, _ = hard_grader(s)
        s.required_actions_done.append("generated_waste_summary_report")
        score2, _ = hard_grader(s)
        self.assertGreater(score2, score1)


# ──────────────────────────────────────────────────────────────────────────
# Environment integration tests (requires pydantic)
# ──────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_HAS_DEPS, "pydantic not installed")
class TestEnvironmentReset(unittest.TestCase):

    def test_reset_returns_observation(self):
        env = InventoryEnvironment("easy_expiry_check")
        obs = env.reset()
        self.assertTrue(obs.success)
        self.assertFalse(obs.done)
        self.assertEqual(obs.reward, 0.0)
        self.assertGreater(obs.total_items_in_stock, 0)

    def test_reset_clears_state(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        env.step(InventoryAction(action_type="list_items"))
        env.reset()
        self.assertEqual(env.state.step_count, 0)
        self.assertEqual(len(env.state.disposed_log), 0)

    def test_reset_with_task_switch(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        obs = env.reset(task_name="medium_restock_transfer")
        self.assertTrue(obs.success)
        self.assertEqual(env.state.task_name, "medium_restock_transfer")


@unittest.skipUnless(_HAS_DEPS, "pydantic not installed")
class TestEnvironmentStep(unittest.TestCase):

    def setUp(self):
        self.env = InventoryEnvironment("easy_expiry_check")
        self.env.reset()

    def test_list_items_action(self):
        obs = self.env.step(InventoryAction(action_type="list_items"))
        self.assertTrue(obs.success)
        self.assertGreater(len(obs.items), 0)
        self.assertEqual(len(obs.items), 24)

    def test_list_items_with_location_filter(self):
        obs = self.env.step(InventoryAction(action_type="list_items", filter_location="WH-A"))
        self.assertTrue(obs.success)
        for item in obs.items:
            self.assertEqual(item.location, "WH-A")

    def test_list_items_low_stock_filter(self):
        obs = self.env.step(InventoryAction(action_type="list_items", filter_low_stock=True))
        self.assertTrue(obs.success)
        for item in obs.items:
            self.assertLess(item.quantity, item.reorder_point)

    def test_get_item_known_sku(self):
        obs = self.env.step(InventoryAction(action_type="get_item", sku="SKU-0001"))
        self.assertTrue(obs.success)
        self.assertEqual(len(obs.items), 1)
        self.assertEqual(obs.items[0].sku, "SKU-0001")

    def test_get_item_unknown_sku(self):
        obs = self.env.step(InventoryAction(action_type="get_item", sku="SKU-9999"))
        self.assertFalse(obs.success)
        self.assertIsNotNone(obs.error)

    def test_restock_increases_quantity(self):
        before = self.env.state.inventory["SKU-0009"]["quantity"]
        obs = self.env.step(InventoryAction(action_type="restock", sku="SKU-0009", quantity=50))
        self.assertTrue(obs.success)
        after = self.env.state.inventory["SKU-0009"]["quantity"]
        self.assertEqual(after, before + 50)

    def test_remove_expired_item(self):
        # SKU-0001 is expired (3 days past)
        obs = self.env.step(InventoryAction(action_type="remove", sku="SKU-0001"))
        self.assertTrue(obs.success)
        self.assertIn("EXPIRED", obs.message)
        self.assertEqual(len(self.env.state.disposed_log), 1)
        self.assertTrue(self.env.state.disposed_log[0]["was_expired"])

    def test_remove_non_expired_warns(self):
        obs = self.env.step(InventoryAction(action_type="remove", sku="SKU-0012"))
        self.assertTrue(obs.success)
        self.assertIn("NOT expired", obs.message)

    def test_flag_expiry_batch(self):
        obs = self.env.step(InventoryAction(action_type="flag_expiry", sku="SKU-0001"))
        self.assertTrue(obs.success)
        self.assertIn("B001", self.env.state.flagged_batches)

    def test_flag_expiry_unknown_raises(self):
        obs = self.env.step(InventoryAction(action_type="flag_expiry", sku="SKU-9999"))
        self.assertFalse(obs.success)

    def test_update_price_markdown(self):
        # SKU-0003 price = 1.80
        obs = self.env.step(InventoryAction(action_type="update_price", sku="SKU-0003", new_price=1.20))
        self.assertTrue(obs.success)
        self.assertIn("markdown", obs.message.lower())
        self.assertAlmostEqual(self.env.state.inventory["SKU-0003"]["unit_price"], 1.20)

    def test_generate_expiry_risk_report(self):
        obs = self.env.step(InventoryAction(action_type="generate_report", report_type="expiry_risk", days_threshold=7))
        self.assertTrue(obs.success)
        self.assertIsNotNone(obs.report)
        self.assertGreater(len(obs.report.items), 0)
        self.assertIn("generated_expiry_risk_report", self.env.state.required_actions_done)

    def test_generate_low_stock_report(self):
        obs = self.env.step(InventoryAction(action_type="generate_report", report_type="low_stock"))
        self.assertTrue(obs.success)
        self.assertIsNotNone(obs.report)
        self.assertIn("generated_low_stock_report", self.env.state.required_actions_done)

    def test_generate_waste_summary_report(self):
        self.env.step(InventoryAction(action_type="remove", sku="SKU-0001"))
        obs = self.env.step(InventoryAction(action_type="generate_report", report_type="waste_summary"))
        self.assertTrue(obs.success)
        self.assertGreater(len(obs.report.items), 0)

    def test_place_order_low_stock_item(self):
        obs = self.env.step(InventoryAction(
            action_type="place_order", sku="SKU-0009",
            order_quantity=100, supplier="TestSupplier"
        ))
        self.assertTrue(obs.success)
        self.assertIsNotNone(obs.order)
        self.assertEqual(obs.order.sku, "SKU-0009")
        self.assertEqual(len(self.env.state.orders), 1)

    def test_transfer_moves_stock(self):
        before_a = self.env.state.inventory["SKU-0011"]["quantity"]
        obs = self.env.step(InventoryAction(
            action_type="transfer", sku="SKU-0011",
            quantity=10, from_location="WH-D", to_location="WH-A"
        ))
        self.assertTrue(obs.success)
        after_a = self.env.state.inventory["SKU-0011"]["quantity"]
        self.assertEqual(after_a, before_a - 10)
        self.assertIn("SKU-0011-WH-A", self.env.state.inventory)
        self.assertEqual(self.env.state.inventory["SKU-0011-WH-A"]["quantity"], 10)

    def test_transfer_insufficient_stock_raises(self):
        obs = self.env.step(InventoryAction(
            action_type="transfer", sku="SKU-0011",
            quantity=99999, from_location="WH-D", to_location="WH-A"
        ))
        self.assertFalse(obs.success)

    def test_done_triggers_grading(self):
        # Do minimum work then call done
        self.env.step(InventoryAction(action_type="list_items"))
        obs = self.env.step(InventoryAction(action_type="done"))
        self.assertTrue(obs.done)
        self.assertTrue(self.env.state.task_completed)
        self.assertGreaterEqual(self.env.state.score, 0.0)
        self.assertLessEqual(self.env.state.score, 1.0)

    def test_step_count_increments(self):
        for i in range(3):
            self.env.step(InventoryAction(action_type="list_items"))
        self.assertEqual(self.env.state.step_count, 3)

    def test_episode_reward_accumulates(self):
        obs1 = self.env.step(InventoryAction(action_type="list_items"))
        obs2 = self.env.step(InventoryAction(action_type="list_items"))
        self.assertGreaterEqual(obs2.episode_reward_so_far, obs1.episode_reward_so_far)

    def test_stats_in_observation(self):
        obs = self.env.step(InventoryAction(action_type="list_items"))
        self.assertGreater(obs.total_items_in_stock, 0)
        self.assertGreater(obs.expired_items_count, 0)
        self.assertGreater(obs.near_expiry_items_count, 0)


@unittest.skipUnless(_HAS_DEPS, "pydantic not installed")
class TestRewardShaping(unittest.TestCase):
    """Verify dense reward function gives correct signals."""

    def test_removing_expired_gives_positive_reward(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        obs = env.step(InventoryAction(action_type="remove", sku="SKU-0001"))
        self.assertGreater(obs.reward, 0, "Disposing an expired item should give positive reward")

    def test_removing_fresh_item_gives_negative_reward(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        obs = env.step(InventoryAction(action_type="remove", sku="SKU-0023"))  # 730 days until expiry
        self.assertLess(obs.reward, 0, "Disposing a fresh item should give negative reward")

    def test_restocking_low_stock_gives_positive_reward(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        # SKU-0009 has qty=5, reorder=30
        obs = env.step(InventoryAction(action_type="restock", sku="SKU-0009", quantity=30))
        self.assertGreater(obs.reward, 0)

    def test_done_with_high_score_gives_large_bonus(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()

        # Complete all objectives
        env.step(InventoryAction(action_type="list_items"))
        for sku, item in env.state.inventory.items():
            if item.get("expiry_date"):
                from tasks import _days_until
                d = _days_until(item["expiry_date"])
                if d < 0:
                    env.step(InventoryAction(action_type="remove", sku=sku))
                if d <= 7:
                    env.step(InventoryAction(action_type="flag_expiry", sku=sku, batch_id=item["batch_id"]))
        env.step(InventoryAction(action_type="generate_report", report_type="expiry_risk"))
        obs = env.step(InventoryAction(action_type="done"))
        self.assertGreater(obs.reward, 1.0, "High-score episode should yield large terminal reward")


@unittest.skipUnless(_HAS_DEPS, "pydantic not installed")
class TestMaxStepTermination(unittest.TestCase):
    def test_easy_terminates_at_max_steps(self):
        env = InventoryEnvironment("easy_expiry_check")
        env.reset()
        obs = None
        for _ in range(20):  # more than max_steps=15
            obs = env.step(InventoryAction(action_type="list_items"))
            if obs.done:
                break
        self.assertIsNotNone(obs)
        self.assertTrue(obs.done)


@unittest.skipUnless(_HAS_DEPS, "pydantic not installed")
class TestAllThreeTasks(unittest.TestCase):
    """Smoke-test that all 3 tasks reset and step without crashing."""

    def _smoke(self, task: str):
        env = InventoryEnvironment(task)
        obs = env.reset()
        self.assertTrue(obs.success)
        obs = env.step(InventoryAction(action_type="list_items"))
        self.assertTrue(obs.success)
        self.assertGreater(len(obs.items), 0)

    def test_easy_task(self):
        self._smoke("easy_expiry_check")

    def test_medium_task(self):
        self._smoke("medium_restock_transfer")

    def test_hard_task(self):
        self._smoke("hard_full_audit")


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run pure-logic tests even without pydantic
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    pure_classes = [TestCatalogAndTasks, TestGraderEasy, TestGraderMedium, TestGraderHard]
    for cls in pure_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    if _HAS_DEPS:
        dep_classes = [
            TestEnvironmentReset, TestEnvironmentStep,
            TestRewardShaping, TestMaxStepTermination, TestAllThreeTasks,
        ]
        for cls in dep_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        print(f"Running ALL tests (pydantic available).")
    else:
        print("pydantic not installed — running pure-logic tests only.")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
