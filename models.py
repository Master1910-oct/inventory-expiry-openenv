"""
Inventory & Expiry Management Environment — Typed Models
=========================================================
Pydantic models for Action, Observation, and State following the OpenEnv spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All legal action verbs the agent may issue."""
    # Read
    LIST_ITEMS      = "list_items"       # List inventory (optionally filtered)
    GET_ITEM        = "get_item"         # Get a single item by SKU
    # Write
    RESTOCK         = "restock"          # Add quantity to existing SKU
    REMOVE          = "remove"           # Mark as removed/disposed (expired / damaged)
    TRANSFER        = "transfer"         # Move quantity between locations
    UPDATE_PRICE    = "update_price"     # Change unit price of a SKU
    FLAG_EXPIRY     = "flag_expiry"      # Flag a batch as expiring soon or expired
    # Analytics / Planning
    GENERATE_REPORT = "generate_report"  # Ask for an expiry / waste / stock report
    PLACE_ORDER     = "place_order"      # Place a replenishment order for a SKU
    # Lifecycle
    DONE            = "done"             # Agent signals episode completion


class ReportType(str, Enum):
    EXPIRY_RISK   = "expiry_risk"   # Items expiring within threshold
    WASTE_SUMMARY = "waste_summary" # Disposed / lost quantities
    LOW_STOCK     = "low_stock"     # Items below reorder point
    FULL_AUDIT    = "full_audit"    # Complete inventory snapshot


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class InventoryAction(BaseModel):
    """Single action the agent sends to the environment each step."""

    action_type: ActionType = Field(..., description="The operation to perform.")

    # Applicable to GET_ITEM / RESTOCK / REMOVE / TRANSFER / UPDATE_PRICE / FLAG_EXPIRY / PLACE_ORDER
    sku: Optional[str] = Field(None, description="Stock-Keeping Unit identifier.")

    # Applicable to RESTOCK / REMOVE / TRANSFER
    quantity: Optional[int] = Field(None, ge=1, description="Number of units.")

    # Applicable to TRANSFER
    from_location: Optional[str] = Field(None, description="Source location / warehouse.")
    to_location: Optional[str] = Field(None, description="Destination location / warehouse.")

    # Applicable to UPDATE_PRICE
    new_price: Optional[float] = Field(None, gt=0, description="New unit price in USD.")

    # Applicable to FLAG_EXPIRY
    batch_id: Optional[str] = Field(None, description="Batch / lot identifier.")
    expiry_date: Optional[str] = Field(None, description="ISO date string YYYY-MM-DD.")

    # Applicable to GENERATE_REPORT
    report_type: Optional[ReportType] = Field(None)
    days_threshold: Optional[int] = Field(
        None, ge=1, le=365,
        description="Days threshold for expiry-risk report.",
    )

    # Applicable to LIST_ITEMS
    filter_location: Optional[str] = None
    filter_expiring_within_days: Optional[int] = Field(None, ge=1)
    filter_low_stock: Optional[bool] = None

    # Applicable to PLACE_ORDER
    order_quantity: Optional[int] = Field(None, ge=1)
    supplier: Optional[str] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ItemRecord(BaseModel):
    sku: str
    name: str
    category: str
    location: str
    quantity: int
    unit_price: float
    reorder_point: int
    expiry_date: Optional[str] = None   # ISO date or None
    batch_id: Optional[str] = None
    days_until_expiry: Optional[int] = None  # Negative = already expired
    is_flagged: bool = False


class OrderRecord(BaseModel):
    order_id: str
    sku: str
    quantity: int
    supplier: str
    status: str  # "pending" | "confirmed" | "delivered"
    placed_at: str  # ISO datetime


class ReportRecord(BaseModel):
    report_type: str
    generated_at: str
    summary: str
    items: List[Dict[str, Any]] = Field(default_factory=list)


class InventoryObservation(BaseModel):
    """Everything the agent can see after each step."""

    # Step result
    success: bool
    done: bool
    reward: float
    message: str                         # Human-readable feedback

    # Inventory snapshot (returned for LIST / GET)
    items: List[ItemRecord] = Field(default_factory=list)

    # Report payload (returned for GENERATE_REPORT)
    report: Optional[ReportRecord] = None

    # Order confirmation (returned for PLACE_ORDER)
    order: Optional[OrderRecord] = None

    # Running episode stats (always present)
    total_items_in_stock: int = 0
    expired_items_count: int = 0
    near_expiry_items_count: int = 0     # Within 7 days
    total_waste_units: int = 0
    total_orders_placed: int = 0
    episode_reward_so_far: float = 0.0

    # Errors
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# State  (server-side, returned by state())
# ---------------------------------------------------------------------------

@dataclass
class InventoryState:
    """Full server-side state of one episode."""
    episode_id: str = ""
    step_count: int = 0
    task_name: str = "easy_expiry_check"
    task_completed: bool = False
    score: float = 0.0

    # Mutable inventory store: sku -> ItemRecord dict
    inventory: Dict[str, Dict] = field(default_factory=dict)

    # Disposed items log
    disposed_log: List[Dict] = field(default_factory=list)

    # Orders placed
    orders: List[Dict] = field(default_factory=list)

    # Flags set by agent
    flagged_batches: List[str] = field(default_factory=list)

    # Grader tracking
    required_actions_done: List[str] = field(default_factory=list)

    # Simulated "today" date for reproducibility
    sim_date: str = ""
