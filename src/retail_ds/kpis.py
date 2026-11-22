"""Formulas for the core KPIs defined in Section 9 of the report."""
import numpy as np


def on_shelf_availability(in_stock_days: int, expected_demand_days: int) -> float:
    return in_stock_days / expected_demand_days


def markdown_rate(markdown_value: float, gross_sales: float) -> float:
    return markdown_value / gross_sales


def promotion_roi(incremental_gross_profit: float, promo_cost: float) -> float:
    return incremental_gross_profit / promo_cost


def retention_rate(active_month12: int, active_month0: int) -> float:
    return active_month12 / active_month0


def top_decile_lift(y_true, y_score) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    k = max(1, len(y_true) // 10)
    top = y_true[np.argsort(-y_score)[:k]]
    return float(top.mean() / y_true.mean())
