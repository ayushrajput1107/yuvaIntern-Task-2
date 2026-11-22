import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from retail_ds import business_case as bc
from retail_ds.data import make_transactions
from retail_ds.experiments import sample_size_per_arm
from retail_ds.forecasting import bias, wape
from retail_ds.kpis import top_decile_lift
from retail_ds.segmentation import build_rfm, segment_customers


def test_run_rate_matches_report():
    assert sum(bc.run_rate_benefits(bc.Assumptions()).values()) == pytest.approx(5.65, abs=0.01)


def test_business_case_positive_npv_and_payback():
    r = bc.evaluate()
    assert r["npv"] > 0 and r["payback"] == "Year 2"


def test_breakeven_between_0_and_1():
    assert 0 < bc.breakeven_factor() < 1


def test_wape_and_bias():
    assert wape([10, 10], [9, 11]) == pytest.approx(0.1)
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)


def test_sample_size_positive():
    assert sample_size_per_arm(0.032, 0.034) > 1000


def test_segmentation_returns_all_customers():
    tx = make_transactions(n_customers=300, days=365)
    seg = segment_customers(build_rfm(tx), n_clusters=4)
    assert seg["segment"].nunique() == 4


def test_top_decile_lift_perfect_model():
    y = [1] * 10 + [0] * 90
    assert top_decile_lift(y, y) == pytest.approx(10.0)
