"""Unit tests for the business-value framework (hand-computed cases, no data download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import value as V  # noqa: E402
from config import CHURN, CREDIT, FRAUD, RETAIL  # noqa: E402


def test_fraud_utilities_and_policy_value():
    y = np.array([1, 0, 1, 0])
    aux = pd.DataFrame({"amount": [100.0, 50.0, 3.0, 10.0]})
    pos, neg, const = V.utilities("fraud", y, aux)
    r = FRAUD["review_cost"]
    assert np.allclose(pos, aux["amount"] - r) and np.allclose(neg, r) and const == 0
    # flag rows 0 and 1: gain (100 - r) on the fraud, lose r on the false alarm
    assert V.policy_value(np.array([1, 1, 0, 0]), y, pos, neg, const) == pytest.approx(100 - 2 * r)
    # oracle ignores frauds smaller than the review cost
    assert V.oracle_value(y, pos, neg, const) == pytest.approx(100 - r)


def test_churn_utilities():
    y = np.array([1, 0])
    aux = pd.DataFrame({"monthly": [80.0, 80.0]})
    pos, neg, _ = V.utilities("churn", y, aux)
    expected = CHURN["accept_rate"] * CHURN["horizon_months"] * 80 - CHURN["offer_cost"]
    assert pos[0] == pytest.approx(expected) and neg[1] == pytest.approx(CHURN["offer_cost"])


def test_credit_constant_is_approve_all():
    y = np.array([1, 0])                    # first loan defaults
    aux = pd.DataFrame({"amount": [1000.0, 2000.0]})
    pos, neg, const = V.utilities("credit", y, aux)
    assert const == pytest.approx(-CREDIT["lgd"] * 1000 + CREDIT["margin"] * 2000)
    # rejecting only the defaulter recovers the oracle
    assert V.policy_value(np.array([1, 0]), y, pos, neg, const) == pytest.approx(V.oracle_value(y, pos, neg, const))


def test_threshold_tuning_finds_separating_cut():
    s = np.array([0.1, 0.2, 0.8, 0.9]); y = np.array([0, 0, 1, 1])
    pos = np.full(4, 10.0); neg = np.full(4, 1.0)
    t = V.tune_global_threshold(s, y, pos, neg, 0.0)
    assert 0.2 < t <= 0.8
    assert V.policy_value(s >= t, y, pos, neg, 0.0) == pytest.approx(20.0)


def test_bayes_rule_matches_break_even():
    # act iff p*Pos > (1-p)*Neg  <=>  p > Neg / (Pos + Neg) = 0.25 here
    pos, neg = np.full(3, 3.0), np.full(3, 1.0)
    p = np.array([0.2, 0.25, 0.3])
    assert V.bayes_flags(p, pos, neg, 1.0).tolist() == [False, False, True]


def test_platt_calibration_is_cross_fitted_and_monotone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(0.5 * y + rng.normal(0.25, 0.2, 400), 0.01, 0.99)
    p = V.crossfit_platt(s, y)
    assert p.shape == s.shape and np.all((p > 0) & (p < 1))
    f = V.fit_platt(s, y)
    grid = np.linspace(0.05, 0.95, 10)
    assert np.all(np.diff(f(grid)) > 0)
    assert V.ece(y, p) < V.ece(y, s) + 0.05


def test_newsvendor_value_and_oracle():
    demand = np.array([10.0, 10.0])
    aux = pd.DataFrame({"unit_cost": [5.0, 5.0], "price": [8.0, 8.0], "lag_7": [12.0, 7.0]})
    # baseline: 2 over (2*h*5) + 3 under (3*3)
    base = 2 * RETAIL["holding_frac"] * 5 + 3 * 3
    assert V.retail_oracle(demand, aux) == pytest.approx(base)
    assert V.retail_value(demand, demand, aux) == pytest.approx(base)
    assert V.newsvendor_row_cost(np.array([12.0]), np.array([10.0]), np.array([5.0]), np.array([8.0]))[0] \
        == pytest.approx(2 * RETAIL["holding_frac"] * 5)
