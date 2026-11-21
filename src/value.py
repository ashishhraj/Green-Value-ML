"""Business-value functions and decision policies.

Unified utility framework for the three classification scenarios
------------------------------------------------------------------
A model's scores are turned into an ACTION on each case (flag a transaction,
target a customer with an offer, reject a loan). The money a policy earns is

    V(policy) = const + sum_{i flagged} [ y_i * Pos_i  -  (1 - y_i) * Neg_i ]

Pos_i : value of acting on a true positive  (example-dependent)
Neg_i : cost of acting on a false positive  (example-dependent)
const : value of the "do nothing" baseline

    scenario  action        Pos_i                         Neg_i              const
    fraud     block/review  amount_i - review_cost        review_cost        0
    churn     retention     accept*horizon*monthly_i - c  c                  0
    credit    reject loan   LGD * amount_i (loss avoided) margin * amount_i  value(approve all)

Because every scenario uses this one equation, value numbers are directly
auditable and the "oracle" (perfect foresight) value is simply
    const + sum_{y_i = 1} max(Pos_i, 0).

For the synthetic retail scenario the action is an order quantity and the
value is the newsvendor cost saved against a seasonal-naive ordering rule.
"""
from __future__ import annotations

import numpy as np

from config import CHURN, CREDIT, FRAUD, RETAIL


# ---------------------------------------------------------------------------
# Classification scenarios
# ---------------------------------------------------------------------------
def utilities(scenario: str, y: np.ndarray, aux) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (Pos_i, Neg_i, const) for the rows in `aux`."""
    if scenario == "fraud":
        amt = aux["amount"].values
        pos = amt - FRAUD["review_cost"]
        neg = np.full_like(amt, FRAUD["review_cost"])
        return pos, neg, 0.0
    if scenario == "churn":
        m = aux["monthly"].values
        pos = CHURN["accept_rate"] * CHURN["horizon_months"] * m - CHURN["offer_cost"]
        neg = np.full_like(m, CHURN["offer_cost"])
        return pos, neg, 0.0
    if scenario == "credit":
        amt = aux["amount"].values
        pos = CREDIT["lgd"] * amt
        neg = CREDIT["margin"] * amt
        approve_all = np.sum(np.where(y == 1, -CREDIT["lgd"] * amt, CREDIT["margin"] * amt))
        return pos, neg, float(approve_all)
    raise ValueError(scenario)


def policy_value(flag: np.ndarray, y: np.ndarray, pos, neg, const) -> float:
    flag = flag.astype(bool)
    return float(const + np.sum(np.where(y[flag] == 1, pos[flag], -neg[flag])))


def oracle_value(y, pos, neg, const) -> float:
    return float(const + np.sum(np.maximum(pos[y == 1], 0)))


def _thr_grid(scores: np.ndarray) -> np.ndarray:
    q = np.unique(np.quantile(scores, np.linspace(0, 1, 401)))
    return np.unique(np.concatenate([q, [0.5, 1.01]]))


def tune_global_threshold(scores, y, pos, neg, const) -> float:
    """D1: one probability cut-off chosen to maximise VALIDATION value."""
    best_t, best_v = 0.5, -np.inf
    for t in _thr_grid(scores):
        v = policy_value(scores >= t, y, pos, neg, const)
        if v > best_v:
            best_t, best_v = t, v
    return float(best_t)


def bayes_flags(scores, pos, neg, lam: float) -> np.ndarray:
    """D2: example-dependent expected-value rule.

    Flag i  iff  p_i * Pos_i  >  lam * (1 - p_i) * Neg_i.
    lam = 1 is the textbook Bayes-optimal rule for calibrated p; lam is tuned on
    validation to absorb mis-calibration of tree/NN scores.
    """
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return (pos > 0) & (p * pos > lam * (1 - p) * neg)


def tune_bayes_lambda(scores, y, pos, neg, const) -> float:
    best_l, best_v = 1.0, -np.inf
    for lam in np.logspace(-2, 2, 81):
        v = policy_value(bayes_flags(scores, pos, neg, lam), y, pos, neg, const)
        if v > best_v:
            best_l, best_v = lam, v
    return float(best_l)


# ---------------------------------------------------------------------------
# Probability quality: checked BEFORE the Bayes-style rule is used
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(scores, y):
    """Platt scaling: logistic regression on logit(score). Returns a callable."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return lambda s: np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(scores)[:, None], y)
    return lambda s: lr.predict_proba(_logit(s)[:, None])[:, 1]


def crossfit_platt(scores, y, seed: int = 0, k: int = 2):
    """Out-of-fold Platt-calibrated validation probabilities (no row is calibrated by itself)."""
    from sklearn.model_selection import StratifiedKFold
    out = np.empty(len(y), dtype=float)
    for fit_idx, app_idx in StratifiedKFold(k, shuffle=True, random_state=seed).split(scores, y):
        out[app_idx] = fit_platt(scores[fit_idx], y[fit_idx])(scores[app_idx])
    return out


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p = np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(bins) if np.any(b == k)))


def crossfit_threshold_value(scores, y, pos, neg, const, seed: int = 0) -> float:
    """Validation value of D1 when the threshold is tuned on one half and scored on the other
    (2-fold, swapped). Makes D1 comparable with the out-of-fold calibrated D2 for rule selection."""
    from sklearn.model_selection import StratifiedKFold
    total = const
    for fit_idx, app_idx in StratifiedKFold(2, shuffle=True, random_state=seed).split(scores, y):
        t = tune_global_threshold(scores[fit_idx], y[fit_idx], pos[fit_idx], neg[fit_idx], 0.0)
        total += policy_value(scores[app_idx] >= t, y[app_idx], pos[app_idx], neg[app_idx], 0.0)
    return float(total)


# ---------------------------------------------------------------------------
# Retail (regression) scenario: newsvendor cost
# ---------------------------------------------------------------------------
def newsvendor_row_cost(order, demand, unit_cost, price) -> np.ndarray:
    over = np.maximum(order - demand, 0) * RETAIL["holding_frac"] * unit_cost   # spoiled stock
    under = np.maximum(demand - order, 0) * (price - unit_cost)                 # lost margin
    return over + under


def newsvendor_cost(order, demand, unit_cost, price) -> float:
    return float(np.sum(newsvendor_row_cost(order, demand, unit_cost, price)))


def retail_value(order, demand, aux) -> float:
    """Cost saved versus the seasonal-naive rule 'order what sold 7 days ago'."""
    base = newsvendor_cost(aux["lag_7"].values, demand, aux["unit_cost"].values, aux["price"].values)
    return base - newsvendor_cost(order, demand, aux["unit_cost"].values, aux["price"].values)


def retail_oracle(demand, aux) -> float:
    return retail_value(demand, demand, aux)


def tune_order_uplift(pred, demand, aux) -> float:
    """Decision layer for forecasting: order = forecast * (1 + alpha), alpha tuned on validation
    (the newsvendor critical ratio says over-ordering pays when margin >> spoilage cost)."""
    best_a, best_v = 0.0, -np.inf
    for a in np.linspace(-0.3, 0.8, 45):
        v = retail_value(np.maximum(pred * (1 + a), 0), demand, aux)
        if v > best_v:
            best_a, best_v = a, v
    return float(best_a)


def crossfit_uplift_value(pred, demand, aux, seed: int = 0) -> float:
    """Validation value of the order-uplift rule, tuned on one random half and scored on the other."""
    half = np.random.default_rng(seed).permutation(len(demand)) < len(demand) // 2
    total = 0.0
    for fit, app in ((half, ~half), (~half, half)):
        a = tune_order_uplift(pred[fit], demand[fit], aux[fit].reset_index(drop=True))
        total += retail_value(np.maximum(pred[app] * (1 + a), 0), demand[app], aux[app].reset_index(drop=True))
    return float(total)
