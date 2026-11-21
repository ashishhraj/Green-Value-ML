"""Guards for the selection protocol: nothing is tuned or selected on the test set."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_price_decisions_ignores_test_labels_for_choices():
    """Changing test labels must not change the threshold, the rule or any validation number."""
    import run_experiments as R
    rng = np.random.default_rng(1)
    n = 300
    y_va, y_te = rng.integers(0, 2, n), rng.integers(0, 2, n)
    s_va = np.clip(0.4 * y_va + rng.uniform(0, 0.6, n), 0, 1)
    s_te = np.clip(0.4 * y_te + rng.uniform(0, 0.6, n), 0, 1)
    a_va = pd.DataFrame({"monthly": rng.uniform(20, 120, n)})
    a_te = pd.DataFrame({"monthly": rng.uniform(20, 120, n)})
    r1 = R.price_decisions("churn", "clf", s_va, y_va, a_va, s_te, y_te, a_te, 0)
    r2 = R.price_decisions("churn", "clf", s_va, y_va, a_va, s_te, 1 - y_te, a_te, 0)
    for k in ("thr_D1", "rule", "val_D1", "val_D2", "val_aware", "val_oracle", "lam_D2raw"):
        assert r1[k] == r2[k], k


def test_analysis_selects_on_validation_columns():
    import analysis as A
    src = inspect.getsource(A.key_models)
    assert 'cap="vcap"' in src and 'acc="vacc"' in src
    main_src = inspect.getsource(A.main)
    assert "key_models(sub.reset_index())" in main_src          # default = validation selection


def test_sensitivity_never_tunes_on_test():
    import sensitivity as S
    src = inspect.getsource(S.price)
    for call in ("tune_order_uplift(s_va", "tune_global_threshold(s_va", "fit_platt(s_va"):
        assert call in src
    assert "s_te, y_te" not in src.split("tune_")[1][:60]
