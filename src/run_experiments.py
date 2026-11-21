"""Main experiment: 4 scenarios x 10 models x 3 seeds, carbon + business value.

Usage:
    python src/run_experiments.py                 # everything (resumable)
    python src/run_experiments.py --scenarios churn credit
    python src/run_experiments.py --ablation      # fraud data-volume ablation

Writes one row per (scenario, seed, model) to results/runs.csv and the test-set
validation and test scores to results/preds/ so every number in the paper can be regenerated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from carbon import measure                                   # noqa: E402
import env_log                                               # noqa: E402
from config import RESULTS_DIR, SEEDS, SEEDS_BY_SCENARIO, SPLIT                 # noqa: E402
from data import LOADERS                                     # noqa: E402
from models import LADDER, build                             # noqa: E402
import value as V                                            # noqa: E402

RUNS = RESULTS_DIR / "runs.csv"
PRED_DIR = RESULTS_DIR / "preds"


# ---------------------------------------------------------------------------
def split(d: dict, seed: int):
    n = len(d["y"])
    idx = np.arange(n)
    if d["task"] == "reg":                      # chronological split, no shuffling
        day = d["aux"]["day"].values
        if d.get("retail_real"):
            f0, f1, f2 = d["rolling_fractions"][seed % len(d["rolling_fractions"])]
            n_days = int(day.max()) + 1
            t0, t1, t2 = int(n_days * f0), int(n_days * f1), int(n_days * f2)
            return idx[day < t0], idx[(day >= t0) & (day < t1)], idx[(day >= t1) & (day < t2)]
        cut1, cut2 = np.quantile(np.unique(day), [SPLIT[0], SPLIT[0] + SPLIT[1]])
        return idx[day < cut1], idx[(day >= cut1) & (day < cut2)], idx[day >= cut2]
    tr, rest = train_test_split(idx, train_size=SPLIT[0], stratify=d["y"], random_state=seed)
    va, te = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                              stratify=d["y"][rest], random_state=seed)
    return tr, va, te


def score(model, X, task):
    return model.predict_proba(X)[:, 1] if task == "clf" else np.maximum(model.predict(X), 0)


def timed_inference(model, X, task):
    """Repeat inference until >= ~2 s so tiny per-prediction energy is measurable."""
    t0 = time.perf_counter(); score(model, X, task); dt = time.perf_counter() - t0
    reps = int(min(50, max(1, np.ceil(2.0 / max(dt, 1e-3)))))
    _, m = measure(lambda: [score(model, X, task) for _ in range(reps)])
    n = reps * len(X)
    return {"infer_kwh_per_1k": m["kwh"] / n * 1000, "infer_kg_per_1k": m["kg"] / n * 1000,
            "infer_cpu_s_per_1k": m["cpu_s"] / n * 1000}


def evaluate(scenario, d, tr, va, te, model_name, seed, train_frac=1.0):
    X, y, aux, task = d["X"], d["y"], d["aux"], d["task"]
    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        tr = rng.choice(tr, size=int(len(tr) * train_frac), replace=False)
    rec = dict(scenario=scenario, seed=seed, model=model_name, train_frac=train_frac,
               n_train=len(tr), n_test=len(te))

    if model_name == "Naive":                   # seasonal-naive forecast, no training
        fit_m = dict(kwh=0.0, kg=0.0, wall_s=0.0, cpu_s=0.0)
        s_va, s_te = X.iloc[va]["lag_7"].values, X.iloc[te]["lag_7"].values
        _, im = measure(lambda: [X.iloc[te]["lag_7"].values.copy() for _ in range(50)])
        n = 50 * len(te)
        rec.update(infer_kwh_per_1k=im["kwh"] / n * 1000, infer_kg_per_1k=im["kg"] / n * 1000,
                   infer_cpu_s_per_1k=im["cpu_s"] / n * 1000)
    else:
        model = build(model_name, task, seed)
        _, fit_m = measure(model.fit, X.iloc[tr], y[tr])
        s_va, s_te = score(model, X.iloc[va], task), score(model, X.iloc[te], task)
        rec.update(timed_inference(model, X.iloc[te], task))
    rec.update(train_kwh=fit_m["kwh"], train_kg=fit_m["kg"], train_wall_s=fit_m["wall_s"],
               train_cpu_s=fit_m["cpu_s"])

    y_va, y_te = y[va], y[te]
    a_va, a_te = aux.iloc[va].reset_index(drop=True), aux.iloc[te].reset_index(drop=True)
    rec.update(price_decisions(scenario, task, s_va, y_va, a_va, s_te, y_te, a_te, seed))
    if task == "clf":
        rec.update(auc=roc_auc_score(y_te, s_te), pr_auc=average_precision_score(y_te, s_te),
                   accuracy=accuracy_score(y_te, s_te >= 0.5), f1=f1_score(y_te, s_te >= 0.5))
    else:
        rec.update(mae=mean_absolute_error(y_te, s_te), rmse=mean_squared_error(y_te, s_te) ** 0.5,
                   wape=np.sum(np.abs(y_te - s_te)) / np.sum(y_te))
    if train_frac == 1.0:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        # validation scores are saved too, so every downstream selection/sensitivity step
        # can be tuned on validation data and only *evaluated* on the test set
        np.savez_compressed(PRED_DIR / f"{scenario}_{model_name}_s{seed}.npz",
                            score=s_te, y=y_te, idx=te, score_va=s_va, y_va=y_va, idx_va=va)
    return rec


def price_decisions(scenario, task, s_va, y_va, a_va, s_te, y_te, a_te, seed):
    """Turn validation/test scores into money under D0, D1 and D2.

    Everything that is *chosen* (threshold, order uplift, calibrator, which rule to use)
    is chosen on validation data. The test set is only evaluated.
      D0  default rule (p >= 0.5; order = forecast)
      D1  one threshold / order uplift tuned on validation
      D2  Bayes rule  p*Pos > (1-p)*Neg  on probabilities Platt-calibrated on validation
          (lambda = 1, no tuning). Probability quality (Brier, ECE) is checked on validation
          before and after calibration and logged.
      D2raw  the earlier lambda-tuned rule on raw scores, kept only for comparison.
    The value-aware rule is whichever of D1/D2 has the higher *cross-fitted* validation value.
    """
    if task == "clf":
        pv, nv, cv = V.utilities(scenario, y_va, a_va)
        pt, nt, ct = V.utilities(scenario, y_te, a_te)
        t1 = V.tune_global_threshold(s_va, y_va, pv, nv, cv)
        lam = V.tune_bayes_lambda(s_va, y_va, pv, nv, cv)
        cal = V.fit_platt(s_va, y_va)                       # applied to test
        pc_va = V.crossfit_platt(s_va, y_va, seed=seed)     # out-of-fold, for validation scoring
        pc_te = cal(s_te)
        r = dict(
            value_D0=V.policy_value(s_te >= 0.5, y_te, pt, nt, ct),
            value_D1=V.policy_value(s_te >= t1, y_te, pt, nt, ct),
            value_D2=V.policy_value(V.bayes_flags(pc_te, pt, nt, 1.0), y_te, pt, nt, ct),
            value_D2raw=V.policy_value(V.bayes_flags(s_te, pt, nt, lam), y_te, pt, nt, ct),
            value_oracle=V.oracle_value(y_te, pt, nt, ct), thr_D1=t1, lam_D2raw=lam,
            val_D0=V.policy_value(s_va >= 0.5, y_va, pv, nv, cv),
            val_D1=V.crossfit_threshold_value(s_va, y_va, pv, nv, cv, seed=seed),
            val_D2=V.policy_value(V.bayes_flags(pc_va, pv, nv, 1.0), y_va, pv, nv, cv),
            val_oracle=V.oracle_value(y_va, pv, nv, cv),
            val_brier_raw=V.brier(y_va, s_va), val_ece_raw=V.ece(y_va, s_va),
            val_brier_cal=V.brier(y_va, pc_va), val_ece_cal=V.ece(y_va, pc_va),
            test_brier_raw=V.brier(y_te, s_te), test_ece_raw=V.ece(y_te, s_te),
            test_brier_cal=V.brier(y_te, pc_te), test_ece_cal=V.ece(y_te, pc_te))
        use_d2 = r["val_D2"] > r["val_D1"]
        r["rule"] = "D2" if use_d2 else "D1"
        r["value_aware"] = r["value_D2"] if use_d2 else r["value_D1"]
        r["val_aware"] = max(r["val_D1"], r["val_D2"])
        return r
    alpha = V.tune_order_uplift(s_va, y_va, a_va)
    r = dict(value_D0=V.retail_value(s_te, y_te, a_te),
             value_D1=V.retail_value(np.maximum(s_te * (1 + alpha), 0), y_te, a_te),
             value_oracle=V.retail_oracle(y_te, a_te), thr_D1=alpha,
             val_D0=V.retail_value(s_va, y_va, a_va),
             val_D1=V.crossfit_uplift_value(s_va, y_va, a_va, seed=seed),
             val_oracle=V.retail_oracle(y_va, a_va), rule="D1")
    r["value_aware"], r["val_aware"] = r["value_D1"], r["val_D1"]
    return r


COLUMNS = ["scenario", "seed", "model", "train_frac", "n_train", "n_test",
           "train_kwh", "train_kg", "train_wall_s", "train_cpu_s",
           "infer_kwh_per_1k", "infer_kg_per_1k", "infer_cpu_s_per_1k",
           "auc", "pr_auc", "accuracy", "f1", "mae", "rmse", "wape",
           "value_D0", "value_D1", "value_D2", "value_D2raw", "value_oracle", "value_aware", "rule",
           "thr_D1", "lam_D2raw", "val_D0", "val_D1", "val_D2", "val_oracle", "val_aware",
           "val_brier_raw", "val_ece_raw", "val_brier_cal", "val_ece_cal",
           "test_brier_raw", "test_ece_raw", "test_brier_cal", "test_ece_cal"]


def append(rec, path):
    """Fixed schema so classification and regression rows share one CSV."""
    df = pd.DataFrame([{c: rec.get(c, np.nan) for c in COLUMNS}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["credit", "churn", "retail", "fraud"])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--runs-file", default=str(RUNS), help="output CSV (default results/runs.csv)")
    ap.add_argument("--seeds", nargs="+", type=int, help="override seeds (smoke tests only)")
    ap.add_argument("--models", nargs="+", help="override model list (smoke tests only)")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs_path = Path(args.runs_file)
    if runs_path == RUNS:
        env_log.write()                      # results/run_config.json: versions, hardware, power model

    if args.ablation:
        out = RESULTS_DIR / "ablation_fraud.csv"
        d = LOADERS["fraud"]()
        for seed in SEEDS:
            tr, va, te = split(d, seed)
            for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
                for m in ("Linear", "RF-100", "HGB-100"):
                    rec = evaluate("fraud", d, tr, va, te, m, seed, train_frac=frac)
                    append(rec, out)
                    print(f"[ablation] s{seed} {frac:.2f} {m:8s} kg={rec['train_kg']:.2e} "
                          f"V={rec['value_aware']:,.0f}", flush=True)
        return

    done = set()
    if runs_path.exists():
        prev = pd.read_csv(runs_path)
        done = set(zip(prev.scenario, prev.seed, prev.model))
    for sc in args.scenarios:
        d = LOADERS[sc]()
        print(f"== {sc}: X={d['X'].shape}, positives={d['y'].mean():.4f}" if d["task"] == "clf"
              else f"== {sc}: X={d['X'].shape}", flush=True)
        names = (["Naive"] if d["task"] == "reg" else []) + [n for n, _ in LADDER]
        if args.models:
            names = [n for n in names if n in args.models]
        # warm-up: import/JIT/thread-pool start-up must not be billed to the first model
        wi = np.r_[np.arange(200), np.flatnonzero(d["y"] == 1)[:20]] if d["task"] == "clf" else np.arange(200)
        warm = build("Linear", d["task"], 0); warm.fit(d["X"].iloc[wi], d["y"][wi])
        measure(lambda: None)
        for seed in (args.seeds if args.seeds is not None else SEEDS_BY_SCENARIO[sc]):
            tr, va, te = split(d, seed)
            for m in names:
                if (sc, seed, m) in done:
                    continue
                rec = evaluate(sc, d, tr, va, te, m, seed)
                append(rec, runs_path)
                print(f"{sc:6s} s{seed} {m:8s} train={rec['train_wall_s']:7.1f}s "
                      f"kg={rec['train_kg']:.2e} V0={rec['value_D0']:12,.0f} "
                      f"Vaware={rec['value_aware']:12,.0f} oracle={rec['value_oracle']:12,.0f}", flush=True)


if __name__ == "__main__":
    main()
