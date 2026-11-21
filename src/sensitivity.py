"""Robustness of the conclusions to the business assumptions (paper Sec. IV-H).

For every alternative cost setting the saved predictions are re-priced exactly as in the main
study, with NO use of the test set for any choice:
    1. decision rules are re-tuned on VALIDATION scores (threshold / order uplift; the Platt
       calibrator does not depend on costs and is refitted on validation),
    2. the rule (D1 vs D2) is chosen on cross-fitted validation value,
    3. the value-best model and the Green Value Point are chosen on mean VALIDATION value capture,
    4. the test set only reports the value of those choices.
No model is retrained, so this analysis costs almost no extra carbon.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import value as V  # noqa: E402
from config import CHURN, CREDIT, EQUIV_MARGIN_PP, FRAUD, GVP_TOL, RESULTS_DIR, RETAIL, ROOT  # noqa: E402
from data import LOADERS  # noqa: E402

GRID = {
    "fraud": [dict(review_cost=c) for c in (2, 5, 10, 20)],
    "churn": [dict(accept_rate=a, offer_cost=c) for a, c in itertools.product((0.15, 0.30, 0.50), (10, 25, 50))],
    "credit": [dict(margin=m, lgd=l) for m, l in itertools.product((0.05, 0.10, 0.20), (0.4, 0.6, 0.8))],
    "retail": [dict(holding_frac=h) for h in (0.1, 0.3, 0.6, 1.0)],
    "retail_real": [dict(holding_frac=h, real_cost_ratio=c)
                    for h, c in itertools.product((0.1, 0.3, 0.6, 1.0), (0.4, 0.6, 0.8))],
}
TARGET = {"fraud": FRAUD, "churn": CHURN, "credit": CREDIT, "retail": RETAIL, "retail_real": RETAIL}


def price(sc, task, z, d, seed, cal_cache):
    """Validation capture (selection) and test capture (reporting) for one saved run."""
    a_va = d["aux"].iloc[z["idx_va"]].reset_index(drop=True)
    a_te = d["aux"].iloc[z["idx"]].reset_index(drop=True)
    s_va, y_va, s_te, y_te = z["score_va"], z["y_va"], z["score"], z["y"]
    if task != "clf":
        if "ref_price" in a_va:
            for a in (a_va, a_te):
                a["unit_cost"] = a["ref_price"] * RETAIL["real_cost_ratio"]
        alpha = V.tune_order_uplift(s_va, y_va, a_va)
        vcap = 100 * V.crossfit_uplift_value(s_va, y_va, a_va, seed=seed) / V.retail_oracle(y_va, a_va)
        tcap = 100 * V.retail_value(np.maximum(s_te * (1 + alpha), 0), y_te, a_te) / V.retail_oracle(y_te, a_te)
        return vcap, tcap
    pv, nv, cv = V.utilities(sc, y_va, a_va)
    pt, nt, ct = V.utilities(sc, y_te, a_te)
    if "pc_va" not in cal_cache:                       # calibration does not depend on business costs
        cal_cache["pc_va"] = V.crossfit_platt(s_va, y_va, seed=seed)
        cal_cache["pc_te"] = V.fit_platt(s_va, y_va)(s_te)
    v1 = V.crossfit_threshold_value(s_va, y_va, pv, nv, cv, seed=seed)
    v2 = V.policy_value(V.bayes_flags(cal_cache["pc_va"], pv, nv, 1.0), y_va, pv, nv, cv)
    if v2 > v1:
        test_v = V.policy_value(V.bayes_flags(cal_cache["pc_te"], pt, nt, 1.0), y_te, pt, nt, ct)
    else:
        t = V.tune_global_threshold(s_va, y_va, pv, nv, cv)
        test_v = V.policy_value(s_te >= t, y_te, pt, nt, ct)
    return 100 * max(v1, v2) / V.oracle_value(y_va, pv, nv, cv), 100 * test_v / V.oracle_value(y_te, pt, nt, ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=GRID, default=list(GRID))
    selected = ap.parse_args().scenarios
    runs = pd.read_csv(RESULTS_DIR / "runs.csv")
    carbon = runs.groupby(["scenario", "model"])["train_kg"].mean() * 1e6
    out = []
    for sc, settings in GRID.items():
        if sc not in selected:
            continue
        d = LOADERS[sc]()
        base = dict(TARGET[sc])
        models = sorted(runs[runs.scenario == sc]["model"].unique())
        seeds = sorted(runs[runs.scenario == sc]["seed"].unique())
        preds = {(m, s): np.load(RESULTS_DIR / "preds" / f"{sc}_{m}_s{s}.npz") for m in models for s in seeds}
        caches = {k: {} for k in preds}
        for setting in settings:
            TARGET[sc].update(setting)
            vc, tc = {m: [] for m in models}, {m: [] for m in models}
            for (m, s), z in preds.items():
                v, t = price(sc, d["task"], z, d, int(s), caches[(m, s)])
                vc[m].append(v); tc[m].append(t)
            vcap = pd.Series({m: np.mean(x) for m, x in vc.items()})
            tcap = pd.Series({m: np.mean(x) for m, x in tc.items()})
            mg = carbon.loc[sc].reindex(vcap.index)
            best = vcap.idxmax(); vmax = vcap.max()
            gvp = mg.loc[vcap[vcap >= vmax - GVP_TOL * abs(vmax)].index].idxmin()
            out.append(dict(scenario=sc, **setting, value_best=best, gvp=gvp,
                            best_test_cap=tcap[best], gvp_test_cap=tcap[gvp],
                            gvp_minus_best_pp=tcap[gvp] - tcap[best],
                            saving_pct=100 * (1 - mg[gvp] / mg[best]),
                            saving_vs_hpo_pct=100 * (1 - mg[gvp] / mg["HGB-HPO"]),
                            gvp_minus_hpo_pp=tcap[gvp] - tcap["HGB-HPO"],
                            rho_carbon_value=spearmanr(mg, tcap).statistic))
            TARGET[sc].clear(); TARGET[sc].update(base)
            print(out[-1], flush=True)
    df = pd.DataFrame(out)
    if len(selected) < len(GRID) and (RESULTS_DIR / "sensitivity.csv").exists():
        previous = pd.read_csv(RESULTS_DIR / "sensitivity.csv")
        df = pd.concat([previous[~previous.scenario.isin(selected)], df], ignore_index=True)
    df = df.sort_values(["scenario"], kind="stable").reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    core = df[df.scenario != "retail_real"]
    real = df[df.scenario == "retail_real"]
    within = df["gvp_minus_best_pp"] >= -EQUIV_MARGIN_PP
    summ = {"n_settings": int(len(df)),
            "gvp_saves_90_pct": int((df["saving_pct"] >= 90).sum()),
            "gvp_is_value_best": int((df["gvp"] == df["value_best"]).sum()),
            "gvp_within_margin": int(within.sum()),
            "gvp_saves_90_and_within": int(((df["saving_pct"] >= 90) & within).sum()),
            "hpo_is_value_best": int((df["value_best"] == "HGB-HPO").sum()),
            "hpo_saves_90": int((df["saving_vs_hpo_pct"] >= 90).sum()),
            "hpo_median_saving": float(df["saving_vs_hpo_pct"].median()),
            "gvp_within_margin_of_hpo": int((df["gvp_minus_hpo_pp"] >= -EQUIV_MARGIN_PP).sum()),
            "gvp_beats_hpo": int((df["gvp_minus_hpo_pp"] >= 0).sum()),
            "median_saving": float(df["saving_pct"].median()),
            "core_n": int(len(core)), "core_saves_90": int((core["saving_pct"] >= 90).sum()),
            "core_median_saving": float(core["saving_pct"].median()),
            "rho_min": float(df["rho_carbon_value"].min()), "rho_max": float(df["rho_carbon_value"].max()),
            "rho_median": float(df["rho_carbon_value"].median()),
            "worst_gvp_minus_best_pp": float(df["gvp_minus_best_pp"].min()),
            "by_scenario_median_saving": df.groupby("scenario")["saving_pct"].median().round(1).to_dict(),
            "distinct_value_best": df.groupby("scenario")["value_best"].nunique().to_dict(),
            "retail_real_n_settings": int(len(real)),
            "retail_real_saves_90_pct": int((real["saving_pct"] >= 90).sum()),
            "retail_real_median_saving": float(real["saving_pct"].median()) if len(real) else float("nan")}
    json.dump(summ, open(RESULTS_DIR / "sensitivity_summary.json", "w"), indent=2)
    macros = {"SensN": summ["n_settings"], "SensNinety": summ["gvp_saves_90_pct"],
              "SensWithin": summ["gvp_within_margin"], "SensNinetyWithin": summ["gvp_saves_90_and_within"],
              "SensSame": summ["gvp_is_value_best"], "SensWorst": f"{summ['worst_gvp_minus_best_pp']:.1f}",
              "SensMedianSave": f"{summ['median_saving']:.0f}", "SensRhoMin": f"{summ['rho_min']:+.2f}",
              "SensRhoMax": f"{summ['rho_max']:+.2f}", "SensRhoMed": f"{summ['rho_median']:+.2f}",
              "SensHpoBest": summ["hpo_is_value_best"], "SensHpoNinety": summ["hpo_saves_90"],
              "SensHpoMedian": f"{summ['hpo_median_saving']:.1f}",
              "SensHpoWithin": summ["gvp_within_margin_of_hpo"], "SensBeatsHpo": summ["gvp_beats_hpo"],
              "SensCoreN": summ["core_n"], "SensCoreNinety": summ["core_saves_90"],
              "SensCoreMedian": f"{summ['core_median_saving']:.0f}",
              "SensCreditDistinct": summ["distinct_value_best"].get("credit", 0),
              "SensRealN": summ["retail_real_n_settings"], "SensRealNinety": summ["retail_real_saves_90_pct"],
              "SensRealMedian": f"{summ['retail_real_median_saving']:.1f}"}
    (ROOT / "paper" / "numbers_sens.tex").write_text(
        "% auto-generated by src/sensitivity.py -- do not edit by hand\n"
        + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
