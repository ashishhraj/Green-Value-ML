"""Turn results/runs.csv into every table, figure and number used in the paper.

Selection protocol (the main analysis)
    * The value-best model, the accuracy-best model, the best model at the default rule and the
      Green Value Point (GVP) are all chosen on VALIDATION data (mean over repeated splits).
    * Test data are only used to REPORT the value of those pre-selected choices, to bootstrap
      paired differences and to run the equivalence (TOST) / non-inferiority tests.
    * For transparency, the choices a test-set selection would have made are logged in
      results/key_numbers.json ("test_selected"), but they are never used in the paper.

Outputs
    results/summary_by_model.csv   per scenario x model means / std (validation and test)
    results/key_numbers.json       every number quoted in the paper text
    paper/tables/*.tex             LaTeX tables (auto-generated, never typed by hand)
    paper/figures/*.pdf|png        paper figures
    paper/numbers.tex              LaTeX macros for every number in the prose
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import value as V  # noqa: E402
from carbon import measure  # noqa: E402
from config import (BOOT_SEED, DEPLOYMENT, EQUIV_MARGIN_PP, FIG_DIR, GRIDS, GVP_TOL,  # noqa: E402
                    N_BOOT, RESULTS_DIR, ROOT, SOCIAL_COST_CARBON)
from data import LOADERS  # noqa: E402
from models import LADDER  # noqa: E402

TAB_DIR = ROOT / "paper" / "tables"
PRED_DIR = RESULTS_DIR / "preds"
SCEN = ["fraud", "churn", "credit", "retail"]
TITLE = {"fraud": "Fraud detection (ULB, real)", "churn": "Churn retention (IBM Telco, real)",
         "credit": "Credit approval (German, real)", "retail": "Perishable ordering (synthetic)"}
CUR = {"fraud": "EUR", "churn": "USD", "credit": "DM", "retail": "INR"}
# fixed FX to USD, used ONLY for the order-of-magnitude carbon-cost ratio (Sec. IV-G)
TO_USD = {"EUR": 1.10, "USD": 1.0, "DM": 0.511 * 1.10, "INR": 0.012}
FAMILY = dict(LADDER) | {"Naive": "naive"}
ORDER = ["Naive"] + [n for n, _ in LADDER]
FAM_STYLE = {"naive": ("#4a3aa7", "P"), "linear": ("#2a78d6", "o"), "tree": ("#eb6834", "v"),
             "bagging": ("#1baf7a", "s"), "boosting": ("#eda100", "D"), "neural": ("#e87ba4", "^"),
             "ensemble": ("#008300", "X"), "search": ("#e34948", "*")}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "pdf.fonttype": 42})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _npz(sc, m, seed):
    return np.load(PRED_DIR / f"{sc}_{m}_s{seed}.npz")


def validation_accuracy(r) -> float:
    """ROC-AUC (classification) or 1-WAPE (forecasting) on the VALIDATION split."""
    z = _npz(r.scenario, r.model, r.seed)
    if r.scenario in ("retail", "retail_real"):
        return float(1 - np.sum(np.abs(z["y_va"] - z["score_va"])) / np.sum(z["y_va"]))
    return float(roc_auc_score(z["y_va"], z["score_va"]))


def load():
    df = pd.read_csv(RESULTS_DIR / "runs.csv")
    df = df[df["scenario"].isin(DEPLOYMENT)].copy()   # UCI check is summarised in retail_real_analysis.py
    df["cap0"] = 100 * df["value_D0"] / df["value_oracle"]                 # test, default rule
    df["cap"] = 100 * df["value_aware"] / df["value_oracle"]               # test, value-aware rule
    df["vcap0"] = 100 * df["val_D0"] / df["val_oracle"]                    # validation, default rule
    df["vcap"] = 100 * df["val_aware"] / df["val_oracle"]                  # validation, value-aware (cross-fitted)
    df["acc"] = np.where(df["auc"].notna(), df["auc"], 1 - df["wape"])     # test accuracy
    df["vacc"] = [validation_accuracy(r) for r in df.itertuples()]         # validation accuracy
    df["train_mg"] = df["train_kg"] * 1e6
    dep = df["scenario"].map(DEPLOYMENT)
    df["annual_train_kg"] = [d["retrains"] * tk for d, tk in zip(dep, df["train_kg"])]
    df["annual_infer_kg"] = [d["annual_predictions"] / 1000 * ik for d, ik in zip(dep, df["infer_kg_per_1k"])]
    df["annual_kg"] = df["annual_train_kg"] + df["annual_infer_kg"]
    df["annual_kwh"] = [d["retrains"] * tk + d["annual_predictions"] / 1000 * ik
                        for d, tk, ik in zip(dep, df["train_kwh"], df["infer_kwh_per_1k"])]
    df["annual_value"] = [d["annual_predictions"] * v / n
                          for d, v, n in zip(dep, df["value_aware"], df["n_test"])]
    return df


def summarise(df):
    g = df.groupby(["scenario", "model"])
    s = g.agg(train_mg=("train_mg", "mean"), train_mg_sd=("train_mg", "std"),
              train_s=("train_wall_s", "mean"), cpu_s=("train_cpu_s", "mean"),
              infer_mg_per_1k=("infer_kg_per_1k", lambda x: 1e6 * x.mean()),
              acc=("acc", "mean"), acc_sd=("acc", "std"), vacc=("vacc", "mean"),
              cap0=("cap0", "mean"), cap=("cap", "mean"), cap_sd=("cap", "std"),
              vcap0=("vcap0", "mean"), vcap=("vcap", "mean"), vcap_sd=("vcap", "std"),
              value=("value_aware", "mean"), value0=("value_D0", "mean"), oracle=("value_oracle", "mean"),
              annual_kg=("annual_kg", "mean"), annual_kwh=("annual_kwh", "mean"),
              annual_train_g=("annual_train_kg", lambda x: 1000 * x.mean()),
              annual_train_g_sd=("annual_train_kg", lambda x: 1000 * x.std()),
              annual_infer_g=("annual_infer_kg", lambda x: 1000 * x.mean()),
              annual_infer_g_sd=("annual_infer_kg", lambda x: 1000 * x.std()),
              annual_value=("annual_value", "mean"), n_seeds=("seed", "nunique")).reset_index()
    s["family"] = s["model"].map(FAMILY)
    s["order"] = s["model"].map({m: i for i, m in enumerate(ORDER)})
    return s.sort_values(["scenario", "order"]).reset_index(drop=True)


def pareto(sub, col="cap"):
    pts = sub.sort_values("train_mg")
    front, best = [], -np.inf
    for _, r in pts.iterrows():
        if r[col] > best:
            front.append(r["model"]); best = r[col]
    return front


def key_models(sub, cap="vcap", acc="vacc"):
    """Accuracy-best, value-best, most carbon-intensive and GVP. Defaults select on VALIDATION."""
    acc_m = sub.loc[sub[acc].idxmax(), "model"]
    val_m = sub.loc[sub[cap].idxmax(), "model"]
    car_m = sub.loc[sub["train_mg"].idxmax(), "model"]
    vmax = sub[cap].max()
    ok = sub[sub[cap] >= vmax - GVP_TOL * abs(vmax)]
    gvp = ok.loc[ok["train_mg"].idxmin(), "model"]
    return acc_m, val_m, car_m, gvp


# ---------------------------------------------------------------------------
# Per-row test utilities and paired bootstrap / TOST
# ---------------------------------------------------------------------------
def test_utility_rows(sc, d, r):
    """Per-test-row money earned by run r under the decision rule it chose on validation."""
    z = _npz(sc, r["model"], int(r["seed"]))
    idx, y, s = z["idx"], z["y"], z["score"]
    aux = d["aux"].iloc[idx].reset_index(drop=True)
    if d["task"] == "clf":
        pos, neg, const = V.utilities(sc, y, aux)
        if r["rule"] == "D2":
            p = V.fit_platt(z["score_va"], z["y_va"])(s)
            flags = V.bayes_flags(p, pos, neg, 1.0)
        else:
            flags = s >= r["thr_D1"]
        u = np.where(flags, np.where(y == 1, pos, -neg), 0.0)
        return u, const, V.oracle_value(y, pos, neg, const)
    q = np.maximum(s * (1 + r["thr_D1"]), 0)
    c, p, lag = aux["unit_cost"].values, aux["price"].values, aux["lag_7"].values
    u = V.newsvendor_row_cost(lag, y, c, p) - V.newsvendor_row_cost(q, y, c, p)
    return u, 0.0, V.retail_oracle(y, aux)


def paired_bootstrap_pp(sc, d, runs, m_a, m_b, B=N_BOOT):
    """Test value capture of m_a minus m_b, in percentage points of the oracle, averaged over all
    repeated splits. Rows are resampled within each split (paired: both models see the same rows).
    Returns point estimate, 95% CI and 90% CI (the TOST interval)."""
    rng = np.random.default_rng(BOOT_SEED)
    seeds = sorted(set(runs[(runs.scenario == sc) & (runs.model == m_a)].seed)
                   & set(runs[(runs.scenario == sc) & (runs.model == m_b)].seed))
    point, boots = 0.0, np.zeros(B)
    for sd in seeds:
        ra = runs[(runs.scenario == sc) & (runs.model == m_a) & (runs.seed == sd)].iloc[0]
        rb = runs[(runs.scenario == sc) & (runs.model == m_b) & (runs.seed == sd)].iloc[0]
        ua, ca, oracle = test_utility_rows(sc, d, ra)
        ub, _, _ = test_utility_rows(sc, d, rb)
        assert np.isclose(ua.sum() + ca, ra["value_aware"], rtol=1e-6, atol=1e-6), (sc, m_a, sd)
        diff = 100 * (ua - ub) / oracle
        n = len(diff)
        point += diff.sum() / len(seeds)
        for c0 in range(0, B, 100):
            k = min(100, B - c0)
            boots[c0:c0 + k] += diff[rng.integers(0, n, (k, n))].sum(axis=1) / len(seeds)
    lo95, hi95, lo90, hi90 = np.percentile(boots, [2.5, 97.5, 5, 95])
    m = EQUIV_MARGIN_PP
    return dict(diff_pp=float(point), lo95=float(lo95), hi95=float(hi95), lo90=float(lo90), hi90=float(hi90),
                equivalent=bool(lo90 > -m and hi90 < m), non_inferior=bool(lo90 > -m),
                significant=bool(lo95 > 0 or hi95 < 0), n_splits=len(seeds))


def decision_layer_carbon(sc, d):
    """Carbon of the value-aware decision tuning itself (seed 0, baseline model, VALIDATION scores)."""
    m = "Naive" if d["task"] == "reg" else "Linear"
    z = _npz(sc, m, 0)
    s, y, idx = z["score_va"], z["y_va"], z["idx_va"]
    aux = d["aux"].iloc[idx].reset_index(drop=True)
    if d["task"] == "clf":
        pos, neg, const = V.utilities(sc, y, aux)
        _, meas = measure(lambda: (V.tune_global_threshold(s, y, pos, neg, const),
                                   V.crossfit_threshold_value(s, y, pos, neg, const),
                                   V.fit_platt(s, y), V.crossfit_platt(s, y)))
    else:
        _, meas = measure(lambda: (V.tune_order_uplift(s, y, aux), V.crossfit_uplift_value(s, y, aux)))
    return meas["kg"] * 1e6


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_carbon_value(s, K):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6))
    for ax, sc in zip(axes.flat, SCEN):
        sub = s[s.scenario == sc].copy()
        sub["x"] = np.maximum(sub["train_mg"], 1e-3)
        front = pareto(sub)
        fr = sub[sub.model.isin(front)].sort_values("x")
        ax.plot(fr["x"], fr["cap"], color="#9a9a9a", lw=1.0, ls="--", zorder=1)
        for _, r in sub.iterrows():
            col, mk = FAM_STYLE[r["family"]]
            ax.plot([r["x"], r["x"]], [r["cap0"], r["cap"]], color=col, lw=0.6, alpha=0.45, zorder=1)
            sz = 1.8 if mk == "*" else 1.0
            ax.scatter(r["x"], r["cap0"], s=22 * sz, marker=mk, facecolors="white", edgecolors=col, lw=0.9, zorder=2)
            ax.errorbar(r["x"], r["cap"], yerr=r["cap_sd"], fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, zorder=2)
            ax.scatter(r["x"], r["cap"], s=30 * sz, marker=mk, color=col, edgecolors="white", lw=0.6, zorder=3)
        ks = K["scen"][sc]
        r = sub[sub.model == ks["gvp"]].iloc[0]
        ax.scatter(r["x"], r["cap"], s=150, facecolors="none", edgecolors="#222222", lw=1.0, zorder=4)
        r = sub[sub.model == ks["acc_best"]].iloc[0]
        ax.scatter(r["x"], r["cap"], s=110, marker="s", facecolors="none", edgecolors="#222222",
                   lw=0.8, ls=(0, (2, 1.5)), zorder=4)
        ax.set_xscale("log")
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_title(TITLE[sc], loc="left")
        ax.set_ylabel("Test value captured (% of oracle)")
        ax.set_xlabel("Training emissions (mg CO$_2$e, log)")
    handles = [plt.Line2D([], [], marker=FAM_STYLE[f][1], color=FAM_STYLE[f][0], ls="", ms=5, label=f)
               for f in FAM_STYLE]
    handles += [plt.Line2D([], [], marker="o", mfc="white", mec="#555", ls="", ms=5, label="default 0.5 rule"),
                plt.Line2D([], [], marker="o", color="#555", ls="", ms=5, label="value-aware rule"),
                plt.Line2D([], [], color="#9a9a9a", ls="--", label="Pareto frontier"),
                plt.Line2D([], [], marker="o", mfc="none", mec="#222", ls="", ms=9, label="GVP (val.-selected)"),
                plt.Line2D([], [], marker="s", mfc="none", mec="#222", ls="", ms=8, label="accuracy-best (val.)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / "fig_carbon_value.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_carbon_value.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_acc_vs_value(s, rho):
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0))
    for ax, sc in zip(axes.flat, SCEN):
        sub = s[s.scenario == sc]
        if sc == "retail":
            sub = sub[sub.model != "Naive"]
        for _, r in sub.iterrows():
            col, mk = FAM_STYLE[r["family"]]
            ax.scatter(r["acc"], r["cap"], s=(30 if mk == "*" else 18), marker=mk, color=col,
                       edgecolors="white", lw=0.4)
        ax.set_title(f"{sc}  ($\\rho$={rho[sc]['acc_value']:+.2f})", loc="left", fontsize=7.5)
        ax.set_xlabel("ROC-AUC" if sc != "retail" else "1 - WAPE", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Value captured (%)", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_DIR / "fig_acc_value.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_acc_value.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    p = RESULTS_DIR / "ablation_fraud.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p)
    a["cap"] = 100 * a["value_aware"] / a["value_oracle"]
    a["mg"] = a["train_kg"] * 1e6
    g = a.groupby(["model", "train_frac"]).agg(cap=("cap", "mean"), sd=("cap", "std"),
                                                mg=("mg", "mean")).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for m in ("Linear", "RF-100", "HGB-100"):
        col, mk = FAM_STYLE[FAMILY[m]]
        sub = g[g.model == m].sort_values("train_frac")
        x = 100 * sub["train_frac"]
        ax1.errorbar(x, sub["cap"], yerr=sub["sd"], color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5, label=m)
        ax2.plot(x, np.maximum(sub["mg"], 1e-3), color=col, marker=mk, ms=3.5, lw=1.1, label=m)
    for ax in (ax1, ax2):
        ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels(["5", "10", "25", "50", "100"])
        ax.set_xlabel("Training data used (%)"); ax.tick_params(labelsize=6.5)
    ax1.set_ylabel("Value captured (%)"); ax1.set_ylim(-5, 100)
    ax2.set_yscale("log"); ax2.set_ylabel("Training mg CO$_2$e")
    ax1.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return g


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def sci(x):
    """LaTeX scientific notation, e.g. 3.1e-07 -> $3.1{\\times}10^{-7}$."""
    if x == 0 or not np.isfinite(x):
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.1f}{{\\times}}10^{{{e}}}$"


def fold(x):
    """A carbon span as a readable fold-change: 920 -> '920', 2.4e5 -> '$2.4{\\times}10^{5}$'."""
    return f"{x:,.0f}" if x < 1e4 else sci(x)


def fmt_mg(x):
    if x < 0.01:
        return "$<$0.01"
    return f"{x:,.2f}" if x < 10 else f"{x:,.0f}"


def sgn(x, d=1):
    """Signed number without a negative zero (-0.0 -> 0.0)."""
    return f"{0.0:.{d}f}" if abs(x) < 0.5 * 10 ** -d else f"{x:+.{d}f}"


def g2(x):
    return f"{x:.2g}" if x >= 0.01 else sci(x)


# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True); TAB_DIR.mkdir(parents=True, exist_ok=True)
    runs = load()
    s = summarise(runs)
    s.to_csv(RESULTS_DIR / "summary_by_model.csv", index=False)
    cfg = json.loads((RESULTS_DIR / "run_config.json").read_text()) if (RESULTS_DIR / "run_config.json").exists() else {}
    pm = cfg.get("codecarbon_power_model", {})
    cv = runs.groupby(["scenario", "model"])["train_kg"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
    K = {"n_runs": int(len(runs)), "total_train_kg": float(runs["train_kg"].sum()),
         "total_train_kwh": float(runs["train_kwh"].sum()),
         "total_infer_kg_per_1k": float(runs["infer_kg_per_1k"].sum()),
         "cpu_tdp_w": pm.get("cpu_tdp_w_used"), "ram_w": pm.get("ram_power_w"),
         "train_carbon_cv_median_pct": float(100 * cv.median()),
         "python": cfg.get("python"), "packages": cfg.get("packages", {})}

    # rank correlations (descriptive; n = 10-11 models per scenario, so power is low) ----------------
    rho = {}
    for sc in SCEN:
        sub = s[s.scenario == sc]
        r_cv = spearmanr(sub["train_mg"], sub["cap"]); r_ca = spearmanr(sub["train_mg"], sub["acc"])
        r_av = spearmanr(sub["acc"], sub["cap"])
        r_av0 = spearmanr(sub[sub.model != "Naive"]["acc"], sub[sub.model != "Naive"]["cap0"])
        rho[sc] = dict(acc_value0=r_av0.statistic, acc_value0_p=r_av0.pvalue, carbon_value=r_cv.statistic, carbon_value_p=r_cv.pvalue,
                       carbon_acc=r_ca.statistic, carbon_acc_p=r_ca.pvalue,
                       acc_value=r_av.statistic, acc_value_p=r_av.pvalue, n_models=len(sub),
                       kendall_carbon_value=kendalltau(sub["train_mg"], sub["cap"]).statistic)
    K["rho"] = rho
    K["energy_vs_cpu_spearman"] = float(spearmanr(runs["train_kwh"], runs["train_cpu_s"]).statistic)

    # key models (validation-selected), bootstrap/TOST, decision layer --------------------------------
    K["scen"] = {}
    for sc in SCEN:
        sub = s[s.scenario == sc].set_index("model")
        acc_m, val_m, car_m, gvp = key_models(sub.reset_index())
        t_acc, t_val, _, t_gvp = key_models(sub.reset_index(), cap="cap", acc="acc")
        d = LOADERS[sc]()
        ks = dict(acc_best=acc_m, value_best=val_m, carbon_max=car_m, gvp=gvp, frontier=pareto(sub.reset_index()),
                  test_selected=dict(acc_best=t_acc, value_best=t_val, gvp=t_gvp))
        for tag, m in (("acc_best", acc_m), ("value_best", val_m), ("gvp", gvp), ("hpo", "HGB-HPO")):
            ks[f"{tag}_mg"] = float(sub.loc[m, "train_mg"]); ks[f"{tag}_cap"] = float(sub.loc[m, "cap"])
            ks[f"{tag}_vcap"] = float(sub.loc[m, "vcap"]); ks[f"{tag}_acc"] = float(sub.loc[m, "acc"])
        ks["saving_vs_valuebest_pct"] = 100 * (1 - ks["gvp_mg"] / ks["value_best_mg"])
        ks["saving_vs_hpo_pct"] = 100 * (1 - ks["gvp_mg"] / ks["hpo_mg"])
        learned = sub.drop(index="Naive", errors="ignore")
        ks["carbon_span_x"] = float(learned["train_mg"].max() / max(learned["train_mg"].min(), 1e-6))
        ks["value_span_pp"] = float(sub["cap"].max() - sub["cap"].min())
        ks["test_best_cap"] = float(sub["cap"].max()); ks["test_best_model"] = sub["cap"].idxmax()
        if gvp != val_m:
            ks["tost_gvp_vs_best"] = paired_bootstrap_pp(sc, d, runs, gvp, val_m)
        if "HGB-HPO" not in (gvp,):
            ks["tost_gvp_vs_hpo"] = paired_bootstrap_pp(sc, d, runs, gvp, "HGB-HPO")
        if acc_m != val_m:
            ks["boot_acc_vs_best"] = paired_bootstrap_pp(sc, d, runs, acc_m, val_m)
        base = "Linear"
        best0 = sub["vcap0"].idxmax()                    # best model at the 0.5 rule, chosen on validation
        ks["model_gain_pp"] = float(sub.loc[best0, "cap0"] - sub.loc[base, "cap0"])
        ks["model_gain_model"] = best0
        ks["model_gain_mg"] = float(sub.loc[best0, "train_mg"] - sub.loc[base, "train_mg"])
        ks["decision_gain_pp"] = float(sub.loc[base, "cap"] - sub.loc[base, "cap0"])
        ks["decision_gain_mean_pp"] = float((sub["cap"] - sub["cap0"]).mean())
        ks["decision_mg"] = decision_layer_carbon(sc, d)
        ks["models_negative_at_default"] = int((sub["cap0"] < 0).sum())
        wins = runs[runs.scenario == sc].loc[lambda x: x.groupby("seed")["vcap"].transform("max") == x["vcap"]]
        ks["winner_counts"] = wins["model"].value_counts().to_dict()
        if d["task"] == "clf":
            rs = runs[runs.scenario == sc]
            ks["d2_chosen_pct"] = float(100 * (rs["rule"] == "D2").mean())
            for col in ("value_D1", "value_D2", "value_D2raw"):
                ks[f"mean_cap_{col}"] = float((100 * rs[col] / rs["value_oracle"]).mean())
        for tag, m in (("value_best", val_m), ("gvp", gvp)):
            kwh = float(sub.loc[m, "annual_kwh"]); val = float(sub.loc[m, "annual_value"])
            ks[f"{tag}_annual_kwh"] = kwh
            ks[f"{tag}_annual_kg"] = {g: kwh * ci for g, ci in GRIDS.items()}
            ks[f"{tag}_annual_value_usd"] = val * TO_USD[CUR[sc]]
            ks[f"{tag}_scc_share_pct"] = 100 * (kwh * GRIDS["India"] / 1000 * SOCIAL_COST_CARBON) / max(abs(val) * TO_USD[CUR[sc]], 1e-9)
            for part in ("train", "infer"):
                ks[f"{tag}_annual_{part}_g"] = float(sub.loc[m, f"annual_{part}_g"])
                ks[f"{tag}_annual_{part}_g_sd"] = float(sub.loc[m, f"annual_{part}_g_sd"])
        d_kg = (ks["value_best_annual_kwh"] - ks["gvp_annual_kwh"]) * GRIDS["India"]
        d_val = ks["value_best_annual_value_usd"] - ks["gvp_annual_value_usd"]
        ks["implied_usd_per_t"] = float(d_val / (d_kg / 1000)) if d_kg > 0 else float("nan")
        ks["acc_best_value_loss_pp"] = ks["value_best_cap"] - ks["acc_best_cap"]
        K["scen"][sc] = ks

    abl = fig_ablation()
    if abl is not None:
        K["ablation"] = abl.to_dict(orient="records")
    fig_carbon_value(s, K)
    fig_acc_vs_value(s, rho)
    write_tables(s, K, runs)
    write_macros(K)
    with open(RESULTS_DIR / "key_numbers.json", "w") as f:
        json.dump(K, f, indent=2, default=float)
    for sc in SCEN:
        k = K["scen"][sc]
        print(sc, {x: k[x] for x in ("acc_best", "value_best", "gvp", "test_selected", "gvp_cap",
                                     "value_best_cap", "saving_vs_valuebest_pct")},
              k.get("tost_gvp_vs_best"), flush=True)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------
def write_macros(K):
    def num(x, d=1):
        return f"{0.0:.{d}f}" if abs(x) < 0.5 * 10 ** -d else f"{x:,.{d}f}"

    def pct(x):
        return "$>$99.9" if x >= 99.95 else f"{x:.1f}"

    def yn(b):
        return "yes" if b else "no"
    M = {"NRuns": str(K["n_runs"]), "TotalWh": num(K["total_train_kwh"] * 1000, 1),
         "TotalG": num(K["total_train_kg"] * 1000, 1), "EnergyCpuRho": num(K["energy_vs_cpu_spearman"], 2),
         "CpuTdp": str(K["cpu_tdp_w"]), "RamW": f"{K['ram_w']:.0f}" if K["ram_w"] else "--",
         "CarbonCv": num(K["train_carbon_cv_median_pct"], 0), "EqMargin": num(EQUIV_MARGIN_PP, 0),
         "PyVer": str(K["python"])}
    for p, v in (K.get("packages") or {}).items():
        M["Ver" + "".join(ch for ch in p.title() if ch.isalpha())] = str(v)
    for sc in SCEN:
        k, r, P = K["scen"][sc], K["rho"][sc], sc.capitalize()
        M.update({f"{P}AccBest": k["acc_best"], f"{P}ValBest": k["value_best"], f"{P}Gvp": k["gvp"],
                  f"{P}GvpCap": num(k["gvp_cap"]), f"{P}BestCap": num(k["value_best_cap"]),
                  f"{P}GvpVcap": num(k["gvp_vcap"]), f"{P}BestVcap": num(k["value_best_vcap"]),
                  f"{P}AccBestCap": num(k["acc_best_cap"]), f"{P}HpoCap": num(k["hpo_cap"]),
                  f"{P}GvpMg": fmt_mg(k["gvp_mg"]), f"{P}HpoMg": fmt_mg(k["hpo_mg"]),
                  f"{P}BestMg": fmt_mg(k["value_best_mg"]), f"{P}AccBestMg": fmt_mg(k["acc_best_mg"]),
                  f"{P}SaveHpo": pct(k["saving_vs_hpo_pct"]), f"{P}SaveBest": pct(k["saving_vs_valuebest_pct"]),
                  f"{P}CarbonSpan": fold(k["carbon_span_x"]), f"{P}ValueSpan": num(k["value_span_pp"]),
                  f"{P}TestBestModel": k["test_best_model"], f"{P}TestBestCap": num(k["test_best_cap"]),
                  f"{P}RhoCV": f"{r['carbon_value']:+.2f}", f"{P}RhoCVp": num(r["carbon_value_p"], 2),
                  f"{P}RhoCA": f"{r['carbon_acc']:+.2f}", f"{P}RhoAV": f"{r['acc_value']:+.2f}",
                  f"{P}RhoAVp": num(r["acc_value_p"], 2), f"{P}RhoAVz": f"{r['acc_value0']:+.2f}",
                  f"{P}DecGain": num(k["decision_gain_pp"]), f"{P}DecGainMean": num(k["decision_gain_mean_pp"]),
                  f"{P}ModGain": num(k["model_gain_pp"]), f"{P}ModGainModel": k["model_gain_model"],
                  f"{P}NegDefault": str(k["models_negative_at_default"]),
                  f"{P}SccShare": sci(k["value_best_scc_share_pct"]),
                  f"{P}AccLoss": num(k["acc_best_value_loss_pp"]),
                  f"{P}AccBestAcc": num(k["acc_best_acc"], 3), f"{P}BestAcc": num(k["value_best_acc"], 3),
                  f"{P}AnnualKgIN": num(k["value_best_annual_kg"]["India"], 2)})
        if "d2_chosen_pct" in k:
            M[f"{P}DTwoPct"] = num(k["d2_chosen_pct"], 0)
            M.update({f"{P}CapDOne": num(k["mean_cap_value_D1"]), f"{P}CapDTwo": num(k["mean_cap_value_D2"]),
                      f"{P}CapDTwoRaw": num(k["mean_cap_value_D2raw"])})
        for key, tag in (("tost_gvp_vs_best", "Tost"), ("tost_gvp_vs_hpo", "TostHpo"), ("boot_acc_vs_best", "BootAcc")):
            if key in k:
                t = k[key]
                M.update({f"{P}{tag}Diff": num(t["diff_pp"]), f"{P}{tag}Lo": num(t["lo95"]),
                          f"{P}{tag}Hi": num(t["hi95"]), f"{P}{tag}LoNinety": num(t["lo90"]),
                          f"{P}{tag}HiNinety": num(t["hi90"]), f"{P}{tag}Eq": yn(t["equivalent"]),
                          f"{P}{tag}Ni": yn(t["non_inferior"])})
    if "ablation" in K:
        a = pd.DataFrame(K["ablation"])
        for m, tag in (("HGB-100", "Hgb"), ("RF-100", "Rf"), ("Linear", "Lin")):
            sub = a[a.model == m].set_index("train_frac")
            M[f"Abl{tag}Five"] = num(sub.loc[0.05, "cap"]); M[f"Abl{tag}Full"] = num(sub.loc[1.0, "cap"])
            M[f"Abl{tag}Ratio"] = f"{sub.loc[1.0, 'mg'] / max(sub.loc[0.05, 'mg'], 1e-6):.0f}"
    lines = ["% auto-generated by src/analysis.py -- do not edit by hand"]
    for k, v in M.items():
        v = str(v).replace("%", r"\%")
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")


def write_tables(s, K, runs):
    # Table: full ladder ------------------------------------------------------------------------
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Model ladder results (mean over repeated splits). C = training emissions in mg CO$_2$e "
             r"(CodeCarbon, Indian grid). Acc = test ROC-AUC (retail: 1$-$WAPE). V$_{0.5}$ / V$^{\ast}$ = test business "
             r"value captured as \% of the perfect-foresight oracle under the default 0.5 rule / the value-aware rule. "
             r"Bold = value-best, \underline{underline} = accuracy-best and $\dagger$ = Green Value Point, all three "
             r"\emph{selected on validation data}; the test columns only report their outcome.}",
             r"\label{tab:ladder}", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{l" + "rrrr" * 4 + "}", r"\toprule"]
    head = " & ".join([r"\multicolumn{4}{c}{" + TITLE[sc].split(" (")[0] + "}" for sc in SCEN])
    lines.append(r" & " + head + r" \\")
    lines.append("".join([rf"\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(4)]))
    lines.append(r"Model & " + " & ".join([r"C & Acc & V$_{0.5}$ & V$^{\ast}$"] * 4) + r" \\ \midrule")
    for m in ORDER:
        cells = [m]
        for sc in SCEN:
            sub = s[(s.scenario == sc) & (s.model == m)]
            if sub.empty:
                cells += ["--"] * 4; continue
            r = sub.iloc[0]; ks = K["scen"][sc]
            acc = f"{r['acc']:.3f}"
            if m == ks["acc_best"]:
                acc = rf"\underline{{{acc}}}"
            cap = f"{r['cap']:.1f}"
            if m == ks["value_best"]:
                cap = rf"\textbf{{{cap}}}"
            if m == ks["gvp"]:
                cap += r"$^\dagger$"
            cells += [fmt_mg(r["train_mg"]), acc, f"{r['cap0']:.1f}", cap]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule", r"Spearman $\rho$(C,V$^{\ast}$) & " + " & ".join(
        [rf"\multicolumn{{4}}{{c}}{{{K['rho'][sc]['carbon_value']:+.2f} (p={K['rho'][sc]['carbon_value_p']:.2f})}}" for sc in SCEN]) + r" \\",
              r"Spearman $\rho$(Acc,V$^{\ast}$) & " + " & ".join(
        [rf"\multicolumn{{4}}{{c}}{{{K['rho'][sc]['acc_value']:+.2f} (p={K['rho'][sc]['acc_value_p']:.2f})}}" for sc in SCEN]) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB_DIR / "ladder.tex").write_text("\n".join(lines) + "\n")

    # Table: GVP equivalence test ---------------------------------------------------------------
    m = EQUIV_MARGIN_PP
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test-set check of the validation-selected Green Value Point (GVP) against the validation-selected "
         r"value-best model (when different) and against hyper-parameter search (HGB-HPO). $\Delta$V = GVP $-$ comparator "
         rf"in pp of oracle value, mean over splits, paired bootstrap ({N_BOOT:,} resamples). TOST: equivalent if the 90\% "
         rf"CI lies inside $\pm${m:.0f} pp. NI = non-inferior (lower 90\% bound $>-${m:.0f} pp).}}",
         r"\label{tab:tost}", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{lllrrrcc}", r"\toprule",
         r"Scen. & GVP & vs. & C saved & $\Delta$V & 90\% CI & TOST & NI \\ \midrule"]
    for sc in SCEN:
        k = K["scen"][sc]
        rows = []
        if "tost_gvp_vs_best" in k:
            rows.append((k["value_best"], k["tost_gvp_vs_best"], k["saving_vs_valuebest_pct"]))
        if "tost_gvp_vs_hpo" in k and k["value_best"] != "HGB-HPO":
            rows.append(("HGB-HPO", k["tost_gvp_vs_hpo"], k["saving_vs_hpo_pct"]))
        if not rows:
            T.append(f"{sc.capitalize()} & {k['gvp']} & (itself) & 0\\% & -- & -- & -- & -- \\\\")
        for i, (comp, t, save) in enumerate(rows):
            lab = (sc.capitalize(), k["gvp"]) if i == 0 else ("", "")
            T.append(f"{lab[0]} & {lab[1]} & {comp} & {('$>$99.9' if save >= 99.95 else f'{save:.0f}')}\\% & {sgn(t['diff_pp'])} & "
                     f"[{sgn(t['lo90'])}, {sgn(t['hi90'])}] & {'yes' if t['equivalent'] else 'no'} & "
                     f"{'yes' if t['non_inferior'] else 'no'} \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tost.tex").write_text("\n".join(T) + "\n")

    # Table: decision layer vs model layer ------------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Where the business value comes from (test, pp of oracle value; C in mg CO$_2$e). Model upgrade = "
         r"Linear$\rightarrow$the model with the best \emph{validation} value at the 0.5 rule. Decision upgrade = "
         r"Linear, 0.5 rule$\rightarrow$value-aware rule (C = metered cost of tuning and calibrating the rule).}",
         r"\label{tab:layers}", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{Model upgrade} & \multicolumn{2}{c}{Decision upgrade} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"Scenario & to & $\Delta$V & $\Delta$C & $\Delta$V & $\Delta$C \\ \midrule"]
    for sc in SCEN:
        k = K["scen"][sc]
        L.append(f"{sc.capitalize()} & {k['model_gain_model']} & {sgn(k['model_gain_pp'])} & {fmt_mg(k['model_gain_mg'])} & "
                 f"{sgn(k['decision_gain_pp'])} & {fmt_mg(k['decision_mg'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "layers.tex").write_text("\n".join(L) + "\n")

    # Table: annual footprint: training vs inference with uncertainty, grids, SCC ----------------
    G = [r"\begin{table}[t]", r"\centering",
         r"\caption{Projected annual footprint of the value-best (B) and GVP (G) models under the illustrative "
         r"deployment profile of Sec.~\ref{sec:lifecycle}, in g CO$_2$e on the Indian grid: training and inference "
         r"shown separately as mean ($\pm$ relative s.d.\ across repeated runs: run-to-run metering spread only; see Sec.~"
         r"\ref{sec:carbon} for the larger power-model uncertainty). FR = total on the French grid. SCC = social cost "
         r"of carbon (\$190/t) as a share of the model's annual modeled value.}",
         r"\label{tab:carbonparts}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Scen. & Model & Training & Inference & FR & SCC/value \\ \midrule"]
    for sc in SCEN:
        k = K["scen"][sc]
        for tag, lab in (("value_best", "B"), ("gvp", "G")):
            if tag == "gvp" and k["gvp"] == k["value_best"]:
                continue
            tr, trs = k[f"{tag}_annual_train_g"], k[f"{tag}_annual_train_g_sd"]
            inf, infs = k[f"{tag}_annual_infer_g"], k[f"{tag}_annual_infer_g_sd"]
            fr = 1000 * k[f"{tag}_annual_kg"]["France"]
            name = sc.capitalize() if tag == "value_best" else ""
            G.append(f"{name} & {lab}: {k[tag]} & {g2(tr)} ($\\pm${100 * trs / tr:.0f}\\%) & "
                     f"{g2(inf)} ($\\pm${100 * infs / inf:.0f}\\%) & {g2(fr)} & {sci(k[f'{tag}_scc_share_pct'])}\\% \\\\")
    G += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB_DIR / "carbon_parts.tex").write_text("\n".join(G) + "\n")

    # Table: probability quality checked BEFORE the Bayes rule ------------------------------------
    C = [r"\begin{table}[t]", r"\centering",
         r"\caption{Probability quality, checked on validation before the Bayes rule D2 is used. ECE (ten bins) "
         r"of raw scores and of Platt-calibrated probabilities (cross-fitted on validation; calibrator then applied "
         r"unchanged to test). Mean over all runs, worst model mean in brackets. D2 = share of runs in which the "
         r"calibrated Bayes rule beat the tuned threshold on validation and was therefore used.}",
         r"\label{tab:calibration}", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Validation ECE} & \multicolumn{2}{c}{Test ECE} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Scenario & raw & calibrated & raw & calibrated & D2 used \\ \midrule"]
    for sc in ("fraud", "churn", "credit"):
        rs = runs[runs.scenario == sc]
        by = rs.groupby("model")[["val_ece_raw", "val_ece_cal", "test_ece_raw", "test_ece_cal"]].mean()
        cells = [f"{rs[c].mean():.3f} [{by[c].max():.3f}]" for c in by.columns]
        C.append(f"{sc.capitalize()} & " + " & ".join(cells) + f" & {100 * (rs['rule'] == 'D2').mean():.0f}\\% \\\\")
    C += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "calibration.tex").write_text("\n".join(C) + "\n")


if __name__ == "__main__":
    main()
