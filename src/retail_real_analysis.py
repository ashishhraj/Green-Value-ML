"""Report UCI Online Retail results with validation-only model selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from config import GVP_TOL  # noqa: E402
from run_experiments import RUNS  # noqa: E402


def main():
    runs = pd.read_csv(RUNS)
    runs = runs[runs.scenario == "retail_real"].copy()
    if runs.empty:
        raise RuntimeError("Run --scenarios retail_real before building this table.")
    # validation capture uses the cross-fitted validation value stored by run_experiments.py
    runs["val_cap"] = 100 * runs.val_aware / runs.val_oracle
    runs["test_cap"] = 100 * runs.value_aware / runs.value_oracle
    runs["train_mg"] = 1e6 * runs.train_kg
    summary = runs.groupby("model").agg(
        train_mg=("train_mg", "mean"),
        acc=("wape", lambda x: 1 - x.mean()),
        val_cap=("val_cap", "mean"), val_sd=("val_cap", "std"),
        test_cap=("test_cap", "mean"), test_sd=("test_cap", "std"),
        folds=("seed", "nunique"),
    ).reset_index()
    best = summary.loc[summary.val_cap.idxmax(), "model"]
    vmax = summary.val_cap.max()
    eligible = summary[summary.val_cap >= vmax - GVP_TOL * abs(vmax)]
    gvp = eligible.loc[eligible.train_mg.idxmin(), "model"]
    summary["model_order"] = summary.model.map({"Naive": 0, "Linear": 1, "Tree-d6": 2,
        "RF-100": 3, "RF-500": 4, "HGB-100": 5, "HGB-500": 6, "MLP-S": 7,
        "MLP-L": 8, "Stack": 9, "HGB-HPO": 10})
    summary = summary.sort_values("model_order")
    lines = [r"\begin{table}[t]", r"\centering",
        r"\caption{UCI Online Retail (real transactions), three rolling-origin folds. The value-best model (bold) and "
        r"the GVP ($\dagger$) are selected on each fold's validation window; capture is then reported on the later test "
        r"window (mean $\pm$ s.d.\ over folds). C = training emissions (mg CO$_2$e). Value uses an assumed cost proxy.}",
        r"\label{tab:retailreal}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & C & $1-$WAPE & Val.\ capture (\%) & Test capture (\%) \\ \midrule"]
    for row in summary.itertuples():
        name = row.model
        if name == best:
            name = rf"\textbf{{{name}}}"
        if row.model == gvp:
            name += r"$^\dagger$"
        c = ("0" if row.train_mg == 0 else "$<$0.01" if row.train_mg < 0.01
             else f"{row.train_mg:.2g}" if row.train_mg < 100 else f"{row.train_mg:,.0f}")
        lines.append(f"{name} & {c} & {row.acc:.3f} & "
                     f"{row.val_cap:.1f} $\\pm$ {row.val_sd:.1f} & "
                     f"{row.test_cap:.1f} $\\pm$ {row.test_sd:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper" / "tables" / "retail_real.tex").write_text("\n".join(lines) + "\n")
    bs = summary.set_index("model")
    report = {"value_best_validation": best, "gvp_validation": gvp,
              "gvp_saving_pct": float(100 * (1 - bs.loc[gvp, "train_mg"] / bs.loc[best, "train_mg"])),
              "best_test_capture": float(bs.loc[best, "test_cap"]),
              "gvp_test_capture": float(bs.loc[gvp, "test_cap"]),
              "folds": int(runs.seed.nunique()), "rows_per_fold": runs.groupby("seed").n_test.first().to_dict(),
              "models": summary.drop(columns="model_order").to_dict(orient="records")}
    (ROOT / "results" / "retail_real_summary.json").write_text(json.dumps(report, indent=2))
    (ROOT / "paper" / "numbers_retail_real.tex").write_text(
        "\\newcommand{\\RRValueBest}{" + best + "}\n" +
        "\\newcommand{\\RRGvp}{" + gvp + "}\n" +
        f"\\newcommand{{\\RRGvpSaving}}{{{report['gvp_saving_pct']:.1f}}}\n" +
        f"\\newcommand{{\\RRBestTest}}{{{report['best_test_capture']:.1f}}}\n" +
        f"\\newcommand{{\\RRGvpTest}}{{{report['gvp_test_capture']:.1f}}}\n")
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
