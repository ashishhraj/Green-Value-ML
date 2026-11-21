"""Dataset loaders (3 public datasets) and the synthetic retail generator.

Each loader returns a dict with
    X        : pd.DataFrame of model features (numeric, no leakage)
    y        : np.ndarray target
    aux      : pd.DataFrame of business columns used ONLY by the value function
               (e.g. transaction amount, monthly charges, loan amount)
    task     : "clf" or "reg"
"""
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from config import DATA_DIR, RETAIL


# ---------------------------------------------------------------------------
# 1. Credit-card fraud (ULB / Kaggle, 284,807 European transactions, Sep 2013)
# ---------------------------------------------------------------------------
def load_fraud() -> dict:
    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    aux = pd.DataFrame({"amount": df["Amount"].values})
    X = df.drop(columns=["Class"]).copy()
    X["hour"] = (X["Time"] // 3600) % 24          # cyclical time-of-day instead of raw seconds
    X["log_amount"] = np.log1p(X["Amount"])
    X = X.drop(columns=["Time", "Amount"])
    return dict(X=X, y=df["Class"].values.astype(int), aux=aux, task="clf")


# ---------------------------------------------------------------------------
# 2. Telco customer churn (IBM sample data, 7,043 customers)
# ---------------------------------------------------------------------------
def load_churn() -> dict:
    df = pd.read_csv(DATA_DIR / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int).values
    aux = pd.DataFrame({"monthly": df["MonthlyCharges"].values})
    X = df.drop(columns=["customerID", "Churn"])
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return dict(X=X, y=y, aux=aux, task="clf")


# ---------------------------------------------------------------------------
# 3. German credit (Statlog, 1,000 applicants); y = 1 means the loan DEFAULTS
# ---------------------------------------------------------------------------
def load_credit() -> dict:
    df = pd.read_csv(DATA_DIR / "german_credit.csv")
    y = (df["credit_risk"] == 0).astype(int).values
    aux = pd.DataFrame({"amount": df["amount"].values.astype(float)})
    X = pd.get_dummies(df.drop(columns=["credit_risk"]), drop_first=True, dtype=float)
    return dict(X=X, y=y, aux=aux, task="clf")


# ---------------------------------------------------------------------------
# 4. Synthetic perishable-retail demand (generated, fully seeded)
# ---------------------------------------------------------------------------
def generate_retail(seed: int = 2026) -> pd.DataFrame:
    """Daily unit demand for N perishable SKUs over 3 years.

    Demand = level * trend * weekly * yearly * promo * price * festival effects,
    drawn from a negative-binomial (over-dispersed) distribution, which is the
    standard count model for retail sales.
    """
    rng = np.random.default_rng(seed)
    n_sku, n_days = RETAIL["n_sku"], RETAIL["n_days"]
    t = np.arange(n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    dow, doy = dates.dayofweek.values, dates.dayofyear.values
    # festival windows (Diwali-like peak in autumn, New-Year peak), same every year
    festival = (((doy >= 295) & (doy <= 305)) | (doy >= 358) | (doy <= 2)).astype(float)

    rows = []
    for s in range(n_sku):
        level = rng.lognormal(3.0, 0.7)
        trend = 1 + rng.normal(0.0, 0.15) * t / n_days
        week_amp = rng.uniform(0.05, 0.35)
        weekly = 1 + week_amp * np.where(dow >= 5, 1.0, -0.4) + rng.normal(0, 0.03, 7)[dow]
        yearly = 1 + rng.uniform(0.0, 0.4) * np.sin(2 * np.pi * (doy + rng.uniform(0, 365)) / 365.25)
        unit_cost = rng.uniform(20, 300)
        base_price = unit_cost * rng.uniform(1.25, 1.8)
        promo = (rng.random(n_days) < 0.08).astype(float)
        discount = np.where(promo == 1, rng.uniform(0.1, 0.3), 0.0)
        price = base_price * (1 - discount) * (1 + rng.normal(0, 0.01, n_days))
        elasticity = rng.uniform(-2.5, -0.8)
        price_eff = (price / base_price) ** elasticity
        fest = 1 + festival * rng.uniform(0.3, 1.2)
        mu = level * trend * weekly * yearly * price_eff * fest
        k = rng.uniform(4, 15)                          # NB dispersion
        demand = rng.negative_binomial(k, k / (k + mu))
        rows.append(pd.DataFrame({
            "sku": s, "day": t, "date": dates, "demand": demand, "promo": promo,
            "price": price, "base_price": base_price, "unit_cost": unit_cost,
            "festival": festival, "dow": dow, "doy": doy,
        }))
    return pd.concat(rows, ignore_index=True)


def load_retail() -> dict:
    raw = generate_retail()
    raw.to_csv(DATA_DIR / "synthetic_retail.csv.gz", index=False, compression="gzip")
    df = raw.sort_values(["sku", "day"]).copy()
    g = df.groupby("sku")["demand"]
    for L in (1, 2, 7, 14, 28):
        df[f"lag_{L}"] = g.shift(L)
    sh = g.shift(1)
    df["roll_mean_7"] = sh.groupby(df["sku"]).transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = sh.groupby(df["sku"]).transform(lambda s: s.rolling(28).mean())
    df["roll_std_7"] = sh.groupby(df["sku"]).transform(lambda s: s.rolling(7).std())
    df["price_ratio"] = df["price"] / df["base_price"]
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df = df.dropna().reset_index(drop=True)

    feats = ["lag_1", "lag_2", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
             "roll_std_7", "promo", "price_ratio", "festival", "dow", "doy_sin", "doy_cos"]
    aux = df[["day", "sku", "unit_cost", "price", "lag_7"]].reset_index(drop=True)
    return dict(X=df[feats].reset_index(drop=True), y=df["demand"].values.astype(float),
                aux=aux, task="reg")


def load_retail_real() -> dict:
    """Build a leak-free daily panel from UCI Online Retail transactions.

    Item selection and reference prices use only the first half-year window.
    Transaction prices are not used as same-day predictive features. Unit cost
    is an explicit proxy (60% of historical median price), not observed COGS.
    """
    path = DATA_DIR / "online_retail.zip"
    with ZipFile(path) as archive:
        workbook = archive.read(archive.namelist()[0])
    raw = pd.read_excel(BytesIO(workbook), sheet_name="Online Retail", engine="openpyxl",
                        usecols=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"])
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw = raw.dropna(subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"])
    raw = raw[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)
              & ~raw["InvoiceNo"].astype(str).str.startswith("C")].copy()
    raw["date"] = raw["InvoiceDate"].dt.normalize()
    first, last = raw["date"].min(), raw["date"].max()
    dates = pd.date_range(first, last, freq="D")
    cutoff = int(len(dates) * 0.50)
    history = raw[raw["date"] < dates[cutoff]]
    selected = history.groupby("StockCode")["Quantity"].sum().nlargest(RETAIL["real_top_skus"]).index
    raw = raw[raw["StockCode"].isin(selected)]
    daily_sales = raw.assign(line_value=raw["Quantity"] * raw["UnitPrice"]).groupby(
        ["StockCode", "date"], as_index=False).agg(demand=("Quantity", "sum"),
                                                       sales_value=("line_value", "sum"))
    daily_sales["transaction_price"] = daily_sales["sales_value"] / daily_sales["demand"]
    panel = pd.MultiIndex.from_product([selected, dates], names=["sku", "date"]).to_frame(index=False)
    panel = panel.merge(daily_sales.rename(columns={"StockCode": "sku"}), on=["sku", "date"], how="left")
    panel["demand"] = panel["demand"].fillna(0.0)
    panel["day"] = (panel["date"] - first).dt.days.astype(int)

    ref = daily_sales[daily_sales["date"] < dates[cutoff]].groupby("StockCode")["transaction_price"].median()
    global_ref = float(ref.median())
    panel["ref_price"] = panel["sku"].map(ref).fillna(global_ref).astype(float)
    # Preserve zero-demand product-days: fill transaction-only fields before
    # the rolling-feature warm-up drops rows with missing lags.
    panel["sales_value"] = panel["sales_value"].fillna(0.0)
    panel["transaction_price"] = panel["transaction_price"].fillna(panel["ref_price"])
    panel["price"] = panel["ref_price"]
    panel["unit_cost"] = panel["ref_price"] * RETAIL["real_cost_ratio"]
    panel = panel.sort_values(["sku", "day"]).reset_index(drop=True)
    g = panel.groupby("sku")["demand"]
    for lag in (1, 2, 7, 14, 28):
        panel[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    panel["roll_mean_7"] = shifted.groupby(panel["sku"]).transform(lambda s: s.rolling(7).mean())
    panel["roll_mean_28"] = shifted.groupby(panel["sku"]).transform(lambda s: s.rolling(28).mean())
    panel["roll_std_7"] = shifted.groupby(panel["sku"]).transform(lambda s: s.rolling(7).std())
    panel["dow"] = panel["date"].dt.dayofweek
    doy = panel["date"].dt.dayofyear
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    panel = panel.dropna().reset_index(drop=True)
    sku_features = pd.get_dummies(panel["sku"].astype(str), prefix="sku", dtype=float)
    feature_cols = ["lag_1", "lag_2", "lag_7", "lag_14", "lag_28", "roll_mean_7",
                    "roll_mean_28", "roll_std_7", "dow", "doy_sin", "doy_cos"]
    X = pd.concat([panel[feature_cols].astype(float), sku_features], axis=1)
    aux = panel[["day", "sku", "unit_cost", "price", "ref_price", "lag_7"]].reset_index(drop=True)
    clean = panel[["sku", "date", "day", "demand", "ref_price", "unit_cost", "price"]]
    clean.to_csv(DATA_DIR / "online_retail_daily.csv.gz", index=False, compression="gzip")
    return dict(X=X.reset_index(drop=True), y=panel["demand"].values.astype(float),
                aux=aux, task="reg", retail_real=True,
                rolling_fractions=((0.50, 0.65, 0.80), (0.60, 0.75, 0.90), (0.70, 0.85, 1.00)))


LOADERS = {"fraud": load_fraud, "churn": load_churn, "credit": load_credit,
           "retail": load_retail, "retail_real": load_retail_real}
