"""Central configuration for the Carbon-vs-Business-Value study.

Every business assumption used to convert model predictions into money
lives here, so that a reader can audit (and change) them in one place.
All monetary values are in the native currency of each dataset
(EUR for fraud, USD for churn, DM for German credit, INR for the
synthetic retail data) and are never mixed across scenarios.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"   # inside paper/ so the folder uploads to Overleaf as-is

SEEDS = [0, 1, 2]                 # repeated stratified splits (large data)
# small datasets get more repeats so that value estimates are stable
SEEDS_BY_SCENARIO = {"credit": list(range(10)), "churn": list(range(5)),
                     "retail": SEEDS, "retail_real": SEEDS, "fraud": SEEDS}
SPLIT = (0.6, 0.2, 0.2)           # train / validation / test
COUNTRY_ISO = "IND"               # grid used by CodeCarbon (India)
N_JOBS = 2                        # fixed so energy is comparable across models

# ---------------------------------------------------------------------------
# Public data sources (downloaded by src/download_data.py)
# ---------------------------------------------------------------------------
SOURCES = {
    "creditcard.csv": "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
    "telco_churn.csv": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "german_credit.csv": "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv",
}

# ---------------------------------------------------------------------------
# Business-value assumptions (cited / justified in the paper, Sec. III-B)
# ---------------------------------------------------------------------------
FRAUD = dict(review_cost=5.0)            # EUR per alert investigated by an analyst
CHURN = dict(offer_cost=25.0,            # USD per customer targeted (contact + discount)
             accept_rate=0.30,           # share of true churners retained by the offer
             horizon_months=12)          # revenue horizon of a retained customer
CREDIT = dict(margin=0.10,               # profit on a repaid loan, fraction of amount
              lgd=0.60)                  # loss-given-default, fraction of amount
RETAIL = dict(n_sku=120, n_days=3 * 365, # synthetic perishable-retail generator
              holding_frac=0.30,         # cost of an unsold unit (spoilage), fraction of unit cost
              real_top_skus=120,          # UCI real-retail validation panel
              real_cost_ratio=0.60,       # assumed unit cost as a share of training-period median price
              )

# Annual deployment profile used to project lifetime carbon & value (Sec. V-D)
DEPLOYMENT = {
    "fraud":  dict(retrains=52, annual_predictions=100_000_000),  # weekly retrain, 100M txns/yr
    "churn":  dict(retrains=12, annual_predictions=1_000_000),   # one campaign/yr over 1M customers
    "credit": dict(retrains=4,  annual_predictions=100_000),      # quarterly retrain, 100k applications
    "retail": dict(retrains=52, annual_predictions=1_000 * 365),  # 1,000 SKUs forecast daily
}

# Grid carbon intensities (kg CO2e / kWh), Ember Global Electricity Review 2024 (2023 data)
GRIDS = {"India": 0.713, "United States": 0.369, "European Union": 0.244, "France": 0.056}
SOCIAL_COST_CARBON = 190.0               # USD / tCO2e, US EPA (2023) central estimate

# ---------------------------------------------------------------------------
# Model-selection and statistics settings (Sec. III-E)
# ---------------------------------------------------------------------------
GVP_TOL = 0.05          # Green Value Point: cheapest model within 5% of the best VALIDATION value
EQUIV_MARGIN_PP = 5.0   # equivalence / non-inferiority margin, percentage points of oracle value
N_BOOT = 2000           # paired bootstrap resamples
BOOT_SEED = 0
