"""Record the run configuration, software versions and the CodeCarbon power model.

Writes results/run_config.json. Called automatically by run_experiments.py, and can be
run on its own:  python src/env_log.py
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "codecarbon", "openpyxl", "psutil"]


def codecarbon_power_model() -> dict:
    """Ask CodeCarbon which CPU/RAM power model it will use on this machine."""
    try:
        from codecarbon.core.cpu import TDP
        from codecarbon.external.ram import RAM
        model, tdp = TDP().model, TDP().tdp
        ram_w = RAM(tracking_mode="machine").total_power().W
        return {"cpu_model": model, "cpu_tdp_w_used": tdp, "ram_power_w": float(ram_w),
                "cpu_power_formula": "TDP * (0.1 + 0.9 * load^3)  [CodeCarbon 'cpu_load' mode, machine tracking]",
                "rapl_available": os.path.exists("/sys/class/powercap/intel-rapl")}
    except Exception as exc:  # pragma: no cover - informational only
        return {"error": repr(exc)}


def collect() -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
        "packages": {p: (md.version(p) if _has(p) else None) for p in PACKAGES},
        "config": {"SEEDS_BY_SCENARIO": C.SEEDS_BY_SCENARIO, "SPLIT": C.SPLIT,
                   "COUNTRY_ISO": C.COUNTRY_ISO, "N_JOBS": C.N_JOBS, "FRAUD": C.FRAUD,
                   "CHURN": C.CHURN, "CREDIT": C.CREDIT, "RETAIL": C.RETAIL,
                   "DEPLOYMENT": C.DEPLOYMENT, "GRIDS": C.GRIDS,
                   "SOCIAL_COST_CARBON": C.SOCIAL_COST_CARBON, "GVP_TOL": C.GVP_TOL,
                   "EQUIV_MARGIN_PP": C.EQUIV_MARGIN_PP},
        "codecarbon_power_model": codecarbon_power_model(),
    }


def _has(p):
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


def write(path: Path | None = None) -> dict:
    info = collect()
    path = path or C.RESULTS_DIR / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))
    return info


if __name__ == "__main__":
    print(json.dumps(write(), indent=2, default=str))
