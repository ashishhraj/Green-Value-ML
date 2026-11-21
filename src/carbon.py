"""Thin wrapper around CodeCarbon that also records hardware-agnostic CPU time.

CodeCarbon estimates energy from CPU/RAM power (RAPL when available, otherwise
a TDP-based model) and converts it to CO2e with the grid intensity of
COUNTRY_ISO. CPU-seconds are logged alongside as an independent proxy, so the
ranking of models can be checked without trusting any one energy model.
"""
from __future__ import annotations

import logging
import os
import time
import warnings

from codecarbon import OfflineEmissionsTracker
import codecarbon.emissions_tracker as cc_tracker
import codecarbon.core.util as cc_util

from config import COUNTRY_ISO

logging.getLogger("codecarbon").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# CodeCarbon's Windows helper shells out to the legacy `powershell` executable
# to count processor sockets. On systems where only PowerShell Core is present,
# that command is absent. This study runs on a single virtual CPU socket, so use
# the documented single-socket fallback rather than silently failing to meter.
if os.name == "nt":
    cc_util._windows_get_physical_sockets = lambda: 1
    cc_tracker.count_physical_cpus = lambda: 1


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def measure(fn, *args, **kwargs):
    """Run fn(*args) under a CodeCarbon tracker.

    Returns (result, dict(kwh, kg_co2e, wall_s, cpu_s)).
    """
    tracker = OfflineEmissionsTracker(country_iso_code=COUNTRY_ISO, measure_power_secs=1,
                                      log_level="error", save_to_file=False)
    c0, w0 = _cpu_seconds(), time.perf_counter()
    tracker.start()
    try:
        out = fn(*args, **kwargs)
    finally:
        kg = tracker.stop()
    w1, c1 = time.perf_counter(), _cpu_seconds()
    kwh = tracker.final_emissions_data.energy_consumed
    return out, dict(kwh=float(kwh), kg=float(kg or 0.0), wall_s=w1 - w0, cpu_s=c1 - c0)
