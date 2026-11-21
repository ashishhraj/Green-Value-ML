# Green Value: Carbon Emissions vs. Business Value of ML Models

Code, data, results and paper for the AMLBT project *"Beyond Accuracy: An Empirical Study of
Machine-Learning Carbon Emissions versus Business Value Fulfilment"* (Goa Institute of Management,
project 21B).

Repository: https://github.com/ashishhraj/Green-Value-ML

**Question.** Green-AI work compares carbon against *accuracy*. Businesses do not earn accuracy — they
earn money from decisions. Does spending more carbon on a model buy more business value?

**Design.** Four modeled business scenarios plus a real transaction-data check × a 10-rung model ladder
(linear → hyper-parameter search) × repeated splits. Every training run and every inference batch is
metered with CodeCarbon, and every prediction is converted into money with an explicit, auditable value
function under three decision rules (default 0.5 cut-off, a validation-tuned threshold, and a Bayes rule on
validation-calibrated probabilities).

**Selection protocol.** Everything that is *chosen* — thresholds, calibrators, which decision rule to use,
the value-best model, the accuracy-best model and the Green Value Point (GVP) — is chosen on
**validation** data. The test set is only used to report the outcome of those choices, to bootstrap paired
differences and to run equivalence (TOST) tests. The business-assumption sensitivity analysis follows the
same protocol. `tests/test_no_test_leakage.py` checks this.

| Scenario | Data | Rows | Business value (vs. do-nothing) |
|---|---|---|---|
| Fraud detection | ULB credit-card fraud (real, public) | 284,807 | fraud € caught − €5 per analyst review |
| Churn retention | IBM Telco churn (real, public) | 7,043 | 30% × 12 months revenue retained − $25 per offer |
| Credit approval | German credit / Statlog (real, public) | 1,000 | 10% margin on repaid loans − 60% LGD on defaults |
| Perishable ordering | Synthetic retail demand (generated, seeded) | 128,040 | newsvendor cost saved vs. seasonal-naive ordering |
| Real retail check | UCI Online Retail transactions, aggregated daily | 41,520 | inventory-style value under stated unit-cost / holding-cost assumptions |

## Repository layout

```
src/config.py              every business assumption, seed, grid factor and selection setting
src/download_data.py       fetches the 3 public datasets and prints SHA-256 checksums
src/data.py                loaders, synthetic retail generator, UCI Online Retail panel builder
src/value.py               utility framework, decision rules D0/D1/D2, Platt calibration, newsvendor cost
src/models.py              the 10-rung model ladder
src/carbon.py              CodeCarbon wrapper (+ CPU-seconds as a hardware-agnostic proxy)
src/env_log.py             writes results/run_config.json (versions, hardware, CodeCarbon power model)
src/run_experiments.py     main study (resumable), real-retail folds and fraud data-volume ablation
src/analysis.py            validation-based selection, bootstrap/TOST, all tables, figures and numbers
src/retail_real_analysis.py  validation-selected UCI Online Retail table
src/sensitivity.py         38 alternative business-cost settings, re-tuned and re-selected on validation
tests/                     unit tests for the value functions and the no-test-leakage protocol
results/                   runs.csv (one row per run), preds/ (validation + test scores), summaries, logs
paper/                     IEEE LaTeX source (main.tex, tables/, figures/, numbers*.tex) and main.pdf
docs/REVIEW_RESPONSE.md    how each first-review comment was addressed
```

Every number in the paper is a LaTeX macro written by the scripts (`paper/numbers*.tex`), and every
table in `paper/tables/` is generated — nothing is typed by hand.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # tested with Python 3.13
pip install -r requirements.txt
make test                    # unit tests, seconds
make data                    # downloads ~100 MB (the fraud CSV is not stored in the repo)
make experiments             # main study + real-retail folds; resumable; ~2.5 h on 2 CPU cores
make ablation                # fraud data-volume ablation
make analysis                # regenerates every table, figure and number in the paper
make paper                   # compiles paper/main.pdf (needs a LaTeX installation)
```

To regenerate the paper from the shipped results without re-training, run `make data analysis paper`
(the fraud CSV is needed because the value function uses transaction amounts). To re-run from scratch,
`make clean-results` first moves the shipped results to `results_shipped/`.

**What should replicate.** Business values are deterministic given the seeds and pinned versions
(re-running a model reproduces its test value exactly). Carbon figures depend on hardware and grid:
`results/run_config.json` records the CPU, the CodeCarbon power model and all versions used. Rankings and
ratios — the paper's claims — are what should replicate; `train_cpu_s` is logged next to every energy
figure so this can be checked independently. `make analysis` re-meters only one quantity, the sub-milligram cost of tuning the decision rule
(ΔC column of the decision-layer table), so that column can differ slightly between runs; every other table,
figure and number regenerates identically from the shipped results.

## How carbon is estimated

The study machine (2 vCPU Intel Xeon @ 2.1 GHz, 7.8 GB RAM) exposes no RAPL energy counters, so
CodeCarbon 3.3.1 uses its fallback power model: CPU power = TDP × (0.1 + 0.9 × load³), with a default TDP
of 4 W per thread (8 W) because this CPU model is not in CodeCarbon's database, plus a constant 10 W for
RAM. Energy is that power integrated over wall-clock time, and CO2e = energy × 0.713 kg/kWh (India).
Absolute values are therefore model-based estimates; see Sec. III-D of the paper for the uncertainty
discussion.

## Changing the business assumptions

All money parameters live in `src/config.py` (`FRAUD`, `CHURN`, `CREDIT`, `RETAIL`). Edit them and run
`make analysis` (decision rules are re-tuned on the saved validation scores, so no re-training is needed),
or run `src/sensitivity.py` to sweep a grid of settings.

## Data sources and licences

- Credit-card fraud: Dal Pozzolo et al., ULB Machine Learning Group (Kaggle "creditcardfraud", DbCL v1.0). Downloaded by `src/download_data.py`.
- Telco churn: IBM sample dataset (IBM/telco-customer-churn-on-icp4d, Apache-2.0).
- German credit: Hofmann, Statlog (German Credit Data), UCI ML Repository (CC BY 4.0).
- Synthetic retail: generated by `src/data.py` with seed 2026.
- Real retail transaction check: [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), UCI ML Repository, DOI `10.24432/C5BW33` (CC BY 4.0), stored as `data/online_retail.zip`. It has no inventory or unit-cost fields, so its monetary results are assumption-based and are not observed savings.

Code is released under the MIT licence (`LICENSE`). Citation metadata: `CITATION.cff`.
