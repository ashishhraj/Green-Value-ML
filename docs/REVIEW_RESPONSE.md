# Response to the first review (project 21B-V1, confirmed mark 21.7 / 30)

Each priority action from the review is listed with what changed in the code, the results and the paper.
All numbers below come from the re-run shipped in `results/` (213 core runs + 33 real-retail runs + fraud ablation).

## Priority action 1: real repository, run configurations and outputs, accurate AI declaration

| Comment | Change |
|---|---|
| Repository URL is a placeholder | Repository packaged for GitHub (README, MIT `LICENSE`, `CITATION.cff`, `.gitignore`, `Makefile`, CI workflow running unit tests). The paper reads the URL from one macro, `\RepoURL`, at the top of `paper/main.tex`. **Set it to the real GitHub URL after pushing.** |
| Run configurations and outputs | `src/env_log.py` writes `results/run_config.json` (hardware, CodeCarbon power model, package versions, seeds, every business parameter). Shipped: `runs.csv`, validation **and** test predictions for every run (`results/preds/`), summaries, sensitivity grid and logs. |
| AI declaration retains unfilled instructions | Replaced with an account of the actual assistance (Claude: idea scoping, rubric self-assessment, this revision; ChatGPT: previous revision). **Authors should check it matches their own use.** |

## Priority action 2: carbon components and uncertainty, TDP, deployment assumptions, validation vs test

| Comment | Change |
|---|---|
| Report training/inference contributions and uncertainty separately | Table VIII now gives annual training and inference emissions separately for the value-best model and the GVP, each with its run-to-run relative s.d. |
| Explain TDP estimation | Sec. III-D explains exactly what CodeCarbon 3.3.1 did on this machine: no RAPL, unknown CPU → default TDP 4 W/thread (8 W), CPU power = TDP·(0.1+0.9·load³), plus a constant 10 W for RAM, integrated over wall time. It states the consequences: energy ≈ proportional to runtime (so the CPU-time check is partly circular), RAM is 56–93 % of estimated power, a 0.5×–3× TDP error shifts full-load power by about −22 % to +89 %. Run-to-run spread: median relative s.d. 11 %. |
| Explain deployment assumptions | Sec. IV-F lists what the projection includes (retrain count × measured training, prediction volume × measured batch inference) and excludes (serving overhead, idle capacity, pipelines, networking, embodied emissions). |
| Distinguish validation-based selection from test-set cross-fitted sensitivity | **Code change.** Validation scores are now saved for every run. All choices (thresholds, calibrator, D1/D2 rule, value-best model, accuracy-best model, best model at 0.5, GVP) are made on validation; the test set only reports outcomes. The sensitivity analysis no longer cross-fits on the test set: in each of the 38 settings rules are re-tuned and models re-selected on validation. `tests/test_no_test_leakage.py` checks that changing test labels cannot change any choice. New Sec. III-E describes the protocol. |
| "No significant difference" is not equivalence | Paired bootstrap pooled over all splits + TOST equivalence / non-inferiority at ±5 pp of oracle value (Table IV). |

## Priority action 3: real retail validation, utility robustness, probability quality before the Bayes rule

| Comment | Change |
|---|---|
| Validate on real retail data | UCI Online Retail check kept (three rolling-origin folds, validation-selected), re-run with the new code (Table VII). |
| Robustness to utility parameters | 38-setting grid, now validation-tuned and validation-selected (Sec. IV-G). The GVP stays within 5 pp of the value-best model in 38/38 settings and within 5 pp of hyper-parameter search in 36/38 while saving ≥90 % of its training carbon in 35/38. |
| Check probability quality before the Bayes-style rule | **Code change.** Before D2 is used, Brier and ECE are measured on validation, Platt scaling is fitted on validation (cross-fitted), ECE is re-measured, and the frozen calibrator is applied to test. D2 then uses calibrated probabilities with no tuned scale factor. Table VI reports raw vs calibrated ECE on validation and test and how often D2 was chosen. The old λ-tuned raw-score rule is logged for comparison only; calibrated D2 beats it on average in all three classification domains. |

## Findings that changed after the fixes (reported honestly in the paper)

- **Fraud:** with validation-only selection the GVP (RF-100) is **not** certified: −3.4 pp vs RF-500, 90 % CI [−8.7, +0.3]. The earlier V1/V2 claim of a 96 % saving with no value loss relied on test-set selection. A test-selected analysis would now even pick Linear as value-best; the paper uses this as the example of why selection must stay on validation.
- **Carbon–value correlation:** fraud is no longer significantly positive (ρ = −0.04); no domain is significant.
- **RQ2 reframed:** accuracy is unrelated to value at the 0.5 rule (ρ = −0.08 fraud, −0.24 credit, +0.49 churn) but tracks it once the rule is value-aware (+0.73, +0.56, +0.76). The accuracy-best vs value-best differences are not significant.
- **Hyper-parameter search** was the value-best model in none of the four domains (and in 3 of 38 sensitivity settings).
- **Decision rule vs model:** unchanged conclusion (credit +89 pp, fraud +28 pp, churn +23 pp from the rule).

## Items still open for the authors

1. Push the repository to GitHub and set `\RepoURL` in `paper/main.tex` (one line), then recompile.
2. Confirm the AI Use Declaration wording. The rubric template places it in Section XI-A; here it is a subsection of Reproducibility.
3. The rubric template's AI-percentage / Turnitin statement is not included because it cannot be verified from the project files; add it only if you have the report.
