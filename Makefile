# Reproduce the whole study:   make reproduce      (about 2.5 h on 2 CPU cores)
# Rebuild only paper numbers:  make analysis       (minutes; uses results/ already in the repo)
PY ?= python

.PHONY: reproduce data experiments ablation analysis paper test clean-results

reproduce: data experiments ablation analysis paper

data:
	$(PY) src/download_data.py

experiments:
	$(PY) src/run_experiments.py --scenarios credit churn retail retail_real fraud

ablation:
	$(PY) src/run_experiments.py --ablation

analysis:
	$(PY) src/analysis.py
	$(PY) src/retail_real_analysis.py
	$(PY) src/sensitivity.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

test:
	$(PY) -m pytest -q tests

# start a fresh run (keeps a copy of the shipped results in results_shipped/)
clean-results:
	mv results results_shipped && mkdir results
