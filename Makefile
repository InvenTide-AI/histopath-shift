# Reproduce the paper's analysis, tables and figures from the shipped run records (CPU only).
#   make records    unpack records/ into $(HISTOPATH_DATA)/v2
#   make test       unit tests of the training code and the statistics
#   make analysis   variance decomposition, contrasts, power, ranks, cross-cohort analysis (all rules)
#   make tables     LaTeX tables + paper_numbers.json (incl. the site acceptance test tables)
#   make sat        re-run the validation of the site acceptance test (~45 min; uses records/v2_test_predictions_*)
#   make figures    every figure of the paper
#   make paper      compile paper/media/main.pdf (needs a TeX installation)
#   make all        records analysis tables figures
HISTOPATH_DATA ?= $(CURDIR)/data
export HISTOPATH_DATA
PY    ?= python
SNAP  := $(HISTOPATH_DATA)/v2/snap_final2_20261005
OUT   := results/v2/analysis
RULES := id ood last oracle

.PHONY: all records test analysis tables figures paper sat
all: records analysis tables figures

records:
	scripts/unpack_records.sh

test:
	$(PY) src/v2/tests/test_v2.py
	$(PY) src/v2/stats_tests/run_all.py

analysis:
	for t in C S; do for s in $(RULES); do \
	  $(PY) src/v2/analyze_v2.py runs --v2-root $(SNAP) --tier $$t --select $$s --out $(OUT) || exit 1; done; done
	for s in $(RULES); do \
	  $(PY) src/v2/analyze_v2.py runs --v2-root $(SNAP) --tier S --select $$s --ladder only --out $(OUT) || exit 1; done
	# un-normalized MIDOG 2021 (compact-tier sensitivity): analysed on its own so it never enters the 2x2 analysis;
	# the figures read it from $(OUT), the tables from results/v2/analysis_sens
	for s in $(RULES); do \
	  $(PY) src/v2/analyze_v2.py runs --v2-root $(SNAP) --tier C --select $$s --cohorts midog21 --out $(OUT) || exit 1; done
	$(PY) src/v2/analyze_v2.py runs --v2-root $(SNAP) --tier C --select id --cohorts midog21 --out results/v2/analysis_sens

tables:
	$(PY) src/v2/paper_tables.py --analysis $(OUT) --snap $(SNAP) --out paper/media/tables \
	  --numbers $(OUT)/paper_numbers.json
	$(PY) src/v2/site_acceptance.py tables --plan-check

sat:
	for t in S C; do $(PY) src/v2/site_acceptance.py evaluate --tier $$t --snap $(SNAP) || exit 1; done

figures:
	$(PY) src/v2/make_figures_v2.py --runs $(SNAP) --analysis $(OUT) --out paper/media/figs --tier both \
	  --domainbed results/v2/external/domainbed_icc.csv

paper:
	$(MAKE) -C paper/media
