# Regenerate every table and figure of the article from the shipped run records (CPU only).
#   make records    verify and unpack records/ into $(HISTOPATH_DATA)/v2
#   make test       unit tests of the training code and the statistics
#   make analysis   variance decomposition, contrasts, power, ranks, cross-cohort analysis (all rules, ~15 min)
#   make sat        validate the site acceptance test on the saved test predictions (both tiers, ~45 min)
#   make tables     LaTeX tables -> outputs/tables, numbers -> results/v2/analysis/paper_numbers.json
#   make figures    every data figure -> outputs/figs
#   make all        records analysis sat tables figures
HISTOPATH_DATA ?= $(CURDIR)/data
export HISTOPATH_DATA
PY    ?= python
SNAP  := $(HISTOPATH_DATA)/v2/snap_final2_20261005
OUT   := results/v2/analysis
SAT   := results/v2/site_acceptance
RULES := id ood last oracle

.PHONY: all records test analysis tables figures sat
all: records analysis sat tables figures

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

sat: $(SAT)/sat_rows_S.csv $(SAT)/sat_rows_C.csv

$(SAT)/sat_rows_%.csv:
	$(PY) src/v2/site_acceptance.py evaluate --tier $* --snap $(SNAP) --out $(SAT)

tables: sat
	$(PY) src/v2/paper_tables.py --analysis $(OUT) --snap $(SNAP) --out outputs/tables \
	  --numbers $(OUT)/paper_numbers.json
	$(PY) src/v2/site_acceptance.py tables --rows $(SAT) --out outputs/tables --plan-check

figures: tables
	$(PY) src/v2/make_figures_v2.py --runs $(SNAP) --analysis $(OUT) --out outputs/figs --tier both \
	  --domainbed results/v2/external/domainbed_icc.csv
