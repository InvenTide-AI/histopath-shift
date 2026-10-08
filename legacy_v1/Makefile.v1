# make check    verify the shipped tables against the published paper
# make figures  regenerate Fig. 1 from the shipped tables
# make all      both
# Nothing here downloads data or trains a model; see README for the full pipeline.

PY ?= python

.PHONY: all check figures lint clean

all: check figures

check:
	$(PY) src/check_release.py

figures: paper/fig1_site_complete_factorial.pdf

paper/fig1_site_complete_factorial.pdf: results/camelyon17_fivehospital/results_folds.csv src/make_figure_fivefold.py
	$(PY) src/make_figure_fivefold.py --out $@

lint:
	$(PY) -m compileall -q src

clean:
	rm -f paper/fig1_site_complete_factorial.png
