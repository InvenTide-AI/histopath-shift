# Build the MedIA manuscript: pdflatex -> bibtex -> pdflatex x2 (latexmk not required).
#   make            build main.pdf and supplementary.pdf (cross-referenced)
#   make cover      build cover_letter.pdf
#   make check      grep the log for errors, warnings, undefined refs/citations
#   make todo       list remaining TODO-AUTHORS / TODO markers
#   make clean      remove auxiliary files
# If main.bib is not here yet, a stub is taken from $(STUBBIB) via BIBINPUTS.
TEXBIN  ?= $(HOME)/.TinyTeX/bin/x86_64-linux
export PATH := $(TEXBIN):$(PATH)
STUBBIB ?= /tmp/media_stubbib
export BIBINPUTS := .:$(STUBBIB):
LATEX   = pdflatex -interaction=nonstopmode -halt-on-error -file-line-error
MAIN    = main

.PHONY: all cover check todo clean supp
all: $(MAIN).pdf supplementary.pdf

# main and supplement reference each other (xr): build both together
SRC = $(MAIN).tex supplementary.tex $(wildcard sections/*.tex) $(wildcard supp/*.tex) $(wildcard tables/*.tex) \
      $(wildcard $(MAIN).bib) $(wildcard figs/*)
$(MAIN).pdf supplementary.pdf: $(SRC)
	$(LATEX) supplementary
	$(LATEX) $(MAIN)
	bibtex supplementary
	bibtex $(MAIN)
	$(LATEX) supplementary
	$(LATEX) $(MAIN)
	$(LATEX) supplementary
	$(LATEX) $(MAIN)

supp: supplementary.pdf

cover: cover_letter.pdf
cover_letter.pdf: cover_letter.tex
	$(LATEX) cover_letter && $(LATEX) cover_letter

check:
	@grep -nE '^!|Warning|Undefined|undefined|Overfull' supplementary.log || echo "supplementary log: no errors or warnings"
	@grep -nE '^!|^[^:]+:[0-9]+:|Warning|Undefined|undefined|Overfull|Underfull' $(MAIN).log || echo "log: no errors or warnings"
	@grep -n -i 'warning' $(MAIN).blg || echo "blg: no warnings"

todo:
	@grep -rn 'TODO' $(MAIN).tex sections submission || true

clean:
	rm -f $(addprefix $(MAIN).,aux bbl blg log out spl toc lof lot)
	rm -f $(addprefix supplementary.,aux bbl blg log out spl toc lof lot)
	rm -f $(addprefix cover_letter.,aux log out)
