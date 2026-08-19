# Makefile for Multi-Mode Mixed-Criticality Scheduling Research

.PHONY: all clean validate compile check-bib sync-bib

all: task_model.pdf evaluation_methodology.pdf mode_optimization.pdf

# Real compile check (undefined macros, missing citations, broken \ref) via
# tectonic, not a syntax-only linter. See validate_latex.py's module docstring
# for why the regex-only version it replaced missed real, live bugs.
validate:
	python3 ../validate_latex.py .

check-bib:
	@echo "Checking bibliography files..."
	@test -f project_bib.bib || (echo "ERROR: project_bib.bib not found" && exit 1)
	@test -f references.bib || (echo "ERROR: references.bib not found -- run 'make sync-bib'" && exit 1)
	@echo "Bibliography check passed"

# Refresh references.bib from the master Overleaf bib project. Run this
# whenever entries have been added there; it is a snapshot copy, not a live
# link, so it does not update itself.
sync-bib:
	git -C .. fetch masterbib
	git -C .. show masterbib/main:references.bib > references.bib
	@echo "references.bib refreshed from the master Overleaf bib project"

task_model.pdf: task_model.tex macros.tex project_bib.bib references.bib
	tectonic task_model.tex
	@echo "Generated task_model.pdf"

evaluation_methodology.pdf: evaluation_methodology.tex macros.tex eval_defaults.tex project_bib.bib references.bib
	tectonic evaluation_methodology.tex
	@echo "Generated evaluation_methodology.pdf"

mode_optimization.pdf: mode_optimization.tex macros.tex project_bib.bib references.bib
	tectonic mode_optimization.tex
	@echo "Generated mode_optimization.pdf"

clean:
	rm -f *.aux *.bbl *.blg *.log *.pdf *.out
