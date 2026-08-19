# Makefile for Multi-Mode Mixed-Criticality Scheduling Research

.PHONY: all clean validate compile check-bib

all: task_model.pdf evaluation_methodology.pdf

validate:
	python3 ../validate_latex.py .

check-bib:
	@echo "Checking bibliography files..."
	@test -f project_bib.bib || (echo "ERROR: project_bib.bib not found" && exit 1)
	@test -f references.bib || (echo "WARNING: references.bib not found, using Overleaf master bib")
	@echo "Bibliography check passed"

task_model.pdf: task_model.tex project_bib.bib
	pdflatex -interaction=nonstopmode task_model.tex
	bibtex task_model
	pdflatex -interaction=nonstopmode task_model.tex
	pdflatex -interaction=nonstopmode task_model.tex
	@echo "Generated task_model.pdf"

evaluation_methodology.pdf: evaluation_methodology.tex project_bib.bib
	pdflatex -interaction=nonstopmode evaluation_methodology.tex
	bibtex evaluation_methodology
	pdflatex -interaction=nonstopmode evaluation_methodology.tex
	pdflatex -interaction=nonstopmode evaluation_methodology.tex
	@echo "Generated evaluation_methodology.pdf"

clean:
	rm -f *.aux *.bbl *.blg *.log *.pdf
