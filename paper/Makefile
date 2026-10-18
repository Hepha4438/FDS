# Build order is fixed for xr-hyper: main -> supplement -> main.
# Cross-references between the two PDFs come from each other's .aux; a single-file build
# silently leaves them stale. Always build with `make`.
TEX = pdflatex -interaction=nonstopmode -halt-on-error

all: main.pdf supplement.pdf

main.pdf supplement.pdf: main.tex supplement.tex refs.bib $(wildcard sections/*.tex)
	$(TEX) main
	$(TEX) supplement
	bibtex main
	bibtex supplement
	$(TEX) main
	$(TEX) supplement
	$(TEX) main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk

.PHONY: all clean
