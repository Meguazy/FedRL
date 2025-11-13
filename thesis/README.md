# Thesis: Federated Reinforcement Learning for Chess

LaTeX source files for the thesis.

## Structure

- `main.tex` - Main thesis document
- `chapters/` - Individual chapter files
- `figures/` - Figures and images
- `bibliography/` - Bibliography files
- `appendices/` - Appendix content

## Building

To compile the thesis:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automated compilation:

```bash
latexmk -pdf main.tex
```
