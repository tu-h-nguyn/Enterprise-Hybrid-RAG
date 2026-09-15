# Design and construction report

`report.tex` is the source; `report.pdf` is the compiled result. The `.tex` is
meant to be edited — it is a plain `article` with no custom class and no external
dependencies beyond the packages named in its preamble.

## Building

```bash
pdflatex report.tex && pdflatex report.tex     # twice, for the table of contents
```

On Debian or Ubuntu the packages needed are:

```bash
apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended lmodern
```

## Where its numbers come from

Every figure in the report is taken from a committed artefact under
`enterprise-hybrid-rag/data/evaluation/` — `results.json`,
`results_fallback.json`, `scale_results.json` — each produced by a run that
actually executed and each recording which backends produced it. Where a figure
is historical, such as a previous image size or an example payload captured at an
earlier default, the report says so in the text.

If the evaluation is re-run and the numbers move, the report does not update
itself. Regenerate the artefacts first, then edit the tables here; the report is
a snapshot with a date on its title page, not a live dashboard.
