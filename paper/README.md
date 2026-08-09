# Paper Build

The manuscript is generated from the confirmatory JSON artifacts, not from manually copied values.

```bash
venv/bin/python -m analysis.paper_assets
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

`analysis.paper_assets` writes `generated_results.tex` and the compact result tables consumed by `main.tex`. Run the complete study first with the scripts documented in the repository README. The compiled manuscript is `paper/main.pdf`.

The author line is intentionally anonymous for review. Replace it only when preparing a named submission.
