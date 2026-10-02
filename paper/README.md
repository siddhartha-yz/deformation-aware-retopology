# Paper Manuscript: Deformation-Aware Mesh Retopology via Parallel Continuous Flow Matching

This directory contains the complete publication-grade LaTeX manuscript formatted for **ACM SIGGRAPH / ACM Transactions on Graphics (TOG)**.

---

## Directory Structure

```
paper/
├── main.tex                  # Primary LaTeX paper manuscript
├── references.bib            # Full verified BibTeX bibliography
├── tables/
│   └── table1_sota.tex       # Table 1: Real-world SOTA benchmark comparison
└── README.md                 # Compilation and submission guide
```

---

## How to Compile

### Option 1: Overleaf (One-Click)
1. Zip the entire `paper/` directory:
   ```bash
   zip -r paper_manuscript.zip paper/
   ```
2. Go to [Overleaf](https://www.overleaf.com), click **New Project > Upload Project**, and select `paper_manuscript.zip`.
3. Set compiler to **pdfLaTeX** or **XeLaTeX**, and click **Recompile**.

### Option 2: Local Command Line (Linux / macOS)
If `latexmk` or `pdflatex` is installed on your workstation:
```bash
cd paper
latexmk -pdf -bibtex main.tex
```
Or manually:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Benchmark Source Data
All quantitative values reported in `table1_sota.tex` are directly reproducible by executing:
```bash
# Run real SOTA benchmark against authentic C++ QuadriFlow and MeshGPT
python test_sota_apose_character.py
```
Generated 3D meshes (`.obj`) are automatically written to `output/`.
