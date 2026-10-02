# Paper draft

`main.tex` is a formulation draft. The numerical SOTA claims are withdrawn. See the status quote after `\maketitle` and the repository `STATUS.md`.

`tables/table1_sota.tex` keeps the first-pass numbers with corrected row names: C++ QuadriFlow, an autoregressive drift surrogate, and an analytic cylinder lattice. It is not a comparison against MeshGPT or against `FlowRetopoDiT`.

`references.bib` was corrected for venue and authorship on QuadriFlow, MeshGPT, PolyGen, Flow Matching, and RigNet. PolyFlow (arXiv:2606.30673) is included because it already covers continuous flow matching for mesh topology.

Compile with `latexmk -pdf -bibtex main.tex` from this directory when a TeX installation is available.
