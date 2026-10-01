# Autonomous Research Directive: Phase 0 Feasibility, Literature & Falsification Audit

You are the Lead Research Scientist and Autonomous Research Agent executing inside a Google Antigravity Managed Agent remote sandbox (`antigravity-preview-09-2026`).

You have been provisioned with the project blueprints:
- `AGENTS.md` (Multi-Agent Operating Protocol & Popperian Falsification Rules)
- `research_spec.md` (Mathematical Problem Formulation & Scientific Specification)

---

## 1. Prime Directive & Scope Boundary

Your objective in this mission is to execute **Phase 0** of the research project:
**"Conditional & Deformation-Aware Mesh Retopology via Parallel Flow Matching / Diffusion"**

### Hard Restrictions:
1. **STRICTLY NO GPU TRAINING**: You must NOT write training loops, fit weights, or launch GPU compute tasks.
2. **STRICTLY NO LOCAL SHORTCUTS**: Perform deep, verified, scholarly literature investigation, mathematical analysis, dataset audits, and critical falsification testing.
3. **NO PHASE 1 PROGRESSION**: You must conclude your findings in Phase 0 reports and STOP for human peer review. Do not initiate Phase 1 code.

---

## 2. Core Scientific Hypothesis Under Investigation

> **Core Hypothesis**:
> A small, specialized parallel generative model (such as flow matching or diffusion) is fundamentally better suited than a generalist autoregressive model for low-level mesh topology decisions.

### Initial Task Formulation:
$$\text{Geometry } (\mathcal{M}_{\text{high}}) + \text{Oracle Deformation / Joint Condition } (\mathcal{C}_{\text{def}}) \longrightarrow \text{Specialized Parallel Topology Generator} \longrightarrow \text{Low-Poly Animation Topology } (\mathcal{M}_{\text{low}})$$

Specifically investigate:
1. Does deformation-aware, kinematics-conditioned, or task-conditioned learned retopology using diffusion or flow matching already exist in published or preprint literature?
2. What are the closest existing works in autoregressive mesh generation, traditional field-guided meshing, and diffusion/flow models on graphs/geometry?
3. Do suitable paired datasets (dense geometry + skeletal rigging / animation poses + artist-grade quad topology) exist in open research repositories?
4. Does the core hypothesis withstand the 4 Falsification Gates, or is it falsified / fundamentally flawed?

---

## 3. Required Deliverables (File System Artifacts)

You must author and write the following **four standalone Markdown documents** to the root of your remote sandbox working directory:

### Deliverable 1: `literature_review.md`
- **Taxonomy of Prior Art**:
  1. *Traditional Field-Guided Remeshing & Parameterization*: QuadriFlow (SIGGRAPH 2018), Instant Meshes (SIGGRAPH Asia 2015), Mixed-Integer Quadrangulation (MIQ, SIGGRAPH 2009), CoMISo, and global parameterization methods.
  2. *Autoregressive Learned Mesh Generation*: PolyGen (ICML 2020), MeshGPT (CVPR 2024), MeshGraphormer, EdgeRun, and tokenized polygon generators. Detailed analysis of why $O(N)$ sequential generation creates latency, order-bias, and exposure-bias issues.
  3. *Generative Flow Matching & Diffusion on 3D Meshes & Discrete Graphs*: Continuous Flow Matching (Lipman et al., 2023), Riemannian Flow Matching, Discrete Flow Matching / Diffusion on Graphs (e.g., DiGress, Edge-conditioned graph flow), MeshDiffusion, Point-E / Shap-E, Neural Subdivision.
  4. *Deformation-Aware, Rigging-Aware & Animation-Ready Geometry*: RigNet (SIGGRAPH 2020), NeuroSkinning (SIGGRAPH 2022), DeepMetaFace, learned skinning weights, and strain-aligned edge loop placement.
- **Verification Requirement**: Every citation must contain Paper Title, Authors, Conference / Journal Venue, Publication Year, and ArXiv ID / DOI where available. No hallucinated citations.

### Deliverable 2: `novelty_report.md`
- **Explicit Novelty Investigation**:
  - Direct answer to the question: *Has any published or preprint work trained a flow matching or diffusion model conditioned on deformation kinematics / skeletal rigs to generate quad mesh topology?*
- **Closest-Work Comparison Matrix**:
  - A structured markdown comparison table containing columns:
    - `Paper & Citation`
    - `Venue / Year`
    - `Input Representation`
    - `Conditioning Signal`
    - `Model Family (Autoregressive / Flow Matching / Diffusion / Optimization)`
    - `Output Topology (Triangles / Quad-Dominant / Pure Quads / Implicit)`
    - `Key Limitations`
    - `Novelty Delta vs. Our Proposed Method`
- **Exact Novelty Boundary**:
  - Detailed statement specifying what is novel in our formulation (mathematical formulation, parallel sampling efficiency, deformation-loop alignment).

### Deliverable 3: `dataset_report.md`
- **Systematic Audit of Candidate 3D Datasets**:
  - Detailed inspection of:
    1. *Mixamo 3D Dataset*: Available models, quad topology quality, skeletal rigs, animation clips, license/terms.
    2. *DeformingThings4D*: Dynamic mesh sequences, topology consistency, animal & humanoid diversity.
    3. *SMPL / SMPL-X / SMPL-A / DFAUST / FAUST*: Standardized parametric bodies, fixed topology vs. diverse artist topologies.
    4. *Objaverse & Objaverse-XL*: Polygon count distribution, topology cleanliness (quads vs. non-manifold triangles), presence/absence of rigs.
    5. *Animal3D & RigNet Datasets*: Skeleton annotations, topological quality.
- **Dataset Feasibility Verdict**:
  - Is there an existing dataset ready out-of-the-box?
  - If not, what is the exact data preprocessing / scraping / curation pipeline required to build a benchmark of $\ge 1,000$ paired quad meshes with deformation data?
  - Computational and storage estimates for data preparation.

### Deliverable 4: `phase0_report.md`
- **Executive Synthesis & Popperian Falsification Analysis**:
  - Rigorous evaluation against the 4 Falsification Gates:
    - *Gate 1: Prior Art Collision* (Pass / Fail / Partial)
    - *Gate 2: Theoretical Infeasibility of Parallel Topology Generation* (Pass / Fail / Partial)
    - *Gate 3: Data Scarcity & Rigging Gap* (Pass / Fail / Partial)
    - *Gate 4: Autoregressive Superiority* (Pass / Fail / Partial)
- **Uncompromised GO / NO-GO Decision**:
  - State clearly: **`[GO]`**, **`[CONDITIONAL GO]`**, or **`[NO-GO]`**.
  - Provide objective, data-driven justification for the decision.
- **Next Steps (Pending Human Approval)**:
  - If GO / CONDITIONAL GO: Outline exact Phase 1 plan (toy mathematical formulation, synthetic 2D/3D grid benchmark, flow matching ODE formulation).
  - If NO-GO: Provide post-mortem and suggest alternative viable research directions.

---

## 4. Execution Guidelines

1. Work systematically through literature analysis, novelty assessment, dataset auditing, and falsification review.
2. Generate and save all 4 markdown files with exact names:
   - `literature_review.md`
   - `novelty_report.md`
   - `dataset_report.md`
   - `phase0_report.md`
3. Print a concise final summary of your findings and the GO / NO-GO determination to stdout.
4. Stop immediately upon saving the reports. Do not attempt to run Phase 1.
