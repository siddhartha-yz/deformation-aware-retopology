# Autonomous Research Directive: Phase 3 Scaling, Kinematic Conditioning & Production Retopology

You are the Lead Research Scientist and Generative Modeling Architect executing inside the Google Antigravity Managed Agent remote sandbox (`antigravity-preview-09-2026`).

---

## 1. Phase 3 Mission & Context

Human review of Phase 2 has officially authorized the transition with an uncompromised **`[GO / APPROVED]`** verdict.
You are now authorized to initiate the final culminating phase: **Phase 3: Scaling, Kinematic Conditioning & Production Retopology**.

You have access to all prior workspace artifacts:
- `AGENTS.md` (Multi-Agent Operating Protocol & Popperian Rules)
- `research_spec.md` (Scientific Problem Formulation & Falsification Gates)
- `runs/latest/phase0_report.md` (Phase 0 Literature & Novelty Audit)
- `runs/latest/phase1_math_spec.md` (Flow Matching & 4-RoSy Strain Mechanics)
- `runs/latest/flow_retopo_model.py` (PyTorch DiT Neural Backbone)
- `runs/latest/sampler_ablation.py` (Parallel ODE Solvers & Pareto Frontier)
- `runs/latest/baseline_comparison.py` (Comparative Study vs. Classical & AR)
- `runs/latest/phase2_report.md` (Phase 2 Empirical Synthesis)

### Core Mandates for Phase 3:
1. **Rig-Retopo-3K Dataset Ingestion Pipeline**: Implement a robust paired dataset preprocessor that extracts high-resolution geometry $\mathcal{M}_{\text{high}}$, kinematic skeleton hierarchies $\mathcal{J}$, bone skinning priors $\mathcal{W}$, deformation strain tensors $\mathbf{C}$, and artist-grade quad topologies $\mathcal{M}_{\text{low}}$.
2. **GPU Training Engine & Recipe**: Construct an efficient PyTorch training harness (`train_flow_retopo.py`) with mixed precision (BF16/FP16), AdamW optimization, cosine learning rate decay, and joint OT-CFM + 4-RoSy strain regularization.
3. **Production DCC Plugin (Blender Add-on)**: Deliver a complete, installable Blender 4.x / 5.x Python add-on (`blender_retopo_addon.py`) enabling 3D technical artists to execute one-click deformation-aware quad retopology directly in Blender's 3D viewport.
4. **Final Scientific Publication Report**: Synthesize the complete project results into a publication-ready final monograph (`phase3_report.md`), summarizing the theoretical proofs, empirical benchmarks, and definitive falsification gate verdicts.

---

## 2. Phase 3 Work Packages (WP)

You must execute the following four work packages:

### [WP 3.1] Rig-Retopo-3K Dataset Ingestion & Preprocessing (`dataset_pipeline.py`)
- Implement a comprehensive data loader and preprocessor supporting:
  - Synthetic articulated benchmarks (Hinge, Cylinder, Articulated Limb) and humanoid character assets (Mixamo / SMPL-X archetype schema).
  - Point cloud extraction: Poisson disk or uniform surface sampling ($M = 2048 \sim 4096$ points + normals).
  - Skeletal hierarchy extraction: Joint coordinate matrix $\mathbf{J} \in \mathbb{R}^{K \times 3}$, parent index map, and bone direction vectors.
  - Skinning weight matrix $\mathcal{W} \in [0, 1]^{N \times K}$ with normalization $\sum_k w_{ik} = 1$.
  - Deformation strain tensor extraction: Cauchy-Green tensor $\mathbf{C} = \mathbf{F}^T \mathbf{F}$ across representative animation flexion poses.
  - PyTorch `Dataset` and `DataLoader` classes with synthetic generation fallback for self-contained execution.

### [WP 3.2] Training Harness & Scaled Model Recipe (`train_flow_retopo.py`)
- Implement the complete PyTorch training engine:
  - Model initialization: `FlowRetopoDiT` supporting GPU acceleration (CUDA) and CPU fallback.
  - Objective: $\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{strain}} \mathcal{L}_{\text{strain}}$ with dynamic loss weighting.
  - Optimization: AdamW ($\text{lr} = 3 \times 10^{-4}$), linear warmup + cosine decay scheduler, gradient clipping (norm $\le 1.0$).
  - Self-contained training validation loop running 100~500 optimization steps, recording loss convergence curves, parameter gradients, and memory footprints.

### [WP 3.3] Blender DCC Production Retopology Add-on (`blender_retopo_addon.py`)
- Implement a production-ready Blender Python Add-on:
  - Standard `bl_info` metadata for Blender 4.x / 5.x.
  - Custom UI Panel in 3D Viewport (`VIEW_3D > Sidebar > RetopoFlow-AI` or `Tool > Flow Retopology`).
  - Operator `FLOWRETOPO_OT_generate`:
    - Reads active selected mesh object as high-poly target ($\mathcal{M}_{\text{high}}$).
    - Detects active or selected Armature in the scene as kinematic skeleton ($\mathcal{J}$).
    - Runs the parallel flow matching solver (using local PyTorch / NumPy engine).
    - Creates a new quad mesh object in the Blender scene with generated vertices, faces, and vertex groups.
    - Computes and displays topology quality metrics: Quad Ratio ($Q_\%$) and Valence-4 percentage ($V_{4\%}$).

### [WP 3.4] Culminating Scientific Monograph & Phase 3 Report (`phase3_report.md`)
- Author a publication-grade comprehensive report detailing:
  - Executive summary and final validation of the Core Hypothesis.
  - End-to-end performance benchmarks across asset classes.
  - Production readiness scorecard (latency, memory, manifoldness, skinning distortion).
  - Open challenges, future directions, and commercial deployment roadmap.

---

## 3. Required Deliverables (File System Artifacts)

You must create and save the following four artifacts in your root working directory:

1. **`dataset_pipeline.py`**:
   - Self-contained, executable PyTorch dataset and preprocessor module with synthetic asset generation and batch loading demo in `main`.
2. **`train_flow_retopo.py`**:
   - Complete executable training script with optimizer, loss logging, gradient validation, and checkpoint management.
3. **`blender_retopo_addon.py`**:
   - Fully compliant Blender Add-on script with UI panel, operators, and standalone test harness.
4. **`phase3_report.md`**:
   - Comprehensive Phase 3 research synthesis report and publication monograph.

In addition, output each artifact inside markdown code blocks in your final response for double redundancy.
Conclude with a clear status update.
