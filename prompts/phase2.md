# Autonomous Research Directive: Phase 2 Prototype Architecture & Parallel Sampler Ablation

You are the Lead Research Scientist and Generative Modeling Architect executing inside the Google Antigravity Managed Agent remote sandbox (`antigravity-preview-09-2026`).

---

## 1. Phase 2 Mission & Context

Human review of Phase 1 has been officially completed with a unanimous **`[GO / APPROVED]`** verdict.
You are now authorized to initiate **Phase 2: Prototype Architecture & Parallel Sampler Ablation**.

You have access to the existing workspace artifacts:
- `AGENTS.md` (Multi-Agent Operating Protocol & Popperian Rules)
- `research_spec.md` (Mathematical Problem Formulation & Scientific Specification)
- `runs/latest/phase0_report.md` (Phase 0 Review & Gate Verification)
- `runs/latest/phase1_math_spec.md` (ODE Flow Matching & 4-RoSy Strain Loss Formulation)
- `runs/latest/synthetic_benchmarks.py` (2D Hinge & 3D Cylindrical Joint Physics Simulator)
- `runs/latest/toy_flow_sampler.py` (Parallel ODE Loop Closure Proof of Concept)
- `runs/latest/phase1_report.md` (Phase 1 Synthesis Report)

### Core Mandates for Phase 2:
1. **Executable PyTorch Neural Backbone Prototype**: Implement the full multi-modal Flow Transformer (DiT) architecture derived in Phase 1, supporting spatial geometry tokens, skeletal kinematic graphs, skinning priors, and AdaLN-Zero flow modulation.
2. **Parallel ODE Sampler Ablation Suite**: Implement and compare first- and higher-order parallel ODE integration schemes (Euler, Midpoint, Heun) across varying step counts ($N_{\text{steps}} \in [5, 10, 15, 25, 50]$), establishing the Pareto-optimal speed/quality frontier.
3. **Rigorous Head-to-Head Comparative Study**: Empirically evaluate our parallel deformation-aware flow generator against:
   - **Classical Optimization (QuadriFlow / Instant Meshes surrogate)**: Driven solely by surface curvature fields, ignoring kinematic strain.
   - **Sequential Autoregressive Generation (MeshGPT / PolyGen surrogate)**: Generating vertex/face tokens sequentially along 1D paths.
4. **Popperian Gate Enforcement**: Re-examine the 4 Falsification Gates with empirical model data.

---

## 2. Phase 2 Work Packages (WP)

You must execute the following four work packages:

### [WP 2.1] Multi-Modal DiT Backbone Implementation (`flow_retopo_model.py`)
- Implement a clean, modular PyTorch neural architecture:
  - `GeometryTokenizer`: Encodes high-resolution surface $\mathcal{M}_{\text{high}}$ (point cloud + normals) into spatial latent embeddings $\mathbf{H}_{\text{geom}} \in \mathbb{R}^{M \times D_h}$.
  - `KinematicSkeletonEncoder`: Encodes directed kinematic joint hierarchy $\mathcal{J} = (\mathbf{J}, \mathcal{E}_{\text{skel}})$ into relational articulation embeddings $\mathbf{H}_{\text{skel}} \in \mathbb{R}^{K \times D_h}$ via Graph Attention (GAT) or relational MLP.
  - `SkinningContextModule`: Injects vertex bone weights $\mathcal{W} \in [0, 1]^{N \times K}$ via bilinear/linear projections into token context.
  - `DiTBlock`: Transformer block with Adaptive LayerNorm (AdaLN-Zero) modulated by flow time $t$, self-attention among state tokens $\mathbf{S}_t \in \mathbb{R}^{N \times (6 + D_z)}$, and cross-attention over $\mathbf{H}_{\text{geom}}$ and $\mathbf{H}_{\text{skel}}$.
  - `FlowVelocityHead`: Output linear projection predicting velocity $d\mathbf{s}/dt = [\mathbf{v}_p, \mathbf{v}_n, \mathbf{v}_z]$.
  - `CombinedFlowLoss`: Joint training loss $\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{strain}} \mathcal{L}_{\text{strain}}$ (with $\lambda_{\text{strain}} = 0.25$). Include an executable demo test block in `if __name__ == "__main__":`.

### [WP 2.2] Parallel Sampler Ablation Suite (`sampler_ablation.py`)
- Implement ODE integration schemes:
  - **Euler Solver** (1st-order): $\mathbf{S}_{k+1} = \mathbf{S}_k + \Delta t \, \mathbf{v}_\theta(\mathbf{S}_k, t_k)$
  - **Midpoint Solver** (2nd-order): $\mathbf{S}_{k+1} = \mathbf{S}_k + \Delta t \, \mathbf{v}_\theta(\mathbf{S}_k + \frac{\Delta t}{2}\mathbf{v}_\theta(\mathbf{S}_k, t_k), t_k + \frac{\Delta t}{2})$
  - **Heun's Predictor-Corrector**: 2-stage Runge-Kutta
- Perform step scaling ablation across $N_{\text{steps}} \in [5, 10, 15, 25, 50]$.
- Measure:
  - Truncation error / Chamfer distance relative to analytical reference
  - Cycle closure error (mm) on closed loops
  - Regular quad valence ratio $V_{4\%}$
  - Inference latency (ms) per asset on CPU/GPU
  - Strain alignment loss $\mathcal{L}_{\text{strain}}$ under joint flexion.

### [WP 2.3] Head-to-Head Comparative Study vs. Baselines (`baseline_comparison.py`)
- Implement a comparative benchmarking harness evaluating 3 distinct paradigms on the Phase 1 benchmark geometries (Planar Hinge and Cylindrical Elbow Joint):
  1. **Classical Curvature Parameterization (QuadriFlow / Instant Meshes surrogate)**:
     - Purely geometric; aligns edges with extrinsic principal curvatures $\kappa_1, \kappa_2$.
     - Evaluated under dynamic flexion: demonstrates shear distortion and diagonal tearing when kinematic axes diverge from principal geometric curvature.
  2. **Sequential Autoregressive Generator (MeshGPT / PolyGen surrogate)**:
     - Sequential token-by-token emission with exposure bias and cumulative random walk drift $\mathcal{O}(\sigma\sqrt{N})$.
     - Evaluated under loop closure: demonstrates seam gaping and severe latency scaling $\mathcal{O}(N)$ vs. $\mathcal{O}(1)$.
  3. **Parallel Deformation-Aware Flow Matching (Ours)**:
     - Bidirectional parallel ODE integration conditioned on kinematic deformation tensor $\mathbf{C}$.
- Output structured comparison tables covering:
  - Inference Latency (ms) vs. Output Complexity
  - Dirichlet Energy ($E_D$) under $90^\circ$ joint flexion
  - Cross-sectional Joint Pinching Ratio ($A_{\text{flex}} / A_0$)
  - Strain Loss $\mathcal{L}_{\text{strain}}$
  - Cycle Closure Gap (mm)

### [WP 2.4] Phase 2 Synthesis & Milestone Report (`phase2_report.md`)
- Synthesize all experimental data, ablation curves, mathematical interpretations, and baseline comparisons.
- Conduct a rigorous audit of the 4 Popperian Falsification Gates.
- Detail the formal Phase 3 Execution Blueprint (Scaling to Rig-Retopo-3K, multi-GPU training recipe, production DCC Blender/Maya plugin).
- Formulate an unambiguous recommendation for advancing to Phase 3.

---

## 3. Required Deliverables (File System Artifacts)

You must create and save the following four artifacts in your root working directory:

1. **`flow_retopo_model.py`**:
   - Complete, self-contained PyTorch implementation of the Multi-Modal DiT Backbone, conditioning encoders, velocity head, and loss modules with unit test in `main`.
2. **`sampler_ablation.py`**:
   - Executable Python script implementing ODE solvers, step sweeps, convergence analysis, and latency profiling.
3. **`baseline_comparison.py`**:
   - Executable benchmark comparing Parallel Flow against Classical (QuadriFlow-style) and Autoregressive (MeshGPT-style) baselines with structured printouts.
4. **`phase2_report.md`**:
   - Comprehensive Phase 2 synthesis report detailing empirical findings, architectural ablation tables, and formal Phase 3 recommendation.

In addition, output each artifact inside markdown code blocks in your final response for double redundancy.
Conclude with a clear status update.
