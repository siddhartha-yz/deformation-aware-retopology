# Research Specification: Conditional & Deformation-Aware Mesh Retopology via Parallel Flow Matching

- **Project Code**: `MESH-FLOW-RETOPOLOGY`
- **Current Phase**: `Phase 0: Feasibility, Systematic Literature Review, Novelty & Falsification`
- **Execution Target**: Google Antigravity Managed Agent (`antigravity-preview-09-2026`) in Remote Sandbox
- **Status**: Active (Phase 0 Planning & Verification)

---

## 1. Problem Definition & Context

### 1.1 The Retopology Bottleneck in 3D Production
In computer animation, visual effects, and real-time interactive graphics (video games, VR/XR), 3D digital assets originate as high-density geometric representations:
- High-poly sculpts (ZBrush, Mudbox) with millions of arbitrary triangles.
- Raw 3D Gaussian Splatting / NeRF reconstructions with noisy, non-manifold point clouds or marching-cubes meshes.
- Generative 3D outputs (text-to-3D, image-to-3D) containing chaotic, dense polygon soups.

Before these assets can be rigged, skinned, and animated in a production engine (Maya, Blender, Unreal Engine), they must undergo **retopology**: constructing a lightweight, clean, quad-dominant surface mesh with regular vertex valences (valence-4 interior vertices) and continuous edge loops.

### 1.2 The Critical Failure of Current Retopology: Articulation Agnosticism
Existing retopology approaches suffer from a fundamental flaw:
1. **Classical Heuristic / Optimization-Based Methods** (e.g., QuadriFlow, Instant Meshes, Mixed-Integer Quadrangulation / MIQ, CoMISo):
   - Rely solely on extrinsic static curvature, principal direction fields, and geodesic metrics.
   - Completely agnostic to whether a region represents an articulated knee joint, an elbow bend, an eye socket, or a rigid armor plate.
   - Result: Edge loops cross diagonally across bending hinges, causing severe self-intersection, volume collapse, and skinning artifacts under skeletal deformation.
2. **Recent Autoregressive (AR) Learned Retopology** (e.g., PolyGen, MeshGPT, MeshGraphormer):
   - Model mesh synthesis as a sequential 1D token stream of vertices and face indices: $P(\mathcal{M}) = \prod_{i=1}^N P(v_i \mid v_{<i}) \prod_{j=1}^M P(f_j \mid f_{<j})$.
   - Suffers from $O(N^2)$ sequential generation latency and quadratic context explosion.
   - Suffers from exposure bias: early vertex misplacement compounds downstream, causing non-manifold edges, topological tearing, and broken edge-loop cycles.
   - Imposes an artificial, arbitrary sequential ordering on an inherently symmetric, non-sequential 2D manifold spatial graph.

---

## 2. Core Hypothesis & Proposed Architecture

### 2.1 The Core Scientific Hypothesis
> **Core Hypothesis**:
> A small, specialized parallel generative model (e.g., Continuous Flow Matching over geometric vector fields or Discrete Flow Matching over graph adjacency/edge attributes) is fundamentally better suited than a generalist autoregressive model for low-level mesh topology decisions.
>
> **Theoretical Rationale**:
> 1. **Global Spatial Constraint Satisfaction**: Quad mesh regularity is a bidirectional, simultaneous constraint satisfaction problem across all cycles of the graph. Bidirectional attention in a diffusion/flow vector field resolves cycle closures simultaneously, eliminating autoregressive order dependency.
> 2. **Efficiency & Scalability**: Parallel flow matching requires a fixed, small number of ODE integration steps (e.g., $T \in [10, 50]$ steps using adaptive Runge-Kutta / Euler solvers) compared to $O(|\mathcal{V}| + |\mathcal{F}|)$ sequential token steps in AR models ($> 5,000$ sequential tokens for even modest meshes).
> 3. **Smooth Kinematic Conditioning**: Continuous conditioning on skeletal kinematics, skinning fields, and strain tensors can be injected directly into cross-attention layers and flow vector fields at each diffusion step.

### 2.2 Mathematical Formulation of the Initial Task
$$\mathcal{M}_{\text{low}} \sim p_\theta\left(\mathcal{M}_{\text{low}} \;\Big|\; \mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}}\right)$$

#### Conditioning Inputs:
1. **Static Surface Geometry ($\mathcal{M}_{\text{high}}$)**:
   - Dense surface mesh or point cloud sampled from the surface $\mathcal{S} \subset \mathbb{R}^3$, encoded via surface feature extractor (e.g., PointNet++, DiffusionNet, or sparse 3D convolutional backbone).
2. **Oracle Deformation / Joint Condition ($\mathcal{C}_{\text{def}}$)**:
   - **Kinematic Skeleton Hierarchy**: Joint positions $\mathbf{J} \in \mathbb{R}^{K \times 3}$ and parent-child kinematic tree $\mathcal{E}_{\text{skel}}$.
   - **Skinning Prior**: Prior skinning weight field $\mathcal{W} \in [0, 1]^{|\mathcal{V}| \times K}$ (Linear Blend Skinning weights or harmonic coordinates).
   - **Representative Deformation Poses**: Oracle set of canonical deformation extremes $\mathcal{P}_{\text{def}} = \{\mathbf{T}_1, \dots, \mathbf{T}_P\}$, indicating principal strain directions $\mathbf{S}(x)$ across the surface manifold during articulation.

#### Generative Target ($\mathcal{M}_{\text{low}}$):
- Vertices $\mathcal{V}_{\text{low}} \in \mathbb{R}^{V \times 3}$ and quad/triangle face topology $\mathcal{F}_{\text{low}}$.
- Topological requirements:
  - Quad dominance ($\ge 90\%$ quads).
  - Valence regularity: interior vertex valences concentrated at 4, with minimal singular vertices (valences 3 and 5) positioned strictly at geometric or kinematic branching nodes.
  - Loop alignment: edge loops parallel and perpendicular to principal deformation axes at articulated joints.

---

## 3. Phase 0: Scope, Deliverables & Constraints

Phase 0 is an autonomous, remote, deep-intellect literature, novelty, and feasibility audit. **It serves as the formal scientific gate before any implementation or training is attempted.**

### 3.1 Hard Constraints
- **NO GPU Training**: Do not train, fine-tune, or fit machine learning models.
- **NO Local Local Execution**: Research harness executes exclusively via Antigravity Managed Agent in the remote Google sandbox.
- **Strict Falsification Standard**: Active search for evidence that kills or necessitates revision of the core hypothesis.
- **Human-in-the-Loop Review**: Phase 0 must terminate upon delivering reports. Phase 1 will NOT begin automatically.

### 3.2 Four Mandatory Deliverables (Sandbox Markdown Reports)
The remote agent must compile, format, and save the following four documents in its workspace:

1. **`literature_review.md`**:
   - Comprehensive, structured taxonomy of the state of the art.
   - Section 1: Classical optimization-based quad meshing (MIQ, CoMISo, QuadriFlow, InstantMeshes, field-aligned parameterization).
   - Section 2: Learned mesh generation (Autoregressive models: PolyGen, MeshGPT, MeshGraphormer, EdgeRun).
   - Section 3: Flow matching and diffusion on 3D geometry and graphs (Continuous Flow Matching, Riemannian Flow Matching, Discrete Diffusion/Flow on graphs, MeshDiffusion).
   - Section 4: Deformation-aware, skinning-aware, and articulation-driven 3D representations (DeepMetaFace, RigNet, NeuroSkinning, Animation-ready avatar retopology).
   - Full citations with titles, authors, venues, years, and ArXiv links.

2. **`novelty_report.md`**:
   - Direct analysis of the proposed hypothesis against published literature.
   - Specific investigation: Does deformation-conditioned learned retopology using flow matching or diffusion already exist?
   - Closest-work comparison matrix detailing: Paper, Venue/Year, Input Representation, Conditioning Signal, Generative Family (AR / Diffusion / Flow / Heuristic), Topology Type (Quad vs. Tri), Key Limitations, and Delta relative to our proposal.
   - Clear definition of the theoretical and empirical novelty boundary.

3. **`dataset_report.md`**:
   - Critical feasibility analysis of training and evaluation datasets.
   - Evaluation of public 3D datasets containing paired (Geometry, Rig/Deformation, Production Quad Topology):
     - Mixamo character repository (clean quad topologies, rigged animations, varied morphology).
     - DeformingThings4D (dense dynamic meshes, non-rigid animal and human sequences).
     - SMPL / SMPL-X / SMPL-A / DFAUST / FAUST (standardized human bodies with fixed topology).
     - Objaverse / Objaverse-XL (massive scale, but highly uncontrolled, noisy topology, unrigged).
     - Commercial / open-source production game asset repositories.
   - Dataset audit: Data availability, licensing, topology quality distribution, rigging consistency, preprocessing pipeline requirements.

4. **`phase0_report.md`**:
   - Executive synthesis of findings.
   - Systematic evaluation against the 4 Falsification Gates (detailed in Section 4).
   - Objective, uncompromised **GO / NO-GO** recommendation.
   - If GO: Concrete mathematical roadmap for Phase 1 (representation sandbox, synthetic benchmark, baseline definition).
   - If NO-GO: Root-cause post-mortem detailing which assumptions were falsified and what alternative scientific avenues exist.

---

## 4. Explicit Scientific Falsification Gates (Kill Criteria)

The remote research agent must rigorously evaluate the project against four non-negotiable falsification criteria:

| Gate | Criterion Name | Condition for NO-GO / Hypothesis Failure |
| :--- | :--- | :--- |
| **Gate 1** | **Prior Art Collision** | A published or pre-printed paper (prior to today) already demonstrates a deformation-conditioned flow matching or parallel diffusion model that generates animation-ready quad mesh topologies with identical scope and formulation, rendering the contribution incremental. |
| **Gate 2** | **Topological Infeasibility of Parallel Generation** | Theoretical or empirical proof that parallel vector/graph flow matching cannot satisfy topological closed-manifold invariants (Euler characteristic, zero self-intersections, face cycle closure) without sequential autoregressive token constraints. |
| **Gate 3** | **Data Scarcity & Rigging Gap** | Inability to identify or assemble a dataset of at least 1,000 diverse 3D assets possessing both clean, artist-grade quad topologies and valid skeletal/deformation pairings under permissible research licenses. |
| **Gate 4** | **Autoregressive Conditioning Superiority** | Evidence demonstrating that conditioning autoregressive models on deformation priors completely resolves their latency and edge-loop placement bottlenecks, rendering flow matching architectures strictly suboptimal or redundant. |

---

## 5. Evaluation Metrics & Success Criteria (For Subsequent Phases)

When the project advances past Phase 0 (upon human approval), the following quantitative metrics will govern benchmarking:

### 5.1 Topological Quality
- **Quad Ratio ($Q_{\%}$)**: $\frac{|\mathcal{F}_{\text{quad}}|}{|\mathcal{F}_{\text{total}}|}$, target $\ge 95\%$.
- **Regular Valence Ratio ($V_{4\%}$)**: Proportion of non-boundary vertices with degree 4, target $\ge 85\%$.
- **Singularity Count & Placement**: Number of valence-3 and valence-5 vertices, and their geodesic distance to geometric/kinematic inflection points.
- **Manifoldness**: $100\%$ zero non-manifold edges, zero duplicate faces, zero non-manifold vertices.

### 5.2 Geometric Fidelity
- **Chamfer Distance ($CD$)**: Bidirectional surface distance between $\mathcal{M}_{\text{high}}$ and $\mathcal{M}_{\text{low}}$.
- **Normal Consistency**: Cosine similarity between surface normals of $\mathcal{M}_{\text{high}}$ and $\mathcal{M}_{\text{low}}$.
- **Hausdorff Distance ($HD$)**: Maximum deviation between surfaces.

### 5.3 Deformation & Animation Readiness
- **Strain-Aligned Edge Flow**: Angular alignment between edge vectors of $\mathcal{M}_{\text{low}}$ and the eigenvectors of the Cauchy-Green deformation strain tensor $\mathbf{C}$ under animation poses $\mathcal{P}_{\text{def}}$.
- **Skinning Distortion Metric**: Mean Dirichlet energy and conformal distortion of faces when animated using Linear Blend Skinning (LBS) across standard motion sequences.
- **Joint Pinching / Volume Loss Ratio**: Volume deviation $\Delta \text{Vol} / \text{Vol}_0$ at maximum joint flexion (e.g. 90-degree elbow/knee bend).

### 5.4 Computational Efficiency
- **Inference Latency**: Generation time (seconds) as a function of output face count compared directly to autoregressive baselines (e.g., PolyGen, MeshGPT).
- **Sampling Step Scalability**: Sensitivity of mesh quality to number of flow integration steps ($N_{\text{steps}} \in [5, 10, 25, 50, 100]$).
