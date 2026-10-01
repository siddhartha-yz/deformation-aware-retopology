# Multi-Agent Research System: Conditional / Deformation-Aware Mesh Retopology

This repository defines the operational blueprint and research protocol for autonomous scientific investigation into **Conditional, Deformation-Aware Mesh Retopology via Parallel Generative Models (Flow Matching / Diffusion)**.

The research execution is conducted within Google's Antigravity Managed Agent environment (`antigravity-preview-09-2026`) hosted in a secure remote Linux sandbox.

---

## 1. Executive Summary & Research Mission

### Core Scientific Hypothesis
> **Core Hypothesis**: A small, specialized parallel generative model (such as continuous or discrete flow matching / diffusion) is fundamentally better suited than a generalist autoregressive model for low-level mesh topology decisions.

### Initial Concrete Task Formulation
$$\text{Input: } (\mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}}) \longrightarrow \text{Specialized Parallel Topology Generator} \longrightarrow \text{Output: } \mathcal{M}_{\text{low}}$$
- $\mathcal{M}_{\text{high}}$: Unstructured high-resolution surface geometry (dense triangle mesh, point cloud, or SDF).
- $\mathcal{C}_{\text{def}}$: Oracle deformation/joint condition (skeletal kinematic graph $\mathcal{J}$, bone skinning weight priors $\mathcal{W}$, or a representative set of pose deformation gradients $\mathcal{P}_{\text{def}}$).
- $\mathcal{M}_{\text{low}}$: Production-ready low-poly quad-dominant mesh whose edge loops actively align with deformation stress zones and articulation axes, minimizing animation distortion while preserving geometric fidelity.

---

## 2. Multi-Agent Roles & Specializations

Inside the Antigravity Managed Agent sandbox, the system orchestrates five specialized persona sub-functions to thoroughly cross-examine hypotheses and prevent confirmation bias.

```
       +-------------------------------------------------------------+
       |             Lead Research Scientist (Agent PI)              |
       |       - Research governance, phase gates, Go/No-Go          |
       +------------------------------+------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
+--------v----------------------+                       +----------v--------------------+
| Geometry & Retopology Agent   |                       | Deformation & Rigging Agent   |
| - Quad valence, singular pts  |                       | - Skeletal kinematics & LBS   |
| - Edge flow & manifold checks |                       | - Principal strain & loops    |
+--------+----------------------+                       +----------+--------------------+
         |                                                         |
         +----------------------------+----------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
+--------v----------------------+                       +----------v--------------------+
| Generative Modeling Architect |                       | Red Team Falsification Critic |
| - Flow matching vs. AR        |                       | - Prior art collisions        |
| - Parallel sampling dynamics  |                       | - Hard falsification gates    |
+-------------------------------+                       +-------------------------------+
```

### 1. Lead Research Scientist (Agent PI)
- **Mandate**: Overall project direction, synthesis of sub-agent outputs, and formulation of uncompromised scientific judgments.
- **Responsibility**: Guarantees adherence to the Scientific Method. Formulates explicit hypotheses and falsification tests. Decides the final **GO / NO-GO** recommendation at the end of each phase.
- **Boundary**: Will NOT authorize transition from Phase 0 to Phase 1 without explicit human sign-off.

### 2. Geometry & Retopology Specialist Agent
- **Mandate**: Geometric and topological validity.
- **Responsibility**: Analyzes quad-dominant meshing, regular vertex valence (valence-4 interior, valence-3/5 singularities), non-manifold edge prevention, face normal consistency, and boundary constraints.
- **Domain Focus**: Assesses why traditional algorithms (QuadriFlow, Instant Meshes, MIQ, CoMISo) fail to predict downstream articulation behavior and where pure geometric curvature fails as a surrogate for kinematic deformation.

### 3. Deformation & Rigging Dynamics Specialist Agent
- **Mandate**: Animation readiness, kinematic conditioning, and skeletal deformation mechanics.
- **Responsibility**: Formulates the conditioning representation ($\mathcal{C}_{\text{def}}$). Evaluates Linear Blend Skinning (LBS), Dual Quaternion Skinning (DQS), skeletal joint hierarchies, and deformation gradients.
- **Domain Focus**: Validates that edge loops are aligned with orthogonal axes of contraction/extension at articulation joints (knees, elbows, shoulders, facial muscular loops).

### 4. Generative Modeling Architect (Flow Matching vs. Autoregressive)
- **Mandate**: Algorithmic formulation of generative flow matching and diffusion on 3D discrete graphs/meshes.
- **Responsibility**: Compares parallel generative paradigms (Continuous Flow Matching on vector fields/coordinates, Discrete Flow Matching on adjacency matrices or hypergraphs, Latent Diffusion over patch atlases) against sequential autoregressive tokenizers (e.g., PolyGen, MeshGPT, MeshGraphormer).
- **Domain Focus**: Evaluates asymptotic computational complexity, $O(1)$ parallel step generation vs. $O(N^2)$ sequential attention, exposure bias, and edge-cycle consistency across mesh patches.

### 5. Red Team & Falsification Critic Agent
- **Mandate**: Relentless Popperian falsification and critical skepticism.
- **Responsibility**: Actively attempts to invalidate the project's core hypothesis. Searches for prior art collisions, overlooked fatal flaws, mathematical inconsistencies, data collection impossibilities, and computational intractability.
- **Domain Focus**: Enforces the 4 Falsification Gates. Prevents confirmation bias and rationalization of weak results.

---

## 3. Scientific Falsification Philosophy & Popperian Rules

In this project, **a negative result that conclusively rules out an invalid hypothesis is a scientific success**. The agent is explicitly rewarded for discovering why an idea fails early, saving substantial computational resources and engineering time.

### Core Working Rules:
1. **Falsification Over Confirmation**: Look for evidence that *disproves* the hypothesis before looking for evidence that supports it.
2. **Zero Citation Hallucination**: Every cited paper must be real, verifiable via ArXiv ID, DOI, or major conference proceedings (SIGGRAPH, CVPR, ICCV, ECCV, NeurIPS, ICLR, Eurographics, SGP, 3DV, TOG).
3. **Transparent Comparison**: When comparing against autoregressive or traditional baselines, present their true strengths and our method's potential weaknesses fairly.
4. **Quantified Evaluation Criteria**: Define clear mathematical definitions for all proposed metrics (valence distributions, Hausdorff distance, skinning distortion error, inference latency).

---

## 4. Phased Research Protocol

```
+-------------------------------------------------------------------------+
| [Phase 0] Feasibility, Literature, Novelty & Falsification (CURRENT)   |
| - Pure remote analysis; STRICTLY NO GPU TRAINING                        |
| - Outputs: literature_review.md, novelty_report.md, dataset_report.md,  |
|            phase0_report.md (GO / NO-GO)                                |
+------------------------------------+------------------------------------+
                                     |
                          [Human Review Gate]
                                     |
+------------------------------------v------------------------------------+
| [Phase 1] Mathematical Formulation & Representation Sandbox             |
| - Flow matching formulation on mesh topologies; synthetic benchmarks    |
+-------------------------------------------------------------------------+
                                     |
+------------------------------------v------------------------------------+
| [Phase 2] Prototype Architecture & Parallel Sampler Ablation            |
| - Small-scale learned model; comparison against autoregressive baseline  |
+-------------------------------------------------------------------------+
                                     |
+------------------------------------v------------------------------------+
| [Phase 3] Scaling, Kinematic Conditioning & Production Retopology       |
| - Large-scale evaluation on animation datasets and real-world assets    |
+-------------------------------------------------------------------------+
```

### Phase 0 Rules (Active Phase)
- **Constraint 1**: NO GPU training or heavy parameter optimization.
- **Constraint 2**: Exhaustive survey of learned retopology, flow matching on geometric graphs, and kinematic mesh conditioning.
- **Constraint 3**: Mandatory delivery of the 4 specified markdown reports in the remote sandbox.
- **Constraint 4**: Unambiguous GO / NO-GO verdict based on the 4 Falsification Gates.
- **Constraint 5**: Halt execution upon completion of Phase 0. Wait for human peer review.
