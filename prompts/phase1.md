# Autonomous Research Directive: Phase 1 Mathematical Formulation & Representation Sandbox

You are the Lead Research Scientist and Generative Modeling Architect executing inside the Google Antigravity Managed Agent remote sandbox (`antigravity-preview-09-2026`).

---

## 1. Phase 1 Mission & Context

Human review of Phase 0 has been officially completed with a unanimous **`[GO]`** verdict.
You are now authorized to initiate **Phase 1: Mathematical Formulation & Representation Sandbox**.

You have access to the existing workspace artifacts:
- `AGENTS.md` (Multi-Agent Operating Protocol)
- `research_spec.md` (Mathematical Problem Formulation & Scientific Specification)
- `runs/latest/phase0_report.md` (Phase 0 Review & Gate Verification)

### Hard Guidelines for Phase 1:
1. **Lightweight Representation Sandbox Only**: Focus on mathematical precision, small-scale toy validation, and algorithm verification. Do NOT run heavy, multi-GPU training clusters.
2. **Deterministic Synthetic Benchmarks**: Implement controlled synthetic geometries (2D Planar Hinge and 3D Cylindrical Joint) with analytical ground truths.
3. **Rigorous Verification**: Compare the parallel flow trajectory against sequential autoregressive accumulation on loop closures.

---

## 2. Phase 1 Work Packages (WP)

You must execute the following four work packages:

### [WP 1.1] Formal ODE Flow Matching Formulation
- Define the continuous state space $\mathbf{s}_i(t) = [\mathbf{p}_i(t), \mathbf{n}_i(t), \mathbf{z}_i(t)] \in \mathbb{R}^{3 + 3 + D_z}$ combining spatial coordinates, surface normals, and latent topology features.
- Formulate the Optimal Transport (OT) velocity field with linear probability paths:
  $$\mathbf{s}_t = (1 - t)\mathbf{s}_0 + t \mathbf{s}_1, \quad \mathbf{u}_t(\mathbf{s}_1 \mid \mathbf{s}_0) = \mathbf{s}_1 - \mathbf{s}_0$$
- Define the conditioning injection mechanism for the skeletal graph $\mathcal{J} = (\mathbf{J}, \mathcal{E}_{\text{skel}})$ and skinning weight priors $\mathcal{W}$ via cross-attention layers.

### [WP 1.2] Synthetic Deformation Benchmarks (Controlled Toy Physics)
- **Benchmark A: The 2D Planar Articulating Hinge**:
  - Rectangular strip subjected to fold angle $\theta \in [0^\circ, 120^\circ]$.
  - Measure whether generated edge loops align parallel to the hinge axis or produce diagonal cross-seams that tear under flexion.
- **Benchmark B: The 3D Cylindrical Articulating Joint**:
  - Hollow cylinder driven by a two-bone kinematic chain.
  - Test whether concentric quad rings are concentrated at the flexion zone, preserving volume under $90^\circ$ rotation without pinching.

### [WP 1.3] Deformation-Strain Regularization Loss ($\mathcal{L}_{\text{strain}}$)
- Formulate the strain penalty based on the right Cauchy-Green deformation tensor $\mathbf{C} = \mathbf{F}^T \mathbf{F}$:
  $$\mathcal{L}_{\text{strain}} = \sum_{e \in \mathcal{E}} \sin^2 \angle (\mathbf{e}, \mathbf{v}_{\text{principal}})$$
  enforcing edge vectors to align strictly parallel or orthogonal to principal elongation/compression directions.

### [WP 1.4] Deliverables & Phase 1 Exit Gate
- Synthesize mathematical proofs, Python benchmark implementations, and experimental observations.
- Save the required deliverables in your working directory.

---

## 3. Required Deliverables (File System Artifacts)

You must create and save the following four artifacts in your root working directory:

1. **`phase1_math_spec.md`**:
   - Complete, publication-ready mathematical derivation of the flow matching formulation, ODE paths, conditioning mechanics, and strain loss.
2. **`synthetic_benchmarks.py`**:
   - Clean, executable Python script implementing the 2D Hinge and 3D Cylindrical Joint generators, LBS deformation simulator, and loop distortion metrics.
3. **`toy_flow_sampler.py`**:
   - Executable lightweight Python prototype simulating the parallel ODE integration (Euler / Midpoint) on the synthetic benchmarks, demonstrating bidirectional loop closure.
4. **`phase1_report.md`**:
   - Comprehensive Phase 1 synthesis report detailing the benchmark results, mathematical verification, and formal recommendation for Phase 2 prototyping.

In addition, output each artifact inside markdown code blocks in your final response for double redundancy.
Conclude with a clear status update.
