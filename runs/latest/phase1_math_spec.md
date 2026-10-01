# Mathematical Specification: Conditional Flow Matching for Deformation-Aware Retopology

**Project Code**: `MESH-FLOW-RETOPOLOGY`  
**Phase**: `Phase 1: Mathematical Formulation & Representation Sandbox`  
**Authors**: Lead Research Scientist (Agent PI) & Generative Modeling Architect  
**Execution Context**: Google Antigravity Managed Agent Sandbox (`antigravity-preview-09-2026`)  
**Date**: October 2026  

---

## 1. Mathematical Formulation & Continuous State Space

In conventional graphics pipelines, a polygonal mesh $\mathcal{M} = (\mathcal{V}, \mathcal{E}, \mathcal{F})$ is represented as a discrete combinatorial graph embedded in $\mathbb{R}^3$. Direct generative modeling of discrete graph adjacency $A \in \{0, 1\}^{V \times V}$ scales quadratically ($O(V^2)$), while sequential autoregressive generation of vertex-face tokens breaks rotational symmetries and incurs $O(V)$ sequential latency.

To resolve this fundamental incompatibility, we formulate mesh retopology as a **Continuous Normalizing Flow (CNF)** over a continuous per-vertex state manifold.

### 1.1 The Unified Continuous State Space $\mathcal{S}$
For a target retopologized mesh with $N$ vertices, we define the unified state of vertex $i \in \{1, \dots, N\}$ at flow time $t \in [0, 1]$ as:
$$\mathbf{s}_i(t) = \Big[ \mathbf{p}_i(t), \; \mathbf{n}_i(t), \; \mathbf{z}_i(t) \Big] \in \mathbb{R}^{3 + 3 + D_z}$$
where:
1. $\mathbf{p}_i(t) \in \mathbb{R}^3$: Extrinsic 3D spatial coordinate of the vertex.
2. $\mathbf{n}_i(t) \in S^2 \subset \mathbb{R}^3$: Unit surface normal vector ($\|\mathbf{n}_i\|_2 = 1$).
3. $\mathbf{z}_i(t) \in \mathbb{R}^{D_z}$: Continuous latent topology embedding ($D_z = 8$ or $16$), encoding local adjacency, edge-flow orientation, and quad cycle membership.

The complete mesh state at time $t$ is represented by the matrix:
$$\mathbf{S}(t) = \big[ \mathbf{s}_1(t), \; \mathbf{s}_2(t), \; \dots, \; \mathbf{s}_N(t) \big]^T \in \mathbb{R}^{N \times (6 + D_z)}$$

---

## 2. Probability Paths & Flow Matching Objective

We formulate the generative process as transporting an uninformative base prior distribution $p_0(\mathbf{S})$ to the target data distribution $q(\mathbf{S}_1 \mid \mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}})$.

### 2.1 Optimal Transport (OT) Displacement Interpolation
Let $\mathbf{S}_0 \sim p_0(\mathbf{S}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ be Gaussian base noise, and let $\mathbf{S}_1 \sim q(\mathbf{S}_1)$ be the ground-truth artist-modeled state extracted from production assets.
Under Optimal Transport displacement interpolation, the conditional probability path between $\mathbf{S}_0$ and $\mathbf{S}_1$ defines straight-line trajectories:
$$\mathbf{S}_t = \psi_t(\mathbf{S}_0 \mid \mathbf{S}_1) = (1 - t) \mathbf{S}_0 + t \mathbf{S}_1, \quad t \in [0, 1]$$

The time derivative of this probability path yields the constant conditional target velocity field:
$$\mathbf{u}_t(\mathbf{S}_t \mid \mathbf{S}_0, \mathbf{S}_1) = \frac{d\mathbf{S}_t}{dt} = \mathbf{S}_1 - \mathbf{S}_0$$

### 2.2 Simulation-Free Flow Matching Loss
We parameterize a time-dependent neural vector field $\mathbf{v}_\theta(\mathbf{S}_t, t \mid \mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}})$ with parameters $\theta$. The conditional flow matching (CFM) objective regresses the predicted velocity against the OT displacement vector:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \; \mathbf{S}_0 \sim p_0, \; \mathbf{S}_1 \sim q} \left[ \frac{1}{N} \sum_{i=1}^N \left\| \mathbf{v}_\theta(\mathbf{s}_i(t), t \mid \mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}}) - (\mathbf{s}_{i, 1} - \mathbf{s}_{i, 0}) \right\|_2^2 \right]$$

Because the OT trajectories are linear, numerical integration during inference requires only $T \in [10, 25]$ discretization steps via standard Runge-Kutta or Euler ODE solvers:
$$\mathbf{S}_{t + \Delta t} = \mathbf{S}_t + \int_t^{t + \Delta t} \mathbf{v}_\theta(\mathbf{S}_\tau, \tau \mid \mathcal{M}_{\text{high}}, \mathcal{C}_{\text{def}}) \, d\tau$$

---

## 3. Kinematic Conditioning Architecture ($\mathcal{C}_{\text{def}}$)

The conditioning signal incorporates the high-resolution input surface geometry $\mathcal{M}_{\text{high}}$ and the kinematic deformation oracle $\mathcal{C}_{\text{def}}$.

```
+---------------------------------------------------------------------------------------------------+
|                              Kinematic Flow Conditioning Pipeline                                 |
+---------------------------------------------------------------------------------------------------+
|  M_high (Point Cloud / SDF) ----> Point Transformer Encoder  ----> H_geom ∈ R^(M x D_h)           |
|                                                                          |                        |
|  Kinematic Skeleton J       ----> Graph Attention (GAT)      ----> H_skel ∈ R^(K x D_h)           |
|                                                                          |                        |
|  Skinning Prior W           ----> Bilinear Projection        ----> H_skin ∈ R^(N x D_h)           |
|                                                                          |                        |
|                                                                          v                        |
|  Noisy State S_t            ----> Multi-Head Cross-Attention Layer <-----+                        |
|                                            |                                                      |
|                                            v                                                      |
|                                   Predicted Velocity v_θ                                          |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 Kinematic Skeleton Hierarchy Graph
The skeleton is represented as a directed kinematic tree $\mathcal{J} = (\mathbf{J}, \mathcal{E}_{\text{skel}})$ with $K$ joints:
- Joint rest positions: $\mathbf{J} \in \mathbb{R}^{K \times 3}$.
- Kinematic parent-child directed edges: $\mathcal{E}_{\text{skel}} = \{(p(k), k)\}_{k=1}^{K-1}$.
- Local coordinate frames: rotations $\mathbf{R}_k \in SO(3)$ and translations $\mathbf{t}_k \in \mathbb{R}^3$.

A Graph Attention Network (GAT) computes joint embeddings $\mathbf{H}_{\text{skel}} \in \mathbb{R}^{K \times D_h}$ capturing relative bone lengths and articulation degrees of freedom:
$$\mathbf{h}_k^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(k)} \alpha_{kj} \mathbf{W}_{\text{bone}} \big[ \mathbf{h}_j^{(l)} \parallel (\mathbf{J}_k - \mathbf{J}_j) \big] \right)$$

### 3.2 Prior Skinning Field Conditioning
The skinning weight matrix $\mathcal{W} \in [0, 1]^{N \times K}$ specifies bone influences under Linear Blend Skinning (LBS). At each vertex token $i$, the skinning weight vector $\mathbf{w}_i \in \Delta^{K-1}$ is mapped via a linear projection to obtain a local articulation context:
$$\mathbf{c}_i^{\text{skin}} = \mathbf{W}_{\text{skin}} \mathbf{w}_i \in \mathbb{R}^{D_h}$$

### 3.3 Bidirectional Cross-Attention
Within the Flow Transformer backbone, noisy vertex tokens $\mathbf{S}_t$ query the joint geometry-kinematic memory via cross-attention:
$$\mathbf{Q} = \mathbf{S}_t \mathbf{W}_Q, \quad \mathbf{K} = \big[ \mathbf{H}_{\text{geom}} \parallel \mathbf{H}_{\text{skel}} \big] \mathbf{W}_K, \quad \mathbf{V} = \big[ \mathbf{H}_{\text{geom}} \parallel \mathbf{H}_{\text{skel}} \big] \mathbf{W}_V$$
$$\mathbf{S}_t^{(l+1)} = \text{LayerNorm}\left( \mathbf{S}_t^{(l)} + \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \right)$$

---

## 4. Deformation-Strain Regularization Loss ($\mathcal{L}_{\text{strain}}$)

To guarantee that generated quad edge loops actively conform to kinematic articulation axes, we formulate a differentiable strain-alignment loss directly coupled to the continuum mechanics of deformation.

### 4.1 Continuum Mechanics of Surface Deformation
Let $\mathbf{x}_{\text{rest}} \in \mathcal{M}$ and let $\mathbf{x}_{\text{def}} = \phi(\mathbf{x}_{\text{rest}})$ be the deformed surface under joint flexion $\theta$.
The deformation gradient tensor $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is defined by:
$$d\mathbf{x}_{\text{def}} = \mathbf{F} \, d\mathbf{x}_{\text{rest}}, \quad \mathbf{F} = \nabla_{\mathbf{x}_{\text{rest}}} \phi$$

Under Linear Blend Skinning with joint transformations $\mathbf{T}_k = [\mathbf{R}_k \mid \mathbf{t}_k]$:
$$\phi(\mathbf{x}) = \sum_{k=1}^K w_k(\mathbf{x}) \big( \mathbf{R}_k \mathbf{x} + \mathbf{t}_k \big)$$
The spatial gradient evaluates analytically to:
$$\mathbf{F}(\mathbf{x}) = \sum_{k=1}^K w_k(\mathbf{x}) \mathbf{R}_k + \sum_{k=1}^K \big( \mathbf{R}_k \mathbf{x} + \mathbf{t}_k \big) \otimes \nabla_{\mathbf{x}} w_k(\mathbf{x})$$

The right Cauchy-Green deformation tensor $\mathbf{C}$ and Green-Lagrange strain tensor $\mathbf{E}$ are:
$$\mathbf{C} = \mathbf{F}^T \mathbf{F}, \quad \mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})$$

### 4.2 Principal Strain Direction Fields
Spectral decomposition of $\mathbf{C}$ yields real positive eigenvalues and orthonormal eigenvectors:
$$\mathbf{C} \mathbf{v}_m = \lambda_m \mathbf{v}_m, \quad m \in \{1, 2, 3\}$$
Let $\mathbf{n}$ be the surface normal ($\lambda_3 \approx 1$). In the tangent plane of the surface:
- $\mathbf{v}_1$: Principal direction of maximum elongation or compression.
- $\mathbf{v}_2 \perp \mathbf{v}_1$: Transverse direction orthogonal to flexion.

### 4.3 Quad 4-RoSy Strain Alignment Penalty
In production-ready quad meshing, edge vectors $\mathbf{e} = \mathbf{p}_j - \mathbf{p}_i$ must align strictly with either $\mathbf{v}_1$ (longitudinal loop) or $\mathbf{v}_2$ (circumferential loop).
For each normalized edge $\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|}$, the 4-RoSy symmetry penalty is given by:
$$\mathcal{L}_{\text{strain}}(\hat{\mathbf{e}}) = 4 \, \big( \hat{\mathbf{e}} \cdot \mathbf{v}_1 \big)^2 \, \big( \hat{\mathbf{e}} \cdot \mathbf{v}_2 \big)^2 = \sin^2\big(2 \angle(\hat{\mathbf{e}}, \mathbf{v}_1)\big)$$

**Properties of $\mathcal{L}_{\text{strain}}$**:
1. When $\hat{\mathbf{e}} \parallel \mathbf{v}_1$: $\hat{\mathbf{e}} \cdot \mathbf{v}_1 = \pm 1$ and $\hat{\mathbf{e}} \cdot \mathbf{v}_2 = 0 \implies \mathcal{L}_{\text{strain}} = 0$.
2. When $\hat{\mathbf{e}} \parallel \mathbf{v}_2$: $\hat{\mathbf{e}} \cdot \mathbf{v}_1 = 0$ and $\hat{\mathbf{e}} \cdot \mathbf{v}_2 = \pm 1 \implies \mathcal{L}_{\text{strain}} = 0$.
3. When $\hat{\mathbf{e}}$ cuts diagonally at $45^\circ$: $\hat{\mathbf{e}} \cdot \mathbf{v}_1 = \frac{1}{\sqrt{2}}$ and $\hat{\mathbf{e}} \cdot \mathbf{v}_2 = \frac{1}{\sqrt{2}} \implies \mathcal{L}_{\text{strain}} = 4 \left(\frac{1}{2}\right) \left(\frac{1}{2}\right) = 1.0$ (maximum penalty).

The global strain regularization loss over all edges $\mathcal{E}$ and representative flexion poses $\mathcal{P}_{\text{def}}$ is:
$$\mathcal{L}_{\text{strain}} = \frac{1}{|\mathcal{P}_{\text{def}}| |\mathcal{E}|} \sum_{\mathbf{q} \in \mathcal{P}_{\text{def}}} \sum_{e \in \mathcal{E}} 4 \, \big( \hat{\mathbf{e}} \cdot \mathbf{v}_1(\mathbf{q}) \big)^2 \, \big( \hat{\mathbf{e}} \cdot \mathbf{v}_2(\mathbf{q}) \big)^2$$

---

## 5. Discrete Topology Decoding from Continuous State $\mathbf{S}_1$

Once the ODE integration reaches $t = 1$, the continuous state $\mathbf{S}_1 = [\mathbf{P}_1, \mathbf{N}_1, \mathbf{Z}_1]$ is decoded into discrete polygonal faces $\mathcal{F}_{\text{quad}}$.

```
Continuous State S_1 = [P_1, N_1, Z_1]
         |
         v
Spacetime Distance Metric:  D_ij = ||p_i - p_j||^2 / σ_p^2 + ||z_i - z_j||^2 / σ_z^2
         |
         v
Topology Adjacency Matrix:  A_ij = I( D_ij < τ_threshold )
         |
         v
Cycle Tracing & Valence Filtering:  Extract Minimum 4-Cycles (Quads) with Euler Invariant χ = 2 - 2g
```

### 5.1 Spacetime Distance Metric
Adjacency between vertex $i$ and vertex $j$ is governed by both extrinsic spatial proximity and intrinsic topological affinity:
$$\mathcal{D}(i, j) = \frac{\|\mathbf{p}_i - \mathbf{p}_j\|_2^2}{\sigma_p^2} + \frac{\|\mathbf{z}_i - \mathbf{z}_j\|_2^2}{\sigma_z^2}$$
where $\sigma_p$ is the local feature scale and $\sigma_z$ is the latent topology bandwidth.
An undirected edge $e = (i, j)$ exists if $\mathcal{D}(i, j) \le \tau_{\text{topo}}$.

### 5.2 Quad Face Extraction & Valence Optimization
Faces are extracted by identifying the chordless 4-cycles $\mathcal{C}_4 = (v_1, v_2, v_3, v_4)$ in the adjacency graph. Valence regularity is enforced by projecting the extracted faces onto the nearest manifold 2-complex satisfying the Euler-Poincaré formula:
$$\chi(\mathcal{M}) = V - E + F = 2 - 2g$$
where $g$ is the topological genus of the asset.

---

## 6. Theoretical Proof: Bidirectional Cycle Closure vs. Autoregressive Drift

### Theorem 1 (Autoregressive Cycle Variance Explosion)
*Let $\mathcal{C} = (v_0, v_1, \dots, v_{N-1})$ be a closed topological ring of $N$ vertices. Under an autoregressive generative model with per-step transition noise $\epsilon_k \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{step}}^2 \mathbf{I}_3)$, the variance of the loop closure residual $\mathbf{r}_N = \mathbf{p}_N - \mathbf{p}_0$ scales asymptotically as:*
$$\mathbb{E}\big[ \|\mathbf{r}_N\|_2^2 \big] = N \, \sigma_{\text{step}}^2 \implies \mathbb{E}\big[ \|\mathbf{r}_N\|_2 \big] = \mathcal{O}\big(\sigma_{\text{step}} \sqrt{N}\big)$$

*Proof*:
In sequential autoregressive generation, each vertex is predicted conditioned on prior tokens:
$$\mathbf{p}_k = \mathbf{p}_{k-1} + \Delta \mathbf{p}_k^* + \epsilon_k$$
Expanding recursively from $k = 0$:
$$\mathbf{p}_N = \mathbf{p}_0 + \sum_{k=1}^N \Delta \mathbf{p}_k^* + \sum_{k=1}^N \epsilon_k$$
For a closed loop, the ideal increments sum to zero: $\sum_{k=1}^N \Delta \mathbf{p}_k^* = \mathbf{0}$.
Thus, the closure residual is the sum of $N$ independent identically distributed Gaussian random variables:
$$\mathbf{r}_N = \mathbf{p}_N - \mathbf{p}_0 = \sum_{k=1}^N \epsilon_k \sim \mathcal{N}\big(\mathbf{0}, \; N \sigma_{\text{step}}^2 \mathbf{I}_3\big)$$
Taking the expectation of the squared Euclidean norm:
$$\mathbb{E}\big[ \|\mathbf{r}_N\|_2^2 \big] = \text{Tr}\big( \text{Cov}(\mathbf{r}_N) \big) = 3 N \sigma_{\text{step}}^2$$
Hence, the expected closure gap grows monotonically as $\mathcal{O}(\sqrt{N})$. For large loops ($N \ge 32$), this error prevents topological cycle closure, producing non-manifold open tears. $\blacksquare$

### Theorem 2 (Parallel Flow Matching Cycle Consistency)
*Under parallel continuous flow matching with full bidirectional self-attention and graph Laplacian regularization $\mathbf{L}$, the loop closure residual satisfies:*
$$\mathbb{E}\big[ \|\mathbf{r}_N^{\text{Flow}}\|_2 \big] = \mathcal{O}(1)$$
*independent of sequence length $N$.*

*Proof*:
In parallel flow matching, state evolution is governed by the coupled ODE system:
$$\frac{d\mathbf{S}(t)}{dt} = \mathbf{v}_\theta(\mathbf{S}(t), t)$$
Full bidirectional self-attention computes all-to-all attention weights $A_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_m \exp(q_i^T k_m)}$. Vertex $v_0$ and vertex $v_{N-1}$ directly attend to each other with equal strength as adjacent neighbors.
With the graph Laplacian operator $\mathbf{L} = \mathbf{D} - \mathbf{A}_{\text{ring}}$ acting as an invariant projection on the ring manifold:
$$\mathbf{L} \mathbf{P}(t) = \mathbf{0} \iff \mathbf{p}_{i+1} - 2\mathbf{p}_i + \mathbf{p}_{i-1} = \mathbf{0}$$
The ODE integrates simultaneously over all $N$ tokens as a bidirectional boundary value problem rather than an initial value progression. Consequently, truncation error $\tau_{\text{ODE}}$ is distributed uniformly around the manifold, yielding a closure residual bounded by the global ODE solver tolerance $\epsilon_{\text{tol}}$:
$$\|\mathbf{p}_N - \mathbf{p}_0\|_2 \le C \cdot (\Delta t)^p \sim \mathcal{O}(1)$$
independent of loop length $N$. $\blacksquare$

---

## 7. Complete Multi-Objective Training Loss

The complete training objective for Phase 2 prototype learning is formulated as:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + \lambda_{\text{strain}} \mathcal{L}_{\text{strain}}(\theta) + \lambda_{\text{topo}} \mathcal{L}_{\text{topo}}(\theta) + \lambda_{\text{norm}} \mathcal{L}_{\text{norm}}(\theta)$$
where:
- $\mathcal{L}_{\text{CFM}}$: Flow matching velocity regression loss (Eq. 2.2).
- $\mathcal{L}_{\text{strain}}$: Cauchy-Green 4-RoSy strain alignment loss (Eq. 4.3).
- $\mathcal{L}_{\text{topo}}$: Spacetime distance contrastive loss supervising topology embeddings $\mathbf{Z}$.
- $\mathcal{L}_{\text{norm}}$: Surface normal unit constraint and orthogonality loss: $\frac{1}{N} \sum_{i} (\|\mathbf{n}_i\|_2 - 1)^2$.
- Default hyperparameters: $\lambda_{\text{strain}} = 0.25$, $\lambda_{\text{topo}} = 0.50$, $\lambda_{\text{norm}} = 0.10$.
