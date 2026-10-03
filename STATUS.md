# Status

The front page is the ring tool: `python demo/tube_retopo.py --gallery`. This file is the research record.

**Verdict: NO-GO** on the empirical claim that a trained kinematic flow-matching model beats QuadriFlow and MeshGPT at deformation-aware quad retopology.

Date: 2026-10-03. This file is the current scientific record. Phase reports under `runs/latest/` are archival logs from the first pass. They are kept so the failed claim stays visible. They are not the verdict.

## What was claimed

The first-pass README, `paper/main.tex`, and `runs/latest/sota_character_benchmark.md` stated that a multi-modal flow-matching DiT, regularized by a 4-RoSy strain potential, had been confirmed on production A-pose assets. Cited advantages included about 53% better Chamfer distance, 22–30% lower Dirichlet energy, eliminated joint pinching, zero limb singularities, and a >200× speedup over QuadriFlow, with MeshGPT as the autoregressive baseline.

## What the code actually runs

Three pieces exist, and they are not connected.

1. **Analytic cylinder lattice.** `RetopoInferenceEngine.generate_quad_topology` in `runs/latest/blender_retopo_addon.py` builds a regular quad cylinder from the point-cloud bounding box and integrates the closed-form field `s_1 - s_0` for a few ODE steps. Quad ratio and valence-4 ratio are 100% because every face is a quad on that lattice. `run_pipeline.py` and the old SOTA scripts still call that lattice. They do not load `FlowRetopoDiT`. The Blender button and `demo/tube_retopo.py` do not use it. They slice the mesh into quad rings with `blender_addon/tube.py`.

2. **Autoregressive surrogate.** The rows labeled MeshGPT in the first-pass tables are a Gaussian drift on a cylinder, plus a fixed latency offset. No MeshGPT or PolyGen weights are loaded.

3. **Flow-matching trainer.** `runs/latest/flow_retopo_model.py` and `runs/latest/train_flow_retopo.py` define a small DiT and an OT-CFM plus strain loss. Training data is a synthetic cylinder. A local 100-step checkpoint may exist under `checkpoints/`, which is gitignored. Demo, add-on, and benchmark entry points do not load it. `sampler_ablation.py` adds the untrained network output at a coefficient of 0.05 on top of the ideal straight-line field.

`benchmark_sota.py` can call real C++ QuadriFlow through `pyQuadriFlow` when that package is installed. That baseline is real. The comparison against it still uses the analytic lattice as “ours”.

## Why the published numbers are not a result

- README, `paper/tables/table1_sota.tex`, and the generated character table do not share one Dirichlet or MeshGPT column. Examples: 90° Dirichlet is 0.1873 / 0.1384 in the old README and 0.1376 / 0.1068 in the generated table. MeshGPT Chamfer is 8.420 mm in the old README and 13.462 mm in the table. MeshGPT latency is 24,650 ms in the old README and about 1,260 ms in the table.
- In `runs/latest/sota_character_benchmark.md` the cylinder lattice has **lower** joint volume retention than QuadriFlow (120°: 0.482 vs 0.516). The same file’s closing paragraph, hardcoded in `test_sota_apose_character.py`, claimed 89.5%–93.1% volume retention, a 62%–72% Dirichlet reduction, and 1.8–2.4 ms latency. Those sentences were not computed from the table.
- `run_pipeline.py --mode benchmark` printed that the falsification gates had passed without testing the thresholds. The topology gates cannot fail for this lattice.

## What remains a research question

PolyFlow (arXiv:2606.30673) already generates meshes by flow matching on continuous per-vertex position, normal, and topology embeddings. The open question for this repository is whether **kinematic conditioning** (skeleton, skinning, deformation strain) improves quad edge flow at articulations beyond curvature-only quadrangulation. That question has not been measured here.

The section-area test below measures that geometric premise directly. It does not support extra pinching from 45° edges under this skinning model.

## Oracle edge orientation

`experiments/oracle_edge_orientation.py` compares grid quads with 45° diamonds on the **same vertices** and the same Linear Blend Skinning. The pre-registered rule was that axis-aligned quads must have strictly lower joint-band Dirichlet energy at 90° and 120° on both a cylinder and a planar hinge.

**Verdict: `NOT_SUPPORTED`.** On the cylinder at 90°, joint Dirichlet is 0.174 for aligned quads and 0.144 for diamonds. At 120° it is 0.158 versus 0.120. On the hinge the joint Dirichlet is 0 for both, because the energy clamps `trace(C) - 3` at zero while the faces are compressing. Joint area ratio does not favor the aligned quads either (hinge at 90°: 0.804 aligned, 0.834 diagonal).

Vertex-radius ratio is identical for the two topologies. It does not measure edge direction. The diamonds also have fewer, longer edges, so this is not a matched-resolution proof that 45° edges are better. It is enough to reject the claim that axis-aligned edges win this test.

A first hinge construction sheared the surface in a direction the bend does not stretch. Its Dirichlet did not move. That log is `runs/oracle_edge_orientation/null_hinge_instrument.md`.

## Oracle material section

`experiments/oracle_section_area.py` bends one cylinder with Linear Blend Skinning. Axis-aligned quads and 45° quads use the same radius, height, and edge length (about 0.08). The measured loop is the material that lay on `z = 0` at rest.

An absolute-area comparison labeled `MECHANISM` because the aligned area was about 0.5% larger at 90° and 120°. That gap is already present at 0° (0.4992 versus 0.4967), and it is the faceting of the rest surface. Changing the skinning width from 0.15 to 0.05 translates the loop and leaves its area unchanged. That log is `runs/oracle_section_area/absolute_area_rule.md`.

Retention, area divided by the same mesh at 0°, matches `cos(angle/2)` for both topologies:

| Angle | Aligned retention | Diagonal retention | `cos(angle/2)` |
| ---: | ---: | ---: | ---: |
| 45° | 0.9239 | 0.9239 | 0.9239 |
| 90° | 0.7071 | 0.7071 | 0.7071 |
| 120° | 0.5000 | 0.5000 | 0.5000 |

**Verdict: `NOT_SUPPORTED`.** Under this skinning model the joint loop is a 50-50 bone blend. Its area shrinks by `cos(angle/2)` whether the edges are rings or 45° diagonals. Edge direction does not add candy-wrapper pinching to this loop.

## Decision

```text
NO-GO
```

The published neural comparison remains withdrawn. The geometric premise that 45° edges pinch a skinned cylinder more than ring edges is not supported for the material joint section. A learned retopology model is not justified by that premise until a deformation model is named in which edge direction changes the section.
