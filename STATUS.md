# Status

**Verdict: NO-GO** on the empirical claim that a trained kinematic flow-matching model beats QuadriFlow and MeshGPT at deformation-aware quad retopology.

Date: 2026-10-03. This file is the current scientific record. Phase reports under `runs/latest/` are archival logs from the first pass. They are kept so the failed claim stays visible. They are not the verdict.

## What was claimed

The first-pass README, `paper/main.tex`, and `runs/latest/sota_character_benchmark.md` stated that a multi-modal flow-matching DiT, regularized by a 4-RoSy strain potential, had been confirmed on production A-pose assets. Cited advantages included about 53% better Chamfer distance, 22–30% lower Dirichlet energy, eliminated joint pinching, zero limb singularities, and a >200× speedup over QuadriFlow, with MeshGPT as the autoregressive baseline.

## What the code actually runs

Three pieces exist, and they are not connected.

1. **Analytic cylinder lattice.** `RetopoInferenceEngine.generate_quad_topology` in `blender_addon/__init__.py` builds a regular quad cylinder from the point-cloud bounding box and integrates the closed-form field `s_1 - s_0` for a few ODE steps. Quad ratio and valence-4 ratio are 100% because every face is a quad on that lattice. This is the path used by `run_pipeline.py`, the Blender add-on, and both SOTA scripts. It does not load `FlowRetopoDiT`.

2. **Autoregressive surrogate.** The rows labeled MeshGPT in the first-pass tables are a Gaussian drift on a cylinder, plus a fixed latency offset. No MeshGPT or PolyGen weights are loaded.

3. **Flow-matching trainer.** `runs/latest/flow_retopo_model.py` and `runs/latest/train_flow_retopo.py` define a small DiT and an OT-CFM plus strain loss. Training data is a synthetic cylinder. A local 100-step checkpoint may exist under `checkpoints/`, which is gitignored. Demo, add-on, and benchmark entry points do not load it. `sampler_ablation.py` adds the untrained network output at a coefficient of 0.05 on top of the ideal straight-line field.

`benchmark_sota.py` can call real C++ QuadriFlow through `pyQuadriFlow` when that package is installed. That baseline is real. The comparison against it still uses the analytic lattice as “ours”.

## Why the published numbers are not a result

- README, `paper/tables/table1_sota.tex`, and the generated character table do not share one Dirichlet or MeshGPT column. Examples: 90° Dirichlet is 0.1873 / 0.1384 in the old README and 0.1376 / 0.1068 in the generated table. MeshGPT Chamfer is 8.420 mm in the old README and 13.462 mm in the table. MeshGPT latency is 24,650 ms in the old README and about 1,260 ms in the table.
- In `runs/latest/sota_character_benchmark.md` the cylinder lattice has **lower** joint volume retention than QuadriFlow (120°: 0.482 vs 0.516). The same file’s closing paragraph, hardcoded in `test_sota_apose_character.py`, claimed 89.5%–93.1% volume retention, a 62%–72% Dirichlet reduction, and 1.8–2.4 ms latency. Those sentences were not computed from the table.
- `run_pipeline.py --mode benchmark` printed that the falsification gates had passed without testing the thresholds. The topology gates cannot fail for this lattice.

## What remains a research question

PolyFlow (arXiv:2606.30673) already generates meshes by flow matching on continuous per-vertex position, normal, and topology embeddings. The open question for this repository is whether **kinematic conditioning** (skeleton, skinning, deformation strain) improves quad edge flow at articulations beyond curvature-only quadrangulation. That question has not been measured here.

A later experiment has to put the trained velocity field on the inference path, compare with a real autoregressive mesh model or drop that baseline, and write the conclusion from the run log.

## Decision

```text
NO-GO
```

The hypothesis is not ruled out. The confirmation is.
