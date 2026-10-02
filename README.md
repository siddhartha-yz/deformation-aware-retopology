# Deformation-Aware Retopology

[![CI](https://github.com/siddhartha-yz/deformation-aware-retopology/actions/workflows/ci.yml/badge.svg)](https://github.com/siddhartha-yz/deformation-aware-retopology/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research repository on **conditional, deformation-aware quad retopology**. The question is whether a small parallel flow-matching model, conditioned on a skeleton and deformation strain, places quad edge loops better than curvature-only quadrangulation.

**Current verdict: [NO-GO](STATUS.md)** on the claim that this has already been shown. The first pass published a SOTA scorecard. The scorecard does not come from the neural model, and the write-up contradicts its own table. Read [STATUS.md](STATUS.md) before the phase reports.

---

## Hypothesis

Input a high-resolution surface and a deformation condition (joints, skinning weights, or pose gradients). Output a quad-dominant mesh whose edge loops follow articulation axes, so Linear Blend Skinning distorts the surface less than a curvature-aligned quad mesh.

Prior art already covers static flow-matching mesh generation, notably PolyFlow (arXiv:2606.30673). A pre-registered oracle test of edge direction, without a neural model, is in `experiments/oracle_edge_orientation.py`. Axis-aligned quads did not beat 45° diamonds on joint Dirichlet. The log is `runs/oracle_edge_orientation/report.md`.

## What is in the tree

| Path | What it is |
| :--- | :--- |
| `blender_addon/` | Blender 4.x/5.x panel. The operator builds an **analytic cylinder lattice** from the bounding box. It does not run the DiT. |
| `run_pipeline.py` | CLI for that lattice (`demo`, `benchmark`) and for the trainer (`train`). |
| `runs/latest/flow_retopo_model.py` | Small DiT, OT-CFM loss, and a 4-RoSy strain term. |
| `runs/latest/train_flow_retopo.py` | Trainer on synthetic cylinders. Checkpoints stay local under `checkpoints/` and are not loaded by demo or benchmarks. |
| `benchmark_sota.py`, `test_sota_apose_character.py` | QuadriFlow is the real C++ binding when `pyQuadriFlow` is installed. The autoregressive row is a **drift surrogate**, not MeshGPT. The “ours” row is the cylinder lattice. |
| `paper/` | Draft manuscript. Numerical SOTA claims in it are withdrawn; see the notice at the top of `paper/main.tex`. |
| `runs/latest/phase*.md` | Archival logs from the first pass, including the confirmation that this status file withdraws. |

Figures in `output/` and `assets/` were rendered from that first pass. They are illustrations of the lattice, not evidence for a trained model.

## Setup

```bash
git clone https://github.com/siddhartha-yz/deformation-aware-retopology.git
cd deformation-aware-retopology
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pyQuadriFlow` is listed for Linux only. Without it, the QuadriFlow rows are skipped.

## Commands

```bash
# Analytic cylinder lattice on a synthetic joint, writes output/*.obj
python run_pipeline.py --mode demo --archetype cylindrical_joint

# Topology counts for four synthetic archetypes. This does not train or load a model.
python run_pipeline.py --mode benchmark

# Train the DiT on synthetic cylinders. Separate from the commands above.
python run_pipeline.py --mode train --train-steps 100
```

Repackage the Blender add-on after editing `blender_addon/`:

```bash
python scripts/package_addon.py
```

## Citation

This repository is a research log, not a paper with a confirmed result.

```bibtex
@misc{yang2026deformation_aware_retopology,
  title        = {Deformation-Aware Retopology},
  author       = {Yang, Zhi},
  year         = {2026},
  howpublished = {\url{https://github.com/siddhartha-yz/deformation-aware-retopology}},
  note         = {Research log. Empirical SOTA claim withdrawn; see STATUS.md}
}
```
