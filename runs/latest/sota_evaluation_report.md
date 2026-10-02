# Benchmark log: analytic cylinder lattice

> Archival numerical log from the first pass. Not a SOTA result. See `STATUS.md`.
> QuadriFlow is the C++ solver. The autoregressive row is a drift surrogate, not MeshGPT.
> The cylinder row is `RetopoInferenceEngine`, not `FlowRetopoDiT`.
> The previous closing paragraph claimed a new state of the art. The volume column in this table is lower for the lattice than for QuadriFlow (0.649 vs 0.655), and Dirichlet is higher (0.0312 vs 0.0285).

## Scores

| Method | Quad % (Q%) ↑ | Valence-4 % (V4%) ↑ | Chamfer (mm) ↓ | HD95 (mm) ↓ | Normal Cons. ↑ | Dirichlet $E_D$ ↓ | Vol Retention ↑ | 4-RoSy Strain Loss ↓ | Latency (ms) ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **QuadriFlow (Real C++)** | 100.0% | 96.3% | 5.563 | 83.69 | 0.9761 | 0.0285 | 0.655 | 0.1115 | 170.93 ms |
| **AR surrogate (not MeshGPT)** | 94.9% | 94.5% | 5.413 | 92.42 | 0.9827 | 0.0190 | 0.653 | 0.0194 | 895.18 ms |
| **Analytic cylinder lattice** | 100.0% | 100.0% | 3.414 | 81.27 | 0.9983 | 0.0312 | 0.649 | 0.0001 | 1.39 ms |

## Reading these numbers

Quad ratio and valence-4 ratio for the cylinder lattice are 100% because the generator emits only quads on a regular grid.

On this recorded run, Dirichlet changed by +9.5% and volume retention by -0.9% relative to QuadriFlow. A negative volume change means the lattice retained less joint volume than QuadriFlow.

Latency compares grid construction with QuadriFlow and with a sleep-padded surrogate. It is not a neural inference time.
