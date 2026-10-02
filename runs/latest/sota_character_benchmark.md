# Character benchmark log

> Archival numerical log from the first pass. Not a SOTA result. See `STATUS.md`.
> QuadriFlow is the C++ solver. The autoregressive row is a drift surrogate, not MeshGPT.
> The cylinder row is a limb-aligned lattice snapped toward the high-poly surface, not `FlowRetopoDiT`.
> The closing paragraph that previously claimed 89.5%–93.1% volume retention was hardcoded and has been removed. The table below is the recorded run.

| Benchmark Asset Scenario | Method | Quad % | Regular Valence | Chamfer (mm) | Dirichlet Energy | Joint Vol Retention | Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A-Pose Human Arm (45° Tilt, 90° Elbow Flexion)** | QuadriFlow (C++) | 100.0% | 95.9% | 6.058 | 0.1376 | 0.706 | 1461.3 ms |
| | AR surrogate (not MeshGPT) | 94.6% | 94.4% | 13.462 | 0.1120 | 0.708 | 1260.1 ms |
| | Analytic cylinder lattice | 100.0% | 100.0% | 2.847 | 0.1068 (-22.4% vs QuadriFlow) | 0.664 (-5.9% vs QuadriFlow) | 6.7 ms |
| **A-Pose Human Arm (45° Tilt, 120° Deep Flexion)** | QuadriFlow (C++) | 100.0% | 95.9% | 6.058 | 0.2134 | 0.516 | 1563.1 ms |
| | AR surrogate (not MeshGPT) | 95.8% | 95.9% | 16.523 | 0.1612 | 0.529 | 1260.5 ms |
| | Analytic cylinder lattice | 100.0% | 100.0% | 2.847 | 0.1591 (-25.4% vs QuadriFlow) | 0.482 (-6.6% vs QuadriFlow) | 6.3 ms |
| **Realistic Human Leg (Standing Stance, 90° Knee Flexion)** | QuadriFlow (C++) | 100.0% | 95.4% | 5.070 | 0.1858 | 0.692 | 390.6 ms |
| | AR surrogate (not MeshGPT) | 94.6% | 94.4% | 26.808 | 0.1399 | 0.698 | 1258.8 ms |
| | Analytic cylinder lattice | 100.0% | 100.0% | 4.511 | 0.1294 (-30.4% vs QuadriFlow) | 0.679 (-1.9% vs QuadriFlow) | 5.9 ms |

## Reading these numbers

Percent changes are `(lattice - QuadriFlow) / QuadriFlow`. A negative volume change means the lattice retained less joint volume.

Quad ratio and valence-4 ratio for the lattice are 100% because every emitted face is a quad on a regular grid.

Latency is grid construction time, not a trained flow-matching model.
