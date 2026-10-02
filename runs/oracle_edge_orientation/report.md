# Oracle edge-orientation test

Verdict: `NOT_SUPPORTED`

MECHANISM if, at every decision angle, joint-band Dirichlet of the axis-aligned quads is strictly lower than the 45-degree quads on both the cylinder and the planar hinge. Otherwise NOT_SUPPORTED. This does not test a neural model.

Both topologies use the same vertices, the same skinning, and the same bend. Aligned quads follow the grid. Diagonal quads are non-overlapping diamonds on that grid, so their edges run at 45 degrees and are longer. A linear shear of the hinge was tried first and left Dirichlet unchanged; that null instrument is kept beside this log. This is not a trained model and not QuadriFlow.

Joint Dirichlet is the mean of `0.5 * max(0, trace(C) - 3)` over faces whose centroid lies in the joint band. Compression that lowers the trace is clamped to zero, so a joint band can report 0 while its area ratio is below 1.
Vertex-radius ratio is a control. On these rings it barely sees connectivity, because Linear Blend Skinning moves vertices, not edges.

| Shape | Topology | Faces | Angle (deg) | Joint Dirichlet | Global Dirichlet | Joint area ratio | Vertex radius ratio | Rest mean edge |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| cylinder | aligned | 320 | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1280 |
| cylinder | diagonal | 80 | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1854 |
| cylinder | aligned | 320 | 45 | 0.126722 | 0.049145 | 0.9147 | 0.9281 | 0.1280 |
| cylinder | diagonal | 80 | 45 | 0.104169 | 0.042396 | 0.9162 | 0.9281 | 0.1854 |
| cylinder | aligned | 320 | 90 | 0.174228 | 0.082985 | 0.7685 | 0.7481 | 0.1280 |
| cylinder | diagonal | 80 | 90 | 0.143576 | 0.072119 | 0.7733 | 0.7481 | 0.1854 |
| cylinder | aligned | 320 | 120 | 0.157560 | 0.093304 | 0.7445 | 0.6111 | 0.1280 |
| cylinder | diagonal | 80 | 120 | 0.120290 | 0.078398 | 0.7449 | 0.6111 | 0.1854 |
| hinge | aligned | 120 | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1000 |
| hinge | diagonal | 30 | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1414 |
| hinge | aligned | 120 | 45 | 0.000000 | 0.012662 | 0.9476 | 0.9765 | 0.1000 |
| hinge | diagonal | 30 | 45 | 0.000000 | 0.012662 | 0.9554 | 0.9765 | 0.1414 |
| hinge | aligned | 120 | 90 | 0.000000 | 0.043232 | 0.8036 | 0.9099 | 0.1000 |
| hinge | diagonal | 30 | 90 | 0.000000 | 0.043232 | 0.8340 | 0.9099 | 0.1414 |
| hinge | aligned | 120 | 120 | 0.000000 | 0.064849 | 0.6767 | 0.8566 | 0.1000 |
| hinge | diagonal | 30 | 120 | 0.000000 | 0.064849 | 0.7290 | 0.8566 | 0.1414 |

Git revision recorded at run start: `a7c8fdda57ea9f408deaf655cced2b86f735d428`
Python 3.14.4, NumPy 2.5.3, Linux-7.0.0-34-generic-x86_64-with-glibc2.43.
