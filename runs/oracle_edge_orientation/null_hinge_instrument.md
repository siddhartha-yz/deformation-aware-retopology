# Null instrument, kept on purpose

The diagonal hinge in this file was a shear. Bending around Y left that shear's Dirichlet unchanged to numerical noise, so the hinge did not test edge direction. The cylinder rows here also move the vertices, unlike the shared-vertex log in `report.md`.

# Oracle edge-orientation test

Verdict: `NOT_SUPPORTED`

MECHANISM if, at every decision angle, joint-band Dirichlet of the axis-aligned quads is strictly lower than the 45-degree quads on both the cylinder and the planar hinge. Otherwise NOT_SUPPORTED. This does not test a neural model.

Both meshes of a shape use the same skinning and the same bend. Diagonal cylinder rings are phase-shifted so the edge between rings is 45 degrees in the unrolled metric. The planar hinge is sheared so the across-hinge edge is 45 degrees. This is not a trained model and not QuadriFlow.

Vertex-radius ratio is a control. On these rings it barely sees connectivity, because Linear Blend Skinning moves vertices, not edges.

| Shape | Topology | Angle (deg) | Joint Dirichlet | Global Dirichlet | Joint area ratio | Vertex radius ratio | Rest mean edge |
|:---|:---|---:|---:|---:|---:|---:|---:|
| cylinder | aligned | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1280 |
| cylinder | diagonal | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1487 |
| cylinder | aligned | 45 | 0.126722 | 0.049145 | 0.9147 | 0.9281 | 0.1280 |
| cylinder | diagonal | 45 | 0.124375 | 0.048178 | 0.9212 | 0.9281 | 0.1487 |
| cylinder | aligned | 90 | 0.174228 | 0.082985 | 0.7685 | 0.7481 | 0.1280 |
| cylinder | diagonal | 90 | 0.171350 | 0.081871 | 0.7751 | 0.7481 | 0.1487 |
| cylinder | aligned | 120 | 0.157560 | 0.093304 | 0.7445 | 0.6111 | 0.1280 |
| cylinder | diagonal | 120 | 0.152044 | 0.091542 | 0.7402 | 0.6111 | 0.1487 |
| hinge | aligned | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1000 |
| hinge | diagonal | 0 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.1207 |
| hinge | aligned | 45 | 0.000000 | 0.012662 | 0.9476 | 0.9765 | 0.1000 |
| hinge | diagonal | 45 | 0.000000 | 0.012662 | 0.9476 | 0.9781 | 0.1207 |
| hinge | aligned | 90 | 0.000000 | 0.043232 | 0.8036 | 0.9099 | 0.1000 |
| hinge | diagonal | 90 | 0.000000 | 0.043232 | 0.8036 | 0.9155 | 0.1207 |
| hinge | aligned | 120 | 0.000000 | 0.064849 | 0.6767 | 0.8566 | 0.1000 |
| hinge | diagonal | 120 | 0.000000 | 0.064849 | 0.6767 | 0.8652 | 0.1207 |

Git revision recorded at run start: `a7c8fdda57ea9f408deaf655cced2b86f735d428`
Python 3.14.4, NumPy 2.5.3, Linux-7.0.0-34-generic-x86_64-with-glibc2.43.
