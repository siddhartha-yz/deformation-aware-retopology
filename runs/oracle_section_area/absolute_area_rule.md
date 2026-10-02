# Absolute-area rule, superseded

This file is the first automatic label. It says MECHANISM because the aligned section is slightly larger at 90° and 120°. That gap is already present at 0°, and changing the skinning width leaves the area unchanged. The claim uses retention against each mesh's own rest section. See report.md.

# Oracle material-section test

Verdict: `MECHANISM`
Decision skin delta: `0.05`

MECHANISM if, at every decision angle, the axis-aligned mesh has strictly larger material-section area than the 45-degree mesh. Both 0-degree sections must be at least rest_area_ratio_min times pi*r^2, otherwise the run is INVALID. If the relative gap stays below sensitivity_if_relative_gap_below at every decision angle, repeat once at sensitivity_delta and apply the same rule. This does not test a neural model.

The section is the image of the rest-pose joint loop z = 0, carried by the edges that cross that plane. Area is the polygon of those points projected onto the bend bisector. Edge length is matched near `target_edge`; the 45-degree edges take a shorter step along z so each edge stays the same length.

| Delta | Topology | Angle (deg) | Section area | Ratio to pi r^2 | Points | Faces | Mean edge |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.15 | aligned | 0 | 0.499220 | 0.9932 | 31 | 775 | 0.0805 |
| 0.15 | diagonal | 0 | 0.496678 | 0.9881 | 31 | 1085 | 0.0809 |
| 0.15 | aligned | 45 | 0.461219 | 0.9176 | 31 | 775 | 0.0805 |
| 0.15 | diagonal | 45 | 0.458870 | 0.9129 | 31 | 1085 | 0.0809 |
| 0.15 | aligned | 90 | 0.353002 | 0.7023 | 31 | 775 | 0.0805 |
| 0.15 | diagonal | 90 | 0.351204 | 0.6987 | 31 | 1085 | 0.0809 |
| 0.15 | aligned | 120 | 0.249610 | 0.4966 | 31 | 775 | 0.0805 |
| 0.15 | diagonal | 120 | 0.248339 | 0.4941 | 31 | 1085 | 0.0809 |
| 0.05 | aligned | 0 | 0.499220 | 0.9932 | 31 | 775 | 0.0805 |
| 0.05 | diagonal | 0 | 0.496678 | 0.9881 | 31 | 1085 | 0.0809 |
| 0.05 | aligned | 45 | 0.461219 | 0.9176 | 31 | 775 | 0.0805 |
| 0.05 | diagonal | 45 | 0.458870 | 0.9129 | 31 | 1085 | 0.0809 |
| 0.05 | aligned | 90 | 0.353002 | 0.7023 | 31 | 775 | 0.0805 |
| 0.05 | diagonal | 90 | 0.351204 | 0.6987 | 31 | 1085 | 0.0809 |
| 0.05 | aligned | 120 | 0.249610 | 0.4966 | 31 | 775 | 0.0805 |
| 0.05 | diagonal | 120 | 0.248339 | 0.4941 | 31 | 1085 | 0.0809 |

Git revision recorded at run start: `6f39b2a35136241ca24152ee3a88349497c53773`
Python 3.14.4, NumPy 2.5.3, Linux-7.0.0-34-generic-x86_64-with-glibc2.43.
