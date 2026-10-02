# Oracle material-section test

Verdict: `NOT_SUPPORTED`
Decision skin delta: `0.15`

Revised after the absolute-area run. That run labeled MECHANISM on a gap of about 0.5 percent that was already present at 0 degrees, and skinning width did not change the area. The claim is now retention: section area divided by that mesh's own 0-degree section area. MECHANISM only if aligned retention exceeds diagonal retention by at least sensitivity_if_relative_gap_below at every decision angle. Otherwise NOT_SUPPORTED. Both 0-degree sections must still be at least rest_area_ratio_min times pi*r^2. The closed form for a 50-50 blend of a joint ring is cos(angle/2).

The section is the image of the rest-pose joint loop z = 0. Retention divides that area by the same mesh at 0 degrees. For a joint ring blended equally between the two bones, the analytic retention is cos(angle/2), independent of edge direction. The absolute-area label is kept in absolute_area_rule.md.

| Delta | Topology | Angle (deg) | Section area | Retention | Analytic cos(angle/2) | Faces | Mean edge |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.15 | aligned | 0 | 0.499220 | 1.0000 | 1.0000 | 775 | 0.0805 |
| 0.15 | diagonal | 0 | 0.496678 | 1.0000 | 1.0000 | 1085 | 0.0809 |
| 0.15 | aligned | 45 | 0.461219 | 0.9239 | 0.9239 | 775 | 0.0805 |
| 0.15 | diagonal | 45 | 0.458870 | 0.9239 | 0.9239 | 1085 | 0.0809 |
| 0.15 | aligned | 90 | 0.353002 | 0.7071 | 0.7071 | 775 | 0.0805 |
| 0.15 | diagonal | 90 | 0.351204 | 0.7071 | 0.7071 | 1085 | 0.0809 |
| 0.15 | aligned | 120 | 0.249610 | 0.5000 | 0.5000 | 775 | 0.0805 |
| 0.15 | diagonal | 120 | 0.248339 | 0.5000 | 0.5000 | 1085 | 0.0809 |
| 0.05 | aligned | 0 | 0.499220 | 1.0000 | 1.0000 | 775 | 0.0805 |
| 0.05 | diagonal | 0 | 0.496678 | 1.0000 | 1.0000 | 1085 | 0.0809 |
| 0.05 | aligned | 45 | 0.461219 | 0.9239 | 0.9239 | 775 | 0.0805 |
| 0.05 | diagonal | 45 | 0.458870 | 0.9239 | 0.9239 | 1085 | 0.0809 |
| 0.05 | aligned | 90 | 0.353002 | 0.7071 | 0.7071 | 775 | 0.0805 |
| 0.05 | diagonal | 90 | 0.351204 | 0.7071 | 0.7071 | 1085 | 0.0809 |
| 0.05 | aligned | 120 | 0.249610 | 0.5000 | 0.5000 | 775 | 0.0805 |
| 0.05 | diagonal | 120 | 0.248339 | 0.5000 | 0.5000 | 1085 | 0.0809 |

Git revision recorded at run start: `6f39b2a35136241ca24152ee3a88349497c53773`
Python 3.14.4, NumPy 2.5.3, Linux-7.0.0-34-generic-x86_64-with-glibc2.43.
