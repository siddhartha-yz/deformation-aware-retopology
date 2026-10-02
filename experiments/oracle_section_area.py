#!/usr/bin/env python3
"""Oracle test: material cross-section of a bending cylinder.

Axis-aligned quads and 45-degree quads share the same cylinder, edge length,
and Linear Blend Skinning. The endpoint is the area of the material loop that
sat in the joint plane at rest, after the bend. No neural network.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "oracle_section_area"


def lbs(vertices: np.ndarray, weights: np.ndarray, transforms: list[np.ndarray]) -> np.ndarray:
    homo = np.hstack([vertices, np.ones((len(vertices), 1))])
    deformed = np.zeros_like(vertices)
    for k, transform in enumerate(transforms):
        deformed += weights[:, k:k + 1] * (transform @ homo.T).T[:, :3]
    return deformed


def bend_around_x(angle_deg: float) -> list[np.ndarray]:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    moving = np.eye(4)
    moving[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return [np.eye(4), moving]


def sigmoid_weights(z: np.ndarray, delta: float) -> np.ndarray:
    w1 = 1.0 / (1.0 + np.exp(-np.clip(z / delta, -20.0, 20.0)))
    return np.column_stack([1.0 - w1, w1])


def make_cylinder(radius: float, height: float, n_around: int, n_along: int, diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    zs = np.linspace(-height / 2.0, height / 2.0, n_along)
    dz = float(zs[1] - zs[0])
    phase = (dz / radius) if diagonal else 0.0
    verts = []
    for j, z in enumerate(zs):
        for i in range(n_around):
            theta = 2.0 * np.pi * i / n_around + j * phase
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z])
    verts_a = np.asarray(verts, dtype=np.float64)
    quads = []
    for j in range(n_along - 1):
        for i in range(n_around):
            i_next = (i + 1) % n_around
            v0 = j * n_around + i
            v1 = j * n_around + i_next
            v2 = (j + 1) * n_around + i_next
            v3 = (j + 1) * n_around + i
            quads.append([v0, v1, v2, v3])
    return verts_a, np.asarray(quads, dtype=np.int32)


def unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges = set()
    for face in faces:
        idx = [int(v) for v in face]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            i, j = idx[a], idx[b]
            edges.add((i, j) if i < j else (j, i))
    return list(edges)


def mean_edge_length(verts: np.ndarray, edges: list[tuple[int, int]]) -> float:
    lengths = [float(np.linalg.norm(verts[i] - verts[j])) for i, j in edges]
    return float(np.mean(lengths))


def resolutions(radius: float, height: float, target_edge: float, diagonal: bool) -> tuple[int, int]:
    circumference = 2.0 * np.pi * radius
    n_around = max(8, int(round(circumference / target_edge)))
    if diagonal:
        dz = target_edge / np.sqrt(2.0)
    else:
        dz = target_edge
    n_along = max(4, int(round(height / dz)) + 1)
    return n_around, n_along


def material_section(rest: np.ndarray, deformed: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    points = []
    for i, j in edges:
        z0 = float(rest[i, 2])
        z1 = float(rest[j, 2])
        if z0 * z1 > 0.0:
            continue
        denom = z0 - z1
        if abs(denom) < 1e-15:
            continue
        t = z0 / denom
        if t < -1e-8 or t > 1.0 + 1e-8:
            continue
        t = min(1.0, max(0.0, t))
        points.append((1.0 - t) * deformed[i] + t * deformed[j])
    if not points:
        return np.zeros((0, 3))
    stacked = np.asarray(points, dtype=np.float64)
    _, keep = np.unique(np.round(stacked, 6), axis=0, return_index=True)
    return stacked[np.sort(keep)]


def section_area(points: np.ndarray, angle_deg: float) -> float:
    if len(points) < 3:
        return 0.0
    phi = np.radians(angle_deg) / 2.0
    normal = np.array([0.0, -np.sin(phi), np.cos(phi)])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    projected = points - np.outer(points @ normal, normal)
    uv = np.column_stack([projected @ e1, projected @ e2])
    order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
    uv = uv[order]
    x, y = uv[:, 0], uv[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def run_delta(config: dict, delta: float, target_edge: float) -> list[dict]:
    radius = float(config["radius"])
    height = float(config["height"])
    rows = []
    meshes = {}
    for diagonal, label in ((False, "aligned"), (True, "diagonal")):
        n_around, n_along = resolutions(radius, height, target_edge, diagonal)
        verts, faces = make_cylinder(radius, height, n_around, n_along, diagonal)
        edges = unique_edges(faces)
        meshes[label] = {
            "verts": verts,
            "edges": edges,
            "weights": sigmoid_weights(verts[:, 2], delta),
            "n_around": n_around,
            "n_along": n_along,
            "n_faces": int(len(faces)),
            "mean_edge": mean_edge_length(verts, edges),
        }
    circle = np.pi * radius * radius
    for angle in config["angles_deg"]:
        transforms = bend_around_x(float(angle))
        for label, mesh in meshes.items():
            deformed = lbs(mesh["verts"], mesh["weights"], transforms)
            points = material_section(mesh["verts"], deformed, mesh["edges"])
            area = section_area(points, float(angle))
            rows.append({
                "skin_delta": delta,
                "target_edge": target_edge,
                "topology": label,
                "angle_deg": float(angle),
                "section_area": area,
                "section_ratio": area / circle,
                "analytic_retention": float(np.cos(np.radians(float(angle)) / 2.0)),
                "n_section_points": int(len(points)),
                "n_vertices": int(len(mesh["verts"])),
                "n_faces": mesh["n_faces"],
                "n_around": mesh["n_around"],
                "n_along": mesh["n_along"],
                "mean_edge": mesh["mean_edge"],
            })
    return rows


def pair(rows: list[dict], angle: float, delta: float) -> tuple[dict, dict]:
    aligned = next(r for r in rows if r["angle_deg"] == angle and r["topology"] == "aligned" and r["skin_delta"] == delta)
    diagonal = next(r for r in rows if r["angle_deg"] == angle and r["topology"] == "diagonal" and r["skin_delta"] == delta)
    return aligned, diagonal


def rest_is_valid(rows: list[dict], delta: float, minimum: float) -> bool:
    aligned, diagonal = pair(rows, 0.0, delta)
    return aligned["section_ratio"] >= minimum and diagonal["section_ratio"] >= minimum


def rest_area(rows: list[dict], delta: float, topology: str) -> float:
    row = next(r for r in rows if r["angle_deg"] == 0.0 and r["skin_delta"] == delta and r["topology"] == topology)
    return float(row["section_area"])


def decide(rows: list[dict], config: dict) -> tuple[str, float]:
    delta = float(config["skin_delta"])
    decision_angles = [float(a) for a in config["decision_angles_deg"]]
    minimum = float(config["rest_area_ratio_min"])
    margin = float(config["sensitivity_if_relative_gap_below"])
    if not rest_is_valid(rows, delta, minimum):
        return "INVALID", delta
    for angle in decision_angles:
        aligned, diagonal = pair(rows, angle, delta)
        aligned_retention = aligned["section_area"] / rest_area(rows, delta, "aligned")
        diagonal_retention = diagonal["section_area"] / rest_area(rows, delta, "diagonal")
        if aligned_retention < diagonal_retention + margin:
            return "NOT_SUPPORTED", delta
    return "MECHANISM", delta


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Oracle material-section test",
        "",
        f"Verdict: `{payload['verdict']}`",
        f"Decision skin delta: `{payload['decision_delta']}`",
        "",
        payload["config"]["decision_rule"],
        "",
        "The section is the image of the rest-pose joint loop z = 0. Retention divides that area by the same mesh at 0 degrees. For a joint ring blended equally between the two bones, the analytic retention is cos(angle/2), independent of edge direction. The absolute-area label is kept in absolute_area_rule.md.",
        "",
        "| Delta | Topology | Angle (deg) | Section area | Retention | Analytic cos(angle/2) | Faces | Mean edge |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    rest = {}
    for row in payload["rows"]:
        if row["angle_deg"] == 0.0:
            rest[(row["skin_delta"], row["topology"])] = row["section_area"]
    for row in payload["rows"]:
        retention = row["section_area"] / rest[(row["skin_delta"], row["topology"])]
        lines.append(
            f"| {row['skin_delta']:.2f} | {row['topology']} | {row['angle_deg']:.0f} | {row['section_area']:.6f} | "
            f"{retention:.4f} | {row['analytic_retention']:.4f} | {row['n_faces']} | {row['mean_edge']:.4f} |"
        )
    lines.extend([
        "",
        f"Git revision recorded at run start: `{payload['environment']['git_revision']}`",
        f"Python {payload['environment']['python']}, NumPy {payload['environment']['numpy']}, {payload['environment']['platform']}.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "oracle_section_area.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))
    target = float(config["target_edge"])
    primary = float(config["skin_delta"])
    rows = run_delta(config, primary, target)
    # The sensitivity repeat is part of the pre-registered rule. Run it always and
    # let decide() say which delta the verdict uses, so the log contains both.
    rows.extend(run_delta(config, float(config["sensitivity_delta"]), target))
    verdict, decision_delta = decide(rows, config)
    payload = {
        "verdict": verdict,
        "decision_delta": decision_delta,
        "config": config,
        "rows": rows,
        "environment": {
            "git_revision": git_revision(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(RUN_DIR / "report.md", payload)
    print(f"verdict={verdict} decision_delta={decision_delta}")
    print(f"wrote {RUN_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
