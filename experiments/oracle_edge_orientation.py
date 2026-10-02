#!/usr/bin/env python3
"""Oracle test: does edge direction change LBS distortion on a fixed surface?

Same surface, same skinning, same bend. The only intended difference is whether
quad edges sit on the bend axes or at 45 degrees to them. No neural network.
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
RUN_DIR = ROOT / "runs" / "oracle_edge_orientation"


def lbs(vertices: np.ndarray, weights: np.ndarray, transforms: list[np.ndarray]) -> np.ndarray:
    homo = np.hstack([vertices, np.ones((len(vertices), 1))])
    deformed = np.zeros_like(vertices)
    for k, transform in enumerate(transforms):
        moved = (transform @ homo.T).T[:, :3]
        deformed += weights[:, k:k + 1] * moved
    return deformed


def bend_transforms(angle_deg: float, axis: str) -> list[np.ndarray]:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    fixed = np.eye(4)
    moving = np.eye(4)
    if axis == "x":
        moving[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == "y":
        moving[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:
        raise ValueError(axis)
    return [fixed, moving]


def sigmoid_weights(coord: np.ndarray, delta: float) -> np.ndarray:
    w1 = 1.0 / (1.0 + np.exp(-np.clip(coord / delta, -20.0, 20.0)))
    return np.column_stack([1.0 - w1, w1])


def quad_faces(n_u: int, n_v: int, wrap_u: bool = False) -> np.ndarray:
    quads = []
    i_limit = n_u if wrap_u else n_u - 1
    for j in range(n_v - 1):
        for i in range(i_limit):
            i_next = (i + 1) % n_u
            v0 = j * n_u + i
            v1 = j * n_u + i_next
            v2 = (j + 1) * n_u + i_next
            v3 = (j + 1) * n_u + i
            quads.append([v0, v1, v2, v3])
    return np.asarray(quads, dtype=np.int32)


def cylinder_vertices(radius: float, height: float, n_along: int, n_around: int) -> np.ndarray:
    zs = np.linspace(-height / 2.0, height / 2.0, n_along)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    verts = []
    for z in zs:
        for theta in thetas:
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z])
    return np.asarray(verts, dtype=np.float64)


def hinge_vertices(length: float, width: float, n_along: int, n_across: int) -> np.ndarray:
    xs = np.linspace(-length / 2.0, length / 2.0, n_along)
    ys = np.linspace(-width / 2.0, width / 2.0, n_across)
    verts = [[x, y, 0.0] for y in ys for x in xs]
    return np.asarray(verts, dtype=np.float64)


def diamond_faces(n_u: int, n_v: int, wrap_u: bool) -> np.ndarray:
    """Non-overlapping quads whose edges are the diagonals of the grid cells."""
    quads = []
    for j in range(1, n_v - 1, 2):
        i_values = range(0, n_u, 2) if wrap_u else range(1, n_u - 1, 2)
        for i in i_values:
            i_next = (i + 1) % n_u if wrap_u else i + 1
            i_prev = (i - 1) % n_u if wrap_u else i - 1
            if not wrap_u and (i_next >= n_u or i_prev < 0):
                continue
            v_s = (j - 1) * n_u + i
            v_e = j * n_u + i_next
            v_n = (j + 1) * n_u + i
            v_w = j * n_u + i_prev
            quads.append([v_s, v_e, v_n, v_w])
    return np.asarray(quads, dtype=np.int32)


def face_dirichlet(rest: np.ndarray, deformed: np.ndarray, faces: np.ndarray) -> np.ndarray:
    energies = np.zeros(len(faces), dtype=np.float64)
    for idx, face in enumerate(faces):
        rv = rest[face]
        dv = deformed[face]
        e1_r, e2_r = rv[1] - rv[0], rv[3] - rv[0]
        e1_d, e2_d = dv[1] - dv[0], dv[3] - dv[0]
        n_r = np.cross(e1_r, e2_r)
        n_d = np.cross(e1_d, e2_d)
        n_r = n_r / np.linalg.norm(n_r) if np.linalg.norm(n_r) > 1e-12 else np.array([0.0, 0.0, 1.0])
        n_d = n_d / np.linalg.norm(n_d) if np.linalg.norm(n_d) > 1e-12 else np.array([0.0, 0.0, 1.0])
        v_rest = np.column_stack([e1_r, e2_r, n_r])
        v_def = np.column_stack([e1_d, e2_d, n_d])
        try:
            deform = v_def @ np.linalg.inv(v_rest)
        except np.linalg.LinAlgError:
            deform = np.eye(3)
        cauchy = deform.T @ deform
        energies[idx] = 0.5 * max(0.0, float(np.trace(cauchy) - 3.0))
    return energies


def face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.zeros(len(faces), dtype=np.float64)
    for idx, face in enumerate(faces):
        p = verts[face]
        areas[idx] = 0.5 * (
            np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
            + np.linalg.norm(np.cross(p[0] - p[3], p[2] - p[3]))
        )
    return areas


def face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return verts[faces].mean(axis=1)


def mean_edge_length(verts: np.ndarray, faces: np.ndarray) -> float:
    lengths = []
    for face in faces:
        p = verts[face]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            lengths.append(float(np.linalg.norm(p[b] - p[a])))
    return float(np.mean(lengths))


def vertex_radius_ratio(rest: np.ndarray, deformed: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return 1.0
    rest_r = np.mean(np.linalg.norm(rest[mask] - rest[mask].mean(axis=0), axis=1))
    def_r = np.mean(np.linalg.norm(deformed[mask] - deformed[mask].mean(axis=0), axis=1))
    if rest_r < 1e-12:
        return 1.0
    return float((def_r / rest_r) ** 2)


def evaluate_pair(name: str, axis: str, band_coord: int, aligned: dict, diagonal: dict, angles: list[float], band: float) -> list[dict]:
    rows = []
    for angle in angles:
        transforms = bend_transforms(angle, axis)
        for label, mesh in (("aligned", aligned), ("diagonal", diagonal)):
            deformed = lbs(mesh["verts"], mesh["weights"], transforms)
            energies = face_dirichlet(mesh["verts"], deformed, mesh["faces"])
            centroids = face_centroids(mesh["verts"], mesh["faces"])
            band_faces = np.abs(centroids[:, band_coord]) < band
            areas_rest = face_areas(mesh["verts"], mesh["faces"])
            areas_def = face_areas(deformed, mesh["faces"])
            area_ratio = areas_def[band_faces] / np.maximum(areas_rest[band_faces], 1e-12)
            vertex_mask = np.abs(mesh["verts"][:, band_coord]) < band
            rows.append({
                "shape": name,
                "topology": label,
                "angle_deg": angle,
                "joint_dirichlet": float(np.mean(energies[band_faces])) if band_faces.any() else None,
                "global_dirichlet": float(np.mean(energies)),
                "joint_area_ratio": float(np.mean(area_ratio)) if band_faces.any() else None,
                "vertex_radius_ratio": vertex_radius_ratio(mesh["verts"], deformed, vertex_mask),
                "n_vertices": int(len(mesh["verts"])),
                "n_faces": int(len(mesh["faces"])),
                "rest_mean_edge_length": mean_edge_length(mesh["verts"], mesh["faces"]),
                "rest_mean_area": float(np.mean(areas_rest)),
            })
    return rows


def decide(rows: list[dict], decision_angles: list[float]) -> str:
    for shape in ("cylinder", "hinge"):
        for angle in decision_angles:
            aligned = next(r for r in rows if r["shape"] == shape and r["topology"] == "aligned" and r["angle_deg"] == angle)
            diagonal = next(r for r in rows if r["shape"] == shape and r["topology"] == "diagonal" and r["angle_deg"] == angle)
            if aligned["joint_dirichlet"] >= diagonal["joint_dirichlet"]:
                return "NOT_SUPPORTED"
    return "MECHANISM"


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Oracle edge-orientation test",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        payload["config"]["decision_rule"],
        "",
        "Both topologies use the same vertices, the same skinning, and the same bend. Aligned quads follow the grid. Diagonal quads are non-overlapping diamonds on that grid, so their edges run at 45 degrees and are longer. A linear shear of the hinge was tried first and left Dirichlet unchanged; that null instrument is kept beside this log. This is not a trained model and not QuadriFlow.",
        "",
        "Joint Dirichlet is the mean of `0.5 * max(0, trace(C) - 3)` over faces whose centroid lies in the joint band. Compression that lowers the trace is clamped to zero, so a joint band can report 0 while its area ratio is below 1.",
        "Vertex-radius ratio is a control. On these rings it barely sees connectivity, because Linear Blend Skinning moves vertices, not edges.",
        "",
        "| Shape | Topology | Faces | Angle (deg) | Joint Dirichlet | Global Dirichlet | Joint area ratio | Vertex radius ratio | Rest mean edge |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['shape']} | {row['topology']} | {row['n_faces']} | {row['angle_deg']:.0f} | {row['joint_dirichlet']:.6f} | "
            f"{row['global_dirichlet']:.6f} | {row['joint_area_ratio']:.4f} | {row['vertex_radius_ratio']:.4f} | "
            f"{row['rest_mean_edge_length']:.4f} |"
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
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "oracle_edge_orientation.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))

    cyl = config["cylinder"]
    hinge = config["hinge"]
    band = float(config["joint_band"])
    delta = float(config["skin_delta"])
    angles = [float(a) for a in config["angles_deg"]]

    def pack(verts: np.ndarray, faces: np.ndarray, coord: int) -> dict:
        return {"verts": verts, "faces": faces, "weights": sigmoid_weights(verts[:, coord], delta)}

    cyl_verts = cylinder_vertices(cyl["radius"], cyl["height"], cyl["n_along"], cyl["n_around"])
    hinge_verts = hinge_vertices(hinge["length"], hinge["width"], hinge["n_along"], hinge["n_across"])
    rows = []
    rows.extend(evaluate_pair(
        "cylinder", "x", 2,
        pack(cyl_verts, quad_faces(cyl["n_around"], cyl["n_along"], wrap_u=True), 2),
        pack(cyl_verts, diamond_faces(cyl["n_around"], cyl["n_along"], wrap_u=True), 2),
        angles, band,
    ))
    rows.extend(evaluate_pair(
        "hinge", "y", 0,
        pack(hinge_verts, quad_faces(hinge["n_along"], hinge["n_across"], wrap_u=False), 0),
        pack(hinge_verts, diamond_faces(hinge["n_along"], hinge["n_across"], wrap_u=False), 0),
        angles, band,
    ))
    verdict = decide(rows, [float(a) for a in config["decision_angles_deg"]])
    payload = {
        "verdict": verdict,
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
    print(f"verdict={verdict}")
    print(f"wrote {RUN_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
