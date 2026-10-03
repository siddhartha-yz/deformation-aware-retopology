#!/usr/bin/env python3
"""Draw a bendable arm and show how the elbow cross-section shrinks.

Ring loops and 45-degree loops are both drawn on the same arm. The pictures
land in docs/figures/. This is a viewer, not a trained retopology model.
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.oracle_section_area import (  # noqa: E402
    bend_around_x,
    lbs,
    material_section,
    section_area,
    sigmoid_weights,
    unique_edges,
)

FIG_DIR = ROOT / "docs" / "figures"
SKIN = np.array([0.93, 0.76, 0.62])
RING_EDGE = "#6B4A36"
DIAG_EDGE = "#1D4E89"


def setup_font() -> str:
    names = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"):
        if candidate in names:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"]


def arm_radius(z: np.ndarray) -> np.ndarray:
    """Shoulder, biceps, elbow, forearm, wrist. z from -1.2 to 1.2."""
    z = np.asarray(z, dtype=np.float64)
    # Wrist at the bottom (negative z), shoulder at the top. Elbow stays at 0.
    knots_z = np.array([-1.20, -0.95, -0.55, -0.12, 0.06, 0.48, 0.90, 1.20])
    knots_r = np.array([0.105, 0.12, 0.155, 0.18, 0.19, 0.27, 0.24, 0.22])
    return np.interp(z, knots_z, knots_r)


def make_arm(n_along: int = 42, n_around: int = 20, diagonal: bool = False, skin_delta: float = 0.12):
    zs = np.linspace(-1.2, 1.2, n_along)
    dz = float(zs[1] - zs[0])
    radii = arm_radius(zs)
    phase = 0.0
    verts = []
    for j, z in enumerate(zs):
        if diagonal and j:
            phase += dz / max(float(radii[j]), 0.06)
        for i in range(n_around):
            theta = 2.0 * np.pi * i / n_around + phase
            radius = float(radii[j])
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z])
    vertices = np.asarray(verts, dtype=np.float64)
    quads = []
    for j in range(n_along - 1):
        for i in range(n_around):
            nxt = (i + 1) % n_around
            v0 = j * n_around + i
            v1 = j * n_around + nxt
            v2 = (j + 1) * n_around + nxt
            v3 = (j + 1) * n_around + i
            quads.append([v0, v1, v2, v3])
    quads_a = np.asarray(quads, dtype=np.int32)
    weights = sigmoid_weights(-vertices[:, 2], skin_delta)
    return vertices, quads_a, weights


def shade_quads(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    light = np.array([0.25, 0.45, 0.85])
    light /= np.linalg.norm(light)
    colors = np.zeros((len(quads), 3))
    for idx, face in enumerate(quads):
        pts = vertices[face]
        normal = np.cross(pts[1] - pts[0], pts[3] - pts[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            colors[idx] = SKIN * 0.5
            continue
        normal /= length
        bright = 0.38 + 0.62 * max(0.0, float(normal @ light))
        colors[idx] = np.clip(SKIN * bright, 0.0, 1.0)
    return colors


def draw_arm(ax, vertices: np.ndarray, quads: np.ndarray, edge: str, title: str, limits=None, elev: float = 18, azim: float = -64) -> None:
    collection = Poly3DCollection(
        vertices[quads],
        facecolors=shade_quads(vertices, quads),
        edgecolors=edge,
        linewidths=0.45,
        alpha=1.0,
    )
    ax.add_collection3d(collection)
    if limits is None:
        limits = bounds_of([vertices])
    center, half = limits
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect(tuple(np.maximum(half, 0.05)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=2)


def bounds_of(clouds: list[np.ndarray], pad: float = 0.12):
    stacked = np.vstack(clouds)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) + pad
    return center, half


def section_polygon(rest: np.ndarray, deformed: np.ndarray, quads: np.ndarray, angle: float) -> np.ndarray:
    points = material_section(rest, deformed, unique_edges(quads))
    if len(points) < 3:
        return np.zeros((0, 2))
    phi = np.radians(angle) / 2.0
    normal = np.array([0.0, -np.sin(phi), np.cos(phi)])
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_y = np.cross(normal, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    projected = points - np.outer(points @ normal, normal)
    uv = np.column_stack([projected @ axis_x, projected @ axis_y])
    order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
    return uv[order]


def retention_table(arm: dict, angles: list[float]) -> list[dict]:
    rest_area = {}
    rows = []
    for label, mesh in arm.items():
        for angle in angles:
            deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
            area = section_area(material_section(mesh["verts"], deformed, mesh["edges"]), angle)
            if angle == 0.0:
                rest_area[label] = area
            rows.append({"label": label, "angle": angle, "area": area, "deformed": deformed})
    for row in rows:
        row["retention"] = row["area"] / rest_area[row["label"]]
        row["formula"] = float(np.cos(np.radians(row["angle"]) / 2.0))
    return rows


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E7EB")
    ax.set_axisbelow(True)


def write_obj(path: Path, vertices: np.ndarray, quads: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# elbow demo mesh\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in quads:
            handle.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_rest(arm: dict, out: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.2), facecolor="white")
    limits = bounds_of([arm["ring"]["verts"], arm["diagonal"]["verts"]], pad=0.06)
    specs = (("ring", RING_EDGE, "环着骨头走"), ("diagonal", DIAG_EDGE, "斜着切一圈"))
    for index, (label, edge, title) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        mesh = arm[label]
        draw_arm(ax, mesh["verts"], mesh["quads"], edge, title, limits=limits)
    fig.suptitle("同一条胳膊，两种布线", fontsize=16)
    save(fig, out)


def figure_wires(arm: dict, out: Path) -> None:
    """Side view of the elbow so the loop direction is obvious."""
    fig = plt.figure(figsize=(8.6, 4.4), facecolor="white")
    elbow = []
    for mesh in arm.values():
        mask = np.abs(mesh["verts"][:, 2]) < 0.42
        elbow.append(mesh["verts"][mask])
    limits = bounds_of(elbow, pad=0.04)
    specs = (("ring", RING_EDGE, "环线：一圈一圈，横着过肘"), ("diagonal", DIAG_EDGE, "斜线：边是斜着跨过肘的"))
    for index, (label, edge, title) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        mesh = arm[label]
        draw_arm(ax, mesh["verts"], mesh["quads"], edge, title, limits=limits, elev=6, azim=-90)
    fig.suptitle("只看肘部：差别在边往哪走", fontsize=16)
    save(fig, out)


def figure_bend(arm: dict, out: Path) -> None:
    angles = [0, 45, 90, 120]
    mesh = arm["ring"]
    posed = []
    kept = []
    rest_area = section_area(material_section(mesh["verts"], mesh["verts"], mesh["edges"]), 0)
    for angle in angles:
        deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
        posed.append(deformed)
        area = section_area(material_section(mesh["verts"], deformed, mesh["edges"]), angle)
        kept.append(area / rest_area)
    limits = bounds_of(posed, pad=0.08)
    fig = plt.figure(figsize=(12.6, 4.6), facecolor="white")
    for index, (angle, deformed, keep) in enumerate(zip(angles, posed, kept), start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        draw_arm(ax, deformed, mesh["quads"], RING_EDGE, f"弯 {angle}°\n切面还剩 {keep:.0%}", limits=limits)
    fig.suptitle("越弯，肘部那一圈越扁", fontsize=16)
    save(fig, out)


def figure_compare(arm: dict, out: Path) -> None:
    slots = [
        ("ring", 90, RING_EDGE, "环线，弯 90°"),
        ("diagonal", 90, DIAG_EDGE, "斜线，弯 90°"),
        ("ring", 120, RING_EDGE, "环线，弯 120°"),
        ("diagonal", 120, DIAG_EDGE, "斜线，弯 120°"),
    ]
    posed = []
    measured = {}
    for label, angle, _edge, _title in slots:
        mesh = arm[label]
        deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
        posed.append(deformed)
        area = section_area(material_section(mesh["verts"], deformed, mesh["edges"]), angle)
        rest = section_area(material_section(mesh["verts"], mesh["verts"], mesh["edges"]), 0)
        measured[(label, angle)] = area / rest
    limits = bounds_of(posed, pad=0.08)
    fig = plt.figure(figsize=(9.4, 8.2), facecolor="white")
    for slot, (label, angle, edge, title) in enumerate(slots, start=1):
        ax = fig.add_subplot(2, 2, slot, projection="3d")
        draw_arm(
            ax,
            posed[slot - 1],
            arm[label]["quads"],
            edge,
            f"{title}  还剩 {measured[(label, angle)]:.0%}",
            limits=limits,
        )
    fig.suptitle("斜着布线，并不会让肘部更捏", fontsize=16)
    save(fig, out)


def figure_sections(arm: dict, out: Path) -> None:
    angles = [0, 45, 90, 120]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3), facecolor="white")
    for ax, angle in zip(axes, angles):
        for label, color, name in (("ring", RING_EDGE, "环线"), ("diagonal", DIAG_EDGE, "斜线")):
            mesh = arm[label]
            deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
            polygon = section_polygon(mesh["verts"], deformed, mesh["quads"], angle)
            closed = np.vstack([polygon, polygon[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.0, label=name)
            if label == "ring":
                ax.fill(polygon[:, 0], polygon[:, 1], color="#F3D5B5", alpha=0.85)
        formula = np.cos(np.radians(angle) / 2.0)
        ax.set_aspect("equal")
        ax.set_title(f"{angle}°   约 {formula:.0%}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#E5E7EB")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("从肘部切开：两条线叠在一起", fontsize=16)
    save(fig, out)


def figure_curve(arm: dict, out: Path) -> None:
    angles = list(range(0, 151, 5))
    rows = retention_table(arm, [float(a) for a in angles])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    for label, color, name in (("ring", RING_EDGE, "环线，实际量出来"), ("diagonal", DIAG_EDGE, "斜线，实际量出来")):
        subset = [row for row in rows if row["label"] == label]
        ax.plot(
            [row["angle"] for row in subset],
            [row["retention"] for row in subset],
            color=color,
            linewidth=2.2,
            label=name,
        )
    formula_x = np.linspace(0, 150, 60)
    ax.plot(formula_x, np.cos(np.radians(formula_x) / 2.0), color="#6B7280", linestyle="--", linewidth=1.4, label="心算：cos(角度÷2)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 150)
    style_axes(ax, "弯曲角度（度）", "肘部切面还剩多少")
    ax.legend(frameon=False)
    ax.set_title("两条布线叠在同一条线上", fontsize=15)
    save(fig, out)


def render_frame(arm: dict, angle: float, limits) -> Image.Image:
    fig = plt.figure(figsize=(8.6, 4.6), facecolor="white")
    for index, (label, edge, name) in enumerate((("ring", RING_EDGE, "环线"), ("diagonal", DIAG_EDGE, "斜线")), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        mesh = arm[label]
        deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
        keep = float(np.cos(np.radians(angle) / 2.0))
        draw_arm(ax, deformed, mesh["quads"], edge, f"{name}  {angle:.0f}°  还剩 {keep:.0%}", limits=limits)
    fig.suptitle("肘部怎么扁，和布线斜不斜没关系", fontsize=14)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def figure_gif(arm: dict, out: Path) -> None:
    angles = list(np.linspace(0, 120, 25)) + list(np.linspace(115, 0, 24))
    clouds = []
    for angle in angles:
        for mesh in arm.values():
            clouds.append(lbs(mesh["verts"], mesh["weights"], bend_around_x(float(angle))))
    limits = bounds_of(clouds, pad=0.08)
    frames = [render_frame(arm, float(angle), limits) for angle in angles]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def build_arm() -> dict:
    arm = {}
    for label, diagonal in (("ring", False), ("diagonal", True)):
        verts, quads, weights = make_arm(diagonal=diagonal)
        arm[label] = {
            "verts": verts,
            "quads": quads,
            "weights": weights,
            "edges": unique_edges(quads),
        }
    return arm


def main() -> None:
    parser = argparse.ArgumentParser(description="画出弯曲的胳膊和肘部切面")
    parser.add_argument("--quick", action="store_true", help="只画一张，给自动化检查用")
    parser.add_argument("--out", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    setup_font()
    arm = build_arm()
    out = args.out
    if args.quick:
        figure_rest(arm, out / "quick_rest.png")
        print(out / "quick_rest.png")
        return
    figure_rest(arm, out / "01_rest.png")
    figure_wires(arm, out / "02_wires.png")
    figure_bend(arm, out / "03_bend.png")
    figure_compare(arm, out / "04_compare.png")
    figure_sections(arm, out / "05_sections.png")
    figure_curve(arm, out / "06_curve.png")
    figure_gif(arm, out / "bend.gif")
    mesh_dir = ROOT / "docs" / "meshes"
    for label, angle in (("ring", 0.0), ("ring", 90.0), ("diagonal", 90.0)):
        mesh = arm[label]
        deformed = lbs(mesh["verts"], mesh["weights"], bend_around_x(angle))
        write_obj(mesh_dir / f"{label}_{int(angle)}.obj", deformed, mesh["quads"])
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
