#!/usr/bin/env python3
"""Put quad rings on a tube-shaped mesh and bend the result.

    python demo/tube_retopo.py --demo
    python demo/tube_retopo.py path/to/limb.obj --bend 90
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blender_addon"))

from blender_addon.tube import long_axis, missed_directions, retopo_tube  # noqa: E402
from demo.armbend import bounds_of, make_arm, setup_font, shade_quads  # noqa: E402
from experiments.oracle_section_area import bend_around_x, lbs, sigmoid_weights  # noqa: E402

SKIN = np.array([0.93, 0.76, 0.62])
SCULPT_EDGE = "#D6B49A"
QUAD_EDGE = "#6B4A36"


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(item) for item in line.split()[1:4]])
        elif line.startswith("f "):
            corners = []
            for item in line.split()[1:]:
                corners.append(int(item.split("/")[0]) - 1)
            if len(corners) == 3:
                faces.append(corners)
            elif len(corners) == 4:
                faces.append([corners[0], corners[1], corners[2]])
                faces.append([corners[0], corners[2], corners[3]])
    if not vertices or not faces:
        raise RuntimeError(f"{path} 里没有三角形")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = np.zeros_like(vertices)
    for face in faces:
        pts = vertices[face]
        face_normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        length = float(np.linalg.norm(face_normal))
        if length > 1e-12:
            face_normal = face_normal / length
            for index in face:
                normals[int(index)] += face_normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# tube retopo\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for normal in normals:
            handle.write(f"vn {normal[0]:.5f} {normal[1]:.5f} {normal[2]:.5f}\n")
        for face in faces:
            ids = " ".join(f"{int(index) + 1}//{int(index) + 1}" for index in face)
            handle.write(f"f {ids}\n")


def tube_from_profile(knots_z: np.ndarray, knots_r: np.ndarray, n_along: int, n_around: int, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    zs = np.linspace(float(knots_z[0]), float(knots_z[-1]), n_along)
    radii = np.interp(zs, knots_z, knots_r)
    vertices = []
    for z, radius in zip(zs, radii):
        for i in range(n_around):
            theta = 2.0 * np.pi * i / n_around
            vertices.append([radius * np.cos(theta), radius * np.sin(theta), z])
    vertices_a = np.asarray(vertices, dtype=np.float64)
    faces = []
    for j in range(n_along - 1):
        for i in range(n_around):
            nxt = (i + 1) % n_around
            v0 = j * n_around + i
            v1 = j * n_around + nxt
            v2 = (j + 1) * n_around + nxt
            v3 = (j + 1) * n_around + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    tilt = np.radians(tilt_deg)
    rotation = np.array(
        [
            [np.cos(tilt), 0.0, np.sin(tilt)],
            [0.0, 1.0, 0.0],
            [-np.sin(tilt), 0.0, np.cos(tilt)],
        ]
    )
    return vertices_a @ rotation.T, np.asarray(faces, dtype=np.int32)


def sculpt_finger() -> tuple[np.ndarray, np.ndarray]:
    knots_z = np.array([-0.9, -0.55, -0.35, -0.05, 0.2, 0.55, 0.9])
    knots_r = np.array([0.055, 0.07, 0.09, 0.062, 0.085, 0.07, 0.05])
    return tube_from_profile(knots_z, knots_r, 56, 20, 18.0)


def sculpt_curved() -> tuple[np.ndarray, np.ndarray]:
    """A hose that is already bent, so a single straight cut would slice through the arc."""
    samples = np.linspace(-1.05, 1.05, 72)
    radius = 0.11
    bend = 0.55
    centers = np.stack([bend * samples**2, np.zeros_like(samples), samples], axis=1)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    helper = np.array([0.0, 1.0, 0.0])
    vertices = []
    around = 18
    for center, tangent in zip(centers, tangents):
        side = np.cross(tangent, helper)
        side /= np.linalg.norm(side)
        up = np.cross(side, tangent)
        for i in range(around):
            theta = 2.0 * np.pi * i / around
            vertices.append(center + radius * (np.cos(theta) * side + np.sin(theta) * up))
    vertices_a = np.asarray(vertices, dtype=np.float64)
    faces = []
    for j in range(len(centers) - 1):
        for i in range(around):
            nxt = (i + 1) % around
            v0 = j * around + i
            v1 = j * around + nxt
            v2 = (j + 1) * around + nxt
            v3 = (j + 1) * around + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return vertices_a, np.asarray(faces, dtype=np.int32)


def sculpt_body(two_arms: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A torso with one arm sticking out. The long direction is the torso, not the arm."""
    torso_z = np.linspace(-0.9, 0.9, 36)
    torso_r = 0.28
    arm_x = np.linspace(0.22, 1.15, 28)
    arm_r = 0.09
    vertices = []
    around = 16

    def add_tube(centers: np.ndarray, radius: float) -> int:
        base = len(vertices)
        tangents = np.gradient(centers, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        helper = np.array([0.0, 1.0, 0.0])
        for center, tangent in zip(centers, tangents):
            if abs(tangent[1]) > 0.9:
                helper_local = np.array([1.0, 0.0, 0.0])
            else:
                helper_local = helper
            side = np.cross(tangent, helper_local)
            side /= np.linalg.norm(side)
            up = np.cross(side, tangent)
            for i in range(around):
                theta = 2.0 * np.pi * i / around
                vertices.append(center + radius * (np.cos(theta) * side + np.sin(theta) * up))
        return base

    torso = np.stack([np.zeros_like(torso_z), np.zeros_like(torso_z), torso_z], axis=1)
    arm = np.stack([arm_x, np.zeros_like(arm_x), np.full_like(arm_x, 0.25)], axis=1)
    torso_base = add_tube(torso, torso_r)
    arm_base = add_tube(arm, arm_r)
    left = np.stack([-arm_x, np.zeros_like(arm_x), np.full_like(arm_x, 0.25)], axis=1)
    left_base = add_tube(left, arm_r) if two_arms else None
    vertices_a = np.asarray(vertices, dtype=np.float64)
    faces = []

    def add_faces(base: int, count: int) -> None:
        for j in range(count - 1):
            for i in range(around):
                nxt = (i + 1) % around
                v0 = base + j * around + i
                v1 = base + j * around + nxt
                v2 = base + (j + 1) * around + nxt
                v3 = base + (j + 1) * around + i
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    add_faces(torso_base, len(torso))
    add_faces(arm_base, len(arm))
    if left_base is not None:
        add_faces(left_base, len(left))
    return vertices_a, np.asarray(faces, dtype=np.int32)


def sculpt_hose() -> tuple[np.ndarray, np.ndarray]:
    knots_z = np.array([-1.3, -0.4, -0.15, 0.0, 0.15, 0.5, 1.3])
    knots_r = np.array([0.11, 0.11, 0.16, 0.18, 0.16, 0.11, 0.11])
    return tube_from_profile(knots_z, knots_r, 60, 22, -22.0)


def sculpt_arm() -> tuple[np.ndarray, np.ndarray]:
    """Dense triangle arm, tilted so it is not lined up with the world axis."""
    vertices, quads, _weights = make_arm(n_along=64, n_around=32, diagonal=False)
    z = vertices[:, 2]
    vertices = vertices.copy()
    vertices[:, 0] += 0.03 * np.exp(-((z - 0.0) / 0.12) ** 2)
    faces = []
    for quad in quads:
        faces.append([quad[0], quad[1], quad[2]])
        faces.append([quad[0], quad[2], quad[3]])
    tilt = np.radians(28.0)
    rotation = np.array(
        [
            [np.cos(tilt), 0.0, np.sin(tilt)],
            [0.0, 1.0, 0.0],
            [-np.sin(tilt), 0.0, np.cos(tilt)],
        ]
    )
    return vertices @ rotation.T, np.asarray(faces, dtype=np.int32)


def bend_tube(vertices: np.ndarray, angle_deg: float, skin_delta: float = 0.12) -> np.ndarray:
    center, direction = long_axis(vertices)
    along = (vertices - center) @ direction
    # Hinge sits on the tube. Nearby points give the local direction, so a
    # curve bends at the joint instead of twisting around the chord.
    pivot = vertices[int(np.argmin(np.abs(along - np.median(along))))]
    near = vertices[np.linalg.norm(vertices - pivot, axis=1) < 0.22 * max(float(np.ptp(along)), 1e-6)]
    if len(near) >= 8:
        _, axes = np.linalg.eigh(np.cov((near - near.mean(axis=0)).T))
        direction = axes[:, -1]
        direction = direction / np.linalg.norm(direction)
    local = (vertices - pivot) @ direction
    weights = sigmoid_weights(-local, skin_delta * float(np.ptp(local)))
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    hinge = np.cross(direction, helper)
    hinge = hinge / np.linalg.norm(hinge)
    theta = np.radians(angle_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = (
        cosine * np.eye(3)
        + sine * np.array(
            [
                [0.0, -hinge[2], hinge[1]],
                [hinge[2], 0.0, -hinge[0]],
                [-hinge[1], hinge[0], 0.0],
            ]
        )
        + (1.0 - cosine) * np.outer(hinge, hinge)
    )
    moved = (vertices - pivot) @ rotation.T + pivot
    return (1.0 - weights[:, 1:2]) * vertices + weights[:, 1:2] * moved


def stand_up(vertices: np.ndarray) -> np.ndarray:
    """Put the long axis upright so a side view shows the loops."""
    center, direction = long_axis(vertices)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(direction, target)
    sine = np.linalg.norm(axis)
    cosine = float(np.clip(direction @ target, -1.0, 1.0))
    if sine < 1e-8:
        rotation = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis = axis / sine
        rotation = (
            cosine * np.eye(3)
            + sine * np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            + (1.0 - cosine) * np.outer(axis, axis)
        )
    return (vertices - center) @ rotation.T


def _draw(ax, vertices: np.ndarray, faces: np.ndarray, edge: str, title: str, limits, elev: float = 16, azim: float = -58, linewidth: float | None = None) -> None:
    polygons = [vertices[face] for face in faces]
    colors = shade_quads(vertices, faces) if faces.shape[1] == 4 else np.tile(SKIN, (len(faces), 1))
    width = 0.35 if faces.shape[1] == 4 else 0.05
    if linewidth is not None:
        width = linewidth
    collection = Poly3DCollection(polygons, facecolors=colors, edgecolors=edge, linewidths=width)
    ax.add_collection3d(collection)
    center, half = limits
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect(tuple(np.maximum(half, 0.05)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)


def save_gallery(items: list[tuple[str, tuple, np.ndarray, np.ndarray, np.ndarray]], out: Path) -> None:
    setup_font()
    fig, axes = plt.subplots(len(items), 2, figsize=(8.6, 12.4), facecolor="white", constrained_layout=True)
    for row, (name, sculpt, quads_v, quads_f, bent) in enumerate(items):
        center, rotation = _upright_frame(quads_v)
        sculpt_v = (sculpt[0] - center) @ rotation.T
        quads_up = (quads_v - center) @ rotation.T
        bent_up = (bent - center) @ rotation.T
        _draw_side(axes[row, 0], sculpt_v, np.asarray(sculpt[1]), quads_up, quads_f, f"{name} · {len(quads_f)} 个四边面")
        _draw_side(axes[row, 1], bent_up, quads_f, bent_up, quads_f, "弯 90°")
    fig.suptitle("胳膊、手指、软管，同一套切法", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _middle_faces(vertices: np.ndarray, faces: np.ndarray, fraction: float = 0.34) -> np.ndarray:
    """Keep a run of whole rings around the middle, so the crop does not slice a ring in half."""
    first = [int(index) for index in faces[0]]
    gaps = [abs(a - b) for a, b in zip(first, first[1:] + first[:1])]
    around = max(gaps)
    if around < 3 or len(vertices) % around != 0:
        return np.asarray(faces)
    n_rings = len(vertices) // around
    keep_rings = max(4, int(round(n_rings * fraction)))
    start = max(0, (n_rings - keep_rings) // 2)
    stop = min(n_rings, start + keep_rings)
    keep = []
    for face in faces:
        rings = [int(index) // around for index in face]
        if min(rings) >= start and max(rings) < stop:
            keep.append(face)
    if len(keep) < 4:
        return np.asarray(faces)
    return np.asarray(keep, dtype=np.int32)


def save_loop_closeup(items: list[tuple[str, tuple, np.ndarray, np.ndarray, np.ndarray]], out: Path) -> None:
    setup_font()
    fig, axes = plt.subplots(1, len(items), figsize=(11.2, 5.2), facecolor="white")
    for ax, (name, _sculpt, quads_v, quads_f, _bent) in zip(axes, items):
        center, rotation = _upright_frame(quads_v)
        upright = (quads_v - center) @ rotation.T
        band = _middle_faces(upright, quads_f)
        _draw_side(ax, upright, band, upright, band, f"{name}的环")
    fig.suptitle("套完之后，边是一圈一圈绕过去的", fontsize=16)
    fig.subplots_adjust(top=0.78, wspace=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _posed_upright(vertices: np.ndarray, angle: float, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    posed = vertices if angle == 0 else bend_tube(vertices, angle)
    return (posed - center) @ rotation.T


def save_bend_gif(vertices: np.ndarray, faces: np.ndarray, out: Path) -> None:
    setup_font()
    angles = list(np.linspace(0, 90, 18)) + list(np.linspace(84, 0, 15))
    center, rotation = _upright_frame(vertices)
    posed = [_posed_upright(vertices, float(angle), center, rotation) for angle in angles]
    span = np.concatenate(posed, axis=0)
    pad = 0.08 * max(float(np.ptp(span[:, 0])), float(np.ptp(span[:, 2])), 1e-6)
    limits = (float(span[:, 0].min()) - pad, float(span[:, 0].max()) + pad, float(span[:, 2].min()) - pad, float(span[:, 2].max()) + pad)
    frames = []
    for angle, posed_verts in zip(angles, posed):
        fig, ax = plt.subplots(figsize=(4.6, 5.2), facecolor="white")
        _draw_side(ax, posed_verts, faces, posed_verts, faces, f"弯 {angle:.0f}°")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


def save_bend_strip(vertices: np.ndarray, faces: np.ndarray, out: Path) -> None:
    setup_font()
    angles = [0, 45, 90, 120]
    center, rotation = _upright_frame(vertices)
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.6), facecolor="white", constrained_layout=True)
    for ax, angle in zip(axes, angles):
        posed = _posed_upright(vertices, angle, center, rotation)
        _draw_side(ax, posed, faces, posed, faces, f"弯 {angle}°")
    fig.suptitle("套完之后可以这样弯", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _upright_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center, direction = long_axis(vertices)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(direction, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(direction @ target, -1.0, 1.0))
    if sine < 1e-8:
        rotation = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis = axis / sine
        rotation = (
            cosine * np.eye(3)
            + sine * np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            + (1.0 - cosine) * np.outer(axis, axis)
        )
    return center, rotation


def _draw_side(ax, body_v: np.ndarray, body_f: np.ndarray, wire_v: np.ndarray | None, wire_f: np.ndarray | None, title: str) -> None:
    xy = np.column_stack([body_v[:, 0], body_v[:, 2]])
    order = np.argsort(body_v[body_f].mean(axis=1)[:, 1])
    for face in body_f[order]:
        poly = xy[face]
        ax.fill(poly[:, 0], poly[:, 1], color="#F3D7C3", edgecolor="#E7C4A8", linewidth=0.12, zorder=1)
    if wire_v is not None and wire_f is not None:
        wire_xy = np.column_stack([wire_v[:, 0], wire_v[:, 2]])
        for face in wire_f:
            pts = wire_v[list(face)]
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if float(normal[1]) <= 0.0:
                continue
            loop = wire_xy[list(face) + [int(face[0])]]
            ax.plot(loop[:, 0], loop[:, 1], color="#6B4A36", linewidth=1.05, solid_capstyle="round", zorder=3)
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_preview(sculpt, quads_v, quads_f, bent, out: Path, angle: float = 90.0) -> None:
    setup_font()
    center, rotation = _upright_frame(quads_v)
    sculpt_v = (sculpt[0] - center) @ rotation.T
    sculpt_f = np.asarray(sculpt[1])
    quads_up = (quads_v - center) @ rotation.T
    bent_up = (bent - center) @ rotation.T
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.6), facecolor="white", constrained_layout=True)
    _draw_side(axes[0], sculpt_v, sculpt_f, quads_up, quads_f, f"环线套在高模上 · {len(quads_f)} 个四边面")
    _draw_side(axes[1], bent_up, quads_f, bent_up, quads_f, f"再弯 {angle:.0f}°")
    fig.suptitle("高模进来，环线四边面出去", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="给管子形状的模型套环线四边面")
    parser.add_argument("mesh", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true", help="用内置的一条胳膊高模")
    parser.add_argument("--gallery", action="store_true", help="胳膊、手指、软管各做一遍")
    parser.add_argument("--check", action="store_true", help="检查指定方向时切出来的是胳膊")
    parser.add_argument("--rings", type=int, default=26)
    parser.add_argument("--around", type=int, default=16)
    parser.add_argument("--bend", type=float, default=90.0)
    parser.add_argument("--axis", type=str, default=None, help="例如 1,0,0，顺着这个方向切细的那根")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.check:
        body_v, body_f = sculpt_body()
        arm_v, arm_f, _ = retopo_tube(body_v, body_f, n_rings=18, n_around=12, axis=np.array([1.0, 0.0, 0.0]))
        extent = arm_v.max(axis=0) - arm_v.min(axis=0)
        if not (extent[0] > extent[1] * 2 and extent[0] > extent[2] * 2 and len(arm_f) >= 48):
            raise SystemExit(f"指定方向没有切出胳膊: extent={extent} faces={len(arm_f)}")
        both_v, both_f = sculpt_body(two_arms=True)
        auto_v, _auto_f, _auto_axis = retopo_tube(both_v, both_f, n_rings=16, n_around=10)
        if float(auto_v[:, 0].min()) < -0.05 and float(auto_v[:, 0].max()) > 0.05:
            raise SystemExit("自动切法把两条胳膊连到一起了")
        if 0 not in missed_directions(both_v, auto_v):
            raise SystemExit("两条胳膊时应该提示左右还没切完")
        one_v, one_f = sculpt_body(False)
        one_cut, _one_faces, _one_axis = retopo_tube(one_v, one_f, n_rings=18, n_around=12)
        if float(one_cut[:, 0].max()) > 0.6:
            raise SystemExit("自动切法从身子拐进了胳膊")
        if 0 not in missed_directions(one_v, one_cut):
            raise SystemExit("身子加一条胳膊时应该提示左右还没切完")
        print(f"ok 胳膊四边面 {len(arm_f)} 个")
        return

    figure_dir = ROOT / "docs" / "figures"
    demo_mesh_dir = ROOT / "docs" / "meshes"
    if args.gallery:
        args.mesh_dir = args.mesh_dir or demo_mesh_dir
        args.out = args.out or figure_dir / "07_retopo.png"
        specs = (
            ("胳膊", sculpt_arm, "arm"),
            ("手指", sculpt_finger, "finger"),
            ("软管", sculpt_hose, "hose"),
        )
        items = []
        for title, builder, stem in specs:
            vertices, faces = builder()
            quads_v, quads_f, _axis = retopo_tube(vertices, faces, n_rings=args.rings, n_around=args.around)
            bent = bend_tube(quads_v, args.bend)
            write_obj(args.mesh_dir / f"{stem}_sculpt.obj", vertices, faces)
            write_obj(args.mesh_dir / f"{stem}_rings.obj", quads_v, quads_f)
            write_obj(args.mesh_dir / f"{stem}_bent.obj", bent, quads_f)
            items.append((title, (vertices, faces), quads_v, quads_f, bent))
            print(f"{stem} quads={len(quads_f)}")
        gallery = args.out if args.out.name != "07_retopo.png" else args.out.with_name("08_shapes.png")
        save_gallery(items, gallery)
        save_loop_closeup(items, gallery.with_name("09_loops.png"))
        arm_center, arm_rotation = _upright_frame(items[0][2])
        arm_up = (items[0][2] - arm_center) @ arm_rotation.T
        elbow_faces = _middle_faces(arm_up, items[0][3], fraction=0.28)
        close, ax = plt.subplots(figsize=(4.2, 6.4), facecolor="white", constrained_layout=True)
        _draw_side(ax, arm_up, elbow_faces, arm_up, elbow_faces, "胳膊肘这一段")
        close.savefig(gallery.with_name("11_arm_close.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(close)
        save_bend_gif(items[0][2], items[0][3], gallery.with_name("retopo_bend.gif"))
        save_bend_strip(items[0][2], items[0][3], gallery.with_name("14_bend_strip.png"))
        save_bend_strip(items[1][2], items[1][3], gallery.with_name("15_finger_bend.png"))
        save_bend_strip(items[2][2], items[2][3], gallery.with_name("16_hose_bend.png"))
        curved_v, curved_f = sculpt_curved()
        curved_q, curved_faces, _axis = retopo_tube(curved_v, curved_f, n_rings=args.rings, n_around=args.around)
        curved_bent = bend_tube(curved_q, args.bend)
        write_obj(args.mesh_dir / "curved_sculpt.obj", curved_v, curved_f)
        write_obj(args.mesh_dir / "curved_rings.obj", curved_q, curved_faces)
        write_obj(args.mesh_dir / "curved_bent.obj", curved_bent, curved_faces)
        setup_font()
        center, rotation = _upright_frame(curved_q)
        curve_fig, ax = plt.subplots(figsize=(5.4, 7.4), facecolor="white", constrained_layout=True)
        _draw_side(
            ax,
            (curved_v - center) @ rotation.T,
            curved_f,
            (curved_q - center) @ rotation.T,
            curved_faces,
            f"环顺着弯走 · {len(curved_faces)} 个四边面",
        )
        curve_fig.suptitle("不用把它扳直再套环", fontsize=16)
        curve_fig.savefig(gallery.with_name("10_curved.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(curve_fig)
        print(f"curved quads={len(curved_faces)}")
        body_v, body_f = sculpt_body()
        body_default, body_default_f, _ = retopo_tube(body_v, body_f, n_rings=18, n_around=12)
        body_arm, body_arm_f, _ = retopo_tube(body_v, body_f, n_rings=18, n_around=12, axis=np.array([1.0, 0.0, 0.0]))
        write_obj(args.mesh_dir / "body_sculpt.obj", body_v, body_f)
        write_obj(args.mesh_dir / "body_auto.obj", body_default, body_default_f)
        write_obj(args.mesh_dir / "body_arm.obj", body_arm, body_arm_f)
        body_fig, body_axes = plt.subplots(1, 2, figsize=(8.6, 7.2), facecolor="white")

        def _flat(points):
            return np.column_stack([points[:, 0] + 0.35 * points[:, 1], points[:, 2] + 0.18 * points[:, 1]])

        body_xy = _flat(body_v)
        body_order = np.argsort(body_v[body_f].mean(axis=1)[:, 1])
        for ax, limb_v, limb_f, edge, title in (
            (body_axes[0], body_default, body_default_f, "#6B4A36", "不指定，停在胳膊下面"),
            (body_axes[1], body_arm, body_arm_f, "#C2410C", "指定向右，切胳膊"),
        ):
            for face in body_f[body_order]:
                poly = body_xy[face]
                ax.fill(poly[:, 0], poly[:, 1], color="#F3D7C3", edgecolor="#E7C4A8", linewidth=0.12, zorder=1)
            flat = _flat(limb_v)
            for face in limb_f:
                pts = limb_v[list(face)]
                normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                if normal[1] > 0.0:
                    continue
                loop = flat[list(face) + [face[0]]]
                ax.plot(loop[:, 0], loop[:, 1], color=edge, linewidth=1.15, solid_capstyle="round", zorder=3)
            ax.set_title(title, fontsize=14, pad=8)
            ax.set_aspect("equal")
            ax.axis("off")
        body_fig.suptitle("想切哪根，就告诉它方向", fontsize=16)
        body_fig.savefig(gallery.with_name("12_aim.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(body_fig)
        arm_only = body_v[body_v[:, 0] > 0.35]
        aim_limits = bounds_of([arm_only, body_arm], pad=0.04)
        aim_fig = plt.figure(figsize=(8.4, 4.4), facecolor="white")
        for index, (verts, faces, edge, title) in enumerate(
            (
                (body_v, body_f, SCULPT_EDGE, "胳膊那一段"),
                (body_arm, body_arm_f, QUAD_EDGE, f"切出来 · {len(body_arm_f)} 个四边面"),
            ),
            start=1,
        ):
            ax = aim_fig.add_subplot(1, 2, index, projection="3d")
            _draw(ax, verts, faces, edge, title, aim_limits, elev=12, azim=-70, linewidth=0.8)
        aim_fig.suptitle("指定向右之后，只剩这条胳膊", fontsize=16)
        aim_fig.savefig(gallery.with_name("13_arm_only.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(aim_fig)
        print(f"body auto={len(body_default_f)} arm={len(body_arm_f)}")
        both_v, both_f = sculpt_body(two_arms=True)
        right_v, right_f, _ = retopo_tube(both_v, both_f, n_rings=18, n_around=12, axis=np.array([1.0, 0.0, 0.0]))
        left_v, left_f, _ = retopo_tube(both_v, both_f, n_rings=18, n_around=12, axis=np.array([-1.0, 0.0, 0.0]))
        write_obj(args.mesh_dir / "both_sculpt.obj", both_v, both_f)
        write_obj(args.mesh_dir / "both_right.obj", right_v, right_f)
        write_obj(args.mesh_dir / "both_left.obj", left_v, left_f)
        both_limits = bounds_of([both_v, right_v, left_v], pad=0.08)
        both_fig = plt.figure(figsize=(10.4, 4.2), facecolor="white")
        for index, (verts, faces, edge, title) in enumerate(
            (
                (both_v, both_f, SCULPT_EDGE, "左右各一条"),
                (right_v, right_f, QUAD_EDGE, "方向 1,0,0 右手"),
                (left_v, left_f, QUAD_EDGE, "方向 -1,0,0 左手"),
            ),
            start=1,
        ):
            ax = both_fig.add_subplot(1, 3, index, projection="3d")
            _draw(ax, verts, faces, edge, title, both_limits)
        both_fig.suptitle("两条胳膊要切两次", fontsize=16)
        both_fig.savefig(gallery.with_name("17_two_arms.png"), dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(both_fig)
        overlay, ax = plt.subplots(figsize=(6.4, 7.2), facecolor="white")

        def _flat(points):
            return np.column_stack([points[:, 0] + 0.42 * points[:, 1], points[:, 2] + 0.22 * points[:, 1]])

        body_xy = _flat(both_v)
        order = np.argsort(both_v[both_f].mean(axis=1)[:, 1])
        for face in both_f[order]:
            poly = body_xy[face]
            ax.fill(poly[:, 0], poly[:, 1], color="#F3D7C3", edgecolor="#E7C4A8", linewidth=0.15, zorder=1)
        for limb_v, limb_f, edge in ((right_v, right_f, "#C2410C"), (left_v, left_f, "#1D4ED8")):
            flat = _flat(limb_v)
            for face in limb_f:
                loop = flat[list(face) + [face[0]]]
                ax.plot(loop[:, 0], loop[:, 1], color=edge, linewidth=1.15, solid_capstyle="round", zorder=3)
        ax.set_title("两次切完，线套在胳膊上", fontsize=15, pad=8)
        ax.set_aspect("equal")
        ax.axis("off")
        overlay.savefig(gallery.with_name("18_both_on_body.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(overlay)
        # Keep the arm preview the README already points at.
        arm = items[0]
        save_preview(
            arm[1],
            arm[2],
            arm[3],
            arm[4],
            args.out if args.out.name == "07_retopo.png" else gallery.with_name("07_retopo.png"),
            angle=args.bend,
        )
        write_obj(args.mesh_dir / "sculpt_arm.obj", arm[1][0], arm[1][1])
        write_obj(args.mesh_dir / "retopo_rings.obj", arm[2], arm[3])
        write_obj(args.mesh_dir / "retopo_bent.obj", arm[4], arm[3])
        print(gallery)
        return

    chosen_axis = None
    axis_tag = ""
    if args.axis:
        chosen_axis = np.array([float(part) for part in args.axis.split(",")], dtype=np.float64)
        dominant = int(np.argmax(np.abs(chosen_axis)))
        axis_tag = f"_{'xyz'[dominant]}" if chosen_axis[dominant] >= 0 else f"_n{'xyz'[dominant]}"

    if args.demo or args.mesh is None:
        mesh_dir = args.mesh_dir or demo_mesh_dir
        preview = args.out or figure_dir / "07_retopo.png"
        vertices, faces = sculpt_arm()
        write_obj(mesh_dir / "sculpt_arm.obj", vertices, faces)
        ring_path = mesh_dir / "retopo_rings.obj"
        bent_path = mesh_dir / "retopo_bent.obj"
        strip = figure_dir / "14_bend_strip.png" if args.out is None else preview.with_name("bend_strip.png")
    else:
        mesh_dir = args.mesh_dir or args.mesh.parent
        stem = f"{args.mesh.stem}{axis_tag}"
        preview = args.out or mesh_dir / f"{stem}_preview.png"
        vertices, faces = read_obj(args.mesh)
        ring_path = mesh_dir / f"{stem}_rings.obj"
        bent_path = mesh_dir / f"{stem}_bent.obj"
        strip = mesh_dir / f"{stem}_bend.png"
    quads_v, quads_f, _axis = retopo_tube(
        vertices, faces, n_rings=args.rings, n_around=args.around, axis=chosen_axis
    )
    bent = bend_tube(quads_v, args.bend)
    save_preview((vertices, faces), quads_v, quads_f, bent, preview, angle=args.bend)
    save_bend_strip(quads_v, quads_f, strip)
    write_obj(ring_path, quads_v, quads_f)
    write_obj(bent_path, bent, quads_f)
    print(f"三角面 {len(faces)} 个，套成四边面 {len(quads_f)} 个")
    if chosen_axis is None:
        hints = (
            ("左右", "1,0,0", "-1,0,0"),
            ("前后", "0,1,0", "0,-1,0"),
            ("上下", "0,0,1", "0,0,-1"),
        )
        for axis in missed_directions(vertices, quads_v):
            name, positive, negative = hints[axis]
            print(f"{name}还没切完。正向 --axis {positive}，反向 --axis {negative}")
    print(ring_path)
    print(bent_path)
    print(preview)
    print(strip)


if __name__ == "__main__":
    main()
