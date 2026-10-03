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

from blender_addon.tube import long_axis, retopo_tube  # noqa: E402
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
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# tube retopo\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        sides = faces.shape[1]
        for face in faces:
            ids = " ".join(str(int(index) + 1) for index in face[:sides])
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
    fig = plt.figure(figsize=(10.6, 9.6), facecolor="white")
    for row, (name, sculpt, quads_v, quads_f, bent) in enumerate(items):
        limits = bounds_of([sculpt[0], quads_v, bent], pad=0.06)
        panels = (
            (sculpt[0], sculpt[1], SCULPT_EDGE, f"{name} · 高模"),
            (quads_v, quads_f, QUAD_EDGE, "环线"),
            (bent, quads_f, QUAD_EDGE, "弯 90°"),
        )
        for col, (verts, faces, edge, title) in enumerate(panels):
            ax = fig.add_subplot(len(items), 3, row * 3 + col + 1, projection="3d")
            _draw(ax, verts, faces, edge, title, limits)
    fig.suptitle("胳膊、手指、软管，同一套切法", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_loop_closeup(items: list[tuple[str, tuple, np.ndarray, np.ndarray, np.ndarray]], out: Path) -> None:
    setup_font()
    fig = plt.figure(figsize=(10.8, 4.2), facecolor="white")
    for index, (name, _sculpt, quads_v, quads_f, _bent) in enumerate(items, start=1):
        upright = stand_up(quads_v)
        band = np.abs(upright[:, 2]) < 0.28 * max(np.ptp(upright[:, 2]), 1e-6)
        if int(band.sum()) < 12:
            band = np.ones(len(upright), dtype=bool)
        limits = bounds_of([upright[band]], pad=0.03)
        ax = fig.add_subplot(1, len(items), index, projection="3d")
        _draw(ax, upright, quads_f, QUAD_EDGE, f"{name}的环", limits, elev=8, azim=-90, linewidth=0.7)
    fig.suptitle("套完之后，边是一圈一圈绕过去的", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_bend_gif(vertices: np.ndarray, faces: np.ndarray, out: Path) -> None:
    setup_font()
    angles = list(np.linspace(0, 90, 18)) + list(np.linspace(84, 0, 15))
    posed = [bend_tube(vertices, float(angle)) for angle in angles]
    limits = bounds_of(posed, pad=0.08)
    frames = []
    for angle, posed_verts in zip(angles, posed):
        fig = plt.figure(figsize=(4.4, 5.0), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _draw(ax, posed_verts, faces, QUAD_EDGE, f"弯 {angle:.0f}°", limits)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


def save_preview(sculpt, quads_v, quads_f, bent, out: Path) -> None:
    setup_font()
    limits = bounds_of([sculpt[0], quads_v, bent], pad=0.08)
    fig = plt.figure(figsize=(11.4, 4.4), facecolor="white")
    panels = (
        (sculpt[0], sculpt[1], SCULPT_EDGE, "高模"),
        (quads_v, quads_f, QUAD_EDGE, "套上环线"),
        (bent, quads_f, QUAD_EDGE, "再弯 90°"),
    )
    for index, (verts, faces, edge, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _draw(ax, verts, faces, edge, title, limits)
    fig.suptitle("高模进来，环线四边面出去", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="给管子形状的模型套环线四边面")
    parser.add_argument("mesh", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true", help="用内置的一条胳膊高模")
    parser.add_argument("--gallery", action="store_true", help="胳膊、手指、软管各做一遍")
    parser.add_argument("--rings", type=int, default=26)
    parser.add_argument("--around", type=int, default=16)
    parser.add_argument("--bend", type=float, default=90.0)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "07_retopo.png")
    parser.add_argument("--mesh-dir", type=Path, default=ROOT / "docs" / "meshes")
    args = parser.parse_args()

    if args.gallery:
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
        save_bend_gif(items[0][2], items[0][3], gallery.with_name("retopo_bend.gif"))
        curved_v, curved_f = sculpt_curved()
        curved_q, curved_faces, _axis = retopo_tube(curved_v, curved_f, n_rings=args.rings, n_around=args.around)
        curved_bent = bend_tube(curved_q, args.bend)
        write_obj(args.mesh_dir / "curved_sculpt.obj", curved_v, curved_f)
        write_obj(args.mesh_dir / "curved_rings.obj", curved_q, curved_faces)
        write_obj(args.mesh_dir / "curved_bent.obj", curved_bent, curved_faces)
        setup_font()
        curve_limits = bounds_of([curved_v, curved_q], pad=0.06)
        curve_fig = plt.figure(figsize=(8.2, 4.2), facecolor="white")
        for index, (verts, faces, edge, title) in enumerate(
            (
                (curved_v, curved_f, SCULPT_EDGE, "本来就是弯的"),
                (curved_q, curved_faces, QUAD_EDGE, "环顺着弯走"),
            ),
            start=1,
        ):
            ax = curve_fig.add_subplot(1, 2, index, projection="3d")
            _draw(ax, verts, faces, edge, title, curve_limits)
        curve_fig.suptitle("不用把它扳直再套环", fontsize=16)
        curve_fig.savefig(gallery.with_name("10_curved.png"), dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(curve_fig)
        print(f"curved quads={len(curved_faces)}")
        # Keep the arm preview the README already points at.
        arm = items[0]
        save_preview(arm[1], arm[2], arm[3], arm[4], args.out if args.out.name == "07_retopo.png" else gallery.with_name("07_retopo.png"))
        write_obj(args.mesh_dir / "sculpt_arm.obj", arm[1][0], arm[1][1])
        write_obj(args.mesh_dir / "retopo_rings.obj", arm[2], arm[3])
        write_obj(args.mesh_dir / "retopo_bent.obj", arm[4], arm[3])
        print(gallery)
        return

    if args.demo or args.mesh is None:
        vertices, faces = sculpt_arm()
        write_obj(args.mesh_dir / "sculpt_arm.obj", vertices, faces)
    else:
        vertices, faces = read_obj(args.mesh)

    quads_v, quads_f, _axis = retopo_tube(vertices, faces, n_rings=args.rings, n_around=args.around)
    bent = bend_tube(quads_v, args.bend)
    save_preview((vertices, faces), quads_v, quads_f, bent, args.out)
    write_obj(args.mesh_dir / "retopo_rings.obj", quads_v, quads_f)
    write_obj(args.mesh_dir / "retopo_bent.obj", bent, quads_f)
    print(f"quads={len(quads_f)} verts={len(quads_v)}")
    print(args.out)


if __name__ == "__main__":
    main()
