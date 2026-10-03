"""Quad rings along the long axis of a tube-shaped triangle mesh.

Slices the mesh with planes, resamples each cut to a fixed number of points,
and connects neighboring cuts into quads. Meant for arms, fingers, hoses, tails.
"""

from __future__ import annotations

import numpy as np


def long_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((vertices - center).T))
    direction = axes[:, -1]
    direction = direction / np.linalg.norm(direction)
    if direction[2] < 0.0:
        direction = -direction
    return center, direction


def _plane_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(direction, helper)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(direction, tangent)
    return tangent, bitangent


def _segment_hits(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    segments = []
    eps = 1e-8
    for face in faces:
        hits = []
        tri = vertices[face]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            start = tri[a]
            end = tri[b]
            da = float(np.dot(start - origin, normal))
            db = float(np.dot(end - origin, normal))
            if abs(da) <= eps and abs(db) <= eps:
                continue
            if abs(da) <= eps:
                hits.append(start)
                continue
            if abs(db) <= eps:
                continue
            if da * db > 0.0:
                continue
            t = da / (da - db)
            hits.append(start + t * (end - start))
        if len(hits) < 2:
            continue
        unique = [hits[0]]
        for point in hits[1:]:
            if all(np.linalg.norm(point - kept) > 1e-7 for kept in unique):
                unique.append(point)
        if len(unique) >= 2:
            segments.append((unique[0], unique[1]))
    return segments


def _stitch_loop(segments: list[tuple[np.ndarray, np.ndarray]], scale: float) -> np.ndarray | None:
    if not segments:
        return None
    tol = max(scale * 1e-5, 1e-6)

    def key(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(point / tol).astype(int))

    points: dict[tuple[int, int, int], np.ndarray] = {}
    neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for start, end in segments:
        ka, kb = key(start), key(end)
        if ka == kb:
            continue
        points.setdefault(ka, start)
        points.setdefault(kb, end)
        neighbors.setdefault(ka, []).append(kb)
        neighbors.setdefault(kb, []).append(ka)
    if not neighbors:
        return None

    start = max(neighbors, key=lambda item: len(neighbors[item]))
    loop = [start]
    previous = None
    current = start
    seen = set()
    for _ in range(len(neighbors) + 2):
        nxt = None
        for candidate in neighbors.get(current, []):
            edge = (current, candidate) if current < candidate else (candidate, current)
            if candidate != previous and edge not in seen:
                nxt = candidate
                seen.add(edge)
                break
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
    if len(loop) < 6:
        return None
    return np.array([points[item] for item in loop], dtype=np.float64)


def _resample_ring(loop: np.ndarray, origin: np.ndarray, direction: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray, count: int) -> np.ndarray:
    # Angle around the middle of this cut, not the centerline guess. An off-center
    # guess packs points on one side and leaves long skinny quads.
    centroid = loop.mean(axis=0)
    centroid = centroid - direction * float(np.dot(centroid - origin, direction))
    offset = loop - centroid
    flat = offset - np.outer(offset @ direction, direction)
    angles = np.arctan2(flat @ bitangent, flat @ tangent)
    order = np.argsort(angles)
    angles = angles[order]
    samples = loop[order]
    angles = np.concatenate([angles, angles[:1] + 2.0 * np.pi])
    samples = np.vstack([samples, samples[:1]])
    targets = np.linspace(-np.pi, np.pi, count, endpoint=False)
    ring = []
    for target in targets:
        shifted = target
        if shifted < angles[0]:
            shifted += 2.0 * np.pi
        index = int(np.searchsorted(angles, shifted, side="right") - 1)
        index = min(max(index, 0), len(samples) - 2)
        span = angles[index + 1] - angles[index]
        mix = 0.0 if span < 1e-8 else (shifted - angles[index]) / span
        ring.append((1.0 - mix) * samples[index] + mix * samples[index + 1])
    return np.asarray(ring, dtype=np.float64)


def _resample_curve(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total < 1e-8:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, total, count)
    samples = []
    for target in targets:
        index = int(np.searchsorted(cumulative, target, side="right") - 1)
        index = min(max(index, 0), len(points) - 2)
        span = cumulative[index + 1] - cumulative[index]
        mix = 0.0 if span < 1e-8 else (target - cumulative[index]) / span
        samples.append((1.0 - mix) * points[index] + mix * points[index + 1])
    return np.asarray(samples, dtype=np.float64)


def _centerline(vertices: np.ndarray, n_rings: int) -> np.ndarray:
    """Walk from one tip to the other so a curved hose is not cut by one straight axis."""
    center = vertices.mean(axis=0)
    start = vertices[int(np.argmax(np.linalg.norm(vertices - center, axis=1)))]
    end = vertices[int(np.argmax(np.linalg.norm(vertices - start, axis=1)))]
    length = float(np.linalg.norm(end - start))
    spread = np.linalg.eigvalsh(np.cov((vertices - center).T))
    reach = max(float(np.sqrt(max(spread[0], 1e-8))) * 3.0, length * 0.08)
    step = length / max(n_rings, 4)
    position = start.copy()
    direction = (end - start) / max(length, 1e-8)
    samples = []
    for _ in range(n_rings * 5):
        local = vertices[np.linalg.norm(vertices - position, axis=1) < reach]
        if len(local) < 8:
            break
        here = local.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov((local - here).T))
        tangent = axes[:, -1]
        tangent = tangent / np.linalg.norm(tangent)
        if float(tangent @ direction) < 0.0:
            tangent = -tangent
        samples.append(here)
        remaining = end - here
        if float(np.linalg.norm(remaining)) < reach:
            samples.append(end)
            break
        direction = tangent
        position = here + tangent * step
    polyline = np.asarray(samples, dtype=np.float64) if samples else np.zeros((0, 3))
    span = float(np.linalg.norm(polyline[-1] - polyline[0])) if len(polyline) > 1 else 0.0
    if len(polyline) < 4 or span < 0.65 * length:
        return np.linspace(start, end, n_rings)
    return _resample_curve(polyline, n_rings)


def _all_loops(segments: list[tuple[np.ndarray, np.ndarray]], scale: float) -> list[np.ndarray]:
    if not segments:
        return []
    tol = max(scale * 1e-5, 1e-6)

    def key(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(point / tol).astype(int))

    points: dict[tuple[int, int, int], np.ndarray] = {}
    neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    unused: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for start, end in segments:
        ka, kb = key(start), key(end)
        if ka == kb:
            continue
        points.setdefault(ka, start)
        points.setdefault(kb, end)
        neighbors.setdefault(ka, []).append(kb)
        neighbors.setdefault(kb, []).append(ka)
        unused.add((ka, kb) if ka < kb else (kb, ka))
    loops = []
    while unused:
        start = next(iter(unused))[0]
        loop = [start]
        previous = None
        current = start
        for _ in range(len(points) + 2):
            nxt = None
            for candidate in neighbors.get(current, []):
                edge = (current, candidate) if current < candidate else (candidate, current)
                if candidate != previous and edge in unused:
                    nxt = candidate
                    unused.remove(edge)
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) >= 6:
            loops.append(np.array([points[item] for item in loop], dtype=np.float64))
    return loops


def _loop_radius(loop: np.ndarray) -> float:
    center = loop.mean(axis=0)
    return float(np.mean(np.linalg.norm(loop - center, axis=1)))


def _retopo_aimed(vertices: np.ndarray, faces: np.ndarray, axis: np.ndarray, n_rings: int, n_around: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow the thin limb in one direction and stop when the cut hits the body."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    projection = vertices @ axis
    # The axis points out along the limb you want. Start at that tip and walk back.
    tip = vertices[int(np.argmax(projection))]
    forward = -axis
    length = float(projection.max() - projection.min())
    step = length / max(n_rings * 2, 8)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    rings = []
    normal = None
    bitangent = None
    tip_radius = None
    origin = tip.copy()
    for _ in range(n_rings * 3):
        loops = _all_loops(_segment_hits(vertices, faces, origin, forward), scale)
        if loops:
            if len(loops) > 1 and len(rings) >= 4:
                break
            loop = min(loops, key=_loop_radius)
            radius = _loop_radius(loop)
            if tip_radius is None:
                tip_radius = radius
            elif radius > tip_radius * 2.4 and len(rings) >= 4:
                break
            if normal is None:
                normal, bitangent = _plane_basis(forward)
            else:
                normal = normal - forward * float(normal @ forward)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-8:
                    normal, bitangent = _plane_basis(forward)
                else:
                    normal = normal / norm
                    bitangent = np.cross(forward, normal)
            ring = _resample_ring(loop, origin, forward, normal, bitangent, n_around)
            edges = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
            if float(edges.min()) >= 0.35 * float(np.median(edges)):
                rings.append(ring)
            if len(rings) >= n_rings:
                break
        origin = origin + forward * step
    if len(rings) < 4:
        raise RuntimeError("这个方向上切不出一条细管子")
    quads = []
    count = n_around
    for j in range(len(rings) - 1):
        for i in range(count):
            nxt = (i + 1) % count
            v0 = j * count + i
            v1 = j * count + nxt
            v2 = (j + 1) * count + nxt
            v3 = (j + 1) * count + i
            pts = np.stack([rings[j][i], rings[j][nxt], rings[j + 1][nxt], rings[j + 1][i]])
            face_normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            outward = pts.mean(axis=0) - 0.5 * (rings[j].mean(axis=0) + rings[j + 1].mean(axis=0))
            if float(face_normal @ outward) < 0.0:
                quads.append([v0, v3, v2, v1])
            else:
                quads.append([v0, v1, v2, v3])
    return np.vstack(rings), np.asarray(quads, dtype=np.int32), forward


def retopo_tube(vertices: np.ndarray, faces: np.ndarray, n_rings: int = 28, n_around: int = 16, axis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quad vertices, quad faces, and a unit axis direction."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if axis is not None:
        return _retopo_aimed(vertices, faces, np.asarray(axis, dtype=np.float64), n_rings, n_around)
    centers = _centerline(vertices, n_rings)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    rings = []
    normal = None
    bitangent = None
    for index, origin in enumerate(centers):
        nxt = centers[min(index + 1, len(centers) - 1)]
        prev = centers[max(index - 1, 0)]
        tangent = nxt - prev
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length < 1e-8:
            continue
        tangent = tangent / tangent_length
        if normal is None:
            normal, bitangent = _plane_basis(tangent)
        else:
            normal = normal - tangent * float(normal @ tangent)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                normal, bitangent = _plane_basis(tangent)
            else:
                normal = normal / norm
                bitangent = np.cross(tangent, normal)
        segments = _segment_hits(vertices, faces, origin, tangent)
        loop = _stitch_loop(segments, scale)
        if loop is None:
            continue
        ring = _resample_ring(loop, origin, tangent, normal, bitangent, n_around)
        edges = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        if float(edges.min()) < 0.35 * float(np.median(edges)):
            continue
        rings.append(ring)
    direction = centers[-1] - centers[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    if len(rings) < 4:
        raise RuntimeError(f"只切出了 {len(rings)} 圈，这条模型不像一根管子")
    quads = []
    count = n_around
    for j in range(len(rings) - 1):
        for i in range(count):
            nxt = (i + 1) % count
            v0 = j * count + i
            v1 = j * count + nxt
            v2 = (j + 1) * count + nxt
            v3 = (j + 1) * count + i
            pts = np.stack([rings[j][i], rings[j][nxt], rings[j + 1][nxt], rings[j + 1][i]])
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            outward = pts.mean(axis=0) - 0.5 * (rings[j].mean(axis=0) + rings[j + 1].mean(axis=0))
            if float(normal @ outward) < 0.0:
                quads.append([v0, v3, v2, v1])
            else:
                quads.append([v0, v1, v2, v3])
    return np.vstack(rings), np.asarray(quads, dtype=np.int32), direction
