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
    offset = loop - origin
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


def retopo_tube(vertices: np.ndarray, faces: np.ndarray, n_rings: int = 28, n_around: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quad vertices, quad faces, and a unit axis direction."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    center, direction = long_axis(vertices)
    tangent, bitangent = _plane_basis(direction)
    coords = (vertices - center) @ direction
    low, high = np.quantile(coords, [0.03, 0.97])
    stations = np.linspace(low, high, n_rings)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    rings = []
    for station in stations:
        origin = center + direction * float(station)
        segments = _segment_hits(vertices, faces, origin, direction)
        loop = _stitch_loop(segments, scale)
        if loop is None:
            continue
        rings.append(_resample_ring(loop, origin, direction, tangent, bitangent, n_around))
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
            quads.append([v0, v1, v2, v3])
    return np.vstack(rings), np.asarray(quads, dtype=np.int32), direction
