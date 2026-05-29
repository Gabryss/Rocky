"""Mesh data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocky.geometry import Vec3


@dataclass
class Mesh:
    """Triangle mesh with simple UVs and per-vertex normals."""

    vertices: list[Vec3] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    face_uvs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)

    def recalculate_normals(self) -> None:
        accum = [Vec3(0.0, 0.0, 0.0) for _ in self.vertices]
        for a, b, c in self.faces:
            va = self.vertices[a]
            vb = self.vertices[b]
            vc = self.vertices[c]
            normal = (vb - va).cross(vc - va).normalized()
            accum[a] = accum[a] + normal
            accum[b] = accum[b] + normal
            accum[c] = accum[c] + normal
        self.normals = [normal.normalized() for normal in accum]

    def bounds(self) -> tuple[Vec3, Vec3]:
        xs = [vertex.x for vertex in self.vertices]
        ys = [vertex.y for vertex in self.vertices]
        zs = [vertex.z for vertex in self.vertices]
        return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))

    def vertex_count(self) -> int:
        return len(self.vertices)

    def face_count(self) -> int:
        return len(self.faces)
