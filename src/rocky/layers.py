"""Generation layers for high-quality procedural rock assets."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from typing import Protocol

import numpy as np
import trimesh

from rocky.geometry import Vec3
from rocky.mesh import Mesh


@dataclass(frozen=True)
class RockParameters:
    """Concrete sampled parameters for one generated rock."""

    name: str
    seed: int
    size_class: str
    archetype: str
    shape_type: str
    material_type: str
    placement_role: str
    max_height: float
    target_height: float
    diameter: float
    base_shape: str
    subdivisions: int
    radius: float
    roughness: float
    angularity: float
    spike_limit: float
    elongation: float
    floor_flattening: float
    fracture_strength: float = 0.0
    pitting_intensity: float = 0.0
    ropy_strength: float = 0.0
    fracture_count: int = 0
    crack_count: int = 0


@dataclass
class RockState:
    """Mutable state passed through pipeline layers."""

    params: RockParameters
    mesh: Mesh | None = None
    material_maps: dict[str, Path] = field(default_factory=dict)


class GenerationLayer(Protocol):
    """Pipeline layer interface."""

    name: str

    def apply(self, state: RockState) -> RockState:
        """Mutate or replace the rock state and return it."""


class TrimeshRockLayer:
    """Generates rocks using ico-sphere deformation with layered procedural masks."""

    name = "trimesh_rock"

    def apply(self, state: RockState) -> RockState:
        p = state.params
        tri_mesh = generate_rock_mesh(
            seed=p.seed,
            subdivisions=p.subdivisions,
            radius=p.radius,
            roughness=p.roughness,
            angularity=p.angularity,
            target_height=p.target_height,
            diameter=p.diameter,
            base_shape=p.base_shape,
            elongation=p.elongation,
            floor_flattening=p.floor_flattening,
            max_height=p.max_height,
            spike_limit=p.spike_limit,
            shape_type=p.shape_type,
            fracture_count=p.fracture_count,
            fracture_strength=p.fracture_strength,
            pitting_intensity=p.pitting_intensity,
            ropy_strength=p.ropy_strength,
        )
        state.mesh = _to_rocky_mesh(tri_mesh)
        return state


class UvProjectionLayer:
    """Builds per-face UVs from final deformed mesh geometry."""

    name = "uv_projection"

    def apply(self, state: RockState) -> RockState:
        assert state.mesh is not None
        mesh = state.mesh
        bounds_min, bounds_max = mesh.bounds()
        size = bounds_max - bounds_min
        scale = max(size.x, size.y, size.z, 0.001)
        tile_scale = 2.25
        face_uvs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []

        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            normal = (vb - va).cross(vc - va).normalized()
            face_uvs.append(
                (
                    _project_face_uv(va, normal, bounds_min, scale, tile_scale),
                    _project_face_uv(vb, normal, bounds_min, scale, tile_scale),
                    _project_face_uv(vc, normal, bounds_min, scale, tile_scale),
                )
            )

        mesh.face_uvs = face_uvs
        return state


def generate_rock_mesh(
    seed: int,
    subdivisions: int,
    radius: float,
    roughness: float,
    angularity: float,
    target_height: float,
    diameter: float,
    base_shape: str,
    elongation: float,
    floor_flattening: float,
    max_height: float,
    spike_limit: float,
    shape_type: str = "rounded_boulder",
    fracture_count: int = 0,
    fracture_strength: float = 0.0,
    pitting_intensity: float = 0.0,
    ropy_strength: float = 0.0,
) -> trimesh.Trimesh:
    """Generate an irregular rock mesh from an ico-sphere or subdivided box."""

    rng = np.random.default_rng(seed)
    mesh = _create_base_mesh(base_shape=base_shape, subdivisions=subdivisions, radius=radius)

    vertices = mesh.vertices.copy()
    normals = mesh.vertex_normals.copy()

    if base_shape == "box":
        vertices = _round_box_vertices(vertices, amount=0.38)

    scale = rng.uniform(0.75, 1.45, size=3)
    scale[0] *= elongation
    scale[1] *= max(0.25, target_height / max(diameter, 0.001))
    scale[2] /= max(0.35, elongation)
    vertices *= scale

    radial = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-12)
    low = multi_sine_noise_3d(radial, rng, octaves=4, base_frequency=1.4)
    medium = multi_sine_noise_3d(radial, rng, octaves=5, base_frequency=4.0)
    high = multi_sine_noise_3d(radial, rng, octaves=4, base_frequency=12.0)
    ridged = ridged_noise(radial, rng, octaves=4, base_frequency=3.0)
    strata = strata_noise(radial, rng, frequency=rng.uniform(12.0, 26.0))
    cracks = crack_mask(radial, rng)

    shape_bias = 1.25 if base_shape == "box" else 1.0
    displacement = 0.55 * low + 0.25 * medium + 0.08 * high + angularity * 0.22 * ridged + 0.08 * strata - 0.07 * cracks
    if ropy_strength > 0.0:
        displacement += _ropy_lava_displacement(vertices, rng) * ropy_strength * 0.14
    if shape_type == "flat_slab":
        displacement += strata * 0.12
    displacement = normalize01(displacement) * 2.0 - 1.0
    displacement = _limit_spikes(displacement, spike_limit)
    vertices += normals * displacement[:, None] * roughness * shape_bias
    vertices = _apply_fracture_planes(vertices, normals, rng, fracture_count, fracture_strength, angularity)
    vertices = _apply_pitting(vertices, normals, rng, pitting_intensity)
    vertices = _scale_to_dimensions(vertices, target_height=target_height, diameter=diameter, max_height=max_height)
    vertices = _flatten_floor(vertices, floor_flattening)
    vertices = _relax_spikes(vertices, mesh.faces, iterations=2 if base_shape == "box" else 2, strength=0.24 if base_shape == "box" else 0.28)

    mesh.vertices = vertices
    mesh.fix_normals()
    return mesh


def _create_base_mesh(base_shape: str, subdivisions: int, radius: float) -> trimesh.Trimesh:
    if base_shape == "box":
        mesh = trimesh.creation.box(extents=(radius * 1.8, radius * 1.2, radius * 1.8))
        for _ in range(max(1, subdivisions)):
            mesh = mesh.subdivide()
        mesh.fix_normals()
        return mesh
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def multi_sine_noise_3d(points: np.ndarray, rng: np.random.Generator, octaves: int = 5, base_frequency: float = 1.0) -> np.ndarray:
    """Lightweight pseudo-noise using many oriented sine waves."""

    points = np.asarray(points)
    result = np.zeros(len(points))
    amplitude = 1.0
    frequency = base_frequency
    amplitude_sum = 0.0

    for _ in range(octaves):
        directions = random_unit_vectors(6, rng)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
        octave = np.zeros(len(points))
        for direction, phase in zip(directions, phases, strict=True):
            octave += np.sin(points @ direction * frequency + phase)
        octave /= len(directions)
        result += amplitude * octave
        amplitude_sum += amplitude
        frequency *= 2.1
        amplitude *= 0.5

    return result / amplitude_sum


def ridged_noise(points: np.ndarray, rng: np.random.Generator, octaves: int = 4, base_frequency: float = 2.0) -> np.ndarray:
    noise = multi_sine_noise_3d(points, rng, octaves=octaves, base_frequency=base_frequency)
    return 1.0 - np.abs(noise)


def strata_noise(points: np.ndarray, rng: np.random.Generator, direction: np.ndarray | None = None, frequency: float = 18.0) -> np.ndarray:
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    base = points @ direction
    warp = multi_sine_noise_3d(points, rng, octaves=3, base_frequency=2.0)
    layers = np.sin(base * frequency + warp * 2.5)
    return _smoothstep(0.15, 0.85, normalize01(layers))


def crack_mask(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n1 = multi_sine_noise_3d(points, rng, octaves=5, base_frequency=5.0)
    n2 = multi_sine_noise_3d(points * 1.7 + 13.0, rng, octaves=4, base_frequency=9.0)
    cracks = np.abs(n1 * 0.7 + n2 * 0.3)
    return 1.0 - _smoothstep(0.05, 0.22, cracks)


def _scale_to_dimensions(vertices: np.ndarray, target_height: float, diameter: float, max_height: float) -> np.ndarray:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    span = np.maximum(bounds_max - bounds_min, 1e-9)
    xz_span = max(span[0], span[2], 1e-9)
    scale = np.array([diameter / xz_span, target_height / span[1], diameter / xz_span])
    vertices = vertices * scale
    height = vertices[:, 1].max() - vertices[:, 1].min()
    if height > max_height:
        center_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        vertices[:, 1] = center_y + (vertices[:, 1] - center_y) * (max_height / height)
    return vertices


def _round_box_vertices(vertices: np.ndarray, amount: float) -> np.ndarray:
    radius = np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-12
    rounded = vertices / radius * np.median(radius)
    return vertices * (1.0 - amount) + rounded * amount


def _limit_spikes(displacement: np.ndarray, spike_limit: float) -> np.ndarray:
    limit = max(0.05, spike_limit)
    median = np.median(displacement)
    centered = displacement - median
    threshold = np.quantile(np.abs(centered), 0.88) * limit
    return median + np.clip(centered, -threshold, threshold)


def _apply_fracture_planes(
    vertices: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    fracture_count: int,
    fracture_strength: float,
    angularity: float,
) -> np.ndarray:
    if fracture_count <= 0 or fracture_strength <= 0.0:
        return vertices
    fractured = vertices.copy()
    span = max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    count = min(fracture_count, 14)
    for _ in range(count):
        plane_normal = rng.normal(size=3)
        plane_normal /= np.linalg.norm(plane_normal) + 1e-12
        offset = rng.uniform(-0.22, 0.22) * span
        signed_distance = fractured @ plane_normal - offset
        band_width = rng.uniform(0.035, 0.085) * span
        band = np.exp(-np.abs(signed_distance) / max(band_width, 1e-9))
        side = np.sign(signed_distance)
        chip_depth = span * rng.uniform(0.010, 0.035) * fracture_strength
        shear = span * rng.uniform(0.004, 0.016) * fracture_strength * max(0.2, angularity)
        fractured -= normals * band[:, None] * chip_depth
        fractured += plane_normal * (band * side)[:, None] * shear
    return fractured


def _apply_pitting(vertices: np.ndarray, normals: np.ndarray, rng: np.random.Generator, pitting_intensity: float) -> np.ndarray:
    if pitting_intensity <= 0.0:
        return vertices
    pitted = vertices.copy()
    radial = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-12)
    span = max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    count = int(8 + pitting_intensity * 42)
    for _ in range(count):
        center = rng.normal(size=3)
        center /= np.linalg.norm(center) + 1e-12
        angular_radius = rng.uniform(0.055, 0.16) * (0.75 + pitting_intensity)
        cosine = np.clip(radial @ center, -1.0, 1.0)
        angle = np.arccos(cosine)
        depression = np.exp(-((angle / angular_radius) ** 2))
        depth = span * rng.uniform(0.006, 0.026) * pitting_intensity
        pitted -= normals * depression[:, None] * depth
    return pitted


def _ropy_lava_displacement(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    primary = np.array([math.cos(angle), 0.0, math.sin(angle)])
    secondary = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    span = max(float(np.ptp(points, axis=0).max()), 1e-9)
    normalized = points / span
    along = normalized @ primary
    across = normalized @ secondary
    warp = multi_sine_noise_3d(normalized, rng, octaves=3, base_frequency=2.5)
    waves = np.sin(across * rng.uniform(10.0, 16.0) + along * rng.uniform(1.0, 2.5) + warp * 1.6)
    broken_mask = _smoothstep(-0.4, 0.8, multi_sine_noise_3d(normalized + 9.0, rng, octaves=4, base_frequency=3.0))
    ridges = (1.0 - np.abs(waves)) * broken_mask
    return normalize01(ridges) * 2.0 - 1.0


def _relax_spikes(vertices: np.ndarray, faces: np.ndarray, iterations: int, strength: float) -> np.ndarray:
    if iterations <= 0 or strength <= 0:
        return vertices
    neighbors = [set() for _ in range(len(vertices))]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    relaxed = vertices.copy()
    for _ in range(iterations):
        updated = relaxed.copy()
        for index, linked in enumerate(neighbors):
            if not linked:
                continue
            centroid = relaxed[list(linked)].mean(axis=0)
            updated[index] = relaxed[index] * (1.0 - strength) + centroid * strength
        relaxed = updated
    return relaxed


def _flatten_floor(vertices: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return vertices
    floor_y = vertices[:, 1].min()
    height = max(vertices[:, 1].max() - floor_y, 1e-9)
    influence = 1.0 - _smoothstep(floor_y, floor_y + height * 0.20, vertices[:, 1])
    vertices[:, 1] = vertices[:, 1] * (1.0 - influence * amount) + floor_y * influence * amount
    return vertices


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _to_rocky_mesh(mesh: trimesh.Trimesh) -> Mesh:
    rocky_mesh = Mesh(
        vertices=[Vec3(float(x), float(y), float(z)) for x, y, z in mesh.vertices],
        uvs=[(0.0, 0.0) for _ in mesh.vertices],
        faces=[(int(a), int(b), int(c)) for a, b, c in mesh.faces],
    )
    rocky_mesh.recalculate_normals()
    return rocky_mesh


def _project_face_uv(vertex: Vec3, normal: Vec3, bounds_min: Vec3, scale: float, tile_scale: float) -> tuple[float, float]:
    ax, ay, az = abs(normal.x), abs(normal.y), abs(normal.z)
    if ay >= ax and ay >= az:
        u = (vertex.x - bounds_min.x) / scale
        v = (vertex.z - bounds_min.z) / scale
    elif ax >= az:
        u = (vertex.z - bounds_min.z) / scale
        v = (vertex.y - bounds_min.y) / scale
    else:
        u = (vertex.x - bounds_min.x) / scale
        v = (vertex.y - bounds_min.y) / scale
    return (u * tile_scale, v * tile_scale)
