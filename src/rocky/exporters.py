"""Asset exporters for generated rocks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from rocky.layers import RockState
from rocky.mesh import Mesh


class ObjExporter:
    """Writes OBJ and MTL files for a generated rock mesh."""

    def export(self, state: RockState, output_dir: Path) -> Path:
        assert state.mesh is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        obj_path = output_dir / f"{state.params.name}.obj"
        mtl_path = output_dir / f"{state.params.name}.mtl"
        self._write_mtl(mtl_path, state.material_maps, output_dir)
        self._write_obj(obj_path, mtl_path.name, state.mesh)
        return obj_path

    def _write_obj(self, path: Path, mtl_name: str, mesh: Mesh) -> None:
        lines = [f"mtllib {mtl_name}", "usemtl rock_material"]
        for vertex in mesh.vertices:
            lines.append(f"v {vertex.x:.6f} {vertex.y:.6f} {vertex.z:.6f}")

        face_records: list[tuple[tuple[int, int, int], tuple[int, int, int], int]] = []
        for face_index, (a, b, c) in enumerate(mesh.faces):
            face_uvs = mesh.face_uvs[face_index] if face_index < len(mesh.face_uvs) else (mesh.uvs[a], mesh.uvs[b], mesh.uvs[c])
            vt_indices: list[int] = []
            for u, v in face_uvs:
                lines.append(f"vt {u:.6f} {1.0 - v:.6f}")
                vt_indices.append(len(vt_indices) + 1 + face_index * 3)

            va = mesh.vertices[a]
            vb = mesh.vertices[b]
            vc = mesh.vertices[c]
            normal = (vb - va).cross(vc - va).normalized()
            lines.append(f"vn {normal.x:.6f} {normal.y:.6f} {normal.z:.6f}")
            face_records.append(((a + 1, b + 1, c + 1), (vt_indices[0], vt_indices[1], vt_indices[2]), face_index + 1))

        lines.append("s off")
        for vertex_indices, uv_indices, normal_index in face_records:
            a, b, c = vertex_indices
            au, bu, cu = uv_indices
            lines.append(f"f {a}/{au}/{normal_index} {b}/{bu}/{normal_index} {c}/{cu}/{normal_index}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_mtl(self, path: Path, material_maps: dict[str, Path], output_dir: Path) -> None:
        lines = [
            "newmtl rock_material",
            "Ka 0.15 0.15 0.15",
            "Kd 0.82 0.82 0.82",
            "Ks 0.04 0.04 0.04",
            "Ns 18.0",
        ]
        if "diffuse" in material_maps:
            lines.append(f"map_Kd {_relative_texture_path(material_maps['diffuse'], output_dir)}")
        if "normal" in material_maps:
            lines.append(f"map_Bump {_relative_texture_path(material_maps['normal'], output_dir)}")
        if "roughness" in material_maps:
            lines.append(f"map_Pr {_relative_texture_path(material_maps['roughness'], output_dir)}")
        if "displacement" in material_maps:
            lines.append(f"disp {_relative_texture_path(material_maps['displacement'], output_dir)}")
        content = "\n".join(lines)
        path.write_text(content + "\n", encoding="utf-8")


class GlbExporter:
    """Writes lightweight GLB mesh exports without embedding 4K texture maps."""

    def export(self, state: RockState, output_dir: Path) -> Path:
        assert state.mesh is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{state.params.name}.glb"
        vertices = np.array([vertex.as_tuple() for vertex in state.mesh.vertices], dtype=np.float64)
        faces = np.array(state.mesh.faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(path)
        return path


class MaterialRecipeExporter:
    """Writes a renderer-agnostic material recipe for downstream importers."""

    def export(self, state: RockState, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "material.json"
        path.write_text(json.dumps(self._payload(state, output_dir), indent=2), encoding="utf-8")
        return path

    def _payload(self, state: RockState, output_dir: Path) -> dict[str, object]:
        maps = {
            name: _material_map_payload(name, texture_path, output_dir)
            for name, texture_path in state.material_maps.items()
        }
        return {
            "schema": "rocky.material.v1",
            "material_name": "rock_material",
            "rock_name": state.params.name,
            "material_type": state.params.material_type,
            "shape_type": state.params.shape_type,
            "shader_model": "pbr_metallic_roughness",
            "uv": {
                "source": "mesh_uv",
                "projection": "generated_triplanar_like_face_projection",
                "tiling": [2.25, 2.25],
                "wrap": "repeat",
            },
            "pbr": {
                "base_color": [1.0, 1.0, 1.0, 1.0],
                "metallic": 0.0,
                "roughness_default": 0.86,
                "normal_strength": 1.0,
                "displacement_midlevel": 0.5,
                "displacement_scale_m": 0.085,
            },
            "maps": maps,
            "import_notes": {
                "diffuse": "Use as base color/albedo.",
                "normal": "OpenGL tangent-space normal when the filename contains '_nor_gl'.",
                "roughness": "Use as non-color scalar roughness.",
                "displacement": "Use as non-color height/displacement. Apply scale conservatively for simulation.",
            },
        }


class TextureReferenceExporter:
    """Registers a selected texture set without copying heavy source files."""

    def export(self, state: RockState, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if state.material_maps:
            return state.material_maps.get("diffuse", next(iter(state.material_maps.values())))

        path = output_dir / f"{state.params.name}_diffuse.png"
        image = Image.fromarray(np.full((64, 64, 3), (95, 92, 84), dtype=np.uint8), mode="RGB")
        image.save(path, format="PNG", optimize=True)
        state.material_maps = {"diffuse": path}
        return path


def _relative_texture_path(texture_path: Path, output_dir: Path) -> str:
    try:
        relative = os.path.relpath(texture_path.resolve(), output_dir.resolve())
    except ValueError:
        relative = str(texture_path)
    return Path(relative).as_posix()


def _material_map_payload(name: str, texture_path: Path, output_dir: Path) -> dict[str, object]:
    color_space = "sRGB" if name == "diffuse" else "linear"
    usage_by_name = {
        "diffuse": "base_color",
        "normal": "normal",
        "roughness": "roughness",
        "displacement": "displacement",
    }
    payload: dict[str, object] = {
        "path": _relative_texture_path(texture_path, output_dir),
        "usage": usage_by_name.get(name, name),
        "color_space": color_space,
    }
    if name == "normal":
        payload["normal_convention"] = "OpenGL" if "_gl" in texture_path.name.lower() else "unknown"
    return payload
