"""Batch report generation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from rocky.layers import RockState


class BatchReporter:
    """Writes machine-readable and human-readable summaries."""

    def write_rock_info(self, state: RockState, output_dir: Path, exports: dict[str, Path]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        info_path = output_dir / "info.json"
        payload = self._state_payload(state)
        payload["exports"] = {name: str(path) for name, path in exports.items()}
        payload["placement"] = self._placement_payload(state)
        info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return info_path

    def write(self, states: list[RockState], output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "report.json"
        markdown_path = output_dir / "report.md"
        payload = {
            "count": len(states),
            "rocks": [self._state_payload(state) for state in states],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path.write_text(self._markdown(payload), encoding="utf-8")
        return json_path, markdown_path

    def _state_payload(self, state: RockState) -> dict[str, Any]:
        assert state.mesh is not None
        bounds_min, bounds_max = state.mesh.bounds()
        return {
            "name": state.params.name,
            "seed": state.params.seed,
            "mesh": {
                "vertices": state.mesh.vertex_count(),
                "faces": state.mesh.face_count(),
                "bounds_min": bounds_min.as_tuple(),
                "bounds_max": bounds_max.as_tuple(),
            },
            "parameters": asdict(state.params),
            "fractures": state.params.fracture_count,
            "cracks": state.params.crack_count,
            "materials": {name: str(path) for name, path in state.material_maps.items()},
        }

    def _placement_payload(self, state: RockState) -> dict[str, Any]:
        assert state.mesh is not None
        bounds_min, bounds_max = state.mesh.bounds()
        width_x = bounds_max.x - bounds_min.x
        height = bounds_max.y - bounds_min.y
        depth_z = bounds_max.z - bounds_min.z
        footprint_radius = max(width_x, depth_z) * 0.5
        return {
            "label": f"{state.params.placement_role}/{state.params.shape_type}",
            "source_label": f"{state.params.size_class}/{state.params.archetype}",
            "size_class": state.params.size_class,
            "archetype": state.params.archetype,
            "shape_type": state.params.shape_type,
            "material_type": state.params.material_type,
            "placement_role": state.params.placement_role,
            "base_shape": state.params.base_shape,
            "height_m": height,
            "width_x_m": width_x,
            "depth_z_m": depth_z,
            "footprint_radius_m": footprint_radius,
            "can_scatter_on_floor": state.params.size_class.startswith("floor_"),
            "is_rover_obstacle": state.params.placement_role == "navigation_obstacle",
            "drive_over": state.params.placement_role in {"floor_scatter", "wheel_hazard"} and height < 0.35,
            "collision_category": state.params.placement_role,
            "roughness_score": state.params.roughness,
            "sharpness_score": state.params.angularity,
            "orientation": self._orientation_payload(state, height, width_x, depth_z, bounds_min.y),
            "max_height_m": state.params.max_height,
        }

    def _orientation_payload(self, state: RockState, height: float, width_x: float, depth_z: float, bottom_y: float) -> dict[str, Any]:
        shape = state.params.shape_type
        role = state.params.placement_role
        flatness = height / max(width_x, depth_z, 1e-9)
        max_pitch_roll = _max_pitch_roll_degrees(shape, role, flatness)
        return {
            "local_up_axis": "Y",
            "stable_axis": "Y",
            "bottom_y": bottom_y,
            "align_local_up_to_terrain_normal": True,
            "allow_random_yaw": True,
            "yaw_range_deg": [0.0, 360.0],
            "max_pitch_deg": max_pitch_roll,
            "max_roll_deg": max_pitch_roll,
            "keep_largest_footprint_down": shape in {"flat_slab", "collapsed_ceiling_block", "ropy_lava_fragment"},
            "flatness_ratio": flatness,
            "stability_hint": _stability_hint(shape, role, flatness),
        }

    def _markdown(self, payload: dict[str, Any]) -> str:
        lines = [
            "# Rocky Batch Report",
            "",
            f"Generated rocks: {payload['count']}",
            "",
            "| Rock | Role | Shape | Material | Class | Archetype | Height | Vertices | Faces | Fractures | Cracks |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for rock in payload["rocks"]:
            bounds_min = rock["mesh"]["bounds_min"]
            bounds_max = rock["mesh"]["bounds_max"]
            height = bounds_max[1] - bounds_min[1]
            params = rock["parameters"]
            lines.append(
                f"| {rock['name']} | {params['placement_role']} | {params['shape_type']} | {params['material_type']} | "
                f"{params['size_class']} | {params['archetype']} | {height:.2f}m | {rock['mesh']['vertices']} | "
                f"{rock['mesh']['faces']} | {rock['fractures']} | {rock['cracks']} |"
            )
        lines.append("")
        lines.append("Inspect `preview.png` for the batch contact sheet and each exported asset folder for OBJ/MTL or GLB files.")
        return "\n".join(lines) + "\n"


def _max_pitch_roll_degrees(shape: str, role: str, flatness: float) -> float:
    if shape == "flat_slab" or flatness < 0.22:
        return 6.0
    if role == "navigation_obstacle":
        return 10.0
    if shape in {"collapsed_ceiling_block", "ropy_lava_fragment"}:
        return 12.0
    if role == "wheel_hazard":
        return 16.0
    return 22.0


def _stability_hint(shape: str, role: str, flatness: float) -> str:
    if shape == "flat_slab" or flatness < 0.22:
        return "lay_flat"
    if role == "navigation_obstacle":
        return "upright_on_flattened_base"
    if shape in {"collapsed_ceiling_block", "ropy_lava_fragment"}:
        return "prefer_broad_face_down"
    return "terrain_aligned"
