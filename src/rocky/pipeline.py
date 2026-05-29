"""Rock generation pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
import random
from pathlib import Path
from typing import Any, Protocol

from rocky.config import BatchConfig
from rocky.exporters import GlbExporter, MaterialRecipeExporter, ObjExporter, TextureReferenceExporter
from rocky.layers import (
    GenerationLayer,
    RockParameters,
    RockState,
    TrimeshRockLayer,
    UvProjectionLayer,
)
from rocky.preview import PngPreviewRenderer
from rocky.report import BatchReporter


class ProgressReporter(Protocol):
    """Receives batch generation progress events."""

    def update(self, current: int, total: int, label: str) -> None:
        """Handle one progress update."""


class RockGenerator:
    """Object-oriented facade for generating complete rock batches."""

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.layers: list[GenerationLayer] = [
            TrimeshRockLayer(),
            UvProjectionLayer(),
        ]
        self.obj_exporter = ObjExporter()
        self.glb_exporter = GlbExporter()
        self.material_exporter = MaterialRecipeExporter()
        self.texture_exporter = TextureReferenceExporter()
        self.preview_renderer = PngPreviewRenderer()
        self.reporter = BatchReporter()
        self.texture_sets = _discover_texture_sets(config.texture_dir)

    def generate_batch(self, progress: ProgressReporter | None = None) -> list[RockState]:
        """Generate, export, preview, and report a configured batch."""

        self.config.validate()
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        states: list[RockState] = []
        rng = random.Random(self.config.seed)
        total_steps = self.config.count * (sum(_layer_steps(layer) for layer in self.layers) + 2) + 2
        completed_steps = 0
        for index in range(self.config.count):
            params = self._sample_parameters(index, rng)
            descriptor = _rock_progress_label(params)
            _notify(progress, completed_steps, total_steps, f"{descriptor} | sampling")
            state = RockState(params=params)
            for layer in self.layers:
                if hasattr(layer, "apply_with_progress"):
                    state = layer.apply_with_progress(
                        state,
                        lambda row, row_total, layer=layer, params=params: _notify(
                            progress,
                            completed_steps + row,
                            total_steps,
                            f"{_rock_progress_label(params)} | {layer.name} {row}/{row_total}",
                        ),
                    )
                    completed_steps += _layer_steps(layer)
                else:
                    state = layer.apply(state)
                    completed_steps += 1
                _notify(progress, completed_steps, total_steps, f"{descriptor} | {layer.name}")
            asset_dir = output_dir / params.name
            state.material_maps = _choose_texture_set(self.texture_sets, rng, params.material_type)
            self.texture_exporter.export(state, asset_dir)
            completed_steps += 1
            texture_name = _texture_label(state.material_maps)
            _notify(progress, completed_steps, total_steps, f"{descriptor} | texture {texture_name}")
            exports = self._export_assets(state, asset_dir)
            self.reporter.write_rock_info(state, asset_dir, exports)
            completed_steps += 1
            _notify(progress, completed_steps, total_steps, f"{descriptor} | exported {', '.join(exports) or 'metadata'}")
            states.append(state)

        self.preview_renderer.render_batch(states, output_dir / "preview.png")
        completed_steps += 1
        _notify(progress, completed_steps, total_steps, f"preview | {len(states)} labeled rocks")
        self.reporter.write(states, output_dir)
        completed_steps += 1
        _notify(progress, completed_steps, total_steps, f"report | {output_dir}")
        return states

    def _export_assets(self, state: RockState, asset_dir: Path) -> dict[str, Path]:
        exports: dict[str, Path] = {"material": self.material_exporter.export(state, asset_dir)}
        formats = set(self.config.export_formats)
        if "obj" in formats:
            exports["obj"] = self.obj_exporter.export(state, asset_dir)
        if "glb" in formats:
            exports["glb"] = self.glb_exporter.export(state, asset_dir)
        return exports

    def generate_one(self, params: RockParameters) -> RockState:
        """Run all configured layers for one rock."""

        state = RockState(params=params)
        for layer in self.layers:
            state = layer.apply(state)
        return state

    def describe_layers(self) -> list[dict[str, str]]:
        """Return a simple description of the active pipeline order."""

        return [{"index": str(index), "name": layer.name, "class": layer.__class__.__name__} for index, layer in enumerate(self.layers)]

    def _sample_parameters(self, index: int, rng: random.Random) -> RockParameters:
        ranges = self.config.ranges
        seed = rng.randrange(1, 2**31 - 1)
        size_class = _choose_weighted(self.config.size_classes, rng)
        archetype = _choose_weighted(self.config.archetypes, rng)
        height = min(rng.uniform(*_profile_range(size_class, "height", (0.1, 1.0))), self.config.max_height)
        diameter = rng.uniform(*_profile_range(size_class, "diameter", (height * 0.7, height * 1.7)))
        roughness_mult = float(archetype.get("roughness", 1.0))
        fracture_mult = float(archetype.get("fractures", 1.0))
        crack_mult = float(archetype.get("cracks", 1.0))
        angularity = max(0.0, float(archetype.get("angularity", 0.65)))
        shape_type = str(archetype.get("shape_type", archetype["name"]))
        material_type = str(archetype.get("material_type", _infer_material_type(str(archetype["name"]))))
        placement_role = str(archetype.get("placement_role", _infer_placement_role(str(size_class["name"]))))
        target_height = min(height, self.config.max_height)
        base_subdivisions = int(size_class.get("subdivisions", 2 if target_height < 0.35 else 3))
        subdivisions = _scaled_subdivisions(base_subdivisions, self.config.resolution_scale)
        base_shape = str(archetype.get("base_shape", "icosphere"))
        spike_limit = float(archetype.get("spike_limit", 1.0))
        fracture_count = max(0, round(rng.randint(int(ranges.fracture_count[0]), int(ranges.fracture_count[1])) * fracture_mult))
        crack_count = max(0, round(rng.randint(int(ranges.crack_count[0]), int(ranges.crack_count[1])) * crack_mult))
        floor_flattening = float(size_class.get("floor_flattening", 0.55 if "floor" in str(size_class["name"]) else 0.25))
        elongation = rng.uniform(*ranges.elongation) * rng.uniform(0.75, 1.25)
        if shape_type == "flat_slab":
            diameter *= rng.uniform(1.15, 1.45)
            target_height = min(target_height, diameter * rng.uniform(0.12, 0.28))
            floor_flattening = max(floor_flattening, 0.55)
        elif shape_type == "ropy_lava_fragment":
            elongation *= rng.uniform(1.25, 1.75)
            floor_flattening = max(floor_flattening, 0.45)
        return RockParameters(
            name=f"rock_{index:03d}",
            seed=seed,
            size_class=str(size_class["name"]),
            archetype=str(archetype["name"]),
            shape_type=shape_type,
            material_type=material_type,
            placement_role=placement_role,
            max_height=self.config.max_height,
            target_height=target_height,
            diameter=diameter,
            base_shape=base_shape,
            subdivisions=subdivisions,
            radius=max(target_height, diameter) * 0.5,
            roughness=rng.uniform(*ranges.roughness) * roughness_mult * max(target_height, diameter),
            angularity=angularity,
            spike_limit=spike_limit,
            elongation=elongation,
            floor_flattening=floor_flattening,
            fracture_strength=float(archetype.get("fracture_strength", 0.08 if fracture_count else 0.0)),
            pitting_intensity=float(archetype.get("pitting_intensity", 0.0)),
            ropy_strength=float(archetype.get("ropy_strength", 0.0)),
            fracture_count=fracture_count,
            crack_count=crack_count,
        )

    def layer_manifest(self, output_path: str | Path) -> Path:
        """Write pipeline layer metadata for documentation or debugging."""

        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"layers": self.describe_layers(), "config": asdict(self.config)}, default=str, indent=2), encoding="utf-8")
        return path


def _notify(progress: ProgressReporter | None, current: int, total: int, label: str) -> None:
    if progress is not None:
        progress.update(current, total, label)


def _rock_progress_label(params: RockParameters) -> str:
    return (
        f"{params.name} | {params.placement_role}/{params.shape_type} | {params.base_shape} | "
        f"h={params.target_height:.2f}m d={params.diameter:.2f}m"
    )


def _texture_label(material_maps: dict[str, Path]) -> str:
    diffuse = material_maps.get("diffuse")
    if diffuse is None:
        return "fallback"
    return diffuse.parent.parent.name if diffuse.parent.name == "textures" else diffuse.stem


def _layer_steps(layer: GenerationLayer) -> int:
    return 1


def _choose_weighted(profiles: tuple[dict[str, Any], ...], rng: random.Random) -> dict[str, Any]:
    total = sum(float(profile.get("weight", 1.0)) for profile in profiles)
    pick = rng.uniform(0.0, total)
    cursor = 0.0
    for profile in profiles:
        cursor += float(profile.get("weight", 1.0))
        if pick <= cursor:
            return profile
    return profiles[-1]


def _profile_range(profile: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = profile.get(key, default)
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    return (float(value[0]), float(value[1]))


def _scaled_subdivisions(base_subdivisions: int, resolution_scale: float) -> int:
    scaled = round(base_subdivisions * resolution_scale)
    return max(0, min(6, scaled))


def _discover_texture_sets(texture_dir: Path) -> list[dict[str, Path]]:
    if not texture_dir.exists():
        return []
    sets: list[dict[str, Path]] = []
    for diffuse in sorted(texture_dir.rglob("*_diff_*")):
        if diffuse.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem_prefix = diffuse.name.split("_diff_")[0]
        siblings = list(diffuse.parent.iterdir())
        maps: dict[str, Path] = {"diffuse": diffuse}
        for sibling in siblings:
            lower = sibling.name.lower()
            if not lower.startswith(stem_prefix.lower()):
                continue
            if "_nor_" in lower or "_normal_" in lower:
                maps["normal"] = sibling
            elif "_rough_" in lower:
                maps["roughness"] = sibling
            elif "_disp_" in lower or "_height_" in lower:
                maps["displacement"] = sibling
        sets.append(maps)
    return sets


def _choose_texture_set(texture_sets: list[dict[str, Path]], rng: random.Random, material_type: str = "") -> dict[str, Path]:
    if not texture_sets:
        return {}
    preferred = [texture_set for texture_set in texture_sets if _texture_matches_material(texture_set, material_type)]
    return dict(rng.choice(preferred or texture_sets))


def _texture_matches_material(texture_set: dict[str, Path], material_type: str) -> bool:
    diffuse = texture_set.get("diffuse")
    if diffuse is None:
        return False
    name = diffuse.as_posix().lower()
    material = material_type.lower()
    if material in {"dark_basalt", "porous_lava"}:
        return "rock_" in name or "boulder" in name or "surface" in name
    if material in {"layered_cliff", "fractured_cliff"}:
        return "cliff" in name or "surface" in name or "rock_" in name
    if material in {"dry_boulder", "dusty_surface"}:
        return "dry" in name or "boulder" in name or "surface" in name
    return False


def _infer_material_type(archetype_name: str) -> str:
    name = archetype_name.lower()
    if "vesicular" in name or "lava" in name:
        return "porous_lava"
    if "fractured" in name or "block" in name:
        return "fractured_cliff"
    if "smooth" in name:
        return "dark_basalt"
    return "dry_boulder"


def _infer_placement_role(size_class_name: str) -> str:
    name = size_class_name.lower()
    if name.startswith("floor_"):
        return "floor_scatter"
    if "obstacle" in name:
        return "navigation_obstacle"
    return "wheel_hazard"
