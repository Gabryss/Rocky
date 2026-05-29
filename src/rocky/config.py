"""Configuration objects for procedural batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


NumberRange = tuple[float, float]
IntRange = tuple[int, int]


@dataclass(frozen=True)
class ParameterRanges:
    """Sampling ranges for a batch of related but distinct rocks."""

    elongation: NumberRange = (0.75, 1.55)
    roughness: NumberRange = (0.08, 0.32)
    fracture_count: IntRange = (3, 9)
    crack_count: IntRange = (5, 16)

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            low, high = value
            if low > high:
                raise ValueError(f"ranges.{name} lower bound is greater than upper bound")
            if "count" in name and int(low) < 0:
                raise ValueError(f"ranges.{name} cannot be negative")
            if "count" not in name and low < 0:
                raise ValueError(f"ranges.{name} cannot be negative")


@dataclass(frozen=True)
class BatchConfig:
    """Top-level batch generation configuration."""

    seed: int = 1337
    count: int = 24
    output_dir: Path = Path("outputs/batch_001")
    texture_dir: Path = Path("textures")
    export_formats: tuple[str, ...] = ("obj",)
    max_height: float = 2.0
    resolution_scale: float = 1.0
    ranges: ParameterRanges = field(default_factory=ParameterRanges)
    size_classes: tuple[dict[str, Any], ...] = field(default_factory=lambda: tuple(_default_size_classes()))
    archetypes: tuple[dict[str, Any], ...] = field(default_factory=lambda: tuple(_default_archetypes()))

    @classmethod
    def from_file(cls, path: str | Path) -> "BatchConfig":
        """Load and validate a batch config from JSON."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls.from_mapping(raw)
        config.validate()
        return config

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "BatchConfig":
        """Build a config from a plain mapping, applying defaults for omissions."""

        ranges = ParameterRanges(
            **{
                key: _range_tuple(value)
                for key, value in raw.get("ranges", {}).items()
            }
        )
        return cls(
            seed=int(raw.get("seed", 1337)),
            count=int(raw.get("count", 24)),
            output_dir=Path(raw.get("output_dir", "outputs/batch_001")),
            texture_dir=Path(raw.get("texture_dir", "textures")),
            export_formats=tuple(raw.get("export_formats", ("obj",))),
            max_height=float(raw.get("max_height", 2.0)),
            resolution_scale=float(raw.get("resolution_scale", 1.0)),
            ranges=ranges,
            size_classes=tuple(raw.get("size_classes", _default_size_classes())),
            archetypes=tuple(raw.get("archetypes", _default_archetypes())),
        )

    def validate(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        if self.max_height <= 0:
            raise ValueError("max_height must be positive")
        if self.resolution_scale <= 0:
            raise ValueError("resolution_scale must be positive")
        invalid_formats = set(self.export_formats) - {"obj", "glb"}
        if invalid_formats:
            raise ValueError(f"unsupported export formats: {sorted(invalid_formats)}")
        self.ranges.validate()
        _validate_weighted_profiles("size_classes", self.size_classes)
        _validate_weighted_profiles("archetypes", self.archetypes)


def _range_tuple(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"range values must be two-item arrays, got {value!r}")
    return (value[0], value[1])


def _validate_weighted_profiles(name: str, profiles: tuple[dict[str, Any], ...]) -> None:
    if not profiles:
        raise ValueError(f"{name} must contain at least one profile")
    for profile in profiles:
        if "name" not in profile:
            raise ValueError(f"{name} profiles require a name")
        if float(profile.get("weight", 1.0)) <= 0:
            raise ValueError(f"{name}.{profile['name']} weight must be positive")


def _default_size_classes() -> list[dict[str, Any]]:
    return [
        {"name": "floor_pebble", "weight": 8, "height": [0.03, 0.12], "diameter": [0.04, 0.18], "floor_flattening": 0.75, "subdivisions": 1},
        {"name": "floor_cobble", "weight": 10, "height": [0.10, 0.35], "diameter": [0.12, 0.55], "floor_flattening": 0.65, "subdivisions": 2},
        {"name": "step_rock", "weight": 5, "height": [0.35, 0.85], "diameter": [0.35, 1.15], "floor_flattening": 0.35, "subdivisions": 3},
        {"name": "rover_obstacle", "weight": 3, "height": [0.85, 1.85], "diameter": [0.75, 2.4], "floor_flattening": 0.2, "subdivisions": 4},
    ]


def _default_archetypes() -> list[dict[str, Any]]:
    return [
        {
            "name": "smooth_basalt",
            "weight": 5,
            "shape_type": "rounded_boulder",
            "material_type": "dark_basalt",
            "base_shape": "icosphere",
            "roughness": 0.45,
            "fractures": 0.45,
            "cracks": 0.45,
            "angularity": 0.35,
            "spike_limit": 0.7,
            "fracture_strength": 0.025,
        },
        {
            "name": "rough_basalt",
            "weight": 6,
            "shape_type": "angular_boulder",
            "material_type": "dark_basalt",
            "base_shape": "icosphere",
            "roughness": 1.05,
            "fractures": 0.9,
            "cracks": 0.9,
            "angularity": 0.7,
            "spike_limit": 0.85,
            "fracture_strength": 0.065,
        },
        {
            "name": "fractured_block",
            "weight": 5,
            "shape_type": "collapsed_ceiling_block",
            "material_type": "fractured_cliff",
            "base_shape": "box",
            "roughness": 0.7,
            "fractures": 1.55,
            "cracks": 1.35,
            "angularity": 1.2,
            "spike_limit": 0.78,
            "fracture_strength": 0.14,
        },
        {
            "name": "flat_lava_slab",
            "weight": 4,
            "shape_type": "flat_slab",
            "material_type": "layered_cliff",
            "base_shape": "box",
            "roughness": 0.58,
            "fractures": 1.25,
            "cracks": 1.15,
            "angularity": 0.9,
            "spike_limit": 0.72,
            "fracture_strength": 0.1,
        },
        {
            "name": "vesicular_lava",
            "weight": 4,
            "shape_type": "vesicular_chunk",
            "material_type": "porous_lava",
            "base_shape": "icosphere",
            "roughness": 0.95,
            "fractures": 0.8,
            "cracks": 1.1,
            "angularity": 0.62,
            "spike_limit": 0.82,
            "fracture_strength": 0.06,
            "pitting_intensity": 0.95,
        },
        {
            "name": "ropy_lava_clast",
            "weight": 3,
            "shape_type": "ropy_lava_fragment",
            "material_type": "porous_lava",
            "base_shape": "icosphere",
            "roughness": 0.85,
            "fractures": 0.65,
            "cracks": 0.85,
            "angularity": 0.5,
            "spike_limit": 0.82,
            "fracture_strength": 0.04,
            "ropy_strength": 0.38,
        },
        {
            "name": "weird_erosion",
            "weight": 2,
            "shape_type": "eroded_irregular",
            "material_type": "dry_boulder",
            "base_shape": "icosphere",
            "roughness": 0.95,
            "fractures": 0.75,
            "cracks": 1.1,
            "angularity": 0.95,
            "spike_limit": 0.78,
            "fracture_strength": 0.08,
            "pitting_intensity": 0.35,
        },
    ]
