"""PNG batch previews."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rocky.geometry import Vec3
from rocky.layers import RockState


class PngPreviewRenderer:
    """Renders a PNG contact sheet for inspecting generated batches."""

    def __init__(self, cell_size: int = 260) -> None:
        self.cell_size = cell_size
        self._texture_cache: dict[Path, Image.Image] = {}

    def render_batch(self, states: list[RockState], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = min(3, max(1, len(states)))
        rows = (len(states) + columns - 1) // columns
        width = columns * self.cell_size
        height = rows * self.cell_size
        image = Image.new("RGB", (width, height), "#f1f0ea")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for index, state in enumerate(states):
            col = index % columns
            row = index // columns
            self._render_cell(draw, state, col * self.cell_size, row * self.cell_size, font)

        image.save(output_path, format="PNG", optimize=True)
        return output_path

    def _render_cell(self, draw: ImageDraw.ImageDraw, state: RockState, ox: int, oy: int, font: ImageFont.ImageFont) -> None:
        assert state.mesh is not None
        mesh = state.mesh
        bounds_min, bounds_max = mesh.bounds()
        center = (bounds_min + bounds_max) * 0.5
        span = max((bounds_max - bounds_min).length(), 0.001)
        scale = self.cell_size * 1.25 / span

        def project(vertex: Vec3) -> tuple[float, float]:
            v = vertex - center
            x = (v.x - v.z * 0.42) * scale + ox + self.cell_size * 0.5
            y = (-v.y + (v.x + v.z) * 0.14) * scale + oy + self.cell_size * 0.53
            return x, y

        draw.rectangle((ox, oy, ox + self.cell_size - 1, oy + self.cell_size - 1), fill="#f7f6f0", outline="#d2cec1")
        polygons: list[tuple[float, tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[int, int, int]]] = []
        light = Vec3(-0.35, 0.85, 0.38).normalized()
        diffuse = self._load_diffuse_texture(state)
        for face_index, (a, b, c) in enumerate(mesh.faces):
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            normal = (vb - va).cross(vc - va).normalized()
            if normal.z < -0.78:
                continue
            shade = max(0.2, min(1.0, 0.45 + 0.55 * normal.dot(light)))
            base_color = _face_texture_color(diffuse, mesh.face_uvs[face_index] if face_index < len(mesh.face_uvs) else None)
            color = _shade_rgb(base_color, shade)
            depth = (va.z + vb.z + vc.z) / 3.0
            polygons.append((depth, (project(va), project(vb), project(vc)), color))

        for _, points, color in sorted(polygons):
            draw.polygon(points, fill=color, outline=(43, 41, 36))

        label_y = oy + self.cell_size - 42
        size = bounds_max - bounds_min
        draw.rectangle((ox + 6, label_y - 2, ox + self.cell_size - 6, oy + self.cell_size - 6), fill="#f7f6f0")
        draw.text((ox + 12, label_y), state.params.name, fill="#2b2924", font=font)
        draw.text((ox + 12, label_y + 12), f"{state.params.placement_role} / {state.params.shape_type}", fill="#5b574f", font=font)
        draw.text((ox + 12, label_y + 24), f"h {size.y:.2f}m  footprint {size.x:.2f}m x {size.z:.2f}m", fill="#5b574f", font=font)

    def _load_diffuse_texture(self, state: RockState) -> Image.Image | None:
        diffuse_path = state.material_maps.get("diffuse")
        if diffuse_path is None:
            return None
        if diffuse_path not in self._texture_cache:
            try:
                self._texture_cache[diffuse_path] = Image.open(diffuse_path).convert("RGB")
            except OSError:
                return None
        return self._texture_cache[diffuse_path]


def _shade_rgb(color: tuple[int, int, int], shade: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * shade))) for channel in color)


def _face_texture_color(
    texture: Image.Image | None,
    face_uvs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
) -> tuple[int, int, int]:
    if texture is None or face_uvs is None:
        return (119, 115, 101)
    u = sum(uv[0] for uv in face_uvs) / 3.0
    v = sum(uv[1] for uv in face_uvs) / 3.0
    wrapped_u = u % 1.0
    wrapped_v = v % 1.0
    x = min(texture.width - 1, max(0, int(wrapped_u * texture.width)))
    y = min(texture.height - 1, max(0, int((1.0 - wrapped_v) * texture.height)))
    r, g, b = texture.getpixel((x, y))
    return (int(r), int(g), int(b))
