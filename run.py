"""Run Rocky with the project-level configs/config.json file."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from rocky.config import BatchConfig
from rocky.layers import RockState
from rocky.pipeline import RockGenerator

DEFAULT_CONFIG_PATH = Path("configs/config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate procedural rock and boulder assets.")
    parser.add_argument("--output-dir", help="Override the output directory from configs/config.json.")
    parser.add_argument("--count", type=int, help="Override the batch count from configs/config.json.")
    parser.add_argument("--seed", type=int, help="Override the batch seed from configs/config.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BatchConfig.from_file(DEFAULT_CONFIG_PATH)
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))
    if args.count is not None:
        config = replace(config, count=args.count)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    config.validate()

    console = Console()
    generator = RockGenerator(config)
    console.print(
        f"Rocky batch: {config.count} rocks | seed {config.seed} | formats {', '.join(config.export_formats)} | "
        f"textures {len(generator.texture_sets)} sets"
    )
    console.print(f"Output: {config.output_dir}")
    progress = RichProgress(console)
    with progress:
        states = generator.generate_batch(progress=progress)
    console.print(f"Generated {len(states)} rocks in {config.output_dir}")
    console.print(f"Preview: {config.output_dir / 'preview.png'}")
    console.print(f"Report:  {config.output_dir / 'report.md'}")
    _print_batch_summary(console, states)
    return 0


class RichProgress:
    """Rich-backed terminal progress display."""

    def __init__(self, console: Console) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.task_id: TaskID | None = None

    def __enter__(self) -> "RichProgress":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.progress.__exit__(exc_type, exc, traceback)

    def update(self, current: int, total: int, label: str) -> None:
        if self.task_id is None:
            self.task_id = self.progress.add_task(label, total=total)
        self.progress.update(self.task_id, completed=current, total=total, description=label)


def _print_batch_summary(console: Console, states: list[RockState]) -> None:
    if not states:
        return
    size_counts = Counter(state.params.size_class for state in states)
    role_counts = Counter(state.params.placement_role for state in states)
    shape_counts = Counter(state.params.shape_type for state in states)
    material_counts = Counter(state.params.material_type for state in states)
    heights = [_height(state) for state in states]
    table = Table(title="Batch Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Height range", f"{min(heights):.2f}m - {max(heights):.2f}m")
    table.add_row("Size classes", _format_counts(size_counts))
    table.add_row("Placement roles", _format_counts(role_counts))
    table.add_row("Shapes", _format_counts(shape_counts))
    table.add_row("Materials", _format_counts(material_counts))
    console.print(table)


def _height(state: RockState) -> float:
    assert state.mesh is not None
    bounds_min, bounds_max = state.mesh.bounds()
    return bounds_max.y - bounds_min.y


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
