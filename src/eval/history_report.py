from __future__ import annotations

from pathlib import Path

from .history_loader import StoredRun, load_history_runs, load_latest_run_id, select_comparable_runs
from .readme_renderer import build_history_readme_block, replace_history_block
from .schemas import RunTrack
from .svg_renderer import build_history_svg


def refresh_history_report(
    *,
    output_root: Path,
    readme_path: Path,
    svg_path: Path,
    track: RunTrack = "release",
) -> tuple[StoredRun, list[StoredRun]]:
    runs = load_history_runs(output_root)
    latest_run_id = load_latest_run_id(output_root, track=track)
    latest, comparable_runs = select_comparable_runs(
        runs,
        track=track,
        latest_run_id=latest_run_id,
    )

    readme_block = build_history_readme_block(
        track=track,
        latest=latest,
        comparable_runs=comparable_runs,
        readme_path=readme_path,
        output_root=output_root,
        svg_path=svg_path,
    )
    svg_content = build_history_svg(comparable_runs)

    readme_text = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(
        replace_history_block(readme_text, readme_block),
        encoding="utf-8",
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_content, encoding="utf-8")
    return latest, comparable_runs
