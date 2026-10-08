"""Locate the assets installed by the isaaclab-skill overlay."""

from pathlib import Path


def skill_asset_path(relative_path: str) -> str:
    """Return an asset path relative to the installed Isaac Lab checkout."""
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "isaaclab.sh").is_file())
    return str(root / "skill_assets" / relative_path)
