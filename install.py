#!/usr/bin/env python3
"""Verify or install the selected Isaac Lab overlay and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((PACKAGE_ROOT / "manifest.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package() -> None:
    for relative, expected in MANIFEST["files"].items():
        path = PACKAGE_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Package file is missing or changed: {relative}")
    print(f"Verified {len(MANIFEST['files'])} package files.")


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def install(target: Path, dry_run: bool) -> None:
    target = target.resolve()
    if not (target / "isaaclab.sh").is_file():
        raise SystemExit(f"Not an Isaac Lab checkout: {target}")
    revision = git_output(target, "rev-parse", "HEAD").decode().strip()
    if not revision.startswith(MANIFEST["upstream_commit"]):
        raise SystemExit(
            f"Isaac Lab revision {revision[:12]} differs from required "
            f"{MANIFEST['upstream_commit']}. Use a clean checkout of that revision."
        )

    pending: list[tuple[Path, Path]] = []
    for relative in MANIFEST["files"]:
        source = PACKAGE_ROOT / relative
        destination = target / Path(relative).relative_to(relative.split("/", 1)[0])
        if destination.is_file() and sha256(source) == sha256(destination):
            continue
        if destination.exists():
            if relative.startswith("checkpoints/"):
                raise SystemExit(f"Existing checkpoint differs: {destination}")
            upstream_relative = relative.removeprefix("overlay/")
            original = subprocess.run(
                ["git", "-C", str(target), "show", f"HEAD:{upstream_relative}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if original.returncode != 0 or destination.read_bytes() != original.stdout:
                raise SystemExit(f"Refusing to overwrite local changes: {destination}")
        pending.append((source, destination))

    print(f"{len(pending)} files to install into {target}")
    if dry_run:
        return
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print("Installation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Check package file hashes")
    install_parser = sub.add_parser("install", help="Install into a matching Isaac Lab checkout")
    install_parser.add_argument("isaaclab", type=Path)
    install_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    verify_package()
    if args.command == "install":
        install(args.isaaclab, args.dry_run)


if __name__ == "__main__":
    main()
