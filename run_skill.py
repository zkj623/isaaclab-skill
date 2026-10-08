#!/usr/bin/env python3
"""Run one of the packaged Go2 skills in an installed Isaac Lab checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


MANIFEST = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", choices=MANIFEST["skills"])
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    root = args.isaaclab.resolve()
    if not (root / "isaaclab.sh").is_file():
        parser.error(f"Not an Isaac Lab checkout: {root}")
    skill = MANIFEST["skills"][args.skill]
    command = [
        "./isaaclab.sh", "-p", "source/standalone/workflows/rsl_rl/play.py",
        "--task", skill["task"], "--num_envs", str(args.num_envs),
        "--load_run", skill["run"], "--checkpoint", skill["checkpoint"],
    ]
    if args.headless:
        command.append("--headless")
    print("Running:", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
