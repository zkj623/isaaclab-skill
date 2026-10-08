#!/usr/bin/env python3
"""Check packaged checkpoints and the low-level TorchScript policy on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SKILLS = json.loads((ROOT / "manifest.json").read_text())["skills"]
EXPECTED_DIMS = {
    "flat_locomotion": (48, 12),
    "rough_locomotion": (235, 12),
    "object_pushing": (16, 3),
    "rough_locomotion_jan2025": (236, 12),
    "door_opening": (52, 12),
    "interaction": (245, 12),
    "arx5_rough_locomotion": (260, 18),
    "arx5_internav": (76, 18),
    "arx5_door_push": (56, 18),
}


def check_checkpoint(name: str, info: dict[str, str]) -> None:
    checkpoint = (
        ROOT / "checkpoints" / "logs" / "rsl_rl" / info["experiment"]
        / info["run"] / info["checkpoint"]
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    actor_layers = sorted(
        (key for key in state if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    expected_input, expected_output = EXPECTED_DIMS[name]
    actual_input = state[actor_layers[0]].shape[1]
    actual_output = state[actor_layers[-1]].shape[0]
    if (actual_input, actual_output) != (expected_input, expected_output):
        raise SystemExit(
            f"{name}: expected {expected_input}→{expected_output}, "
            f"got {actual_input}→{actual_output}"
        )
    print(f"{name}: checkpoint loaded, actor {actual_input}→{actual_output}")


def main() -> None:
    for name, info in SKILLS.items():
        check_checkpoint(name, info)
    policy = (
        ROOT / "checkpoints/logs/rsl_rl/unitree_go2_rough"
        / "walking_stable_100/exported/policy.pt"
    )
    model = torch.jit.load(str(policy), map_location="cpu").eval()
    with torch.no_grad():
        output = model(torch.zeros(1, EXPECTED_DIMS["rough_locomotion"][0]))
    if tuple(output.shape) != (1, EXPECTED_DIMS["rough_locomotion"][1]):
        raise SystemExit(f"Low-level policy returned unexpected shape: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise SystemExit("Low-level policy returned non-finite values")
    print("object_pushing low-level policy: CPU forward passed")


if __name__ == "__main__":
    main()
