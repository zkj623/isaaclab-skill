# IsaacLab Skill: Unitree Go2

This repository packages nine Unitree Go2 policies and the matching Isaac Lab
source changes. It is an overlay for a pinned Isaac Lab checkout, rather than a
copy of the complete Isaac Lab repository.

| Skill | Isaac Lab task | Checkpoint | Status |
| --- | --- | --- | --- |
| Flat locomotion | `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` | `model_299.pt` | Three GPU simulation steps passed on the source machine |
| Rough locomotion | `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` | `model_7850.pt` | Three GPU simulation steps passed on the source machine |
| Object pushing | `Isaac-Object-Flat-Unitree-Go2-Play-v0` | `model_7000.pt` | Three GPU simulation steps passed on the source machine |
| Position-command rough locomotion (Jan 2025) | `Isaac-Position-Rough-Unitree-Go2-Play-v0` | `model_9700.pt` | Three GPU steps passed in a clean checkout |
| Door opening | `Isaac-Door-Open-Unitree-Go2-Play-v0` | `model_4200.pt` | Three GPU steps passed in a clean checkout |
| Interaction | `Isaac-Interaction-Unitree-Go2-Play-v0` | `model_8650.pt` | Three GPU steps passed in a clean checkout |
| Go2+ARX5 rough locomotion | `Isaac-Velocity-Rough-Unitree-Go2-ARX5-Play-v0` | `model_4500.pt` | Three GPU steps passed in a clean checkout |
| Go2+ARX5 internav | `Isaac-Internav-Unitree-Go2-ARX5-v0` | `model_9100.pt` | Three GPU steps passed in a clean checkout |
| Go2+ARX5 door push | `Isaac-DoorPush-Unitree-Go2-ARX5-v0` | `model_15300.pt` | Three GPU steps passed in a clean checkout |

The object-pushing task also uses the exported rough-locomotion policy as its
low-level controller. The package includes that file under `checkpoints/`.

## Requirements

- Isaac Lab at upstream commit `43a3ce9af` (the source version from which the
  overlay was prepared)
- Isaac Sim 4.2.0, a compatible NVIDIA GPU and driver, and RSL-RL dependencies
- Access to the Isaac Sim assets referenced by the environment configurations

The package includes the custom ARX5 robot USD and local Library USD scenes
used by the six added policies under `assets/skill_assets/`. The ARX5 asset
comes from RoboDuet; its MIT notice is included alongside the file. Isaac Sim's
standard NVIDIA assets remain a separate dependency. The source machine used
an Isaac Sim 4.2 asset root on a local Nucleus server. Set up equivalent asset
access on the destination machine. If the default cloud asset server is
inaccessible, pass its equivalent local root with `--asset-root`, for example
`--asset-root omniverse://localhost/NVIDIA/Assets/Isaac/4.2`.

## Install into a fresh Isaac Lab checkout

For a binary Isaac Sim 4.2.0 installation, follow the [installation guide for
the pinned Isaac Lab version](https://github.com/isaac-sim/IsaacLab/blob/43a3ce9af/docs/source/setup/installation/binaries_installation.rst).
From a directory where you want both repositories:

```bash
git clone https://github.com/zkj623/isaaclab-skill.git
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 43a3ce9af
cd IsaacLab
ln -s /absolute/path/to/isaac-sim-4.2.0 _isaac_sim  # replace with your installation path
./isaaclab.sh -c isaaclab
conda activate isaaclab
cd ..
python isaaclab-skill/install.py verify
python isaaclab-skill/install.py install IsaacLab --dry-run
python isaaclab-skill/install.py install IsaacLab
cd IsaacLab
./isaaclab.sh -i rsl_rl
cd ..
python isaaclab-skill/verify_models.py
```

If Isaac Sim was installed with pip, follow the pinned upstream pip installation
guide instead of creating `_isaac_sim`. The conda environment must be active
when installing dependencies and playing a skill.

Place `isaaclab-skill` next to `IsaacLab`, or replace the paths above. The
installer verifies the SHA-256 hash of every packaged file, checks the pinned
Isaac Lab revision, and refuses to overwrite locally changed files. The
`overlay/` tree maps directly to paths in Isaac Lab; the `checkpoints/` tree
maps to its `logs/` directory. Re-running the installer is safe when files
already match.

## Play a skill

From the parent directory of both repositories:

```bash
python isaaclab-skill/run_skill.py flat_locomotion --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py rough_locomotion --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py object_pushing --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py rough_locomotion_jan2025 --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py door_opening --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py interaction --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py arx5_rough_locomotion --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py arx5_internav --isaaclab IsaacLab --num-envs 1
python isaaclab-skill/run_skill.py arx5_door_push --isaaclab IsaacLab --num-envs 1
```

Run the skills one at a time. Use `--headless` for a machine without a display;
increase `--num-envs` after a one-environment run succeeds. The wrapper supplies
the exact task, run directory, and checkpoint name to `play.py`. Add
`--follow-camera` to track the first robot in the GUI viewport; it has no effect
in headless mode. Use `--asset-root` when your Isaac Sim assets are hosted at a
different URI.

The January 2025 rough policy has 236 observations from a position-command
environment. The older rough policy has 235 observations from a velocity-command
environment. They use separate task registrations so that neither checkpoint
is played with the wrong observation configuration.

The original working object-pushing command is equivalent to:

```bash
cd IsaacLab
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Isaac-Object-Flat-Unitree-Go2-Play-v0 \
  --num_envs 100 --load_run 2024-12-14_13-42-28 --checkpoint model_7000.pt
```

## Package contents and provenance

- `overlay/`: selected Python source files from the local Isaac Lab checkout.
  Task registries include the nine skills above. Machine-specific paths are
  resolved from the destination Isaac Lab checkout.
- `checkpoints/`: one selected training checkpoint for each skill and the
  exported low-level policy required by object pushing. Other training logs,
  checkpoints, CARLA/Matterport assets, and unrelated experiments are omitted.
- `assets/`: custom robot and local scene USDs required by these tasks. Library
  USDs were flattened to remove their source Nucleus layer references.
- `manifest.json`: exact source revision, task names, run names, and SHA-256
  hashes for package verification.

Copied Isaac Lab source retains its copyright headers. The upstream
BSD-3-Clause license is included in [LICENSE](LICENSE). The trained model
files came from the local experiment runs listed in `manifest.json`.

## Validation

All nine checkpoints loaded on CPU with expected actor dimensions. The six
added tasks loaded their checkpoints and completed three GPU simulation steps
in the `isaaclab-python` conda environment using a clean pinned checkout and
local Isaac Sim assets. The original three tasks also passed three GPU steps
in a clean checkout. These short runs verify startup and stepping, not policy
quality or a full episode. GUI camera following has not been tested, and
another machine still needs its own Isaac Sim installation, dependencies, and
standard asset access.
