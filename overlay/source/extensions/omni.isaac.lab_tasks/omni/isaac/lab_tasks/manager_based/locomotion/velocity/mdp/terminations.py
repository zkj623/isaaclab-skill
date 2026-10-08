# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def front_feet_object_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg_FL: SceneEntityCfg,
    sensor_cfg_FR: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when front feet contact the Interactive_object.

    This termination function detects when the robot's front feet (FL or FR) make contact
    with the Interactive_object using filtered contact sensors. This encourages the robot
    to push the object with its body or arm, not with its front feet.

    Args:
        env: The environment instance.
        sensor_cfg_FL: Front-left foot contact sensor configuration.
        sensor_cfg_FR: Front-right foot contact sensor configuration.
        threshold: Contact force threshold in Newtons (default: 1.0N).

    Returns:
        Boolean tensor of shape (num_envs,). True indicates termination.
    """
    # Get contact sensors for front feet
    contact_sensor_FL = env.scene.sensors[sensor_cfg_FL.name]
    contact_sensor_FR = env.scene.sensors[sensor_cfg_FR.name]

    # Get contact force matrix (filtered for Interactive_object only)
    # force_matrix_w shape: (num_envs, num_bodies=1, num_filtered_objects=1, 3)
    force_FL = contact_sensor_FL.data.force_matrix_w  # (num_envs, 1, 1, 3)
    force_FR = contact_sensor_FR.data.force_matrix_w  # (num_envs, 1, 1, 3)

    # Calculate force magnitude for each foot
    force_magnitude_FL = torch.norm(force_FL[:, 0, 0, :], dim=-1)  # (num_envs,)
    force_magnitude_FR = torch.norm(force_FR[:, 0, 0, :], dim=-1)  # (num_envs,)

    # Terminate if either front foot has contact force above threshold
    has_contact_FL = force_magnitude_FL > threshold
    has_contact_FR = force_magnitude_FR > threshold

    # Return True if any front foot is in contact
    terminate = torch.logical_or(has_contact_FL, has_contact_FR)

    return terminate
