# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    # print(contacts)
    # print(reward)
    return reward

def stand_still_joint_vel_penalty(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities when velocity commands are near zero.

    This function penalizes joint velocities when the commanded velocity is small,
    preventing the robot from stepping in place.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command (should contain lin_vel_x, lin_vel_y, ang_vel_z).
        asset_cfg: Scene entity configuration for the robot (default: "robot").

    Returns:
        Penalty tensor with shape (num_envs,). Higher when joints move with zero command.
    """
    asset = env.scene[asset_cfg.name]

    # Get velocity commands (first 3 dims: lin_vel_x, lin_vel_y, ang_vel_z)
    vel_command = env.command_manager.get_command(command_name)[:, :3]  # (num_envs, 3)

    # Check if commands are small (near zero)
    command_magnitude = torch.norm(vel_command[:, :2], dim=-1)  # Linear velocity magnitude
    ang_command_magnitude = torch.abs(vel_command[:, 2])  # Angular velocity magnitude
    small_commands = (command_magnitude < 0.1) & (ang_command_magnitude < 0.1)  # (num_envs,)

    # Get joint velocities for the specified joints (usually leg joints)
    if asset_cfg.joint_ids is not None:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # (num_envs, num_joints)
    else:
        # If no specific joints specified, use all joints
        joint_vel = asset.data.joint_vel  # (num_envs, num_joints)

    # Compute penalty as sum of absolute joint velocities
    joint_vel_penalty = torch.sum(torch.abs(joint_vel), dim=1)  # (num_envs,)

    # Only apply penalty when commands are small
    return joint_vel_penalty * small_commands.float()


def stand_still_joint_pos_penalty(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint position deviation from default pose when velocity commands are near zero.

    This function penalizes joint position deviations from the default pose when the
    commanded velocity is small, encouraging the robot to return to default standing pose.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command (should contain lin_vel_x, lin_vel_y, ang_vel_z).
        asset_cfg: Scene entity configuration for the robot (default: "robot").

    Returns:
        Penalty tensor with shape (num_envs,). Higher when joints deviate from default with zero command.
    """
    asset = env.scene[asset_cfg.name]

    # Get velocity commands (first 3 dims: lin_vel_x, lin_vel_y, ang_vel_z)
    vel_command = env.command_manager.get_command(command_name)[:, :3]  # (num_envs, 3)

    # Check if commands are small (near zero)
    command_magnitude = torch.norm(vel_command[:, :2], dim=-1)  # Linear velocity magnitude
    ang_command_magnitude = torch.abs(vel_command[:, 2])  # Angular velocity magnitude
    small_commands = (command_magnitude < 0.1) & (ang_command_magnitude < 0.1)  # (num_envs,)

    # Get joint positions for the specified joints (usually leg joints)
    if asset_cfg.joint_ids is not None:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (num_envs, num_joints)
        default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # (num_envs, num_joints)
    else:
        # If no specific joints specified, use all joints
        joint_pos = asset.data.joint_pos  # (num_envs, num_joints)
        default_joint_pos = asset.data.default_joint_pos  # (num_envs, num_joints)

    # Compute penalty as sum of absolute joint position deviations from default
    joint_pos_deviation = joint_pos - default_joint_pos  # (num_envs, num_joints)
    joint_pos_penalty = torch.sum(torch.abs(joint_pos_deviation), dim=1)  # (num_envs,)

    # Only apply penalty when commands are small
    return joint_pos_penalty * small_commands.float()

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def action_l2_first_three(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3]), dim=1)

def action_rate_l2_first_three(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the first three action dimensions using L2 squared kernel.

    This function only penalizes the rate of change for the first three dimensions of the action,
    which typically correspond to velocity commands (lin_vel_x, lin_vel_y, ang_vel_z).
    """
    return torch.sum(torch.square(env.action_manager.action[:, :3] - env.action_manager.prev_action[:, :3]), dim=1)


def action_l2_arm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize arm actions (last 7 dims: position xyz + quaternion wxyz) using L2 squared kernel.

    Used for curriculum learning: by adjusting the weight, you can control the arm usage.
    - High weight (-10.0): almost fix the arm
    - Medium weight (-1.0): allow necessary arm movement
    - Low weight (-0.1): full freedom
    """
    return torch.sum(torch.square(env.action_manager.action[:, 3:]), dim=1)


def action_rate_l2_arm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of arm actions (last 7 dims) using L2 squared kernel.

    Encourages smooth arm movements and prevents jittering.
    """
    return torch.sum(
        torch.square(env.action_manager.action[:, 3:] - env.action_manager.prev_action[:, 3:]),
        dim=1
    )


def action_l2_arm_ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize arm position actions (dims 3-5: position xyz) using L2 squared kernel.

    Used when action space is 6D (3 velocity + 3 position), without quaternion.
    Default position is (0.4, 0., 0.3).
    """
    default_pos = torch.tensor([0.4, 0., 0.3], device=env.device)
    # print(env.action_manager.action[:, 3:6])
    return torch.sum(torch.square(env.action_manager.action[:, 3:6] - default_pos), dim=1)


def action_rate_l2_arm_ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of arm position actions (dims 3-5) using L2 squared kernel.

    Encourages smooth arm position movements and prevents jittering.
    Used when action space is 6D (3 velocity + 3 position), without quaternion.
    """
    return torch.sum(
        torch.square(env.action_manager.action[:, 3:6] - env.action_manager.prev_action[:, 3:6]),
        dim=1
    )

def action_rate_l2_arm_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of arm position actions (dims 3-5) using L2 squared kernel.

    Encourages smooth arm position movements and prevents jittering.
    Used when action space is 6D (3 velocity + 3 position), without quaternion.
    """
    return torch.sum(
        torch.square(env.action_manager.action[:, 3:9] - env.action_manager.prev_action[:, 3:9]),
        dim=1
    )


def arm_end_effector_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.1,
    max_height: float = 1.0,
) -> torch.Tensor:
    """Penalize arm end-effector height when it's too low or too high.

    This function penalizes the end-effector when:
    - It's below min_height (too close to ground) - quadratic penalty
    - It's above max_height (too high) - quadratic penalty
    - Otherwise, no penalty

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot (body_names should be the end-effector).
        min_height: Minimum allowed height (default: 0.1m). Below this, penalty increases.
        max_height: Maximum allowed height (default: 0.6m). Above this, penalty increases.

    Returns:
        Penalty tensor with shape (num_envs,).
    """
    # Extract the asset
    asset = env.scene[asset_cfg.name]

    # Get end-effector (gripper base) position and orientation in world frame
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Get height of actual gripper tip (z-coordinate)
    ee_height = ee_tip_pos_w[:, 2]  # z-coordinate
    # print(ee_height)

    # Compute penalty
    # Below min_height: quadratic penalty based on violation
    below_min = torch.clamp(min_height - ee_height, min=0.0)
    penalty_low = torch.square(below_min)

    # Above max_height: quadratic penalty based on violation
    above_max = torch.clamp(ee_height - max_height, min=0.0)
    penalty_high = torch.square(above_max)

    # Total penalty
    return penalty_low + penalty_high


def robot_object_distance_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peak_distance: float = 0.5,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward robot for approaching the object with a peak at specified distance.

    This function rewards the robot for being close to the object:
    - Distance > peak_distance: reward increases as robot gets closer (exponential kernel)
    - Distance <= peak_distance: reward stays at peak value (maximum reward)

    This encourages the robot to approach the object until it reaches the peak distance,
    after which it can get even closer without penalty.

    Args:
        env: The environment instance.
        object_cfg: Scene entity configuration for the object.
        robot_cfg: Scene entity configuration for the robot (default: "robot").
        peak_distance: Distance at which reward reaches peak (default: 0.5m).
                      Below this distance, reward stays at peak.
        std: Standard deviation for exponential kernel (default: 1.0).

    Returns:
        Reward tensor with shape (num_envs,).
    """
    # Get robot base position
    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_link_state_w[:, :3]

    # Get object position
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_link_state_w[:, :3]

    # Calculate distance (2D distance in xy plane)
    distance = torch.norm(robot_pos[:, :2] - obj_pos[:, :2], dim=1)

    # Compute effective distance for reward calculation
    # If distance < peak_distance, clamp it to peak_distance (to maintain peak reward)
    # If distance >= peak_distance, use actual distance
    effective_distance = torch.clamp(distance, min=peak_distance)

    # Exponential reward based on effective distance
    # At peak_distance or closer: reward = exp(0) = 1.0 (maximum)
    # Beyond peak_distance: reward decreases exponentially
    distance_error = effective_distance - peak_distance
    reward = torch.exp(-distance_error / std)

    return reward


def arm_end_effector_object_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.1,
    num_surface_points: int = 20,
) -> torch.Tensor:
    """Reward arm end-effector proximity to object surface (side faces only).

    This function rewards the robot when its arm end-effector is close to the object's side surfaces.
    It samples points on the object's 4 side faces (excluding top/bottom) at different heights,
    and computes the minimum distance from the end-effector to these surface points.
    This is designed for pushing tasks where only side contact is relevant.

    The reward uses an exponential kernel: reward = exp(-min_distance^2 / std^2)
    - Distance = 0: reward = 1.0 (maximum, touching the object)
    - Distance increases: reward decreases exponentially

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.
                   body_names should specify the end-effector link (e.g., "zarx_body6").
        object_cfg: Scene entity configuration for the target object.
        std: Standard deviation for exponential kernel (default: 0.1m).
             Smaller values make the reward decay faster with distance.
        num_surface_points: Number of surface points to sample (default: 20).
                           Points are sampled on the 4 side faces at multiple heights.

    Returns:
        Reward tensor with shape (num_envs,).
    """
    from omni.isaac.lab.utils.math import quat_rotate

    # Get end-effector position
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Get object position and size
    obj = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_link_state_w[:, :3]  # (num_envs, 3)
    obj_quat_w = obj.data.root_link_state_w[:, 3:7]  # (num_envs, 4)

    num_envs = ee_tip_pos_w.shape[0]
    device = ee_tip_pos_w.device

    # Get object sizes dynamically for each environment
    # Assuming the object sizes are stored in a tensor (based on observations.py)
    from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import size_tensor

    # Get the size tensor and extract sizes for current environments
    env_num = env.num_envs
    size_data = size_tensor
    while len(size_data) < env_num:
        size_data = torch.cat([size_data, size_data], dim=0)
    obj_sizes = size_data[:num_envs]  # (num_envs, 3) - full sizes (not half)
    obj_half_sizes = obj_sizes / 2.0  # (num_envs, 3) - convert to half extents

    # Sample surface points on side faces only (for pushing tasks, assuming Z is vertical)
    # 20 points: concentrated near face centers, avoiding far corners to reduce contact misalignment
    surface_points_local = torch.tensor([
        # Four side face centers at mid-height (4 points)
        [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],  # +X, -X faces
        [0.0, 0.5, 0.0], [0.0, -0.5, 0.0],  # +Y, -Y faces

        # Four near-center points at mid-height (4 points) - closer to center
        [0.5, 0.2, 0.0], [0.5, -0.2, 0.0],
        [-0.5, 0.2, 0.0], [-0.5, -0.2, 0.0],

        # Four side face centers at upper height (4 points)
        [0.5, 0.0, 0.25], [-0.5, 0.0, 0.25],
        [0.0, 0.5, 0.25], [0.0, -0.5, 0.25],

        # Four side face centers at lower height (4 points)
        [0.5, 0.0, -0.25], [-0.5, 0.0, -0.25],
        [0.0, 0.5, -0.25], [0.0, -0.5, -0.25],

        # Four near-center points at upper height (4 points) - closer to center
        [0.5, 0.2, 0.25], [0.5, -0.2, 0.25],
        [-0.5, 0.2, 0.25], [-0.5, -0.2, 0.25],
    ], device=device, dtype=torch.float32)  # (20, 3)

    # Scale surface points by object size for each environment
    # obj_half_sizes: (num_envs, 3), surface_points_local: (20, 3)
    # Broadcast: (num_envs, 20, 3)
    surface_points_local_scaled = surface_points_local.unsqueeze(0) * obj_half_sizes.unsqueeze(1)  # (num_envs, 20, 3)

    # Transform surface points to world frame
    # Rotate by object quaternion
    surface_points_world = quat_rotate(
        obj_quat_w.unsqueeze(1).expand(-1, num_surface_points, -1).reshape(-1, 4),  # (num_envs*20, 4)
        surface_points_local_scaled.reshape(-1, 3)  # (num_envs*20, 3)
    ).reshape(num_envs, num_surface_points, 3)  # (num_envs, 20, 3)

    # Add object position offset
    surface_points_world = surface_points_world + obj_pos_w.unsqueeze(1)  # (num_envs, 20, 3)

    # Compute distances from end-effector tip to all surface points
    ee_tip_pos_expanded = ee_tip_pos_w.unsqueeze(1).expand(-1, num_surface_points, -1)  # (num_envs, 20, 3)
    distances = torch.norm(surface_points_world - ee_tip_pos_expanded, dim=-1)  # (num_envs, 20)

    # Get minimum distance to any surface point
    min_distance = torch.min(distances, dim=1)[0]  # (num_envs,)

    # print("Min distances:", min_distance)

    # Compute reward using exponential kernel
    reward = torch.exp(-torch.square(min_distance) / (std ** 2))

    return reward


def arm_end_effector_forward_reach_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_x: float = 0.3,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward arm end-effector for reaching forward (positive x direction relative to robot base).

    This function rewards the robot when its arm end-effector extends forward (x > target_x).
    The reward uses an exponential kernel centered at target_x:
    - x >= target_x: reward = 1.0 (maximum, arm is extended forward)
    - x < target_x: reward decreases exponentially as x decreases

    This encourages the robot to keep the arm extended forward, which is useful for:
    1. Pushing tasks (need to reach forward to contact object)
    2. Manipulation tasks (need to extend arm to reach target)

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.
                   body_names should specify the end-effector link (e.g., "zarx_body6").
        target_x: Target x position (forward direction) in robot frame (default: 0.3m).
                  When x >= target_x, reward is maximum.
        std: Standard deviation for exponential kernel (default: 0.1m).
             Smaller values make the reward decay faster when x < target_x.

    Returns:
        Reward tensor with shape (num_envs,).
    """
    # Get end-effector position relative to robot base
    asset = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    # Get end-effector (gripper base) position and orientation in world frame
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    # Local offset vector (0.14m forward in gripper frame)
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction

    # Transform offset to world frame using gripper orientation
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)

    # Actual gripper tip position in world frame
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Get robot base position and orientation in world frame
    robot_pos_w = robot.data.root_link_state_w[:, :3]  # (num_envs, 3)
    robot_quat_w = robot.data.root_link_quat_w  # (num_envs, 4)

    # Compute relative position in world frame (using actual gripper tip position)
    ee_pos_rel_w = ee_tip_pos_w - robot_pos_w  # (num_envs, 3)

    # Transform to robot frame (rotate by inverse of robot orientation)
    ee_pos_rel_robot = quat_rotate_inverse(robot_quat_w, ee_pos_rel_w)  # (num_envs, 3)

    # Extract x coordinate (forward direction in robot frame)
    ee_x = ee_pos_rel_robot[:, 0]  # (num_envs,)
    # print(ee_x)

    # Compute reward
    # If x >= target_x: reward = 1.0 (maximum)
    # If x < target_x: reward = exp(-(target_x - x)^2 / std^2)
    x_error = torch.clamp(target_x - ee_x, min=0.0)  # Only penalize when x < target_x
    reward = torch.exp(-torch.square(x_error) / (std ** 2))

    return reward


def arm_end_effector_handle_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
    open_angle_threshold: float = 0.785,  # 45度 = π/4 ≈ 0.785弧度
    initial_door_yaw: float = None,  # 门的初始yaw角度（自动检测）
) -> torch.Tensor:
    """Reward arm end-effector proximity to door handle (only when door is not open enough).

    This function rewards the robot when its arm end-effector is close to the door handle.
    The handle position is computed by applying a local offset to the door object position
    and rotating it according to the door's orientation.

    **Important**: The reward is only given when the door is not open enough yet.
    Once the door has been opened beyond open_angle_threshold (default: 45 degrees),
    this reward becomes zero to avoid encouraging the robot to keep holding the handle.

    The reward uses an exponential kernel: reward = exp(-distance^2 / std^2)
    - Distance = 0: reward = 1.0 (maximum, touching the handle)
    - Distance increases: reward decreases exponentially
    - Door open >= threshold: reward = 0.0 (no need to approach handle anymore)

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.
                   body_names should specify the end-effector link (e.g., "zarx_body6").
        object_cfg: Scene entity configuration for the door object.
        robot_cfg: Scene entity configuration for the robot base (default: "robot").
        std: Standard deviation for exponential kernel (default: 0.1m).
             Smaller values make the reward decay faster with distance.
        open_angle_threshold: Door opening angle threshold in radians (default: 0.785 rad = 45°).
                             When door angle >= threshold, reward becomes zero.
        initial_door_yaw: Initial yaw angle of the door in radians (optional, auto-detected on first call).

    Returns:
        Reward tensor with shape (num_envs,).
    """
    from omni.isaac.lab.utils.math import quat_rotate, euler_xyz_from_quat

    # Get door object position and orientation
    object = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]        # (num_envs, 3)
    object_quat_w = object.data.root_quat_w             # (num_envs, 4)

    # Calculate door's current yaw angle from quaternion
    _, _, current_door_yaw = euler_xyz_from_quat(object_quat_w)  # (num_envs,)

    # Store initial door yaw on first call (per environment)
    if not hasattr(env, '_initial_door_yaw'):
        env._initial_door_yaw = current_door_yaw.clone()

    # Calculate door opening angle (difference from initial yaw)
    door_opening_angle = torch.abs(current_door_yaw - env._initial_door_yaw)  # (num_envs,)

    # Normalize angle to [0, π] range
    door_opening_angle = torch.min(door_opening_angle, 2 * torch.pi - door_opening_angle)

    # print(door_opening_angle)

    # Check if door has been opened enough
    door_open_enough = door_opening_angle >= open_angle_threshold  # (num_envs,)

    # Get end-effector position
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Define handle offset in door's local coordinate system
    # 把手在门局部坐标系中的偏移
    handle_local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    handle_local_offset[:, 0] = 0.0      # x方向(门的前方)
    handle_local_offset[:, 1] = -0.8965 * 0.8  # y方向(门的左侧)
    handle_local_offset[:, 2] = 1.017 * 0.8    # z方向(门的上方)

    # Transform handle offset to world coordinate system
    # 将局部偏移旋转到世界坐标系
    handle_world_offset = quat_rotate(object_quat_w, handle_local_offset)  # (num_envs, 3)

    # Compute handle position in world frame
    # 计算把手在世界坐标系中的真实位置
    handle_pos_w = object_pos_w + handle_world_offset  # (num_envs, 3)

    # Compute distance from end-effector tip to handle
    distance = torch.norm(ee_tip_pos_w - handle_pos_w, dim=1)  # (num_envs,)
    # print(distance)
    # print(door_open_enough)

    # Compute reward using exponential kernel
    reward = torch.exp(-torch.square(distance) / (std ** 2))
    # print(reward)

    # Zero out reward for doors that have been opened enough
    reward = torch.where(door_open_enough, torch.zeros_like(reward), reward)

    # Debug print (optional, uncomment to see door opening status)
    # if env.episode_length_buf[0] % 100 == 0:
    #     print(f"门已打开足够角度的数量: {door_open_enough.sum().item()}/{env.num_envs}")
    #     print(f"门的打开角度(度) (前3个): {(door_opening_angle[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"初始yaw (前3个): {(env._initial_door_yaw[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"当前yaw (前3个): {(current_door_yaw[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"阈值角度: {open_angle_threshold * 180 / 3.14159:.1f}度")

    return reward


def arm_end_effector_to_contact_points_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
    num_points: int = 1024,
    min_height: float = 0.2,
    max_height: float = 0.7,
    top_k_closest: int = 16,
    surface_edge_margin: float = 0.15,
) -> torch.Tensor:
    """Reward arm end-effector proximity to object contact points.

    This function rewards the robot when its arm end-effector is close to the contact points
    on the object surface. Contact points are sampled from the object's surface mesh.
    The reward is based on the distance to the closest contact point.

    **Important**: This function uses the same contact points as the observation function
    (object_shape_pc_relative_optimized). The top_k_closest points are selected based on
    their distance to the robot base (consistent with observations), not the end-effector.

    The reward uses an exponential kernel: reward = exp(-distance^2 / std^2)
    - Distance = 0: reward = 1.0 (maximum, touching the surface)
    - Distance increases: reward decreases exponentially

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.
                   body_names should specify the end-effector link (e.g., "zarx_body6").
        object_cfg: Scene entity configuration for the interactive object.
        robot_cfg: Scene entity configuration for the robot base (default: "robot").
        std: Standard deviation for exponential kernel (default: 0.1m).
             Smaller values make the reward decay faster with distance.
        num_points: Number of points to sample from object surface (default: 1024).
        min_height: Minimum height for sampling contact points (default: 0.2m).
        max_height: Maximum height for sampling contact points (default: 0.7m).
        top_k_closest: Only consider the k closest points to robot base (default: 16).
                      This matches the observation function's behavior.
        surface_edge_margin: Surface sampling edge margin ratio (default: 0.15).

    Returns:
        Reward tensor with shape (num_envs,).
    """
    from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse
    from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import (
        object_shape_pc_relative_optimized,
    )

    # Get end-effector position
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Get robot base position and orientation
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # Get object contact points in world frame
    object = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_quat_w = object.data.root_quat_w  # (num_envs, 4)

    # Get contact points in object's local frame from the shape
    from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import (
        _object_type_point_clouds, _env_to_object_type, _point_cloud_initialized,
        generate_all_point_clouds, calculate_object_type_mapping,
    )

    # Initialize point clouds if needed
    if not _point_cloud_initialized:
        generate_all_point_clouds(env, object_cfg, num_points, min_height, max_height, surface_edge_margin)

    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    env_cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"

    num_envs = env.num_envs
    device = env.device
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)

    # Get point cloud for each environment
    env_ids_by_type = {}
    env_to_type = _env_to_object_type.get(env_cache_key, {})

    if not env_to_type:
        env_to_type = calculate_object_type_mapping(env, object_cfg)

    for env_id, object_type in env_to_type.items():
        if env_id >= num_envs:
            continue
        if object_type not in env_ids_by_type:
            env_ids_by_type[object_type] = []
        env_ids_by_type[object_type].append(env_id)

    for object_type, env_ids in env_ids_by_type.items():
        env_indices = torch.tensor(env_ids, dtype=torch.long, device=device)

        if object_type in _object_type_point_clouds[cache_key]:
            point_cloud = _object_type_point_clouds[cache_key][object_type]
        else:
            point_cloud = _object_type_point_clouds[cache_key]["default"]

        point_cloud_3d.index_copy_(0, env_indices, point_cloud.unsqueeze(0).expand(len(env_indices), -1, -1))

    # Transform point cloud from object local frame to world frame
    # Apply rotation
    object_quat_expanded = object_quat_w.unsqueeze(1).expand(-1, num_points, -1)  # (num_envs, num_points, 4)
    object_quat_flat = object_quat_expanded.reshape(-1, 4)  # (num_envs*num_points, 4)
    point_cloud_flat = point_cloud_3d.reshape(-1, 3)  # (num_envs*num_points, 3)

    rotated_points_flat = quat_rotate(object_quat_flat, point_cloud_flat)  # (num_envs*num_points, 3)
    rotated_points = rotated_points_flat.reshape(num_envs, num_points, 3)  # (num_envs, num_points, 3)

    # Apply translation to get points in world frame
    contact_points_w = rotated_points + object_pos_w.unsqueeze(1)  # (num_envs, num_points, 3)

    # Transform contact points to robot frame (to match observation function's coordinate system)
    contact_points_relative_w = contact_points_w - robot_pos_w.unsqueeze(1)  # (num_envs, num_points, 3)

    # Rotate to robot's local frame
    robot_quat_expanded = robot_quat_w.unsqueeze(1).expand(-1, num_points, -1)  # (num_envs, num_points, 4)
    robot_quat_flat = robot_quat_expanded.reshape(-1, 4)  # (num_envs*num_points, 4)
    contact_points_relative_flat = contact_points_relative_w.reshape(-1, 3)  # (num_envs*num_points, 3)

    contact_points_robot_flat = quat_rotate_inverse(robot_quat_flat, contact_points_relative_flat)  # (num_envs*num_points, 3)
    contact_points_robot = contact_points_robot_flat.reshape(num_envs, num_points, 3)  # (num_envs, num_points, 3)

    # Select the same top_k closest points as in the observation function
    # (closest to robot base origin in robot frame)
    if top_k_closest is not None and top_k_closest < num_points:
        # Compute distance from robot base (origin in robot frame) to each contact point
        distances_to_robot = torch.norm(contact_points_robot, dim=2)  # (num_envs, num_points)

        # Get top k closest points to robot base (matching observation function)
        _, top_k_indices = torch.topk(distances_to_robot, k=top_k_closest, dim=1, largest=False)  # (num_envs, top_k)

        # Gather the closest points in world frame
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 3)  # (num_envs, top_k, 3)
        contact_points_w = torch.gather(contact_points_w, dim=1, index=top_k_indices_expanded)  # (num_envs, top_k, 3)

    # Compute distance from end-effector tip to each selected contact point
    distances = torch.norm(contact_points_w - ee_tip_pos_w.unsqueeze(1), dim=2)  # (num_envs, num_points_selected)

    # Get the minimum distance (closest contact point)
    min_distance, _ = torch.min(distances, dim=1)  # (num_envs,)

    # Compute reward using exponential kernel
    reward = torch.exp(-torch.square(min_distance) / (std ** 2))

    return reward


def object_lateral_distance_from_path_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
    std: float = 0.3,
    saturation_distance: float = 0.8,
    angle_threshold: float = 1.047,  # 60 degrees in radians
    min_displacement_threshold: float = 0.0,  # minimum object displacement before giving reward
) -> torch.Tensor:
    """Reward object being laterally away from the robot-to-goal path (with saturation and angle check).

    This function rewards when the object is pushed away from the direct line between
    the robot and its goal position. The reward increases with lateral distance until
    it reaches a saturation threshold, after which the reward stays at maximum.

    Additionally, the reward is disabled when the angle between (robot→goal) and
    (robot→object) exceeds a threshold. This prevents the robot from continuously
    pushing the object when it's already clear from the path and the robot should
    focus on reaching the goal instead.

    The reward uses an exponential kernel with saturation and angle gating:
    - angle < angle_threshold AND lateral_distance < saturation_distance:
        reward = 1.0 - exp(-distance^2 / std^2)
    - angle >= angle_threshold: reward = 1.0 (object not blocking, focus on goal)
    - lateral_distance >= saturation_distance: reward = maximum (no need to push farther)
    - object_displacement < min_displacement_threshold: reward = 0 (object hasn't moved enough yet)

    Args:
        env: The environment instance.
        object_cfg: Scene entity configuration for the object.
        robot_cfg: Scene entity configuration for the robot base (default: "robot").
        command_name: Name of the pose command (default: "pose_command").
        std: Standard deviation for exponential kernel (default: 0.3m).
             Controls how quickly reward increases with distance.
        saturation_distance: Distance threshold where reward reaches maximum (default: 0.8m).
                            Beyond this distance, pushing farther gives no additional reward.
        angle_threshold: Maximum angle between robot→goal and robot→object vectors (default: 1.047 rad = 60°).
                        When angle exceeds this, reward becomes 1.0 (object no longer blocking path).
        min_displacement_threshold: Minimum displacement of object from initial position (default: 0.0m).
                                   The lateral reward is ZERO until object moves at least this distance.
                                   This prevents robot from cheating by moving away without pushing.

    Returns:
        Reward tensor with shape (num_envs,).
    """
    # Get robot position and orientation
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - only x, y
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # Get object position in world frame
    object_asset = env.scene[object_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :2]  # (num_envs, 2)

    # Check object displacement from initial position (if threshold is set)
    if min_displacement_threshold > 0.0:
        # Initialize or update object initial position storage
        if not hasattr(env, '_object_initial_pos'):
            # First call: initialize storage
            env._object_initial_pos = object_pos_w.clone()
            env._object_initial_pos_episode_id = env.episode_length_buf.clone()

        # Update initial position for environments that just reset (episode_length_buf == 1)
        # This handles reset events automatically without needing a separate event function
        reset_envs = env.episode_length_buf == 0  # Reset happens when episode_length_buf is 0
        if reset_envs.any():
            env._object_initial_pos[reset_envs] = object_pos_w[reset_envs].clone()

        # Compute displacement from initial position
        object_displacement = torch.norm(object_pos_w - env._object_initial_pos, dim=1)  # (num_envs,)

        # print(object_displacement)

        # Check if object has moved enough
        object_moved_enough = object_displacement >= min_displacement_threshold  # (num_envs,) boolean
    else:
        # No threshold: always consider object as moved enough
        object_moved_enough = torch.ones(object_pos_w.shape[0], dtype=torch.bool, device=object_pos_w.device)

    # Get goal position from command
    # NOTE: command returns position in BASE FRAME (robot-relative coordinates)
    # We need to transform it to world frame for unified geometric calculations
    command = env.command_manager.get_command(command_name)
    robot_to_goal_b = command[:, :2]  # (num_envs, 2) - vector in robot frame

    # Transform goal vector from robot frame to world frame
    from omni.isaac.lab.utils.math import quat_rotate, yaw_quat
    robot_to_goal_b_3d = torch.cat([robot_to_goal_b, torch.zeros_like(robot_to_goal_b[:, :1])], dim=1)  # (num_envs, 3)
    robot_to_goal_w_3d = quat_rotate(yaw_quat(robot_quat_w), robot_to_goal_b_3d)  # (num_envs, 3)
    robot_to_goal_w = robot_to_goal_w_3d[:, :2]  # (num_envs, 2)

    # Compute goal position in world frame
    goal_pos_w = robot_pos_w + robot_to_goal_w  # (num_envs, 2)

    # Compute vector from robot to object in world frame
    robot_to_object_w = object_pos_w - robot_pos_w  # (num_envs, 2)

    # Normalize vectors for angle calculation (all in world frame now)
    robot_to_goal_norm = torch.norm(robot_to_goal_w, dim=1, keepdim=True)  # (num_envs, 1)
    robot_to_goal_norm = torch.clamp(robot_to_goal_norm, min=0.01)  # Avoid division by zero
    robot_to_goal_normalized = robot_to_goal_w / robot_to_goal_norm  # (num_envs, 2)

    robot_to_object_norm = torch.norm(robot_to_object_w, dim=1, keepdim=True)  # (num_envs, 1)
    robot_to_object_norm = torch.clamp(robot_to_object_norm, min=0.01)
    robot_to_object_normalized = robot_to_object_w / robot_to_object_norm  # (num_envs, 2)

    # Compute angle between robot→goal and robot→object using dot product
    # cos(angle) = dot(v1, v2) / (|v1| * |v2|)
    # Since vectors are normalized, cos(angle) = dot(v1, v2)
    cos_angle = torch.sum(robot_to_goal_normalized * robot_to_object_normalized, dim=1)  # (num_envs,)

    # Clamp to valid range for acos (numerical stability)
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)

    # Compute angle in radians
    angle = torch.acos(cos_angle)  # (num_envs,)

    # Check if object is still in the path (angle < threshold)
    object_in_path = angle < angle_threshold  # (num_envs,) boolean

    # Project robot_to_object onto the path direction (along-path distance)
    # All in world frame now
    along_path_distance = torch.sum(robot_to_object_w * robot_to_goal_normalized, dim=1, keepdim=True)  # (num_envs, 1)

    # Compute the projection point on the path (in world frame)
    projection_on_path_w = robot_pos_w + along_path_distance * robot_to_goal_normalized  # (num_envs, 2)

    # Compute lateral (perpendicular) distance from object to path
    lateral_distance = torch.norm(object_pos_w - projection_on_path_w, dim=1)  # (num_envs,)

    # Clamp lateral distance to saturation threshold
    # This makes the reward reach maximum and stay there beyond the threshold
    lateral_distance_clamped = torch.clamp(lateral_distance, max=saturation_distance)

    # Reward: higher when object is farther from the path (up to saturation)
    # Use clamped distance so reward doesn't increase beyond saturation_distance
    reward = 1.0 - torch.exp(-torch.square(lateral_distance_clamped) / (std ** 2))

    # When object is not in the path (angle too large), give maximum reward
    # This tells the robot: "Good job! Object is cleared, now focus on reaching the goal"
    # Important: We give MAX reward (1.0), not zero, to avoid penalizing the desired outcome
    max_reward = torch.ones_like(reward)
    reward = torch.where(object_in_path, reward, max_reward)

    # Apply displacement threshold: reward is ZERO until object moves enough
    # This prevents robot from "cheating" by moving away without actually pushing the object
    zero_reward = torch.zeros_like(reward)
    reward = torch.where(object_moved_enough, reward, zero_reward)

    print(reward)

    return reward


def object_displacement_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    std: float = 0.3,
    saturation_distance: float = 0.5,
    decay_end_distance: float = 1.0,
) -> torch.Tensor:
    """Reward object displacement from its initial position (with decay and penalty).

    This function rewards object movement with three phases:
    1. 0 to saturation_distance: Positive reward (exponential growth)
    2. saturation_distance to decay_end_distance: Decay to zero (linear decay)
    3. Beyond decay_end_distance: Negative penalty (linear penalty increasing with distance)

    This encourages pushing the object to an optimal distance range, discouraging
    pushing it too far away.

    Args:
        env: The environment instance.
        object_cfg: Scene entity configuration for the object.
        std: Standard deviation for exponential kernel (default: 0.3m).
             Controls how quickly reward increases with displacement in phase 1.
        saturation_distance: Distance where reward reaches maximum and starts to decay (default: 0.5m).
        decay_end_distance: Distance where reward decays to zero and penalty begins (default: 1.0m).

    Returns:
        Reward tensor with shape (num_envs,).
    """
    # Get object position in world frame
    object_asset = env.scene[object_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :2]  # (num_envs, 2)

    # Initialize or update object initial position storage
    if not hasattr(env, '_object_initial_pos_for_displacement'):
        # First call: initialize storage
        env._object_initial_pos_for_displacement = object_pos_w.clone()

    # Update initial position for environments that just reset (episode_length_buf == 0)
    reset_envs = env.episode_length_buf == 0
    if reset_envs.any():
        env._object_initial_pos_for_displacement[reset_envs] = object_pos_w[reset_envs].clone()

    # Compute displacement from initial position
    displacement = torch.norm(object_pos_w - env._object_initial_pos_for_displacement, dim=1)  # (num_envs,)

    # Phase boundaries (now configurable via parameters)
    phase1_end = saturation_distance  # End of reward growth phase
    phase2_end = decay_end_distance  # End of decay phase, start of penalty

    # Calculate reward based on displacement range
    reward = torch.zeros_like(displacement)

    # Phase 1: 0 to saturation_distance - Exponential growth reward
    phase1_mask = displacement <= phase1_end
    if phase1_mask.any():
        displacement_phase1 = displacement[phase1_mask]
        reward[phase1_mask] = 1.0 - torch.exp(-torch.square(displacement_phase1) / (std ** 2))

    # Phase 2: saturation_distance to decay_end_distance - Linear decay from max reward to 0
    phase2_mask = (displacement > phase1_end) & (displacement <= phase2_end)
    if phase2_mask.any():
        displacement_phase2 = displacement[phase2_mask]
        # Calculate max reward at phase1_end
        max_reward = 1.0 - torch.exp(-torch.tensor(phase1_end ** 2) / (std ** 2))
        # Linear decay: reward goes from max_reward to 0
        decay_ratio = (phase2_end - displacement_phase2) / (phase2_end - phase1_end)
        reward[phase2_mask] = max_reward * decay_ratio

    # Phase 3: beyond decay_end_distance - Linear penalty (negative reward)
    phase3_mask = displacement > phase2_end
    if phase3_mask.any():
        displacement_phase3 = displacement[phase3_mask]
        # Linear penalty that increases with distance
        penalty_rate = 2.0  # Penalty increases at rate of -2.0 per meter
        reward[phase3_mask] = -penalty_rate * (displacement_phase3 - phase2_end)

    return reward


def arm_object_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    threshold: float = 0.1,
    command_name: str = "pose_command",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_distance_threshold: float = 0.8,
    goal_distance_threshold: float = 1.0,
) -> torch.Tensor:
    """奖励机械臂与物体的接触（仅在需要推动时）。

    只在障碍物挡路时才奖励接触，障碍物已移开或接近目标后不再奖励。
    这样可以避免机器人不必要地接触物体。

    奖励条件：
    1. 机械臂与物体有接触
    2. 障碍物还在路径上（横向距离 < lateral_distance_threshold）
    3. 距离目标还较远（距离 > goal_distance_threshold）

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置（机械臂部分）
        object_cfg: 物体配置
        threshold: 接触力阈值（N）
        command_name: 命令名称
        robot_cfg: 机器人配置
        lateral_distance_threshold: 认为障碍物"挡路"的横向距离阈值（米）
        goal_distance_threshold: 认为"接近目标"的距离阈值（米）

    Returns:
        奖励tensor，形状为(num_envs,)。只在需要推动时接触才给奖励。
    """
    from omni.isaac.lab.assets import RigidObject

    # 获取接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 1. 检测是否有接触
    net_contact_forces = contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    contact_force_magnitude = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1)  # (num_envs, num_arm_bodies)
    has_contact = torch.any(contact_force_magnitude > threshold, dim=-1).float()  # (num_envs,)

    # 2. 判断是否需要推动（障碍物是否挡路）
    object_view: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    # 获取机器人位置和朝向
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # 获取目标向量（机器人坐标系）并转换到世界坐标系
    from omni.isaac.lab.utils.math import quat_rotate, yaw_quat
    robot_to_goal_b = env.command_manager.get_command(command_name)[:, :2]  # (num_envs, 2) - robot frame
    robot_to_goal_b_3d = torch.cat([robot_to_goal_b, torch.zeros_like(robot_to_goal_b[:, :1])], dim=1)  # (num_envs, 3)
    robot_to_goal_w_3d = quat_rotate(yaw_quat(robot_quat_w), robot_to_goal_b_3d)  # (num_envs, 3)
    robot_to_goal_w = robot_to_goal_w_3d[:, :2]  # (num_envs, 2)
    direction_normalized_w = robot_to_goal_w / (torch.norm(robot_to_goal_w, dim=1, keepdim=True) + 1e-6)  # (num_envs, 2)

    # 获取障碍物位置（世界坐标系）
    object_pos_w = object_view.data.root_pos_w[:, :2]  # (num_envs, 2)
    robot_to_object_w = object_pos_w - robot_pos_w  # (num_envs, 2)

    # 计算障碍物到路径的横向距离（在世界坐标系中）
    longitudinal = torch.sum(robot_to_object_w * direction_normalized_w, dim=1, keepdim=True)  # (num_envs, 1)
    lateral_vec = robot_to_object_w - longitudinal * direction_normalized_w  # (num_envs, 2)
    lateral_distance = torch.norm(lateral_vec, dim=1)  # (num_envs,)

    # 3. 判断障碍物是否挡路（横向距离小）
    obstacle_blocking = lateral_distance < lateral_distance_threshold

    # 4. 判断是否距离目标还远（使用机器人坐标系的command即可）
    distance_to_goal = torch.norm(robot_to_goal_b, dim=1)
    far_from_goal = distance_to_goal > goal_distance_threshold

    # 5. 只有在需要推动时（障碍物挡路且距离目标远）才奖励接触
    should_push = obstacle_blocking & far_from_goal

    # 6. 最终奖励：有接触 且 需要推动
    reward = has_contact * should_push.float()

    print(reward)

    return reward


def door_opening_angle_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    open_angle_threshold: float = 0.785,  # 45度 = π/4 ≈ 0.785弧度
    std: float = 0.3,  # 标准差，控制奖励增长速度
) -> torch.Tensor:
    """Reward door opening angle (only before reaching threshold).

    This function rewards the robot for opening the door. The reward increases with the door opening angle
    up to the threshold angle. Once the door is opened beyond the threshold, the reward reaches maximum.

    The reward uses an exponential kernel based on the remaining angle to target:
    - Door closed (angle = 0): reward = 0 (minimum)
    - Door opening (0 < angle < threshold): reward increases exponentially
    - Door opened enough (angle >= threshold): reward = 1.0 (maximum)

    Args:
        env: The environment instance.
        object_cfg: Scene entity configuration for the door object.
        open_angle_threshold: Door opening angle threshold in radians (default: 0.785 rad = 45°).
                             When door angle >= threshold, reward reaches maximum.
        std: Standard deviation for exponential kernel (default: 0.3).
             Controls how quickly the reward grows with opening angle.

    Returns:
        Reward tensor with shape (num_envs,).
    """
    from omni.isaac.lab.utils.math import euler_xyz_from_quat

    # Get door object orientation
    object = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w  # (num_envs, 4)

    # Calculate door's current yaw angle from quaternion
    _, _, current_door_yaw = euler_xyz_from_quat(object_quat_w)  # (num_envs,)

    # Store initial door yaw on first call (per environment)
    if not hasattr(env, '_initial_door_yaw_for_opening_reward'):
        env._initial_door_yaw_for_opening_reward = current_door_yaw.clone()

    # Calculate door opening angle (difference from initial yaw)
    door_opening_angle = torch.abs(current_door_yaw - env._initial_door_yaw_for_opening_reward)  # (num_envs,)

    # Normalize angle to [0, π] range
    door_opening_angle = torch.min(door_opening_angle, 2 * torch.pi - door_opening_angle)

    # Calculate reward based on opening angle
    # If angle >= threshold: reward = 1.0 (maximum)
    # If angle < threshold: reward grows exponentially as angle approaches threshold

    # Calculate the "remaining angle" to reach threshold
    remaining_angle = torch.clamp(open_angle_threshold - door_opening_angle, min=0.0)  # (num_envs,)

    # Exponential reward: as remaining_angle -> 0, reward -> 1.0
    # exp(-remaining_angle^2 / std^2)
    reward = torch.exp(-torch.square(remaining_angle) / (std ** 2))

    # print(reward)

    # Alternative: Linear reward (uncomment if you prefer linear growth)
    # reward = door_opening_angle / open_angle_threshold
    # reward = torch.clamp(reward, 0.0, 1.0)

    # Debug print (optional, uncomment to see door opening progress)
    # if env.episode_length_buf[0] % 100 == 0:
    #     print(f"门打开角度(度) (前3个): {(door_opening_angle[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"奖励值 (前3个): {reward[:3].cpu().numpy()}")
    #     print(f"剩余角度(度) (前3个): {(remaining_angle[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"阈值角度: {open_angle_threshold * 180 / 3.14159:.1f}度")

    return reward


def track_pos_when_door_open(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    open_angle_threshold: float = 0.785,  # 45度 = π/4 ≈ 0.785弧度
    std: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of target position only when door is opened beyond threshold.

    This function rewards the robot for reaching a target position, but only provides
    the reward when the door has been opened beyond the specified angle threshold.
    Before the door is opened enough, the reward is zero to encourage the robot to
    open the door first.

    The position tracking reward uses: reward = 1 - (distance_to_target / range)
    where the target position comes from the command manager.

    Args:
        env: The environment instance.
        command_name: Name of the command term to get the position command.
        object_cfg: Scene entity configuration for the door object.
        asset_cfg: Scene entity configuration for the robot (default: "robot").
        open_angle_threshold: Door opening angle threshold in radians (default: 0.785 rad = 45°).
                             Only give reward when door angle >= threshold.

    Returns:
        Reward tensor with shape (num_envs,). Zero when door not open enough.
    """
    from omni.isaac.lab.utils.math import euler_xyz_from_quat

    # Get door object orientation
    door_obj = env.scene[object_cfg.name]
    door_quat_w = door_obj.data.root_quat_w  # (num_envs, 4)

    # Calculate door's current yaw angle from quaternion
    _, _, current_door_yaw = euler_xyz_from_quat(door_quat_w)  # (num_envs,)

    # Store initial door yaw on first call (per environment)
    if not hasattr(env, '_initial_door_yaw_for_track_pos'):
        env._initial_door_yaw_for_track_pos = current_door_yaw.clone()

    # Calculate door opening angle (difference from initial yaw)
    door_opening_angle = torch.abs(current_door_yaw - env._initial_door_yaw_for_track_pos)  # (num_envs,)

    # Normalize angle to [0, π] range
    door_opening_angle = torch.min(door_opening_angle, 2 * torch.pi - door_opening_angle)

    # Check if door has been opened enough
    door_open_enough = door_opening_angle >= open_angle_threshold  # (num_envs,)

    # Get position tracking reward
    # The command is the remaining distance to the target position
    command = env.command_manager.get_command(command_name)  # (num_envs, >=2)

    # Compute distance to target (using x, y components only)
    distance_to_target = torch.norm(command[:, :2], dim=1)  # (num_envs,)

    # Position tracking reward: closer to target = higher reward
    # Assuming the command range is normalized, we use a simple linear reward
    pos_reward = 1 - distance_to_target/std

    # Only return reward when door is open enough, otherwise zero
    return torch.where(door_open_enough, pos_reward, torch.zeros_like(pos_reward))


def object_orientation_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_tilt_angle: float = 0.3,
) -> torch.Tensor:
    """惩罚物体倾斜/翻倒,鼓励物体保持直立。

    该函数通过检查物体的朝向来判断物体是否倾斜或翻倒:
    - 完全直立(z轴朝上): 惩罚 = 0 (无惩罚)
    - 轻微倾斜(角度 < max_tilt_angle): 惩罚较小 (二次增长)
    - 严重倾斜或翻倒: 惩罚较大 (角度越大惩罚越大)

    实现原理:
    1. 获取物体的四元数姿态
    2. 将物体局部z轴(向上方向)旋转到世界坐标系
    3. 计算物体z轴与世界z轴的夹角
    4. 根据倾斜角度计算惩罚值

    Args:
        env: 环境实例
        object_cfg: 物体的场景实体配置 (如 SceneEntityCfg("interactive_object"))
        max_tilt_angle: 最大允许倾斜角度(弧度),超过此角度惩罚快速增加 (默认: 0.3 rad ≈ 17度)

    Returns:
        惩罚张量,形状为 (num_envs,)
    """
    # 获取物体
    obj = env.scene[object_cfg.name]

    # 获取物体的四元数姿态 (世界坐标系)
    obj_quat_w = obj.data.root_link_quat_w  # (num_envs, 4) - [w, x, y, z]

    # 物体局部坐标系的z轴向量 (向上)
    local_z = torch.zeros((env.num_envs, 3), device=env.device)
    local_z[:, 2] = 1.0  # (0, 0, 1)

    # 将物体局部z轴旋转到世界坐标系
    obj_z_world = quat_rotate(obj_quat_w, local_z)  # (num_envs, 3)

    # 世界坐标系的z轴 (向上)
    world_z = torch.zeros((env.num_envs, 3), device=env.device)
    world_z[:, 2] = 1.0

    # 计算物体z轴与世界z轴的点积 (余弦值)
    # 如果物体完全直立,点积=1; 如果物体倾斜90度,点积=0; 如果物体翻倒180度,点积=-1
    cos_angle = torch.sum(obj_z_world * world_z, dim=1)  # (num_envs,)

    # 将余弦值限制在[-1, 1]范围内,避免数值误差
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # 计算倾斜角度 (弧度)
    tilt_angle = torch.acos(cos_angle)  # (num_envs,) - 范围 [0, π]
    # print(tilt_angle)

    # 计算惩罚
    # 方法: 当倾斜角度超过max_tilt_angle时,惩罚快速增加
    # penalty = (max(0, tilt_angle - max_tilt_angle))^2
    angle_violation = torch.clamp(tilt_angle - max_tilt_angle, min=0.0)
    penalty = torch.square(angle_violation)

    return penalty


def robot_object_goal_alignment_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
    std: float = 0.5,
    min_displacement_threshold: float = 0.5,
) -> torch.Tensor:
    """惩罚物体位于机器人到目标的路径上（夹角小）。

    该函数计算两个向量之间的夹角：
    1. 机器人→目标 的向量
    2. 机器人→物体 的向量

    当夹角较小时（物体在机器人前往目标的路径上），给予惩罚。
    当夹角较大时（物体已经偏离路径），惩罚较小或为零。

    这鼓励机器人将物体推离到达目标的路径，避免物体阻挡。

    **重要**：惩罚只在物体移动≥min_displacement_threshold米后才生效，
    这样可以先让机器人推动物体，然后再惩罚物体仍在路径上的情况。

    惩罚使用指数核函数：
    - 夹角 = 0°（物体正好在路径上）: penalty = 1.0 (最大惩罚)
    - 夹角增大: penalty 指数衰减
    - 夹角 = 90°或更大: penalty ≈ 0 (物体已经偏离路径，无需惩罚)
    - 物体位移 < min_displacement_threshold: penalty = 0 (还未推动足够远，暂不惩罚)

    Args:
        env: 环境实例
        object_cfg: 物体的场景实体配置（如 SceneEntityCfg("interactive_object")）
        robot_cfg: 机器人的场景实体配置（默认: "robot"）
        command_name: 命令名称，用于获取目标位置（默认: "pose_command"）
        std: 标准差，控制惩罚衰减速度（默认: 0.5弧度 ≈ 28.6度）
             较小的std会使惩罚衰减更快
        min_displacement_threshold: 最小位移阈值（米），默认0.5m
                                   物体必须移动≥此距离后才开始惩罚

    Returns:
        惩罚张量，形状为 (num_envs,)
    """
    from omni.isaac.lab.utils.math import quat_rotate, yaw_quat

    # 获取机器人位置和朝向
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - 只考虑x, y
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # 获取物体位置
    object_asset = env.scene[object_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :2]  # (num_envs, 2)

    # 检查物体位移是否达到阈值
    if min_displacement_threshold > 0.0:
        # 初始化或更新物体初始位置存储
        if not hasattr(env, '_object_initial_pos_for_alignment_penalty'):
            # 首次调用：初始化存储
            env._object_initial_pos_for_alignment_penalty = object_pos_w.clone()

        # 更新刚重置的环境的初始位置（episode_length_buf == 0）
        reset_envs = env.episode_length_buf == 0
        if reset_envs.any():
            env._object_initial_pos_for_alignment_penalty[reset_envs] = object_pos_w[reset_envs].clone()

        # 计算物体从初始位置的位移
        object_displacement = torch.norm(object_pos_w - env._object_initial_pos_for_alignment_penalty, dim=1)  # (num_envs,)

        # 检查物体是否移动足够远
        object_moved_enough = object_displacement >= min_displacement_threshold  # (num_envs,) boolean
    else:
        # 无位移阈值：总是启用惩罚
        object_moved_enough = torch.ones(object_pos_w.shape[0], dtype=torch.bool, device=object_pos_w.device)

    # 获取目标位置命令（在机器人坐标系中）
    command = env.command_manager.get_command(command_name)
    robot_to_goal_b = command[:, :2]  # (num_envs, 2) - 机器人坐标系

    # 将目标向量从机器人坐标系转换到世界坐标系
    robot_to_goal_b_3d = torch.cat([robot_to_goal_b, torch.zeros_like(robot_to_goal_b[:, :1])], dim=1)  # (num_envs, 3)
    robot_to_goal_w_3d = quat_rotate(yaw_quat(robot_quat_w), robot_to_goal_b_3d)  # (num_envs, 3)
    robot_to_goal_w = robot_to_goal_w_3d[:, :2]  # (num_envs, 2)

    # 计算机器人→物体的向量（世界坐标系）
    robot_to_object_w = object_pos_w - robot_pos_w  # (num_envs, 2)

    # 归一化两个向量
    robot_to_goal_norm = torch.norm(robot_to_goal_w, dim=1, keepdim=True)  # (num_envs, 1)
    robot_to_goal_norm = torch.clamp(robot_to_goal_norm, min=0.01)  # 避免除零
    robot_to_goal_normalized = robot_to_goal_w / robot_to_goal_norm  # (num_envs, 2)

    robot_to_object_norm = torch.norm(robot_to_object_w, dim=1, keepdim=True)  # (num_envs, 1)
    robot_to_object_norm = torch.clamp(robot_to_object_norm, min=0.01)  # 避免除零
    robot_to_object_normalized = robot_to_object_w / robot_to_object_norm  # (num_envs, 2)

    # 计算夹角的余弦值
    # cos(angle) = dot(v1, v2) / (|v1| * |v2|)
    # 由于向量已归一化，cos(angle) = dot(v1, v2)
    cos_angle = torch.sum(robot_to_goal_normalized * robot_to_object_normalized, dim=1)  # (num_envs,)

    # 限制在有效范围内（数值稳定性）
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)

    # 计算夹角（弧度）
    angle = torch.acos(cos_angle)  # (num_envs,) - 范围 [0, π]

    # 使用指数核计算惩罚
    # 夹角越小（物体在路径上），惩罚越高
    # penalty = exp(-angle^2 / std^2)
    penalty = torch.exp(-torch.square(angle) / (std ** 2))

    # 只有在物体移动足够远后才应用惩罚，否则惩罚为0
    zero_penalty = torch.zeros_like(penalty)
    penalty = torch.where(object_moved_enough, penalty, zero_penalty)

    # 调试输出（可选，取消注释以查看角度信息）
    # if env.episode_length_buf[0] % 100 == 0:
    #     print(f"夹角(度) (前3个): {(angle[:3] * 180 / 3.14159).cpu().numpy()}")
    #     print(f"惩罚值 (前3个): {penalty[:3].cpu().numpy()}")
    #     print(f"物体已移动足够远 (前3个): {object_moved_enough[:3].cpu().numpy()}")

    return penalty



def object_near_goal_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
    min_distance: float = 0.5,
    std: float = 0.3,
) -> torch.Tensor:
    """惩罚物体离目标点太近。

    该函数计算物体与目标点之间的距离，当物体太靠近目标时给予惩罚。
    这鼓励机器人将障碍物推离目标区域，避免障碍物堵在目标附近。

    惩罚使用指数核函数：
    - 距离 = 0（物体正好在目标点）: penalty = 1.0 (最大惩罚)
    - 距离 < min_distance: penalty 较大，随距离增加而衰减
    - 距离 >= min_distance: penalty = 0 (距离足够远，不惩罚)

    应用场景：
    - 防止机器人将障碍物推到目标点附近
    - 确保目标区域保持清空
    - 配合其他reward，引导机器人将障碍物推向合适的位置

    Args:
        env: 环境实例
        object_cfg: 物体的场景实体配置（如 SceneEntityCfg("interactive_object")）
        robot_cfg: 机器人的场景实体配置（默认: "robot"）
        command_name: 命令名称，用于获取目标位置（默认: "pose_command"）
        min_distance: 最小安全距离（米），默认0.5m
                     物体距离目标 < min_distance 时会受到惩罚
                     距离 >= min_distance 时惩罚为0
        std: 标准差，控制惩罚衰减速度（默认: 0.3米）
             较小的std会使惩罚衰减更快

    Returns:
        惩罚张量，形状为 (num_envs,)
    """
    from omni.isaac.lab.utils.math import quat_rotate, yaw_quat

    # 获取机器人位置和朝向
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - 只考虑x, y
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # 获取物体位置
    object_asset = env.scene[object_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :2]  # (num_envs, 2)

    # 获取目标位置命令（在机器人坐标系中）
    command = env.command_manager.get_command(command_name)
    robot_to_goal_b = command[:, :2]  # (num_envs, 2) - 机器人坐标系

    # 将目标向量从机器人坐标系转换到世界坐标系
    robot_to_goal_b_3d = torch.cat([robot_to_goal_b, torch.zeros_like(robot_to_goal_b[:, :1])], dim=1)  # (num_envs, 3)
    robot_to_goal_w_3d = quat_rotate(yaw_quat(robot_quat_w), robot_to_goal_b_3d)  # (num_envs, 3)
    robot_to_goal_w = robot_to_goal_w_3d[:, :2]  # (num_envs, 2)

    # 计算目标点在世界坐标系中的位置
    goal_pos_w = robot_pos_w + robot_to_goal_w  # (num_envs, 2)

    # 计算物体到目标点的距离
    object_to_goal_distance = torch.norm(object_pos_w - goal_pos_w, dim=1)  # (num_envs,)

    # 计算惩罚
    # 只有当距离 < min_distance 时才有惩罚
    # 使用 clamp 将超出 min_distance 的距离设为 min_distance（这样惩罚为0）
    effective_distance = torch.clamp(object_to_goal_distance, max=min_distance)

    # 计算距离违反量（离目标越近，违反越严重）
    distance_violation = min_distance - effective_distance  # (num_envs,)

    # 使用指数核计算惩罚
    # penalty = exp(-distance_violation^2 / std^2)
    # 当 distance_violation = 0（距离刚好等于min_distance）时，penalty = exp(0) = 1.0
    # 但我们想要距离 >= min_distance 时 penalty = 0，所以需要调整公式
    # penalty = 1 - exp(-distance_violation^2 / std^2) 会在distance_violation=0时给penalty=0
    # 但这个公式不够直观，改用更简单的：
    # 当距离 < min_distance 时，penalty = exp(-(min_distance - distance)^2 / std^2)
    penalty = torch.exp(-torch.square(distance_violation) / (std ** 2))

    # 当距离 >= min_distance 时，强制惩罚为0
    penalty = torch.where(object_to_goal_distance >= min_distance, torch.zeros_like(penalty), penalty)

    # 调试输出（可选，取消注释以查看距离信息）
    # if env.episode_length_buf[0] % 100 == 0:
    #     print(f"物体到目标距离(m) (前3个): {object_to_goal_distance[:3].cpu().numpy()}")
    #     print(f"惩罚值 (前3个): {penalty[:3].cpu().numpy()}")

    return penalty


def track_velocity_towards_target_when_door_open(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    open_angle_threshold: float = 0.785,
    std: float = 0.5,
) -> torch.Tensor:
    """在门打开后，奖励机器人朝向目标位置运动的速度。

    这个函数仅在门打开超过阈值后才给予奖励，鼓励机器人在开门后
    以合适的速度朝向目标移动。奖励基于机器人速度与目标方向的点积。

    Args:
        env: 环境实例
        command_name: 命令名称（用于获取目标位置）
        object_cfg: 门对象的场景实体配置
        asset_cfg: 机器人的场景实体配置（默认: "robot"）
        open_angle_threshold: 门打开角度阈值（弧度），默认0.785 rad = 45°
        std: 标准差，用于指数奖励核（默认: 0.5）

    Returns:
        奖励张量，形状为 (num_envs,)。门未充分打开时返回0。
    """
    from omni.isaac.lab.utils.math import euler_xyz_from_quat

    # 获取门对象姿态
    door_obj = env.scene[object_cfg.name]
    door_quat_w = door_obj.data.root_quat_w  # (num_envs, 4)

    # 计算门的当前yaw角度
    _, _, current_door_yaw = euler_xyz_from_quat(door_quat_w)  # (num_envs,)

    # 存储初始门yaw角度（每个环境）
    if not hasattr(env, '_initial_door_yaw_for_vel_track'):
        env._initial_door_yaw_for_vel_track = current_door_yaw.clone()

    # 计算门开启角度（与初始yaw的差值）
    door_opening_angle = torch.abs(current_door_yaw - env._initial_door_yaw_for_vel_track)  # (num_envs,)

    # 归一化角度到 [0, π] 范围
    door_opening_angle = torch.min(door_opening_angle, 2 * torch.pi - door_opening_angle)

    # 检查门是否已经打开足够
    door_open_enough = door_opening_angle >= open_angle_threshold  # (num_envs,)

    # 获取机器人
    robot = env.scene[asset_cfg.name]

    # 获取目标位置命令（在机器人坐标系中）
    command = env.command_manager.get_command(command_name)  # (num_envs, >=2)
    target_direction = command[:, :2]  # (num_envs, 2) - x, y方向

    # 归一化目标方向（避免除零）
    target_direction_norm = torch.norm(target_direction, dim=1, keepdim=True)  # (num_envs, 1)
    target_direction_normalized = torch.where(
        target_direction_norm > 0.01,
        target_direction / target_direction_norm,
        torch.zeros_like(target_direction)
    )  # (num_envs, 2)

    # 获取机器人在yaw-aligned坐标系中的速度
    robot_vel_world = robot.data.root_com_lin_vel_w[:, :2]  # (num_envs, 2) - xy速度（世界坐标系）
    robot_quat = robot.data.root_link_quat_w  # (num_envs, 4)

    # 将速度从世界坐标系转换到机器人yaw-aligned坐标系
    robot_vel_yaw_frame = quat_rotate_inverse(yaw_quat(robot_quat),
                                              torch.cat([robot_vel_world,
                                                        torch.zeros((env.num_envs, 1), device=env.device)],
                                                       dim=1))[:, :2]  # (num_envs, 2)

    # 计算速度与目标方向的点积（衡量是否朝向目标运动）
    velocity_alignment = torch.sum(robot_vel_yaw_frame * target_direction_normalized, dim=1)  # (num_envs,)

    # 使用指数核奖励：当速度与目标方向对齐且速度适中时奖励最高
    # 期望速度约为 std，太快或太慢都会降低奖励
    velocity_magnitude = torch.norm(robot_vel_yaw_frame, dim=1)  # (num_envs,)

    # 方向对齐奖励（0到1之间）
    direction_reward = torch.clamp(velocity_alignment, min=0.0)  # 只奖励朝向目标的运动

    # 速度大小奖励（使用指数核，期望速度为std）
    speed_reward = torch.exp(-((velocity_magnitude - std) ** 2) / (2 * std ** 2))

    # 组合奖励
    reward = direction_reward * speed_reward

    # 只在门打开足够时给予奖励，否则为0
    return torch.where(door_open_enough, reward, torch.zeros_like(reward))


def forward_velocity_towards_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """奖励机器人正面朝向目标并向前移动（不斜着走）。

    这个函数同时奖励两个方面：
    1. 机器人朝向与目标方向对齐（heading alignment）
    2. 机器人沿自身朝向前进，而不是侧向或后退（forward motion）

    通过结合这两个奖励，鼓励机器人：
    - 转向目标
    - 正面向前走向目标
    - 避免横着走、斜着走或后退

    Args:
        env: 环境实例
        command_name: 命令名称（用于获取目标位置）
        asset_cfg: 机器人的场景实体配置（默认: "robot"）
        std: 期望的前进速度（m/s），默认0.5

    Returns:
        奖励张量，形状为 (num_envs,)
    """
    from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

    # 获取机器人
    robot = env.scene[asset_cfg.name]

    # 获取目标位置命令（在机器人坐标系中）
    command = env.command_manager.get_command(command_name)  # (num_envs, >=2)
    target_direction_body = command[:, :2]  # (num_envs, 2) - 机器人坐标系中的目标方向

    # 计算到目标的距离
    distance_to_goal = torch.norm(target_direction_body, dim=1)  # (num_envs,)

    # 归一化目标方向
    target_direction_normalized = torch.where(
        distance_to_goal.unsqueeze(1) > 0.01,
        target_direction_body / distance_to_goal.unsqueeze(1),
        torch.zeros_like(target_direction_body)
    )  # (num_envs, 2)

    # 1. 朝向对齐奖励：目标在机器人正前方（x轴）时奖励最高
    # target_direction_normalized[0] = x方向分量，越接近1说明目标在正前方
    heading_alignment = torch.clamp(target_direction_normalized[:, 0], min=0.0)  # (num_envs,)

    # 2. 前进速度奖励：获取机器人速度（机器人坐标系）
    robot_vel_world = robot.data.root_com_lin_vel_w[:, :2]  # (num_envs, 2) - 世界坐标系
    robot_quat = robot.data.root_link_quat_w  # (num_envs, 4)

    # 转换到机器人yaw-aligned坐标系
    robot_vel_body = quat_rotate_inverse(
        yaw_quat(robot_quat),
        torch.cat([robot_vel_world, torch.zeros((env.num_envs, 1), device=env.device)], dim=1)
    )[:, :2]  # (num_envs, 2)

    # 前向速度（x方向）
    forward_velocity = robot_vel_body[:, 0]  # (num_envs,)

    # 侧向速度（y方向，绝对值，用于惩罚横着走）
    lateral_velocity = torch.abs(robot_vel_body[:, 1])  # (num_envs,)

    # 3. 速度质量奖励
    # 奖励前进速度接近期望速度std
    forward_speed_reward = torch.exp(-((forward_velocity - std) ** 2) / (2 * 0.1 ** 2))

    # 惩罚侧向速度（横着走）
    lateral_penalty = torch.exp(-lateral_velocity / (0.3 * std))  # 侧向速度越大，惩罚越大

    # 4. 综合奖励
    # heading_alignment: 鼓励转向目标 (0-1)
    # forward_speed_reward: 鼓励合适的前进速度 (0-1)
    # lateral_penalty: 惩罚横向移动 (0-1)
    reward = heading_alignment * forward_speed_reward * lateral_penalty

    # 只在距离目标较远时才给奖励，避免到达目标后还在原地乱动
    reward = torch.where(distance_to_goal > 0.3, reward, torch.zeros_like(reward))

    # print(reward)

    return reward


def arm_joint_range_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """惩罚机械臂关节超出合法范围的动作。

    ARX5机械臂关节角度范围（从度数转换为弧度）:
    - 关节1: -150° ~ 180° → -2.618 ~ 3.142 rad
    - 关节2:    0° ~ 210° →  0.0 ~ 3.665 rad
    - 关节3:    0° ~ 180° →  0.0 ~ 3.142 rad
    - 关节4:  -90° ~  90° → -1.571 ~ 1.571 rad
    - 关节5:  -90° ~  90° → -1.571 ~ 1.571 rad
    - 关节6:  -90° ~  90° → -1.571 ~ 1.571 rad

    该函数计算超出范围部分的平方，并求和作为惩罚。

    Args:
        env: 环境实例

    Returns:
        惩罚张量，形状为 (num_envs,)。关节在范围内时惩罚为0。
    """
    import math

    # 获取机械臂关节动作 (action dim 3-8, 共6个关节)
    arm_actions = env.action_manager.action[:, 3:]  # (num_envs, 6)

    # 定义每个关节的范围 (最小值, 最大值，单位: 弧度)
    # 度数到弧度的转换: rad = deg * π / 180
    joint_limits = torch.tensor([
        [-2.618, 3.142],   # 关节1: -150° ~ 180°
        [0.0, 3.665],      # 关节2:    0° ~ 210°
        [0.0, 3.142],      # 关节3:    0° ~ 180°
        [-1.571, 1.571],   # 关节4:  -90° ~  90°
        [-1.571, 1.571],   # 关节5:  -90° ~  90°
        [-1.571, 1.571],   # 关节6:  -90° ~  90°
    ], device=env.device, dtype=torch.float32)  # (6, 2)

    # 计算每个关节的范围违反情况
    min_limits = joint_limits[:, 0]  # (6,)
    max_limits = joint_limits[:, 1]  # (6,)

    # 计算低于最小值的违反量
    below_min = torch.clamp(min_limits - arm_actions, min=0.0)  # (num_envs, 6)
    penalty_low = torch.square(below_min)  # (num_envs, 6)

    # 计算高于最大值的违反量
    above_max = torch.clamp(arm_actions - max_limits, min=0.0)  # (num_envs, 6)
    penalty_high = torch.square(above_max)  # (num_envs, 6)

    # 总惩罚: 求和所有关节的违反量
    total_penalty = torch.sum(penalty_low + penalty_high, dim=1)  # (num_envs,)

    return total_penalty


def stand_still_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 1.0
) -> torch.Tensor:
    """到达目标后惩罚关节速度，避免原地踏步。

    当机器人距离目标点小于阈值时，惩罚其关节速度。
    这样可以避免机器人到达目标后腿部还在运动导致原地踏步。

    原理：
    - 机器人可能基座速度为0（停在原地），但腿部关节还在运动
    - 这会导致"原地踏步"现象
    - 通过惩罚关节速度，鼓励机器人完全静止

    Args:
        env: 环境实例
        command_name: 命令管理器中的命令名称（如"pose_command"）
        asset_cfg: 机器人资产配置
        distance_threshold: 认为"到达目标"的距离阈值（米），默认1.0米

    Returns:
        惩罚值tensor，形状为(num_envs,)。只有到达目标的环境才有非零惩罚。
    """
    from omni.isaac.lab.assets import Articulation

    # 获取机器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算到目标的距离（只考虑xy平面）
    goal_track_error_pos = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)

    # 判断哪些环境已到达目标
    reached_mark = goal_track_error_pos < distance_threshold

    # 初始化惩罚为0
    penalty = torch.zeros(env.num_envs, device=env.device)

    # 只对到达目标的环境计算惩罚
    # 惩罚所有关节的速度（主要是腿部关节）
    if torch.any(reached_mark):
        # 获取所有关节的速度 (num_envs, num_joints)
        joint_vel = asset.data.joint_vel

        # 计算关节速度的L2范数作为惩罚
        # 只惩罚到达目标的环境
        penalty[reached_mark] = torch.norm(joint_vel[reached_mark, :], dim=-1)

    return penalty


def joint_pos_to_default_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """到达目标后奖励关节角度回到默认位置。

    当机器人到达目标点时，鼓励所有关节（特别是机械臂）回到默认姿态。
    这样可以让机器人在完成任务后呈现标准的站立姿态。

    奖励计算：使用负的L2距离，关节越接近默认位置，奖励越大
    reward = -||joint_pos - default_pos||

    Args:
        env: 环境实例
        command_name: 命令管理器中的命令名称（如"pose_command"）
        asset_cfg: 机器人资产配置
        distance_threshold: 认为"到达目标"的距离阈值（米），默认0.5米

    Returns:
        奖励tensor，形状为(num_envs,)。只有到达目标的环境才有非零奖励。
        奖励值为负的关节位置偏差（越接近默认位置，奖励越大）
    """
    from omni.isaac.lab.assets import Articulation

    # 获取机器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算到目标的距离（只考虑xy平面）
    goal_track_error_pos = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)

    # 判断哪些环境已到达目标
    reached_mark = goal_track_error_pos < distance_threshold

    # 初始化奖励为0
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # 只对到达目标的环境计算奖励
    if torch.any(reached_mark):
        # 获取当前关节位置 (num_envs, num_joints)
        joint_pos = asset.data.joint_pos

        # 获取默认关节位置
        default_joint_pos = asset.data.default_joint_pos

        # 扩展default_joint_pos到所有环境
        if default_joint_pos.dim() == 1:
            # (num_joints,) -> (1, num_joints) -> (num_envs, num_joints)
            default_joint_pos = default_joint_pos.unsqueeze(0).expand(env.num_envs, -1)

        # 计算关节位置与默认位置的偏差
        joint_pos_error = joint_pos - default_joint_pos  # (num_envs, num_joints)

        # 计算L2范数（偏差的大小） - 沿着关节维度
        joint_pos_deviation = torch.norm(joint_pos_error, dim=1)  # (num_envs,)

        # 只对到达目标的环境赋值奖励
        reward[reached_mark] = -joint_pos_deviation[reached_mark]

    return reward


def front_feet_object_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg_FL: SceneEntityCfg,
    sensor_cfg_FR: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize front feet contact with the Interactive_object.

    This penalty discourages the robot from using its front feet (FL or FR) to push
    the object. The robot should learn to use its arm or body to manipulate the object.

    Args:
        env: The environment instance.
        sensor_cfg_FL: Front-left foot contact sensor configuration (filtered for Interactive_object).
        sensor_cfg_FR: Front-right foot contact sensor configuration (filtered for Interactive_object).
        threshold: Contact force threshold in Newtons (default: 1.0N).

    Returns:
        Penalty value tensor of shape (num_envs,). Returns the number of front feet
        in contact with the object (0, 1, or 2).
    """
    # Get contact sensors for front feet (filtered for Interactive_object only)
    contact_sensor_FL: ContactSensor = env.scene.sensors[sensor_cfg_FL.name]
    contact_sensor_FR: ContactSensor = env.scene.sensors[sensor_cfg_FR.name]

    # Get contact force matrix (filtered for Interactive_object only)
    # force_matrix_w shape: (num_envs, num_bodies=1, num_filtered_objects=1, 3)
    force_FL = contact_sensor_FL.data.force_matrix_w  # (num_envs, 1, 1, 3)
    force_FR = contact_sensor_FR.data.force_matrix_w  # (num_envs, 1, 1, 3)

    # Compute contact force magnitude for each foot
    # torch.norm computes L2 norm across the last dimension (xyz forces)
    force_norm_FL = torch.norm(force_FL, dim=-1).squeeze(-1).squeeze(-1)  # (num_envs,)
    force_norm_FR = torch.norm(force_FR, dim=-1).squeeze(-1).squeeze(-1)  # (num_envs,)

    # Check if contact force exceeds threshold
    is_contact_FL = force_norm_FL > threshold  # (num_envs,)
    is_contact_FR = force_norm_FR > threshold  # (num_envs,)

    # Count number of front feet in contact (0, 1, or 2)
    num_feet_in_contact = is_contact_FL.float() + is_contact_FR.float()

    return num_feet_in_contact


def excessive_velocity_penalty(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear velocity exceeding a threshold.

    This penalty encourages the robot to move at a safe speed. When the velocity
    exceeds the threshold, a penalty proportional to the excess velocity is applied.

    Args:
        env: The environment instance.
        velocity_threshold: Maximum allowed velocity magnitude in m/s (default: 0.8).
        asset_cfg: Asset configuration (default: robot).

    Returns:
        Penalty value tensor of shape (num_envs,). Returns 0 if velocity is below
        threshold, otherwise returns the squared excess velocity.
    """
    # Get robot asset
    asset = env.scene[asset_cfg.name]

    # Get base linear velocity in world frame (x, y, z)
    # We only care about horizontal velocity (x, y)
    lin_vel_w = asset.data.root_lin_vel_w[:, :2]  # (num_envs, 2)

    # Compute velocity magnitude (speed)
    velocity_magnitude = torch.norm(lin_vel_w, dim=1)  # (num_envs,)

    # Calculate excess velocity (how much over the threshold)
    excess_velocity = torch.clamp(velocity_magnitude - velocity_threshold, min=0.0)

    # Return squared excess velocity as penalty
    return torch.square(excess_velocity)
