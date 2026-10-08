# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_link_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""


class PreTrainedLocomanipPolicyAction(ActionTerm):
    r"""Pre-trained loco-manipulation policy action term for legged robots with arms.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy, including both base velocity
    commands (3D: lin_vel_x, lin_vel_y, ang_vel_z) and end-effector pose commands (7D: pos_xyz + quat_wxyz).

    High-level action dimension: 10 (3 for velocity + 7 for ee_pose)
    """

    cfg: PreTrainedLocomanipPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedLocomanipPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # split raw actions into velocity and pose commands
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._ee_pose_commands = torch.zeros(self.num_envs, 7, device=self.device)   # [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]

        # store ee_pose range limits as tensors if configured
        if cfg.ee_pose_pos_range is not None:
            self._ee_pos_min = torch.tensor(
                [cfg.ee_pose_pos_range[0][0], cfg.ee_pose_pos_range[1][0], cfg.ee_pose_pos_range[2][0]],
                device=self.device
            )
            self._ee_pos_max = torch.tensor(
                [cfg.ee_pose_pos_range[0][1], cfg.ee_pose_pos_range[1][1], cfg.ee_pose_pos_range[2][1]],
                device=self.device
            )
        else:
            self._ee_pos_min = None
            self._ee_pos_max = None

        # store fixed quaternion if configured
        if cfg.ee_pose_fixed_quat is not None:
            self._ee_fixed_quat = torch.tensor(cfg.ee_pose_fixed_quat, device=self.device)
        else:
            self._ee_fixed_quat = None

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._velocity_commands
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.pose_command.func = lambda dummy_env: self._ee_pose_commands
        cfg.low_level_observations.pose_command.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 10  # 3 (velocity) + 7 (ee_pose)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # clip velocity commands (first 3 dims)
        if self.cfg.velocity_clip is not None:
            actions[:, :3] = actions[:, :3].clip(-self.cfg.velocity_clip, self.cfg.velocity_clip)

        # handle ee_pose commands (last 7 dims)
        if self.cfg.ee_pose_pos_range is not None:
            # clip position with custom ranges for x, y, z
            actions[:, 3:6] = torch.max(torch.min(actions[:, 3:6], self._ee_pos_max), self._ee_pos_min)
        elif self.cfg.ee_pose_clip is not None:
            # fallback to symmetric clipping for position
            actions[:, 3:6] = actions[:, 3:6].clip(-self.cfg.ee_pose_clip, self.cfg.ee_pose_clip)

        # handle quaternion: either fix it or clip it
        if self.cfg.ee_pose_fixed_quat is not None:
            # override quaternion with fixed value
            actions[:, 6:10] = self._ee_fixed_quat
        elif self.cfg.ee_pose_clip is not None:
            # clip quaternion
            actions[:, 6:10] = actions[:, 6:10].clip(-self.cfg.ee_pose_clip, self.cfg.ee_pose_clip)

        # fallback to unified clipping if nothing else is specified
        if (self.cfg.velocity_clip is None and self.cfg.ee_pose_clip is None and
            self.cfg.ee_pose_pos_range is None and self.cfg.clip is not None):
            actions = actions.clip(-self.cfg.clip, self.cfg.clip)

        # split actions into velocity and pose commands
        self._velocity_commands[:] = actions[:, :3]  # first 3 dims: velocity
        self._ee_pose_commands[:] = actions[:, 3:]   # last 7 dims: ee_pose

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._velocity_commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_link_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedLocomanipPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained loco-manipulation policy action term.

    See :class:`PreTrainedLocomanipPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedLocomanipPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    clip: float | None = None
    """Clip all actions to [-clip, clip]. Defaults to None (no clipping).
    Note: If velocity_clip or ee_pose_clip is set, this parameter is ignored."""
    velocity_clip: float | None = None
    """Clip velocity commands (first 3 dims) to [-velocity_clip, velocity_clip].
    Defaults to None (no clipping)."""
    ee_pose_clip: float | None = None
    """Clip end-effector pose commands (last 7 dims) to [-ee_pose_clip, ee_pose_clip].
    Defaults to None (no clipping). Ignored if ee_pose_pos_range is set."""
    ee_pose_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    """Custom range for end-effector position (x, y, z). Each tuple is (min, max).
    Format: ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    If set, this overrides ee_pose_clip for position components. Defaults to None."""
    ee_pose_fixed_quat: tuple[float, float, float, float] | None = None
    """Fixed quaternion for end-effector orientation in (w, x, y, z) format.
    If set, the orientation will be fixed to this value regardless of action input.
    For example, (1, 0, 0, 0) represents no rotation (roll=pitch=yaw=0).
    Defaults to None (orientation follows action input)."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""


class PreTrainedLocomanipPolicyActionVelocityAndPosition(ActionTerm):
    r"""Pre-trained loco-manipulation policy action term - VELOCITY AND POSITION version.

    This action term accepts 6D commands: 3D velocity commands (lin_vel_x, lin_vel_y, ang_vel_z)
    and 3D end-effector position (pos_x, pos_y, pos_z). The end-effector orientation is fixed.

    High-level action dimension: 6 (3 velocity + 3 position)
    """

    cfg: PreTrainedLocomanipPolicyActionVelocityAndPositionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedLocomanipPolicyActionVelocityAndPositionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # split raw actions into velocity and pose commands
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._ee_pose_commands = torch.zeros(self.num_envs, 7, device=self.device)   # [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]

        # store ee_pos range limits as tensors if configured
        if cfg.ee_pos_range is not None:
            self._ee_pos_min = torch.tensor(
                [cfg.ee_pos_range[0][0], cfg.ee_pos_range[1][0], cfg.ee_pos_range[2][0]],
                device=self.device
            )
            self._ee_pos_max = torch.tensor(
                [cfg.ee_pos_range[0][1], cfg.ee_pos_range[1][1], cfg.ee_pos_range[2][1]],
                device=self.device
            )
        else:
            self._ee_pos_min = None
            self._ee_pos_max = None

        # 固定机械臂末端姿态
        if cfg.ee_pose_fixed_quat is not None:
            self._ee_fixed_quat = torch.tensor(cfg.ee_pose_fixed_quat, device=self.device)
        else:
            # 默认姿态（无旋转）
            self._ee_fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # 初始化末端执行器姿态命令的四元数部分为固定值
        self._ee_pose_commands[:, 3:] = self._ee_fixed_quat

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._velocity_commands
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.pose_command.func = lambda dummy_env: self._ee_pose_commands
        cfg.low_level_observations.pose_command.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 6  # 3 (velocity) + 3 (position)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # clip velocity commands (first 3 dims)
        if self.cfg.velocity_clip is not None:
            actions[:, :3] = actions[:, :3].clip(-self.cfg.velocity_clip, self.cfg.velocity_clip)

        # 设置速度命令
        self._velocity_commands[:] = actions[:, :3]

        # ee_default_pos = torch.tensor([0.4, 0.0, 0.3], device=self.device)
        # actions[:, 3:6] = actions[:, 3:6] + ee_default_pos

        # handle ee_position commands (dims 3-5)
        if self._ee_pos_min is not None and self._ee_pos_max is not None:
            # clip position with custom ranges for x, y, z
            actions[:, 3:6] = torch.max(torch.min(actions[:, 3:6], self._ee_pos_max), self._ee_pos_min)

        # print("actions after clipping:", actions)
        # 设置末端位置命令（位置部分）
        self._ee_pose_commands[:, :3] = actions[:, 3:6]

        # 机械臂末端姿态保持固定（四元数部分在__init__中已初始化）

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._velocity_commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert to world frame
        base_quat_w = self.robot.data.root_link_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

@configclass
class PreTrainedLocomanipPolicyActionVelocityAndPositionCfg(ActionTermCfg):
    """Configuration for pre-trained loco-manipulation policy action term - VELOCITY AND POSITION version.

    See :class:`PreTrainedLocomanipPolicyActionVelocityAndPosition` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedLocomanipPolicyActionVelocityAndPosition
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    velocity_clip: float | None = None
    """Clip velocity commands (first 3 dims) to [-velocity_clip, velocity_clip].
    Defaults to None (no clipping)."""
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    """Custom range for end-effector position (x, y, z). Each tuple is (min, max).
    Format: ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    Defaults to None (no position clipping)."""
    ee_pose_fixed_quat: tuple[float, float, float, float] | None = None
    """Fixed quaternion for end-effector orientation in (w, x, y, z) format.
    If None, defaults to (1, 0, 0, 0) which represents no rotation (roll=pitch=yaw=0)."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""



class PreTrainedLocomanipPolicyActionFullyFixed(ActionTerm):
    r"""Pre-trained loco-manipulation policy action term - FULLY FIXED version (Version 1).

    This action term has NO learnable actions. Both base velocity and end-effector pose
    are completely fixed. Useful for baseline testing.

    Fixed velocity: (0.5, 0.0, 0.0) - moves forward at 0.5 m/s
    Fixed arm pose: position (0.4, 0.0, 0.2) + quaternion (1, 0, 0, 0)

    High-level action dimension: 0 (no actions, fully fixed)
    """

    cfg: PreTrainedLocomanipPolicyActionFullyFixedCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedLocomanipPolicyActionFullyFixedCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # 固定速度命令
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        if cfg.fixed_velocity is not None:
            fixed_vel = torch.tensor(cfg.fixed_velocity, device=self.device)
            self._velocity_commands[:] = fixed_vel
        else:
            # 默认：向前移动 0.5 m/s
            self._velocity_commands[:, 0] = 0.5

        # 固定机械臂末端姿态
        self._ee_pose_commands = torch.zeros(self.num_envs, 7, device=self.device)

        # 固定位置
        if cfg.ee_pose_fixed_pos is not None:
            fixed_pos = torch.tensor(cfg.ee_pose_fixed_pos, device=self.device)
            self._ee_pose_commands[:, :3] = fixed_pos
        else:
            self._ee_pose_commands[:, :3] = torch.tensor([0.4, 0.0, 0.2], device=self.device)

        # 固定姿态
        if cfg.ee_pose_fixed_quat is not None:
            fixed_quat = torch.tensor(cfg.ee_pose_fixed_quat, device=self.device)
            self._ee_pose_commands[:, 3:] = fixed_quat
        else:
            self._ee_pose_commands[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._velocity_commands
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.pose_command.func = lambda dummy_env: self._ee_pose_commands
        cfg.low_level_observations.pose_command.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 0  # 无动作输入，完全固定

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # 忽略输入的actions，所有命令都是固定的
        pass

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._velocity_commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_link_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedLocomanipPolicyActionFullyFixedCfg(ActionTermCfg):
    """Configuration for pre-trained loco-manipulation policy action term - FULLY FIXED version.

    See :class:`PreTrainedLocomanipPolicyActionFullyFixed` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedLocomanipPolicyActionFullyFixed
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    fixed_velocity: tuple[float, float, float] | None = None
    """Fixed velocity command in (lin_vel_x, lin_vel_y, ang_vel_z) format.
    If None, defaults to (0.5, 0.0, 0.0) - moving forward at 0.5 m/s."""
    ee_pose_fixed_pos: tuple[float, float, float] | None = None
    """Fixed position for end-effector in (x, y, z) format.
    If None, defaults to (0.4, 0.0, 0.2)."""
    ee_pose_fixed_quat: tuple[float, float, float, float] | None = None
    """Fixed quaternion for end-effector orientation in (w, x, y, z) format.
    If None, defaults to (1, 0, 0, 0) which represents no rotation."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""


class PreTrainedLocomanipPolicyActionVelocityAndJointAngles(ActionTerm):
    r"""Pre-trained loco-manipulation policy action term - VELOCITY + JOINT ANGLES.

    This action term implements hybrid control where:
    1. High-level policy outputs: 3D base velocity + 5D (or 6D) arm joint angles
    2. Low-level policy: receives velocity commands, outputs 12D leg joint positions (ignores arm)
    3. Arm: directly controlled by high-level joint angles (bypasses low-level policy)

    Key difference from other modes:
    - Low-level policy ONLY controls legs (12 joints)
    - Arm is directly controlled by high-level policy (5 or 6 joints)

    High-level action dimension: 8 (3 velocity + 5 arm joints) or 9 (3 velocity + 6 arm joints)
    """

    cfg: PreTrainedLocomanipPolicyActionVelocityAndJointAnglesCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedLocomanipPolicyActionVelocityAndJointAnglesCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load low-level policy (only for leg control)
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # high-level actions: 3D velocity + arm joint angles
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # split actions into velocity and arm joint angles
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._arm_joint_angles = torch.zeros(self.num_envs, cfg.num_arm_joints, device=self.device)  # [j1, j2, j3, j4, j5(, j6)]

        # ee_pose commands for low-level policy (fixed values)
        self._ee_pose_commands = torch.zeros(self.num_envs, 7, device=self.device)

        # fixed end-effector position for low-level policy observation
        if cfg.ee_pose_fixed_pos is not None:
            self._ee_fixed_pos = torch.tensor(cfg.ee_pose_fixed_pos, device=self.device)
        else:
            self._ee_fixed_pos = torch.tensor([0.3, 0.0, 0.3], device=self.device)

        # fixed end-effector quaternion (orientation)
        if cfg.ee_pose_fixed_quat is not None:
            self._ee_fixed_quat = torch.tensor(cfg.ee_pose_fixed_quat, device=self.device)
        else:
            self._ee_fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # initialize ee pose commands (fixed values for low-level policy)
        self._ee_pose_commands[:, :3] = self._ee_fixed_pos
        self._ee_pose_commands[:, 3:] = self._ee_fixed_quat

        # prepare low level actions (will contain 12 leg joints + 6 arm joints = 18 total)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # determine arm joint indices in the full action space (18 joints)
        # Assuming: Legs are 0-11, Arm is 12-17 (zarx_j1 to zarx_j6)
        if cfg.arm_joint_indices is not None:
            self._arm_joint_indices = list(cfg.arm_joint_indices)
        else:
            # default: assume last 5 or 6 joints are arm joints
            if cfg.num_arm_joints == 5:
                self._arm_joint_indices = [8, 13, 14, 15, 16]  # zarx_j1 到 zarx_j5
            elif cfg.num_arm_joints == 6:
                self._arm_joint_indices = [8, 13, 14, 15, 16, 17]  # zarx_j1 到 zarx_j6
            else:
                raise ValueError(f"num_arm_joints must be 5 or 6, got {cfg.num_arm_joints}")

        # fixed joint angle for j6 if using 5-joint mode
        if cfg.num_arm_joints == 5:
            self._fixed_j6_angle = cfg.fixed_j6_angle if cfg.fixed_j6_angle is not None else 0.0

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._velocity_commands
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.pose_command.func = lambda dummy_env: self._ee_pose_commands
        cfg.low_level_observations.pose_command.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3 + self.cfg.num_arm_joints  # 3 velocity + 5 or 6 arm joint angles

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process high-level actions: 3D velocity + arm joint angles."""
        self._raw_actions[:] = actions

        # split actions: first 3 dims are velocity, rest are arm joint angles
        velocity_actions = actions[:, :3]
        arm_joint_actions = actions[:, 3:]

        # clip velocity commands if specified
        if self.cfg.velocity_clip is not None:
            velocity_actions = velocity_actions.clip(-self.cfg.velocity_clip, self.cfg.velocity_clip)

        # clip arm joint angles if specified
        if self.cfg.joint_angle_clip is not None:
            arm_joint_actions = arm_joint_actions.clip(-self.cfg.joint_angle_clip, self.cfg.joint_angle_clip)

        # store processed commands
        self._velocity_commands[:] = velocity_actions
        self._arm_joint_angles[:] = arm_joint_actions

    def apply_actions(self):
        """Apply actions: use low-level policy for legs ONLY, direct joint angles for arm."""
        if self._counter % self.cfg.low_level_decimation == 0:
            # compute low-level policy output (outputs 18D for all joints)
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            ll_policy_output = self.policy(low_level_obs)

            # ll_policy_output has shape [num_envs, 18]
            # 腿部关节索引: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
            # 机械臂关节索引: [8, 13, 14, 15, 16, 17]

            leg_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]

            # Copy leg joint commands from low-level policy output directly
            for leg_idx in leg_joint_indices:
                self.low_level_actions[:, leg_idx] = ll_policy_output[:, leg_idx]

            # Set arm joint commands from high-level policy (indices [8, 13, 14, 15, 16(, 17)])
            for i, joint_idx in enumerate(self._arm_joint_indices):
                if i < self._arm_joint_angles.shape[1]:  # safety check
                    self.low_level_actions[:, joint_idx] = self._arm_joint_angles[:, i]

            # if using 5-joint mode, fix j6 (index 17)
            if self.cfg.num_arm_joints == 5:
                self.low_level_actions[:, 17] = self._fixed_j6_angle

            # process and apply the combined actions
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # get marker location
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._velocity_commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _update_ee_pose_from_actual_state(self):
        """Update ee_pose_commands based on actual end-effector state from robot.

        This ensures the low-level policy receives accurate information about the current
        arm configuration, even though it doesn't directly control the arm.
        """
        # Get end-effector link name from config
        if hasattr(self.cfg, 'ee_body_name'):
            ee_body_name = self.cfg.ee_body_name
        else:
            ee_body_name = "zarx_body6"  # default for ARX5 arm

        # Find the body index for the end-effector
        body_names = self.robot.data.body_names
        try:
            ee_body_idx = body_names.index(ee_body_name)
        except ValueError:
            # If not found, use fixed values as fallback
            return

        # Get actual end-effector position in world frame
        ee_pos_w = self.robot.data.body_link_state_w[:, ee_body_idx, :3]  # (num_envs, 3)
        ee_quat_w = self.robot.data.body_link_quat_w[:, ee_body_idx, :]  # (num_envs, 4) in wxyz format

        # Get robot base position and orientation
        robot_pos_w = self.robot.data.root_link_state_w[:, :3]  # (num_envs, 3)
        robot_quat_w = self.robot.data.root_link_quat_w  # (num_envs, 4)

        # Compute relative position in robot frame
        ee_pos_rel_w = ee_pos_w - robot_pos_w
        ee_pos_rel_robot = math_utils.quat_rotate_inverse(robot_quat_w, ee_pos_rel_w)

        # Compute relative orientation in robot frame
        ee_quat_rel_robot = math_utils.quat_mul(math_utils.quat_conjugate(robot_quat_w), ee_quat_w)

        # Update ee_pose_commands with actual state
        self._ee_pose_commands[:, :3] = ee_pos_rel_robot  # position (x, y, z)
        self._ee_pose_commands[:, 3:] = ee_quat_rel_robot  # quaternion (w, x, y, z)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert to world frame
        base_quat_w = self.robot.data.root_link_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedLocomanipPolicyActionVelocityAndJointAnglesCfg(ActionTermCfg):
    """Configuration for velocity + joint angles action term.

    High-level policy outputs: 3D velocity + 5D (or 6D) arm joint angles.
    Low-level policy: controls 12D leg joints based on velocity commands.
    Arm: directly controlled by high-level joint angles (bypasses low-level).

    See :class:`PreTrainedLocomanipPolicyActionVelocityAndJointAngles` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedLocomanipPolicyActionVelocityAndJointAngles
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    num_arm_joints: int = 5
    """Number of arm joints to control (5 or 6). Default is 5 (fixes j6)."""
    arm_joint_indices: tuple[int, ...] | None = None
    """Indices of arm joints in the robot's joint array.
    If None, assumes joints 12-16 for 5 joints or 12-17 for 6 joints."""
    fixed_j6_angle: float | None = 0.0
    """Fixed angle for j6 when num_arm_joints=5. Defaults to 0.0."""
    velocity_clip: float | None = None
    """Clip velocity commands (first 3 dims) to [-velocity_clip, velocity_clip].
    Defaults to None (no clipping)."""
    joint_angle_clip: float | None = None
    """Clip joint angle commands to [-clip, clip]. Defaults to None (no clipping)."""
    ee_pose_fixed_pos: tuple[float, float, float] | None = None
    """⚠️ DEPRECATED: This parameter is now ignored. EE pose is computed from actual robot state.
    Fixed position for end-effector in low-level policy (x, y, z) format.
    If None, defaults to (0.3, 0.0, 0.3)."""
    ee_pose_fixed_quat: tuple[float, float, float, float] | None = None
    """⚠️ DEPRECATED: This parameter is now ignored. EE pose is computed from actual robot state.
    Fixed quaternion for end-effector orientation in low-level policy (w, x, y, z) format.
    If None, defaults to (1, 0, 0, 0) which represents no rotation."""
    ee_body_name: str = "zarx_body6"
    """Name of the end-effector body link in the robot. Used to query actual EE state.
    Defaults to 'zarx_body6' for ARX5 arm."""
    debug_vis: bool = True
    """Whether to visualize debug information (velocity arrows). Defaults to True."""
