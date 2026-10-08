# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omni.isaac.lab.utils.skill_assets import skill_asset_path
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt
import omni.isaac.core.utils.prims as prim_utils
import omni.usd

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
# import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG, NAV_TERRAINS_CFG, INTERACTION_TERRAINS_CFG, OBJECT_PUSH_TERRAINS_CFG  # isort: skip
import numpy as np
import re
import random

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2_arx5.rough_env_cfg import UnitreeGo2ARX5RoughEnvCfg

# 底层行走策略环境配置
LOW_LEVEL_ENV_CFG = UnitreeGo2ARX5RoughEnvCfg()
##
# Scene definition
##

def _generate_mixed_object_configs():
    """生成混合物体配置：多种类型 + 多种尺寸的cardbox"""
    configs = []

    # original_objects = [
    #     skill_asset_path("library/local_cardbox.usd"),
    #     skill_asset_path("library/local_cardbox_T.usd"),
    #     skill_asset_path("library/local_barrel_big.usd"),
    #     skill_asset_path("library/local_barrel_small.usd"),
    #     skill_asset_path("library/local_chair_T.usd"),
    #     skill_asset_path("library/local_chair_2_T.usd"),
    # ]

    # 1. 添加原始物体类型 (各一个)
    # original_objects = [
    #     skill_asset_path("library/cardbox.usd"),
    #     skill_asset_path("library/cardbox_T.usd"),
    #     skill_asset_path("library/barrel_big.usd"),
    #     skill_asset_path("library/barrel_small.usd"),
    #     skill_asset_path("library/chair_T.usd"),
    #     skill_asset_path("library/chair_2_T.usd"),
    # ]

    original_objects = [
        skill_asset_path("library/barrel_big_2.usd"),
        skill_asset_path("library/barrel_small_2.usd"),
        skill_asset_path("library/cardbox_2.usd"),
        skill_asset_path("library/chair_3.usd"),
        skill_asset_path("library/chair_4.usd"),
        skill_asset_path("library/wetfloor.usd"),
        skill_asset_path("library/traffic_cone.usd"),
    ]

    for obj_path in original_objects:
        configs.append(sim_utils.UsdFileCfg(usd_path=obj_path))

    # 2. 添加cardbox的多种尺寸变体
    x_orig, y_orig, z_orig = 0.7, 0.5, 0.5
    x_sizes = [0.4, 0.5, 0.6, 0.7]
    y_sizes = [0.4, 0.5, 0.6, 0.7]
    z_sizes = [0.3, 0.4, 0.5]

    for x_target in x_sizes:
        for y_target in y_sizes:
            for z_target in z_sizes:
                x_scale = round(x_target / x_orig, 2)
                y_scale = round(y_target / y_orig, 2)
                z_scale = round(z_target / z_orig, 2)

                configs.append(
                    sim_utils.UsdFileCfg(
                        usd_path=skill_asset_path("library/local_cardbox.usd"),
                        scale=(x_scale, y_scale, z_scale),
                    )
                )

    # 2. 添加其他物体，每个重复48次（与cardbox变体数量相同）
    other_objects = [
        skill_asset_path("library/local_cardbox_T.usd"),
        skill_asset_path("library/local_barrel_big.usd"),
        skill_asset_path("library/local_barrel_small.usd"),
        skill_asset_path("library/local_chair_T.usd"),
        skill_asset_path("library/local_chair_2_T.usd"),
    ]

    num_cardbox_variants = len(x_sizes) * len(y_sizes) * len(z_sizes)  # 48

    for obj_path in other_objects:
        for _ in range(num_cardbox_variants):
            configs.append(sim_utils.UsdFileCfg(usd_path=obj_path))

    return configs

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=OBJECT_PUSH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # target object
    interactive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Interactive_object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            # 方案1: 混合多种物体 + 多种尺寸cardbox (共114个选项)
            assets_cfg=_generate_mixed_object_configs(),
            random_choice=False,  # 随机选择
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=[1, 0, 0, 0]),
    )

    wall1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=skill_asset_path("library/Wall.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.18, 0.0), rot=[0.707, 0, 0, 0.707]),
    )

    wall2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=skill_asset_path("library/Wall.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.38, 0.0), rot=[0.707, 0, 0, 0.707]),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # 前足-物体接触传感器（只检测前两只脚与Interactive_object的接触）
    # 用于终止条件：前足接触物体时重置环境
    # 注意：由于ContactSensor限制，每个足端需要单独的传感器
    feet_object_contact_sensor_FL = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Interactive_object"],
        history_length=0,
        track_air_time=False,
    )
    feet_object_contact_sensor_FR = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Interactive_object"],
        history_length=0,
        track_air_time=False,
    )

    # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-2.0, -1.5), pos_y=(-0.5, 0.5), heading=(-math.pi, math.pi)),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi) # arbitrary walking
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0, 0), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi) # climb stairs
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!.*zarx_j[78]).*$"],  # 排除zarx_j7和zarx_j8
        scale=0.5,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        actions = ObsTerm(func=mdp.last_action)

        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("interactive_object")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        # object_yaw = ObsTerm(
        #     func=mdp.object_yaw,
        #     params={"object_cfg": SceneEntityCfg("interactive_object")},
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )

        object_contact_points = ObsTerm(
            func=mdp.object_shape_pc_relative_optimized,
            params={
                "object_cfg": SceneEntityCfg("interactive_object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "num_points": 1024,  # 减少点数以提高性能
                "min_height": 0.2,
                "max_height": 0.7,
                "top_k_closest": 8,
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_door_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("interactive_object"),
        },
    )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.0), "yaw": (0, 0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("box_1"),
    #     },
    # )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("interactive_object"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset", #start up
        params={
            "asset_cfg": SceneEntityCfg("interactive_object"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.1, 0.1),
            "operation": "add",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ============================================================================
    # -- 阶段1: 接触和推动障碍物奖励 - Object Manipulation Rewards
    # ============================================================================

    # 1.1 机械臂末端接近障碍物接触点
    # 鼓励机械臂靠近物体表面，为推动做准备
    arm_end_effector_approach_object = RewTerm(
        func=mdp.arm_end_effector_to_contact_points_distance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "object_cfg": SceneEntityCfg("interactive_object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "std": 0.3,  # 距离衰减速度
            "num_points": 1024,  # 采样点数
            "min_height": 0.2,  # 最小采样高度
            "max_height": 0.7,  # 最大采样高度
            "top_k_closest": 8,  # 只考虑距离机器人基座最近的16个点（与观测一致）
            "surface_edge_margin": 0.15,  # 表面采样边缘内缩比例
        },
    )

    # 1.2 机械臂向前伸展奖励
    # 鼓励机械臂至少向前伸展到一定距离（x>=0.4时达到最大奖励）
    # 防止机械臂一直缩在身体附近，提供基础行为先验
    arm_forward_reach = RewTerm(
        func=mdp.arm_end_effector_forward_reach_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "target_x": 0.4,  # 最小前伸距离阈值（机器人坐标系）
            "std": 0.2,
        },
    )

    arm_height_penalty = RewTerm(
        func=mdp.arm_end_effector_height_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "min_height": 0.25,  # 最小高度
            "max_height": 0.9,  # 最大高度
        },
    )

    # 1.3 机械臂与物体接触奖励（仅在需要推动时）
    # 只在障碍物挡路且距离目标较远时才奖励接触
    # 障碍物移开或接近目标后，停止奖励接触，鼓励收回机械臂
    # arm_object_contact = RewTerm(
    #     func=mdp.arm_object_contact_reward,
    #     weight=0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="zarx.*"),  # 机械臂的所有部分
    #         "object_cfg": SceneEntityCfg("interactive_object"),
    #         "threshold": 0.1,  # 接触力阈值（N）
    #         "command_name": "pose_command",
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "lateral_distance_threshold": 0.8,  # 障碍物横向距离<0.8米时认为挡路，需要推动
    #         "goal_distance_threshold": 1.0,  # 距离目标>1.0米时才奖励接触
    #     },
    # )

    # 1.4 物体位移奖励（带衰减和惩罚的推动奖励）
    # 三阶段奖励机制，精确控制推动距离：
    # 奖励特点：
    #   - 0-saturation_distance：正向奖励（指数增长），鼓励推动物体
    #   - saturation_distance到decay_end_distance：线性衰减到0，避免推太远
    #   - >decay_end_distance：负向惩罚（线性增加），明确抑制过度推动
    #   - 目标：让机器人学会推动物体到合适距离
    object_displacement = RewTerm(
        func=mdp.object_displacement_reward,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("interactive_object"),
            "std": 0.3,  # 控制第一阶段的奖励增长速度
            "saturation_distance": 0.6,  # 奖励增长结束点（m），开始衰减阶段
            "decay_end_distance": 1.0,  # 衰减结束点（m），之后变为惩罚
        },
    )

    # 1.5 物体离开路径奖励（带位移门槛，防止机器人钻空子）
    # 鼓励将障碍物推离机器人到目标的直线路径
    # 三个奖励条件：
    #   1. 物体必须先移动≥0.8m，才开始给予横向距离奖励（防止机器人后退钻空子）
    #   2. 横向距离达到saturation_distance后奖励饱和至1.0，防止浪费时间推得太远
    #   3. 物体与前进方向夹角超过angle_threshold后奖励直接为1.0（物体已不挡路，任务完成）
    # 注意：角度超阈值给满分而非0，避免agent害怕推过头
    # object_cleared_from_path = RewTerm(
    #     func=mdp.object_lateral_distance_from_path_reward,
    #     weight=0.8,
    #     params={
    #         "object_cfg": SceneEntityCfg("interactive_object"),
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "command_name": "pose_command",
    #         "std": 0.3,  # 横向距离标准差，控制奖励增长速度
    #         "saturation_distance": 0.8,  # 饱和距离阈值（m），超过此距离奖励达到1.0
    #         "angle_threshold": 1.047,  # 角度阈值（弧度，60度），超过此角度说明物体不挡路，奖励直接为1.0
    #         "min_displacement_threshold": 0.8,  # 位移门槛（m），物体必须移动≥0.8m才开始给奖励（防钻空子）
    #     },
    # )

    # 1.6 物体阻挡路径惩罚
    # 惩罚物体位于机器人到目标的路径上（机器人→目标 和 机器人→物体 的夹角小）
    # 这鼓励机器人将物体推离到达目标的路径，避免物体阻挡前进
    # **重要**：此惩罚只在物体移动≥0.5m后才生效，这样可以：
    #   1. 先鼓励机器人推动物体（通过object_displacement奖励）
    #   2. 然后惩罚物体仍在路径上，促使继续推离路径
    # 与横向距离奖励协同：横向奖励鼓励"推离路径"，此惩罚加强"不要留在路径上"
    object_blocking_path_penalty = RewTerm(
        func=mdp.robot_object_goal_alignment_penalty,
        weight=-1.0,  # 负权重表示惩罚，可根据效果调整（建议范围：-0.5到-2.0）
        params={
            "object_cfg": SceneEntityCfg("interactive_object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "command_name": "pose_command",
            "std": 0.5,  # 标准差（弧度，约28.6度），控制惩罚衰减速度
            "min_displacement_threshold": 0.5,  # 位移门槛（m），物体必须移动≥0.5m后才开始惩罚
        },
    )

    # ============================================================================
    # -- 阶段2: 到达目标奖励 - Goal Reaching Rewards
    # ============================================================================

    position_tracking = RewTerm(
        func=mdp.track_pos,
        weight=2.0, # 2.0
        params={"std": 2.0, "command_name": "pose_command"},
    )

    # 2.2 正面朝向目标前进奖励
    # 同时奖励：1) 机器人朝向对齐目标方向  2) 正向前进而不是横着走或斜着走
    # 这鼓励机器人像人一样"转身面向目标，然后向前走"，而不是螃蟹步
    # forward_velocity_towards_goal = RewTerm(
    #     func=mdp.forward_velocity_towards_goal,
    #     weight=1.0,  # 权重可调，建议范围：0.5-3.0
    #     params={
    #         "command_name": "pose_command",
    #         "std": 0.5,  # 期望前进速度（m/s）
    #     },
    # )

    # 2.4 门打开后，奖励朝向目标运动的速度
    # 这个奖励鼓励机器人在门打开后以合适的速度朝向目标移动
    # velocity_towards_target = RewTerm(
    #     func=mdp.track_velocity_towards_target_when_door_open,
    #     weight=1.0,
    #     params={
    #         "std": 0.5,  # 期望速度约为0.5 m/s
    #         "object_cfg": SceneEntityCfg("interactive_object"),
    #         "command_name": "pose_command",
    #         "open_angle_threshold": 0.785,  # 45度
    #     },
    # )

    # ============================================================================
    # -- 整形奖励 (rs) - Shaping Rewards
    # ============================================================================
    # 论文: 用于规范化行为，使策略尊重硬件限制并安全部署

    # locomotion reward
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # 速度过大惩罚
    # 当机器人的水平速度超过0.8m/s时施加惩罚，鼓励安全速度运动
    excessive_velocity = RewTerm(
        func=mdp.excessive_velocity_penalty,
        weight=-2.0,  # 负权重，惩罚过快的速度
        params={
            "velocity_threshold": 0.8,  # 速度阈值（m/s）
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=0.5,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    foot_joint_pos = RewTerm(
        func=spot_mdp.foot_joint_position_penalty,
        weight=-1.5, #-0.7
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.1,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "target_height": 0.32,
        },
    )


    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)  # -0.01

    # 添加关节速度惩罚，减少腿部颤抖
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # 惩罚后腿拖地滑动 - 只惩罚RL和RR两条后腿
    feet_slide_rear = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,  # 负权重表示惩罚，可以根据效果调整（建议范围：-0.1 到 -1.0）
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),  # 只匹配RL_foot和RR_foot
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
        },
    )

    # 2.2 到达目标后惩罚关节速度，避免原地踏步
    # 机器人到达目标后，基座可能停止但腿部关节仍在运动
    # 这会导致"原地踏步"现象，通过惩罚关节速度来解决
    # stand_still_at_goal = RewTerm(
    #     func=mdp.stand_still_velocity_penalty,
    #     weight=-0.1,  # 负权重表示惩罚，可根据效果调整（建议范围：-1.0到-5.0）
    #     params={
    #         "command_name": "pose_command",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "distance_threshold": 0.1,  # 距离目标0.5米内开始惩罚关节运动
    #     },
    # )

    # 2.3 到达目标后关节回到默认位置
    # 鼓励机器人（特别是机械臂）在到达目标后恢复标准站立姿态
    # 关节越接近默认位置，奖励越大
    # joint_default_position_at_goal = RewTerm(
    #     func=mdp.joint_pos_to_default_at_goal,
    #     weight=0.2,  # 正权重表示奖励，可根据效果调整（建议范围：0.5到2.0）
    #     params={
    #         "command_name": "pose_command",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "distance_threshold": 0.5,  # 距离目标0.5米内开始奖励默认姿态
    #     },
    # )

    # 防止物体倾倒惩罚
    # 惩罚物体倾斜或翻倒，鼓励机器人在推动物体时保持物体直立
    # 通过检查物体的z轴朝向与世界z轴的夹角来判断物体是否倾斜
    object_upright_penalty = RewTerm(
        func=mdp.object_orientation_penalty,
        weight=-2.0,  # 负权重表示惩罚，可根据效果调整（建议范围：-1.0到-5.0）
        params={
            "object_cfg": SceneEntityCfg("interactive_object"),
            "max_tilt_angle": 0.5,  # 最大允许倾斜角度（弧度），约28.6度，超过此角度惩罚快速增加
        },
    )

    undesired_contacts_hip = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold": 1.0},
    )
    undesired_contacts_calf = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    undesired_contacts_head_upper = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_upper"), "threshold": 1.0},
    )
    undesired_contacts_head_lower = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_lower"), "threshold": 1.0},
    )

    # 前脚与物体接触惩罚
    # 惩罚前脚（FL和FR）与Interactive_object的接触，鼓励用机械臂或身体推动
    # 返回值：接触的前脚数量（0, 1, 或 2），乘以负权重成为惩罚
    front_feet_object_contact_penalty = RewTerm(
        func=mdp.front_feet_object_contact_penalty,
        weight=-100.0,  # 负权重：每只前脚接触物体惩罚10分
        params={
            "sensor_cfg_FL": SceneEntityCfg("feet_object_contact_sensor_FL"),
            "sensor_cfg_FR": SceneEntityCfg("feet_object_contact_sensor_FR"),
            "threshold": 1.0,  # 接触力阈值（N）
        },
    )

    # 3.6 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    head_upper_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_upper"), "threshold": 1.0},
    )
    head_lower_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_lower"), "threshold": 1.0},
    )
    calf_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )
    thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold": 1.0},
    )

    # 前足-物体接触终止条件
    # 当前足（FL或FR）接触Interactive_object时重置环境
    # 鼓励机器人用身体或机械臂推动物体，而不是用前脚踢
    # front_feet_object_contact = DoneTerm(
    #     func=mdp.front_feet_object_contact,
    #     params={
    #         "sensor_cfg_FL": SceneEntityCfg("feet_object_contact_sensor_FL"),
    #         "sensor_cfg_FR": SceneEntityCfg("feet_object_contact_sensor_FR"),
    #         "threshold": 1.0,  # 接触力阈值（N）
    #     },
    # )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),"limit_angle": math.pi/2},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class InteractionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.decimation = self.decimation * 10
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
