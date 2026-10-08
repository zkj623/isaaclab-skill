# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Door Opening Task Configuration for Legged Manipulator

"""

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
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2_arx5.rough_env_cfg import UnitreeGo2ARX5RoughEnvCfg
# import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG, NAV_TERRAINS_CFG, INTERACTION_TERRAINS_CFG, OBJECT_PUSH_TERRAINS_CFG  # isort: skip
import numpy as np
import re
import random

# 默认使用支持机械臂的底层策略配置（Go2 + ARX5）
# 如果需要使用其他机器人，可以在子类中覆盖此配置
LOW_LEVEL_ENV_CFG = UnitreeGo2ARX5RoughEnvCfg()

##
# Scene definition
##


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

    # door: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Door",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
    #     # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0, 0, 0, 1]),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=skill_asset_path("library/wooden_door_new.usd"),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             # solver_position_iteration_count=16,
    #             # solver_velocity_iteration_count=1,
    #             # max_angular_velocity=1000.0,
    #             # max_linear_velocity=1000.0,
    #             # max_depenetration_velocity=5.0,
    #             # disable_gravity=False,
    #             angular_damping=10.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     debug_vis=False,
    # )

    door: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                # training
                # skill_asset_path("library/wooden_door_new.usd"),
                # skill_asset_path("library/wooden_door_new3.usd"),
                # skill_asset_path("library/low_door_push_left.usd"),
                # skill_asset_path("library/low_door_push_right2.usd"),
                skill_asset_path("library/local_door.usd"),
                skill_asset_path("library/local_door2.usd"),
            ],
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # 注意：这个init_state只在spawn时使用，reset时会被reset_door_position事件覆盖
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=[0.707, 0, 0, 0.707]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.95, 0.0), rot=[1, 0, 0, 0]),
    )

    wall1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=skill_asset_path("library/Wall.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.7, 0.1, 0.0), rot=[1, 0, 0, 0]),
    )

    wall2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=skill_asset_path("library/Wall.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.75, 0.1, 0.0), rot=[1, 0, 0, 0]),
    )

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
        # push door
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-0.5, -0.5), pos_y=(1.0, 1.0), heading=(-math.pi, math.pi)),
        # pull door
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(2.0, 2.0), pos_y=(-0.5, -0.5), heading=(-math.pi, math.pi)),
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
    # ============================================================================
    # 🎯 Action配置：3维速度 + 6维机械臂关节角（版本3）
    # 动作空间：9维（3维速度 + 6维机械臂关节角）
    # 可学习：速度命令（lin_vel_x, lin_vel_y, ang_vel_z）+ 机械臂关节角（zarx_j1-j6）
    # 底层策略：根据速度命令控制12个腿关节
    # 机械臂：直接由高层策略的关节角控制
    # ============================================================================
    # pre_trained_policy_action = mdp.PreTrainedLocomanipPolicyActionVelocityAndJointAnglesCfg(
    #     asset_name="robot",
    #     policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_arx5_rough/2025-11-03_15-04-29/exported/policy.pt",
    #     low_level_decimation=4,
    #     low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
    #     low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    #     num_arm_joints=6,  # 控制6个关节（j1-j6）
    #     arm_joint_indices=None,  # None表示自动检测：12-17 for 6 joints
    #     ee_body_name="zarx_body6",  # 末端执行器body名称，用于查询实际状态
    #     debug_vis=True,  # 显示速度箭头可视化
    # )

    # 版本2：速度 + 末端位置 - 训练速度和臂位置（课程学习阶段2）
    # 动作空间：6维（3维速度 + 3维末端位置）
    # 可学习：速度命令 + 机械臂末端位置
    # 固定姿态：四元数固定为 (1, 0, 0, 0)
    # 适用于推物体等任务，姿态影响不大
    #
    # ============================================================================
    # pre_trained_policy_action: mdp.PreTrainedLocomanipPolicyActionVelocityAndPositionCfg = mdp.PreTrainedLocomanipPolicyActionVelocityAndPositionCfg(
    #     asset_name="robot",
    #     # policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_arx5_rough/2025-09-23_20-58-26/exported/policy.pt",
    #     # policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_arx5_rough/2025-10-31_13-53-43/exported/policy.pt",
    #     policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_arx5_rough/2025-11-03_15-04-29/exported/policy.pt",
    #     low_level_decimation=4,
    #     low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
    #     low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    #     # velocity_clip=0.5,   # 速度命令限制s
    #     # 🔥 关键修改：给机械臂真正的活动范围，而不是固定值
    #     # ee_pos_range=((0.3, 0.5), (-0.15, 0.15), (0.15, 0.35)),
    #     # ee_pos_range=((0.4, 0.4), (0., 0.), (0.3, 0.3)),
    #     # 解释：x: 0.3-0.5m (前后), y: -0.15~0.15m (左右), z: 0.15-0.35m (上下，避免碰地)
    #     ee_pose_fixed_quat=(1.0, 0.0, 0.0, 0.0),  # 固定姿态（无旋转）
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # ============================================================================
        # -- 本体感知 (Proprioception)
        # ============================================================================

        # 基座方向 (通过投影重力隐式表示)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        # 基座线速度和角速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)

        # ============================================================================
        # -- 外感知 (Exteroception) - 门的位置信息
        # ============================================================================

        # 门框位置 - 相对于机器人 (用于穿过阶段的导航)
        door_hinge_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("door")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        door_handle_position = ObsTerm(
            func=mdp.handle_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("door")},
            # params={"object_cfg": SceneEntityCfg("door", object_collection_names="SM_DoorB_Handle")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        # # 门的偏航角 - 用于判断门的开启程度 (关键观测!)
        door_yaw = ObsTerm(
            func=mdp.object_yaw,
            params={"object_cfg": SceneEntityCfg("door")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 注意：不包含门把手观测，因为门是虚掩的，不需要转动把手
        # 注意：不包含门的质量、阻力等特权信息，策略需要在线推断这些属性

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
            # "pose_range": {"x": (-0., 0.), "y": (-0., 0.)},
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
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_door_position = EventTerm(
        func=mdp.reset_door_state_by_usd_path,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "usd_yaw_mapping": {
                skill_asset_path("library/local_door2.usd"): 3.14,  # door2.usd使用yaw=3.14
                skill_asset_path("library/local_door.usd"): -1.57,   # door.usd使用yaw=1.57
            },
            "asset_cfg": SceneEntityCfg("door"),
        },
    )

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="setup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("door"),
    #         "static_friction_range": (1.3, 1.3),
    #         "dynamic_friction_range": (1.2, 1.2),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    参考论文: Learning to Open and Traverse Doors with a Legged Manipulator (ETH Zurich)
    两阶段奖励结构: 开门阶段 (ro) + 穿过阶段 (rp) + 整形奖励 (rs)

    简化版本: 无门把手操作，门是虚掩的，直接推/拉即可
    """

    # ============================================================================
    # -- 阶段1: 开门奖励 (ro) - Opening Stage Rewards
    # ============================================================================

    # 1.1 门开启角度奖励 (rod) - 核心任务奖励！
    # 论文: 使用指数奖励鼓励将门打开
    door_opening = RewTerm(
        func=mdp.door_opening_angle_reward,
        weight=2.0,  # 论文中权重为3，这是主要任务奖励
        params={
            "std": 0.3,  # 奖励衰减速度
            "object_cfg": SceneEntityCfg("door"),
            "open_angle_threshold": 0.785, # 0.523
        },
    )

    # 1.2 机械臂末端接近门奖励 (rehd变体)
    # 论文: rehd = exp(-||e - h||²) - 末端接近把手
    # 我们的版本: 末端接近门表面中心位置 (无把手，接近门板侧面)
    arm_end_effector_approach_handle = RewTerm(
        func=mdp.arm_end_effector_handle_distance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "object_cfg": SceneEntityCfg("door"),
            "std": 0.3,  # 距离衰减速度
            "open_angle_threshold": 0.785,
        },
    )

    # 1.3 机械臂向前伸展奖励
    # 鼓励机械臂向门方向伸展以接触门
    arm_forward_reach = RewTerm(
        func=mdp.arm_end_effector_forward_reach_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "target_x": 0.4,  # 目标x位置（机器人坐标系）
            "std": 0.2,
        },
    )

    arm_height_penalty = RewTerm(
        func=mdp.arm_end_effector_height_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
            "min_height": 0.4,  # 最小高度
            "max_height": 0.9,  # 最大高度
        },
    )

    # ============================================================================
    # -- 阶段2: 穿过门奖励 (rp) - Passing Stage Rewards
    # ============================================================================

    position_tracking = RewTerm(
        func=mdp.track_pos,
        weight=2.0, # 2.0
        params={"std": 2.0, "command_name": "pose_command"},
    )

    # position_tracking = RewTerm(
    #     func=mdp.track_pos_when_door_open,
    #     weight=2.0,
    #     params={
    #         "std": 2.0,  # 奖励衰减速度
    #         "object_cfg": SceneEntityCfg("door"),
    #         "command_name": "pose_command",
    #         "open_angle_threshold": 0.785, # 0.523
    #     },
    # )

    # 2.2 门打开后，奖励朝向目标运动的速度
    # 这个奖励鼓励机器人在门打开后以合适的速度朝向目标移动
    velocity_towards_target = RewTerm(
        func=mdp.track_velocity_towards_target_when_door_open,
        weight=1.0,
        params={
            "std": 0.8,  # 期望速度约为0.5 m/s
            "object_cfg": SceneEntityCfg("door"),
            "command_name": "pose_command",
            "open_angle_threshold": 1.0,  # 45度
        },
    )

    # track_lin_vel_xy_with_pose_command = RewTerm(
    #     func=mdp.track_lin_vel_xy_with_pose_command, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    # ============================================================================
    # -- 整形奖励 (rs) - Shaping Rewards
    # ============================================================================
    # 论文: 用于规范化行为，使策略尊重硬件限制并安全部署

    # locomotion reward
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.1,
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


    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # 增加动作平滑性惩罚，从-0.01增加到-0.05，减少颤抖

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

    # 3.3 动作平滑性 (防止抖动)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2_first_three, weight=-0.1)

    # action_rate_l2_arm_ee_pos = RewTerm(func=mdp.action_rate_l2_arm_ee_pos, weight=-0.001)

    # action_l2_arm_pos = RewTerm(func=mdp.action_l2_arm_ee_pos, weight=-0.1)

    # positive_x_vel = RewTerm(
    #     func=mdp.positive_x_vel,
    #     weight=1, # 1 for commad[:, :3]
    # )

    # 3.4 惩罚手臂过度伸展 (rpsa)
    # 论文: rpsa = -clip((||e - s|| - (0.7 - 0.1)) / 0.1, 0, 1)
    # 防止奇异臂构型
    # arm_overextension_penalty = RewTerm(
    #     func=mdp.arm_end_effector_height_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="zarx_body6"),
    #         "min_height": 0.1,  # 最小高度
    #         "max_height": 1.0,  # 最大高度
    #     },
    # )

    # 3.5 惩罚碰撞 (rpc)
    # 论文中惩罚base, thighs, arm的碰撞
    # 我们已经在terminations中处理了大部分碰撞
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

    # 3.6 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    # ============================================================================
    # -- 论文设计要点总结
    # ============================================================================
    #
    # 论文中的关键设计:
    # 1. 两阶段结构: 开门阶段(门角度<70°) -> 穿过阶段(门角度>70°)
    # 2. 阶段转换时给予最大ro奖励，避免策略拒绝转换
    # 3. 对于拉门，需要额外奖励机器人绕到门后方(rZ1, rZ2)
    # 4. 所有奖励需要考虑任务阶段，避免conflicting objectives
    #
    # 我们的简化版本:
    # - 无门把手操作奖励(rhm中的rth, reho, rhg, rplg)
    # - 门是虚掩的，直接推/拉即可
    # - 如果需要实现完整的两阶段奖励，需在mdp/rewards.py中实现条件奖励函数
    # ============================================================================

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
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),"limit_angle": 1.0},  #0.4
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class DoorPushEnvCfg(ManagerBasedRLEnvCfg):
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
            # self.scene.height_scanner.update_period = self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
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
