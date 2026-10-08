# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for door opening task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_door_state_by_usd_path(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    usd_yaw_mapping: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
):
    """根据不同的USD文件路径设置不同的yaw角度来重置门的状态。

    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        pose_range: 位置范围字典（x, y, z, roll, pitch）
        velocity_range: 速度范围字典
        usd_yaw_mapping: USD路径到yaw角度的映射字典
        asset_cfg: 资产配置（默认为"door"）
    """
    # 获取asset实例
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # 获取默认根状态
    root_states = asset.data.default_root_state[env_ids].clone()

    # 位置采样（x, y, z, roll, pitch）
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 5), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # 对每个环境ID检查其使用的USD文件，并设置对应的yaw角度
    yaw_angles = torch.zeros(len(env_ids), device=asset.device)

    # 通过检查prim的引用来确定USD路径
    for i, env_id in enumerate(env_ids):
        # 获取该环境的门prim路径
        env_prim_path = asset.root_physx_view.prim_paths[env_id.item()]

        # 获取prim并检查它的引用
        prim = prim_utils.get_prim_at_path(env_prim_path)
        if prim and prim.IsValid():
            # 获取prim的引用列表
            prim_stack = prim.GetPrimStack()

            # 遍历引用栈查找USD文件路径
            usd_path_found = None
            for prim_spec in prim_stack:
                layer_path = prim_spec.layer.realPath
                # 检查这个layer路径是否在我们的映射中
                for usd_path in usd_yaw_mapping.keys():
                    if usd_path in layer_path:
                        usd_path_found = usd_path
                        break
                if usd_path_found:
                    break

            # 如果找到了匹配的USD路径，使用对应的yaw角度
            if usd_path_found:
                yaw_angles[i] = usd_yaw_mapping[usd_path_found]
            else:
                # 如果没找到，使用第一个映射的默认值
                yaw_angles[i] = list(usd_yaw_mapping.values())[0] if usd_yaw_mapping else 0.0
        else:
            # 如果prim无效，使用默认值
            yaw_angles[i] = list(usd_yaw_mapping.values())[0] if usd_yaw_mapping else 0.0

    # 从roll, pitch, yaw创建四元数
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3],  # roll
        rand_samples[:, 4],  # pitch
        yaw_angles  # yaw（根据USD文件设置）
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # 速度采样
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # 写入物理仿真
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)
