# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.usd
import numpy as np

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_rotate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

x = 0.3 + 0.6 * torch.arange(10) / 10
y = 0.3 + 0.6 * torch.arange(10) / 10
z = 0.2 + 0.2 * torch.arange(10) / 10

size_tensor = torch.cartesian_prod(
    z.to('cuda:0'),
    y.to('cuda:0'),
    x.to('cuda:0')
)

size_tensor = size_tensor[:, [2, 1, 0]]

def object_yaw(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The yaw of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, _, object_yaw = euler_xyz_from_quat(object.data.root_quat_w)
    # print("object_yaw: ", object_yaw.unsqueeze(-1))
    return object_yaw.unsqueeze(-1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    # print(object_pos_w)
    return object_pos_b

def handle_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]        # (num_envs, 3)
    object_quat_w = object.data.root_quat_w             # (num_envs, 4)

    # 2. 定义把手在物体局部坐标系中的偏移
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.0      # x方向(物体前方)
    local_offset[:, 1] = -0.8965  # y方向(物体左侧)
    local_offset[:, 2] = 1.017    # z方向(物体上方)

    # 3. 将局部偏移旋转到世界坐标系 ⭐ 关键步骤!
    world_offset = quat_rotate(object_quat_w, local_offset)  # (num_envs, 3)

    # 4. 计算把手在世界坐标系中的真实位置
    handle_pos_w = object_pos_w + world_offset  # (num_envs, 3)

    # 5. 转换到机器人坐标系
    handle_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        handle_pos_w
    )

    # _, _, object_yaw = euler_xyz_from_quat(object.data.root_quat_w)

    # print(f"物体朝向: {object_yaw}")
    # print(f"世界坐标偏移: {world_offset}")
    # print(f"把手世界位置: {handle_pos_w}")

    return handle_pos_b

def relative_object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_rel = object.data.root_pos_w[:, :2] - robot.data.root_state_w[:, :2]
    angle = torch.atan2(object_pos_rel[:, 1], object_pos_rel[:, 0])
    # print("angle: ", angle)
    robot_heading = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])[2]
    # print("robot_heading: ", robot_heading)
    object_angle_rel = (angle - robot_heading).unsqueeze(1) % (2 * torch.pi)
    object_angle_rel -= 2 * torch.pi * (object_angle_rel > torch.pi)
    object_pos_rel = torch.cat([object_pos_rel, object_angle_rel], dim=1)
    # object_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    # )
    # print(object_pos_rel)
    return object_pos_rel

def object_size(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tensor: torch.Tensor = size_tensor
) -> torch.Tensor:
    """The size of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    env_num = env.num_envs
    # object_size = object.data
    # object_size = object.cfg.prim_path
    while len(tensor) < env_num:
        tensor = torch.cat([tensor, tensor], dim=0)
    object_size = tensor[:env_num]
    # print("object_size: ", object_size)
    return object_size

def object_size_specific(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The size of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    object_size = object.cfg.spawn.size
    object_size = torch.tensor(object_size, device='cuda:0').unsqueeze(0)
    # print("object_size: ", object_size)
    return object_size

def robot_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the robot in the world frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :7]
    # print("robot_pos_w: ", robot_pos_w)
    return robot_pos_w

def object_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3] - env.scene.env_origins
    # print("object_pos_w: ", object_pos_w)
    return object_pos_w


def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_object_pos_w = subtract_frame_transforms(
        object.data.root_state_w[:, :3], object.data.root_state_w[:, 3:7], command[:, :3]
    )
    # object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_object_pos_w
    )
    # print(object_pos_b)
    return object_pos_b


def target_object_pos_w(env: ManagerBasedRLEnv, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, _, yaw = euler_xyz_from_quat(object.data.root_quat_w)
    x_pos = command[:, 0]*torch.cos(yaw) - command[:, 1]*torch.sin(yaw)
    y_pos = command[:, 0]*torch.sin(yaw) + command[:, 1]*torch.cos(yaw)
    rel_pos = torch.stack([x_pos, y_pos], dim=1)
    tar_pos_w = object.data.root_pos_w[:, :2] + rel_pos - env.scene.env_origins[:, :2]
    # print("tar_pos_w: ", tar_pos_w)
    return tar_pos_w

# 全局缓存变量 - 直接按环境ID索引
_object_category_cache = {}

def object_category(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """获取交互物体的类别ID。

    将物体分为三类：
    - 类型0: 包含"CardBox"的物体
    - 类型1: 包含"Barel"的物体
    - 类型2: 包含"Chair"的物体

    Args:
        env: 环境实例。
        object_cfg: 物体的配置。默认为"object"。

    Returns:
        tensor: 物体类别的ID编码 形状为(num_envs, 1)。
    """
    global _object_category_cache

    # 准备环境ID和结果tensor
    num_envs = env.num_envs
    device = env.device
    result = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

    # 获取物体实例
    object_view = env.scene[object_cfg.name]

    # 缓存键 - 直接使用物体名称
    cache_key = object_cfg.name

    # 初始化缓存（如果不存在）
    if cache_key not in _object_category_cache:
        _object_category_cache[cache_key] = {}

    # 检查哪些环境需要计算类别
    env_ids_to_process = []
    for env_id in range(num_envs):
        if env_id in _object_category_cache[cache_key]:
            # 已有缓存，直接使用
            result[env_id, 0] = _object_category_cache[cache_key][env_id]
        else:
            # 没有缓存，需要处理
            env_ids_to_process.append(env_id)

    # 如果所有环境都有缓存，直接返回
    if not env_ids_to_process:
        return result

    # 获取stage
    stage = omni.usd.get_context().get_stage()

    # 为未缓存的环境计算类别
    for env_id in env_ids_to_process:
        try:
            # 获取物体的prim路径
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)

            category_id = 0  # 默认类别

            if prim.IsValid():
                # 获取子Prim（通常第一个子Prim是mesh）
                children = list(prim.GetChildren())

                # 跳过"Looks"和其他非网格prim
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]

                if mesh_children:
                    # 取第一个mesh子prim的名称
                    child_name = mesh_children[0].GetName().lower()

                    # 简单规则匹配
                    if "cardbox" in child_name:
                        category_id = 0  # 纸箱类型
                    elif "barel" in child_name:
                        category_id = 1  # 桶类型
                    elif "chair" in child_name:
                        category_id = 2  #
                    else:
                        category_id = 1  # 默认桶类型

            # 更新结果和缓存
            result[env_id, 0] = float(category_id)
            _object_category_cache[cache_key][env_id] = category_id

        except Exception as e:
            print(f"处理环境{env_id}时出错: {e}")
            # 出错时使用默认值0
            result[env_id, 0] = 0.0
            _object_category_cache[cache_key][env_id] = 0

    # print("object_category: ", result)
    return result

# 点云缓存
_point_cloud_cache = {}

# 存储每种物体类型的基准缩放（用于生成缓存点云的第一个实例的缩放）
_object_type_base_scales = {}

# 🚀 新增：缓存每个环境的相对缩放，避免每步都读取USD（关键性能优化）
_env_relative_scales_cache = {}
_scales_cache_initialized = False

def object_shape_pc(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = 0.0,
    max_height: float = 0.5
) -> torch.Tensor:
    """获取物体在特定高度区间内的形状点云，并展平为一维向量。

    通过对网格表面采样，获取更完整的点云表示。

    Args:
        env: 环境实例。
        object_cfg: 物体的配置。
        num_points: 采样的点云数量。
        min_height: 过滤的最小高度 相对于物体中心 默认为0.0
        max_height: 过滤的最大高度 相对于物体中心 默认为0.5

    Returns:
        tensor: 形状为 (num_envs, num_points*3) 的展平点云数据。
    """
    global _point_cloud_cache

    # 获取物体实例
    object_view = env.scene[object_cfg.name]

    # 准备结果tensor - 注意这里改为展平形状
    num_envs = env.num_envs
    device = env.device
    # 先创建3D点云
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)

    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"

    # 初始化缓存（如果不存在）
    if cache_key not in _point_cloud_cache:
        _point_cloud_cache[cache_key] = {}

    # 检查哪些环境需要计算点云
    env_ids_to_process = []
    for env_id in range(num_envs):
        # 创建环境特定的缓存键
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)

            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    # 用第一个mesh子prim的名称作为缓存键的一部分
                    object_type = mesh_children[0].GetName().lower()
                    cache_id = f"{env_id}_{object_type}"

                    if cache_id in _point_cloud_cache[cache_key]:
                        # 已有缓存，直接使用
                        point_cloud_3d[env_id] = _point_cloud_cache[cache_key][cache_id]
                    else:
                        # 没有缓存，需要处理
                        env_ids_to_process.append((env_id, object_type, prim_path))
                else:
                    env_ids_to_process.append((env_id, "unknown", prim_path))
            else:
                env_ids_to_process.append((env_id, "invalid", prim_path))
        except Exception as e:
            print(f"处理环境{env_id}的缓存键时出错: {e}")
            env_ids_to_process.append((env_id, "error", ""))

    # 如果所有环境都有缓存，直接返回展平的结果
    if not env_ids_to_process:
        return point_cloud_3d.reshape(num_envs, num_points * 3)

    # 处理需要计算的环境
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    import numpy as np

    # 获取stage
    stage = omni.usd.get_context().get_stage()

    for env_id, object_type, prim_path in env_ids_to_process:
        try:
            # 固定随机种子，确保相同物体类型生成相同的点云
            seed = int(stable_hash(object_type) % 10000000)
            np_state = np.random.get_state()
            np.random.seed(seed)  # 使用物体类型作为随机种子

            prim = stage.GetPrimAtPath(prim_path)

            # 收集所有网格的点和面
            all_points = []
            all_faces = []
            all_face_counts = []
            point_offset = 0

            # 遍历物体的层级结构
            def process_prim_recursively(current_prim, parent_transform=None):
                nonlocal point_offset

                # 跳过特定类型或名称的prim
                if "Looks" in str(current_prim.GetPath()):
                    return

                # 获取当前Prim的局部变换
                xformable = UsdGeom.Xformable(current_prim)
                local_transform = xformable.GetLocalTransformation()
                local_matrix = np.array(local_transform).reshape(4, 4)

                # 计算当前Prim相对于根节点的累积变换
                if parent_transform is not None:
                    current_transform = np.dot(parent_transform, local_matrix)
                else:
                    current_transform = local_matrix

                # 检查是否为Mesh类型
                if current_prim.GetTypeName() == "Mesh":
                    mesh = UsdGeom.Mesh(current_prim)

                    # 获取顶点
                    vertices = mesh.GetPointsAttr().Get()
                    if vertices is not None and len(vertices) > 0:
                        # 转换为numpy数组
                        vertices_np = np.array([(v[0], v[1], v[2]) for v in vertices])

                        # 获取面信息
                        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

                        if face_vertex_counts is not None and face_vertex_indices is not None:
                            # 转换顶点到世界坐标系
                            homogeneous = np.ones((len(vertices_np), 4))
                            homogeneous[:, :3] = vertices_np
                            transformed_vertices = np.dot(homogeneous, current_transform.T)[:, :3]

                            all_points.append(transformed_vertices)

                            # 记录面信息，用于后续采样
                            face_start = 0
                            for face_verts in face_vertex_counts:
                                # 提取当前面的顶点索引
                                face_indices = face_vertex_indices[face_start:face_start + face_verts]
                                face_indices = [idx + point_offset for idx in face_indices]
                                all_faces.append(face_indices)
                                all_face_counts.append(face_verts)
                                face_start += face_verts

                            # 更新顶点偏移量
                            point_offset += len(vertices_np)

                # 递归处理子Prim，传递当前累积变换
                for child in current_prim.GetChildren():
                    process_prim_recursively(child, current_transform)

            # 开始递归处理
            if prim.IsValid():
                process_prim_recursively(prim)

            # 合并所有点和处理面信息
            if all_points and len(all_points) > 0:
                # 合并所有顶点
                all_vertices = np.vstack(all_points)

                # 使用三角形面来生成更密集的点云
                surface_points = []

                # 遍历所有面，对每个面进行采样
                face_start_idx = 0
                for face_idx, face_verts in enumerate(all_face_counts):
                    # 获取当前面的顶点索引
                    face_indices = all_faces[face_idx]

                    # 对三角形面进行处理
                    if face_verts == 3:  # 三角形
                        triangle_vertices = all_vertices[face_indices]
                        # 对三角形表面进行均匀采样
                        surface_points.extend(sample_triangle(triangle_vertices, 10))  # 每个三角形采样10个点
                    elif face_verts == 4:  # 四边形 - 拆分为两个三角形
                        quad_vertices = all_vertices[face_indices]
                        # 将四边形拆分为两个三角形
                        triangle1 = quad_vertices[[0, 1, 2]]
                        triangle2 = quad_vertices[[0, 2, 3]]
                        # 采样两个三角形
                        surface_points.extend(sample_triangle(triangle1, 5))
                        surface_points.extend(sample_triangle(triangle2, 5))
                    elif face_verts > 4:  # 多边形 - 使用简单的扇形三角剖分
                        for i in range(1, face_verts - 1):
                            tri_vertices = all_vertices[[face_indices[0], face_indices[i], face_indices[i+1]]]
                            surface_points.extend(sample_triangle(tri_vertices, 3))

                # 添加原始顶点，确保关键点被保留
                surface_points.extend(all_vertices)

                # 转换为numpy数组
                if surface_points:
                    sampled_points = np.array(surface_points)

                    # 过滤高度范围
                    height_mask = (sampled_points[:, 2] >= min_height) & (sampled_points[:, 2] <= max_height)
                    filtered_points = sampled_points[height_mask]

                    # 如果过滤后点太少，使用全部点
                    if len(filtered_points) < num_points * 0.5 and len(sampled_points) >= num_points:
                        filtered_points = sampled_points

                    # 确保有足够的点
                    if len(filtered_points) >= num_points:
                        # 随机采样所需数量的点
                        indices = np.random.choice(len(filtered_points), num_points, replace=False)
                        final_points = filtered_points[indices]
                    elif len(filtered_points) > 0:
                        # 重复采样以达到所需数量
                        indices = np.random.choice(len(filtered_points), num_points, replace=True)
                        final_points = filtered_points[indices]
                    else:
                        # 如果过滤后没有点，使用原始点
                        if len(sampled_points) >= num_points:
                            indices = np.random.choice(len(sampled_points), num_points, replace=False)
                            final_points = sampled_points[indices]
                        elif len(sampled_points) > 0:
                            indices = np.random.choice(len(sampled_points), num_points, replace=True)
                            final_points = sampled_points[indices]
                        else:
                            # 如果没有点，创建空点云
                            final_points = np.zeros((num_points, 3))

                    # 保存到结果和缓存
                    points_tensor = torch.tensor(final_points, dtype=torch.float32, device=device)
                    point_cloud_3d[env_id] = points_tensor

                    # 存入缓存
                    cache_id = f"{env_id}_{object_type}"
                    _point_cloud_cache[cache_key][cache_id] = points_tensor.clone()
                else:
                    # 没有成功生成点，使用替代方案
                    substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
                    point_cloud_3d[env_id] = substitute_points

                    # 存入缓存
                    cache_id = f"{env_id}_{object_type}"
                    _point_cloud_cache[cache_key][cache_id] = substitute_points.clone()
            else:
                # 没有找到有效的网格，使用替代方案
                substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
                point_cloud_3d[env_id] = substitute_points

                # 存入缓存
                cache_id = f"{env_id}_{object_type}"
                _point_cloud_cache[cache_key][cache_id] = substitute_points.clone()

            # 恢复随机状态
            np.random.set_state(np_state)

        except Exception as e:
            print(f"处理环境{env_id}的点云时出错: {e}")
            import traceback
            traceback.print_exc()

            # 出错时使用替代方案
            seed = int(stable_hash(f"error_{object_type}") % 10000000)
            substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
            point_cloud_3d[env_id] = substitute_points

    # 将3D点云展平为1D向量并返回
    return point_cloud_3d.reshape(num_envs, num_points * 3)

def sample_triangle(vertices, num_samples, edge_margin=0.15):
    """对三角形表面进行均匀采样，避开边缘区域

    Args:
        vertices: 三角形的三个顶点
        num_samples: 采样点数
        edge_margin: 边缘内缩比例(0-0.33)，0表示可以采样到边缘，0.33表示只采样三角形中心点
                     默认0.15，即避开边缘15%的区域
    """
    if len(vertices) != 3:
        return []

    samples = []
    max_attempts = num_samples * 10  # 最大尝试次数
    attempts = 0

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        # 生成重心坐标
        r1 = np.random.random()
        r2 = np.random.random()

        # 确保是均匀分布
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2

        # 计算第三个重心坐标
        r3 = 1 - r1 - r2

        # 边缘过滤：只接受距离三个边都足够远的点
        # 重心坐标中，任意一个坐标接近0表示接近对边
        if edge_margin > 0:
            # 检查是否太靠近任何一条边（重心坐标 < edge_margin）
            if r1 < edge_margin or r2 < edge_margin or r3 < edge_margin:
                continue  # 跳过靠近边缘的点

        # 使用重心坐标计算点
        point = vertices[0] * r1 + vertices[1] * r2 + vertices[2] * r3
        samples.append(point)

    # 如果由于edge_margin太大导致采样不够，降低要求重新采样
    if len(samples) < num_samples * 0.5 and edge_margin > 0:
        remaining = num_samples - len(samples)
        samples.extend(sample_triangle(vertices, remaining, edge_margin=edge_margin * 0.5))

    return samples

def create_substitute_point_cloud(num_points, min_height, max_height, device):
    """当无法获取真实点云时，创建一个简单的替代点云。"""
    import numpy as np

    # 创建一个简单的盒状点云
    x = np.random.uniform(-0.5, 0.5, num_points)
    y = np.random.uniform(-0.5, 0.5, num_points)
    z = np.random.uniform(min_height, max_height, num_points)

    points = np.column_stack([x, y, z])
    return torch.tensor(points, dtype=torch.float32, device=device)


#######################################################
# 全局缓存变量 - 按物体类型缓存
_object_type_point_clouds = {}

# 全局缓存变量 - 环境到物体类型的映射
_env_to_object_type = {}

# 第一次访问时初始化的标志
_point_cloud_initialized = False

def calculate_object_type_mapping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object")
) -> dict:
    """计算每个环境实例对应的物体类型，只需计算一次。

    Args:
        env: 环境实例
        object_cfg: 物体配置

    Returns:
        dict: 环境ID到物体类型的映射
    """
    global _env_to_object_type

    # 缓存键
    cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"

    # 如果已经缓存，直接返回
    if cache_key in _env_to_object_type:
        return _env_to_object_type[cache_key]

    # 初始化映射
    _env_to_object_type[cache_key] = {}

    # 获取物体实例
    object_view = env.scene[object_cfg.name]

    # 获取stage
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # 为每个环境获取物体类型
    for env_id in range(env.num_envs):
        try:
            # 获取物体的prim路径
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)

            object_type = "unknown"
            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    object_type = mesh_children[0].GetName().lower()

            # 存储映射
            _env_to_object_type[cache_key][env_id] = object_type

        except Exception as e:
            print(f"获取环境{env_id}的物体类型时出错: {e}")
            _env_to_object_type[cache_key][env_id] = "unknown"

    # 返回映射
    return _env_to_object_type[cache_key]

def stable_hash(text):
    value = 0
    for char in text:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
    return value

def generate_point_cloud_for_type(
    env: ManagerBasedRLEnv,
    object_type: str,
    prim_path: str,
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5,
    device: torch.device = None,
    surface_edge_margin: float = 0.15
) -> torch.Tensor:
    """为特定类型的物体生成点云。

    Args:
        env: 环境实例
        object_type: 物体类型名称
        prim_path: 物体的prim路径
        num_points: 点数
        min_height: 最小高度
        max_height: 最大高度
        device: 计算设备
        surface_edge_margin: 表面采样边缘内缩比例(0-0.33)，避开三角形边缘区域。默认0.15

    Returns:
        torch.Tensor: 生成的点云
    """
    import numpy as np

    seed = int(stable_hash(object_type) % 10000000)
    np_state = np.random.get_state()
    np.random.seed(seed)  # 使用物体类型作为随机种子

    if device is None:
        device = env.device

    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    import numpy as np

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    # 收集点和面
    all_points = []
    all_faces = []
    all_face_counts = []
    point_offset = 0

    # 递归处理prim
    def process_prim_recursively(current_prim, parent_transform=None):
        nonlocal point_offset

        # 跳过Looks
        if "Looks" in str(current_prim.GetPath()):
            return

        # 处理变换
        xformable = UsdGeom.Xformable(current_prim)
        local_transform = xformable.GetLocalTransformation()
        local_matrix = np.array(local_transform).reshape(4, 4)

        # 计算累积变换
        if parent_transform is not None:
            current_transform = np.dot(parent_transform, local_matrix)
        else:
            current_transform = local_matrix

        # 处理Mesh
        if current_prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(current_prim)

            vertices = mesh.GetPointsAttr().Get()
            if vertices is not None and len(vertices) > 0:
                vertices_np = np.array([(v[0], v[1], v[2]) for v in vertices])

                face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

                if face_vertex_counts is not None and face_vertex_indices is not None:
                    # 变换顶点
                    homogeneous = np.ones((len(vertices_np), 4))
                    homogeneous[:, :3] = vertices_np
                    transformed_vertices = np.dot(homogeneous, current_transform.T)[:, :3]

                    all_points.append(transformed_vertices)

                    # 处理面
                    face_start = 0
                    for face_verts in face_vertex_counts:
                        face_indices = face_vertex_indices[face_start:face_start + face_verts]
                        face_indices = [idx + point_offset for idx in face_indices]
                        all_faces.append(face_indices)
                        all_face_counts.append(face_verts)
                        face_start += face_verts

                    point_offset += len(vertices_np)

        # 递归处理子节点
        for child in current_prim.GetChildren():
            process_prim_recursively(child, current_transform)

    # 开始处理
    process_prim_recursively(prim)

    # 检查是否有有效数据
    if not all_points:
        # 没有找到有效的网格，使用替代方案
        return create_substitute_point_cloud(num_points, min_height, max_height, device)

    # 合并顶点
    all_vertices = np.vstack(all_points)

    # 采样面 - 使用基于面积的均匀采样
    surface_points = []

    # 首先计算所有三角形及其面积
    triangles_with_area = []

    for face_idx, face_verts in enumerate(all_face_counts):
        face_indices = all_faces[face_idx]

        if face_verts == 3:  # 三角形
            triangle_vertices = all_vertices[face_indices]
            # 计算三角形面积
            v0, v1, v2 = triangle_vertices
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            triangles_with_area.append((triangle_vertices, area))

        elif face_verts == 4:  # 四边形
            quad_vertices = all_vertices[face_indices]
            # 分解为两个三角形
            triangle1 = quad_vertices[[0, 1, 2]]
            triangle2 = quad_vertices[[0, 2, 3]]

            # 计算面积
            v0, v1, v2 = triangle1
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            triangles_with_area.append((triangle1, area1))

            v0, v1, v2 = triangle2
            area2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            triangles_with_area.append((triangle2, area2))

        elif face_verts > 4:  # 多边形
            # 简单的扇形三角剖分
            for i in range(1, face_verts - 1):
                tri_vertices = all_vertices[[face_indices[0], face_indices[i], face_indices[i+1]]]
                v0, v1, v2 = tri_vertices
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                triangles_with_area.append((tri_vertices, area))

    # 计算总面积
    total_area = sum(area for _, area in triangles_with_area)

    if total_area > 0:
        # 根据面积分配采样点数（预留一些用于后续采样）
        target_surface_points = int(num_points * 2.0)  # 采样2倍的点数，后续会筛选

        for triangle_vertices, area in triangles_with_area:
            # 根据面积占比分配点数
            num_samples = max(1, int(target_surface_points * (area / total_area)))
            surface_points.extend(sample_triangle(triangle_vertices, num_samples, edge_margin=surface_edge_margin))

    # 注意：不添加原始顶点，只依赖表面采样以获得更均匀的分布

    # 检查是否成功生成点
    if not surface_points:
        return create_substitute_point_cloud(num_points, min_height, max_height, device)

    # 转换为numpy数组
    sampled_points = np.array(surface_points)

    # 过滤高度
    height_mask = (sampled_points[:, 2] >= min_height) & (sampled_points[:, 2] <= max_height)
    filtered_points = sampled_points[height_mask]

    # 如果过滤后点太少，使用全部点
    if len(filtered_points) < num_points * 0.5 and len(sampled_points) >= num_points:
        filtered_points = sampled_points

    # 采样最终点云
    if len(filtered_points) >= num_points:
        indices = np.random.choice(len(filtered_points), num_points, replace=False)
        final_points = filtered_points[indices]
    elif len(filtered_points) > 0:
        indices = np.random.choice(len(filtered_points), num_points, replace=True)
        final_points = filtered_points[indices]
    else:
        return create_substitute_point_cloud(num_points, min_height, max_height, device)

    # 转换为tensor
    points_tensor = torch.tensor(final_points, dtype=torch.float32, device=device)

    np.random.set_state(np_state)
    return points_tensor

def invalidate_scales_cache():
    """🔄 清除缩放缓存（在reset时调用）"""
    global _env_relative_scales_cache, _scales_cache_initialized
    _env_relative_scales_cache.clear()
    _scales_cache_initialized = False
    print("🔄 缩放缓存已清除")

def compute_and_cache_relative_scales(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    cache_key: str = "",
) -> torch.Tensor:
    """🚀 一次性计算并缓存所有环境的相对缩放（关键性能优化）

    这个函数只在第一次调用或reset后调用，避免每步都读取USD。

    Args:
        env: 环境实例
        object_cfg: 物体配置
        cache_key: 缓存键

    Returns:
        相对缩放tensor，形状为(num_envs, 3)
    """
    global _env_relative_scales_cache, _object_type_base_scales, _scales_cache_initialized

    import omni.usd
    from pxr import UsdGeom
    import numpy as np

    num_envs = env.num_envs
    device = env.device

    # 检查是否已缓存
    scales_cache_key = f"{cache_key}_{id(env)}"
    if scales_cache_key in _env_relative_scales_cache and _scales_cache_initialized:
        return _env_relative_scales_cache[scales_cache_key]

    print(f"🔧 首次计算并缓存{num_envs}个环境的相对缩放...")
    import time
    start_time = time.time()

    object_view = env.scene[object_cfg.name]
    stage = omni.usd.get_context().get_stage()
    relative_scales = torch.ones((num_envs, 3), dtype=torch.float32, device=device)

    # 获取环境到物体类型的映射
    env_cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"
    env_to_type = _env_to_object_type.get(env_cache_key, {})
    if not env_to_type:
        env_to_type = calculate_object_type_mapping(env, object_cfg)

    for env_id in range(num_envs):
        try:
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)

                # 读取当前实例的缩放
                current_scale = torch.ones(3, device=device)
                scale_op = None
                for xform_op in xformable.GetOrderedXformOps():
                    if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = xform_op.Get()
                        break

                if scale_op is not None:
                    current_scale = torch.tensor([scale_op[0], scale_op[1], scale_op[2]], device=device)
                else:
                    local_transform = xformable.GetLocalTransformation()
                    transform_matrix = np.array(local_transform)
                    scale_x = np.linalg.norm(transform_matrix[0, :3])
                    scale_y = np.linalg.norm(transform_matrix[1, :3])
                    scale_z = np.linalg.norm(transform_matrix[2, :3])
                    current_scale = torch.tensor([scale_x, scale_y, scale_z], device=device)

                # 获取该环境的物体类型
                object_type = env_to_type.get(env_id, "unknown")
                base_scale_key = f"{cache_key}_{object_type}"

                # 计算相对缩放
                if base_scale_key in _object_type_base_scales:
                    base_scale = _object_type_base_scales[base_scale_key]
                    # 相对缩放 = 当前缩放 / 基准缩放
                    relative_scales[env_id] = current_scale / base_scale
                else:
                    # 如果找不到基准缩放，使用当前缩放（假设基准是1）
                    relative_scales[env_id] = current_scale
        except Exception as e:
            # 出错时使用默认缩放 (1, 1, 1)
            pass

    # 缓存结果
    _env_relative_scales_cache[scales_cache_key] = relative_scales
    _scales_cache_initialized = True

    elapsed_time = time.time() - start_time
    print(f"✅ 相对缩放缓存完成，耗时: {elapsed_time:.3f}秒")

    return relative_scales

def generate_all_point_clouds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5,
    surface_edge_margin: float = 0.15
) -> None:
    """为所有类型的物体生成点云，并存储在全局缓存中。

    Args:
        env: 环境实例
        object_cfg: 物体配置
        num_points: 点数量
        min_height: 最小高度
        max_height: 最大高度
        surface_edge_margin: 表面采样边缘内缩比例(0-0.33)，避开三角形边缘区域。默认0.15
    """
    global _object_type_point_clouds, _point_cloud_initialized, _object_type_base_scales

    # 如果已经初始化过，跳过
    if _point_cloud_initialized:
        return

    # 设置标志
    _point_cloud_initialized = True

    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"

    # 初始化缓存
    if cache_key not in _object_type_point_clouds:
        _object_type_point_clouds[cache_key] = {}

    # 获取环境到物体类型的映射
    env_to_type = calculate_object_type_mapping(env, object_cfg)

    # 获取物体实例
    object_view = env.scene[object_cfg.name]

    # 收集所有唯一的物体类型
    unique_types = set(env_to_type.values())

    print(f"开始为{len(unique_types)}种物体类型生成点云...")
    import time
    start_time = time.time()

    # 为每种类型生成点云
    import omni.usd
    from pxr import UsdGeom
    import numpy as np
    stage = omni.usd.get_context().get_stage()

    for object_type in unique_types:
        # 查找该类型的第一个实例
        for env_id, type_name in env_to_type.items():
            if type_name == object_type:
                # 获取物体的prim路径
                prim_path = object_view.root_physx_view.prim_paths[env_id]

                # 🔧 获取第一个实例的缩放（作为基准缩放）
                base_scale = torch.tensor([1.0, 1.0, 1.0], device=env.device)
                try:
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        xformable = UsdGeom.Xformable(prim)
                        scale_op = None
                        for xform_op in xformable.GetOrderedXformOps():
                            if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                scale_op = xform_op.Get()
                                break
                        if scale_op is not None:
                            base_scale = torch.tensor([scale_op[0], scale_op[1], scale_op[2]], device=env.device)
                        else:
                            local_transform = xformable.GetLocalTransformation()
                            transform_matrix = np.array(local_transform)
                            scale_x = np.linalg.norm(transform_matrix[0, :3])
                            scale_y = np.linalg.norm(transform_matrix[1, :3])
                            scale_z = np.linalg.norm(transform_matrix[2, :3])
                            base_scale = torch.tensor([scale_x, scale_y, scale_z], device=env.device)
                except Exception as e:
                    print(f"  警告: 无法读取基准缩放: {e}")

                # 生成点云
                print(f"生成类型 '{object_type}' 的点云...")
                print(f"  基准缩放: [{base_scale[0]:.3f}, {base_scale[1]:.3f}, {base_scale[2]:.3f}]")
                try:
                    point_cloud = generate_point_cloud_for_type(
                        env, object_type, prim_path, num_points, min_height, max_height,
                        surface_edge_margin=surface_edge_margin
                    )

                    # 存储到缓存
                    _object_type_point_clouds[cache_key][object_type] = point_cloud
                    # 🔧 存储基准缩放
                    _object_type_base_scales[cache_key + "_" + object_type] = base_scale
                    print(f"  成功生成 '{object_type}' 的点云，包含 {num_points} 个点")

                except Exception as e:
                    print(f"  为类型 '{object_type}' 生成点云时出错: {e}")
                    import traceback
                    traceback.print_exc()

                    # 使用默认点云
                    _object_type_point_clouds[cache_key][object_type] = create_substitute_point_cloud(
                        num_points, min_height, max_height, env.device
                    )
                    _object_type_base_scales[cache_key + "_" + object_type] = base_scale

                # 每种类型只处理一次
                break

    # 添加默认点云
    _object_type_point_clouds[cache_key]["default"] = create_substitute_point_cloud(
        num_points, min_height, max_height, env.device
    )

    elapsed_time = time.time() - start_time
    print(f"点云生成完成，耗时: {elapsed_time:.2f}秒")

def object_shape_pc_optimized(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5,
    surface_edge_margin: float = 0.15
) -> torch.Tensor:
    """获取物体点云的优化版本。

    通过预计算和类型映射大幅提高性能。

    Args:
        env: 环境实例
        object_cfg: 物体配置
        num_points: 点数量
        min_height: 最小高度
        max_height: 最大高度
        surface_edge_margin: 表面采样边缘内缩比例(0-0.33)，避开三角形边缘区域。默认0.15

    Returns:
        torch.Tensor: 点云数据，形状为(num_envs, num_points*3)
    """
    global _object_type_point_clouds, _env_to_object_type, _point_cloud_initialized

    # 初始化所有点云（如果尚未完成）
    if not _point_cloud_initialized:
        generate_all_point_clouds(env, object_cfg, num_points, min_height, max_height, surface_edge_margin)

    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    env_cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"

    # 准备输出tensor
    num_envs = env.num_envs
    device = env.device
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)

    # 按类型批量分配点云
    env_ids_by_type = {}
    env_to_type = _env_to_object_type.get(env_cache_key, {})

    # 检查是否已计算环境到类型的映射
    if not env_to_type:
        env_to_type = calculate_object_type_mapping(env, object_cfg)

    # 按类型分组环境ID
    for env_id, object_type in env_to_type.items():
        if env_id >= num_envs:
            continue

        if object_type not in env_ids_by_type:
            env_ids_by_type[object_type] = []
        env_ids_by_type[object_type].append(env_id)

    # 批量分配点云
    for object_type, env_ids in env_ids_by_type.items():
        # 转换为torch.LongTensor以便索引
        env_indices = torch.tensor(env_ids, dtype=torch.long, device=device)

        # 获取点云
        if object_type in _object_type_point_clouds[cache_key]:
            point_cloud = _object_type_point_clouds[cache_key][object_type]
        else:
            point_cloud = _object_type_point_clouds[cache_key]["default"]

        # 一次性分配给所有相同类型的环境
        point_cloud_3d.index_copy_(0, env_indices, point_cloud.unsqueeze(0).expand(len(env_indices), -1, -1))

    # 展平并返回
    return point_cloud_3d.reshape(num_envs, num_points * 3)


def object_shape_pc_relative_optimized(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5,
    top_k_closest: int | None = None,
    surface_edge_margin: float = 0.15
) -> torch.Tensor:
    """获取物体点云的优化版本（相对于机器人坐标系）- 批量操作版本

    Args:
        env: 环境对象
        object_cfg: 物体配置
        robot_cfg: 机器人配置
        num_points: 采样的点云数量
        min_height: 最小高度
        max_height: 最大高度
        top_k_closest: 选取距离机器人最近的k个点。如果为None，则返回所有点
        surface_edge_margin: 表面采样边缘内缩比例(0-0.33)，避开三角形边缘区域。默认0.15
    """

    from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse

    global _object_type_point_clouds, _env_to_object_type, _point_cloud_initialized

    if not _point_cloud_initialized:
        generate_all_point_clouds(env, object_cfg, num_points, min_height, max_height, surface_edge_margin)

    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    env_cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"

    num_envs = env.num_envs
    device = env.device
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)

    # 获取点云
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

    # ⭐ 批量坐标系转换（无循环）
    robot = env.scene[robot_cfg.name]
    object_view = env.scene[object_cfg.name]

    # 获取位置和方向
    object_pos_w = object_view.data.root_pos_w  # (num_envs, 3)
    object_quat_w = object_view.data.root_quat_w  # (num_envs, 4)
    robot_pos_w = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # 🚀 使用缓存的相对缩放（关键性能优化：避免每步都读取USD）
    relative_scales = compute_and_cache_relative_scales(env, object_cfg, cache_key)

    # 1. 点云从物体局部坐标系→世界坐标系
    # 步骤：相对缩放 → 旋转 → 平移

    # 首先应用相对缩放（将缓存点云调整到当前实例的尺寸）
    scaled_point_cloud = point_cloud_3d * relative_scales.unsqueeze(1)  # (num_envs, num_points, 3)

    # 然后应用旋转
    object_quat_expanded = object_quat_w.unsqueeze(1).expand(-1, num_points, -1)  # (num_envs, num_points, 4)
    object_quat_flat = object_quat_expanded.reshape(-1, 4)  # (num_envs*num_points, 4)
    point_cloud_flat = scaled_point_cloud.reshape(-1, 3)  # (num_envs*num_points, 3)

    rotated_points_flat = quat_rotate(object_quat_flat, point_cloud_flat)  # (num_envs*num_points, 3)
    rotated_points = rotated_points_flat.reshape(num_envs, num_points, 3)  # (num_envs, num_points, 3)

    # 最后应用平移
    point_cloud_world = rotated_points + object_pos_w.unsqueeze(1)  # (num_envs, num_points, 3)

    # 2. 点云从世界坐标系→机器人坐标系
    point_cloud_relative_w = point_cloud_world - robot_pos_w.unsqueeze(1)  # (num_envs, num_points, 3)

    # 为所有点复制机器人四元数
    robot_quat_expanded = robot_quat_w.unsqueeze(1).expand(-1, num_points, -1)  # (num_envs, num_points, 4)
    robot_quat_flat = robot_quat_expanded.reshape(-1, 4)  # (num_envs*num_points, 4)
    point_cloud_relative_flat = point_cloud_relative_w.reshape(-1, 3)  # (num_envs*num_points, 3)

    point_cloud_robot_flat = quat_rotate_inverse(robot_quat_flat, point_cloud_relative_flat)  # (num_envs*num_points, 3)
    point_cloud_robot = point_cloud_robot_flat.reshape(num_envs, num_points, 3)  # (num_envs, num_points, 3)

    # 如果指定了top_k_closest，选取距离原点（机器人）最近的k个点
    if top_k_closest is not None and top_k_closest < num_points:
        # 计算每个点到原点的距离
        distances = torch.norm(point_cloud_robot, dim=2)  # (num_envs, num_points)

        # 对每个环境，获取距离最小的top_k个点的索引
        _, top_k_indices = torch.topk(distances, k=top_k_closest, dim=1, largest=False)  # (num_envs, top_k_closest)

        # 使用gather选取对应的点
        # 扩展索引维度以匹配点云的3D坐标
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 3)  # (num_envs, top_k_closest, 3)
        point_cloud_robot = torch.gather(point_cloud_robot, dim=1, index=top_k_indices_expanded)  # (num_envs, top_k_closest, 3)

        # 展平并返回
        return point_cloud_robot.reshape(num_envs, top_k_closest * 3)

    # 展平并返回所有点
    return point_cloud_robot.reshape(num_envs, num_points * 3)


def arm_end_effector_position_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="zarx_body6"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the 3D position of the arm end-effector relative to robot base in robot frame.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.
                   body_names should specify the end-effector link (e.g., "zarx_body6").
        robot_cfg: Scene entity configuration for the robot base.

    Returns:
        Tensor of shape (num_envs, 3) containing the relative end-effector position (x, y, z) in robot frame.
    """
    from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse

    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Get end-effector (gripper base) position and orientation in world frame
    ee_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    ee_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # (num_envs, 4)

    # Get robot base position and orientation in world frame
    robot_pos_w = robot.data.root_link_state_w[:, :3]  # (num_envs, 3)
    robot_quat_w = robot.data.root_link_quat_w  # (num_envs, 4)

    # Compute relative position BEFORE offset (gripper base position)
    ee_base_pos_rel_w = ee_pos_w - robot_pos_w  # (num_envs, 3)
    ee_base_pos_rel_robot = quat_rotate_inverse(robot_quat_w, ee_base_pos_rel_w)  # (num_envs, 3)

    # Add 0.14m offset in gripper's local x-direction to get actual gripper tip position
    local_offset = torch.zeros((env.num_envs, 3), device=env.device)
    local_offset[:, 0] = 0.14  # 0.14m in local x-direction
    world_offset = quat_rotate(ee_quat_w, local_offset)  # (num_envs, 3)
    ee_tip_pos_w = ee_pos_w + world_offset  # (num_envs, 3)

    # Compute relative position AFTER offset (gripper tip position)
    ee_tip_pos_rel_w = ee_tip_pos_w - robot_pos_w  # (num_envs, 3)
    ee_tip_pos_rel_robot = quat_rotate_inverse(robot_quat_w, ee_tip_pos_rel_w)  # (num_envs, 3)

    # Print comparison (only first environment for clarity)
    # print(f"\n{'='*80}")
    # print(f"机械臂末端位置对比 (相对于机器人基座的坐标, 机器人坐标系):")
    # print(f"  [处理前] 夹爪基座位置: {ee_base_pos_rel_robot[0]}")
    # print(f"  [处理后] 夹爪末端位置: {ee_tip_pos_rel_robot[0]}")
    # print(f"  [偏移量] 差值: {ee_tip_pos_rel_robot[0] - ee_base_pos_rel_robot[0]}")
    # print(f"{'='*80}\n")

    return ee_tip_pos_rel_robot
