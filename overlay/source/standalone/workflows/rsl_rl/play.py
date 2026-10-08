# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--follow-camera", action="store_true", help="Keep the viewport camera on the first robot.")
parser.add_argument("--asset-root", type=str, default=None, help="Isaac Sim asset root, such as a local Nucleus URI.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.asset_root:
    import carb

    carb.settings.get_settings().set("/persistent/isaac/asset_root/cloud", args_cli.asset_root.rstrip("/"))

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import object_shape_pc, object_shape_pc_optimized, object_shape_pc_relative_optimized

def visualize_point_cloud(
    env,
    object_name="interactive_object",
    num_points=1024,
    min_height=-0.5,
    max_height=0.5,
    num_envs_to_show=6,
    save_path="/tmp/point_cloud_test.png",
    show_plot=True,
    use_optimized=False
):
    """可视化物体点云的函数。

    Args:
        env: 环境实例
        object_name: 要可视化的物体名称
        num_points: 点云中的点数量
        min_height: 高度过滤最小值
        max_height: 高度过滤最大值
        num_envs_to_show: 要可视化的环境数量
        save_path: 图像保存路径
        show_plot: 是否显示图像
        use_optimized: 是否使用优化版本的函数（需要预先初始化缓存）

    Returns:
        point_cloud_3d: 形状为 (num_envs, num_points, 3) 的点云数据张量
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # 获取原始环境实例
    unwrapped_env = getattr(env, "unwrapped", env)

    # 导入必要的函数
    from omni.isaac.lab.managers import SceneEntityCfg

    # 创建物体配置
    object_cfg = SceneEntityCfg(object_name)

    # 获取点云数据
    print(f"Getting point cloud for '{object_name}' with {num_points} points...")
    print(f"Height range: [{min_height}, {max_height}]")
    print(f"使用{'优化版本' if use_optimized else '手动转换版本'}获取点云")

    # 创建机器人配置
    robot_cfg = SceneEntityCfg("robot")

    # 🔧 无论使用哪个版本，都先获取缩放信息（用于可视化显示）
    import omni.usd
    from pxr import UsdGeom
    import numpy as np
    import torch

    object_view = unwrapped_env.scene[object_cfg.name]
    stage = omni.usd.get_context().get_stage()
    num_envs_actual = unwrapped_env.num_envs
    device = unwrapped_env.device
    object_scales = torch.ones((num_envs_actual, 3), dtype=torch.float32, device=device)

    print("  读取物体缩放信息...")
    for env_id in range(num_envs_actual):
        try:
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)

                # 方法1: 尝试直接读取 scale 属性
                scale_op = None
                for xform_op in xformable.GetOrderedXformOps():
                    if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = xform_op.Get()
                        break

                if scale_op is not None:
                    # 找到了 scale 操作
                    scale_x, scale_y, scale_z = scale_op[0], scale_op[1], scale_op[2]
                else:
                    # 方法2: 从变换矩阵提取
                    local_transform = xformable.GetLocalTransformation()
                    transform_matrix = np.array(local_transform)
                    scale_x = np.linalg.norm(transform_matrix[0, :3])
                    scale_y = np.linalg.norm(transform_matrix[1, :3])
                    scale_z = np.linalg.norm(transform_matrix[2, :3])

                object_scales[env_id] = torch.tensor([scale_x, scale_y, scale_z], device=device)

                if env_id < 3:  # 只打印前3个环境的缩放信息
                    print(f"    Env {env_id}: scale = [{scale_x:.3f}, {scale_y:.3f}, {scale_z:.3f}]")
                    print(f"              prim_path = {prim_path}")
        except Exception as e:
            if env_id == 0:
                print(f"    警告: 无法读取缩放信息: {e}")
            import traceback
            if env_id == 0:
                traceback.print_exc()

    if use_optimized:
        # 使用训练环境中使用的优化版本函数
        print("  调用 object_shape_pc_relative_optimized...")
        # 首先获取所有点（用于可视化背景）
        pc_flat_all = object_shape_pc_relative_optimized(
            unwrapped_env,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            num_points=num_points,
            min_height=min_height,
            max_height=max_height,
            top_k_closest=None  # 不限制，获取所有点
        )
        # 然后获取选中的最近的点
        pc_flat = object_shape_pc_relative_optimized(
            unwrapped_env,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            num_points=num_points,
            min_height=min_height,
            max_height=max_height,
            top_k_closest=16  # 只选择最近的16个点
        )
    else:
        # 手动转换版本（不依赖缓存）
        print("  调用 object_shape_pc + 手动坐标转换...")
        # 先获取相对于物体的点云
        pc_flat_object = object_shape_pc(unwrapped_env, object_cfg, num_points, min_height, max_height)

        # 转换到机器人坐标系
        from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse

        point_cloud_3d = pc_flat_object.reshape(-1, num_points, 3)
        robot = unwrapped_env.scene[robot_cfg.name]

        # 获取位置和方向
        object_pos_w = object_view.data.root_pos_w  # (num_envs, 3)
        object_quat_w = object_view.data.root_quat_w  # (num_envs, 4)
        robot_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

        # 1. 点云从物体局部坐标系→世界坐标系
        # 步骤：缩放 → 旋转 → 平移
        # 因为缓存的点云是按类型共享的（原始尺寸），每个实例需要应用自己的缩放

        # 首先应用缩放
        scaled_point_cloud = point_cloud_3d * object_scales.unsqueeze(1)

        # 然后应用旋转
        object_quat_expanded = object_quat_w.unsqueeze(1).expand(-1, num_points, -1)
        object_quat_flat = object_quat_expanded.reshape(-1, 4)
        point_cloud_flat = scaled_point_cloud.reshape(-1, 3)

        rotated_points_flat = quat_rotate(object_quat_flat, point_cloud_flat)
        rotated_points = rotated_points_flat.reshape(-1, num_points, 3)

        # 最后应用平移
        point_cloud_world = rotated_points + object_pos_w.unsqueeze(1)

        # 2. 点云从世界坐标系→机器人坐标系
        point_cloud_relative_w = point_cloud_world - robot_pos_w.unsqueeze(1)
        robot_quat_expanded = robot_quat_w.unsqueeze(1).expand(-1, num_points, -1)
        robot_quat_flat = robot_quat_expanded.reshape(-1, 4)
        point_cloud_relative_flat = point_cloud_relative_w.reshape(-1, 3)

        point_cloud_robot_flat = quat_rotate_inverse(robot_quat_flat, point_cloud_relative_flat)
        point_cloud_robot = point_cloud_robot_flat.reshape(-1, num_points, 3)

        # 展平结果
        pc_flat = point_cloud_robot.reshape(-1, num_points * 3)
        pc_flat_all = pc_flat  # 非优化版本不支持选择，所有点都一样

    # 计算实际返回的点数
    actual_num_points = pc_flat.shape[1] // 3
    all_num_points = pc_flat_all.shape[1] // 3
    print(f"  选中点数: {actual_num_points}, 总点数: {all_num_points}")

    # 重塑为3D点云格式
    point_cloud_3d = pc_flat.reshape(-1, actual_num_points, 3)  # 选中的点
    point_cloud_3d_all = pc_flat_all.reshape(-1, all_num_points, 3)  # 所有点

    # 🔍 为了准确计算物体尺寸，需要将点云转回物体局部坐标系
    # 使用所有点来计算物体完整尺寸
    from omni.isaac.lab.utils.math import quat_rotate, quat_rotate_inverse

    robot = unwrapped_env.scene[robot_cfg.name]

    # 获取位置和方向
    object_pos_w = object_view.data.root_pos_w  # (num_envs, 3)
    object_quat_w = object_view.data.root_quat_w  # (num_envs, 4)
    robot_pos_w = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    # 转换所有点到物体局部坐标系（用于尺寸验证）
    # 从机器人坐标系转回世界坐标系
    robot_quat_expanded_all = robot_quat_w.unsqueeze(1).expand(-1, all_num_points, -1)
    robot_quat_flat_all = robot_quat_expanded_all.reshape(-1, 4)
    point_cloud_flat_all = point_cloud_3d_all.reshape(-1, 3)

    pc_world_flat_all = quat_rotate(robot_quat_flat_all, point_cloud_flat_all)
    pc_world_all = pc_world_flat_all.reshape(-1, all_num_points, 3)
    pc_world_all = pc_world_all + robot_pos_w.unsqueeze(1)

    # 从世界坐标系转到物体局部坐标系
    pc_relative_w_all = pc_world_all - object_pos_w.unsqueeze(1)

    object_quat_expanded_all = object_quat_w.unsqueeze(1).expand(-1, all_num_points, -1)
    object_quat_flat_all = object_quat_expanded_all.reshape(-1, 4)
    pc_relative_flat_all = pc_relative_w_all.reshape(-1, 3)

    pc_object_flat_all = quat_rotate_inverse(object_quat_flat_all, pc_relative_flat_all)
    point_cloud_3d_object_frame = pc_object_flat_all.reshape(-1, all_num_points, 3)

    # 限制环境数量
    num_envs_to_show = min(num_envs_to_show, unwrapped_env.num_envs)

    # 创建图形
    fig = plt.figure(figsize=(15, 5 * num_envs_to_show))

    for env_id in range(num_envs_to_show):
        # 获取点云 - 所有点和选中的点
        points_selected = point_cloud_3d[env_id].cpu().numpy()  # 选中的点
        points_all = point_cloud_3d_all[env_id].cpu().numpy()  # 所有点

        # 判断是否有筛选（选中点数小于总点数）
        has_selection = (actual_num_points < all_num_points)

        # 获取物体类型信息
        object_type = "Unknown"
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim_path = unwrapped_env.scene[object_name].root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    object_type = mesh_children[0].GetName()
        except Exception as e:
            print(f"Error getting object type: {e}")

        # 创建三个视图
        ax1 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 1)
        ax2 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 2)
        ax3 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 3, projection='3d')

        # 设置标题
        title_suffix = f" (Showing {actual_num_points}/{all_num_points} closest points)" if has_selection else ""
        fig.suptitle(f"Object Point Cloud Visualization (Height Range: {min_height}-{max_height}m){title_suffix}", fontsize=16)

        # 俯视图 (XY平面)
        if has_selection:
            # 先绘制所有点（灰色背景）
            ax1.scatter(points_all[:, 0], points_all[:, 1], c='lightgray', s=15, alpha=0.3, label='All points')
            # 再绘制选中的点（彩色突出）
            sc1 = ax1.scatter(points_selected[:, 0], points_selected[:, 1], c=points_selected[:, 2],
                             cmap='viridis', s=80, alpha=1.0, edgecolors='red', linewidths=2, label='Selected points')
        else:
            sc1 = ax1.scatter(points_all[:, 0], points_all[:, 1], c=points_all[:, 2],
                             cmap='viridis', s=40, alpha=0.8)
        ax1.set_title(f'Env {env_id} - {object_type} - Top View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True)
        ax1.axis('equal')
        if has_selection:
            ax1.legend(loc='upper right', fontsize=8)
        plt.colorbar(sc1, ax=ax1, label='Height (Z)')

        # 侧视图 (XZ平面)
        if has_selection:
            # 先绘制所有点（灰色背景）
            ax2.scatter(points_all[:, 0], points_all[:, 2], c='lightgray', s=15, alpha=0.3, label='All points')
            # 再绘制选中的点（彩色突出）
            sc2 = ax2.scatter(points_selected[:, 0], points_selected[:, 2], c=points_selected[:, 1],
                             cmap='plasma', s=80, alpha=1.0, edgecolors='red', linewidths=2, label='Selected points')
        else:
            sc2 = ax2.scatter(points_all[:, 0], points_all[:, 2], c=points_all[:, 1],
                             cmap='plasma', s=40, alpha=0.8)
        ax2.set_title(f'Env {env_id} - Side View')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.grid(True)
        ax2.axis('equal')
        if has_selection:
            ax2.legend(loc='upper right', fontsize=8)
        plt.colorbar(sc2, ax=ax2, label='Y')

        # 3D视图
        if has_selection:
            # 先绘制所有点（灰色背景）
            ax3.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2],
                       c='lightgray', s=10, alpha=0.3, label='All points')
            # 再绘制选中的点（彩色突出）
            sc3 = ax3.scatter(points_selected[:, 0], points_selected[:, 1], points_selected[:, 2],
                             c=points_selected[:, 2], cmap='viridis', s=80, alpha=1.0,
                             edgecolors='red', linewidths=2, label='Selected points')
        else:
            sc3 = ax3.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2],
                             c=points_all[:, 2], cmap='viridis', s=30, alpha=0.8)
        ax3.set_title('3D View')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.grid(True)
        if has_selection:
            ax3.legend(loc='upper right', fontsize=8)

        # 添加坐标系参考线
        if len(points_all) > 0:
            # 计算点云包围盒（使用所有点来确定视角范围）
            min_coords = np.min(points_all, axis=0)
            max_coords = np.max(points_all, axis=0)
            center = (min_coords + max_coords) / 2

            # 设置更好的视角
            max_range = np.max(max_coords - min_coords) * 0.6
            ax3.set_xlim(center[0] - max_range, center[0] + max_range)
            ax3.set_ylim(center[1] - max_range, center[1] + max_range)
            ax3.set_zlim(center[2] - max_range, center[2] + max_range)

            # 设置视角
            ax3.view_init(elev=30, azim=45)

        # 输出点云统计信息和缩放信息
        print(f"\n{'='*70}")
        print(f"Point Cloud Stats - Env {env_id} - {object_type}:")

        # 显示物体缩放（object_scales 在外部作用域中定义）
        try:
            scale = object_scales[env_id].cpu().numpy()
            print(f"  🔍 Object scale: [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]")
        except Exception as e:
            print(f"  ⚠️  Object scale: 无法读取 ({e})")

        if has_selection:
            print(f"  📊 Points count: {len(points_selected)} selected / {len(points_all)} total")
        else:
            print(f"  📊 Points count: {len(points_all)}")

        # 机器人坐标系下的点云（用于可视化）
        # 显示所有点的范围
        if len(points_all) > 0:
            min_coords_robot = np.min(points_all, axis=0)
            max_coords_robot = np.max(points_all, axis=0)
            range_coords_robot = max_coords_robot - min_coords_robot
            center_robot = (min_coords_robot + max_coords_robot) / 2

            print(f"  📏 All points in robot frame:")
            print(f"     X: [{min_coords_robot[0]:6.3f}, {max_coords_robot[0]:6.3f}] → width  = {range_coords_robot[0]:.3f}")
            print(f"     Y: [{min_coords_robot[1]:6.3f}, {max_coords_robot[1]:6.3f}] → length = {range_coords_robot[1]:.3f}")
            print(f"     Z: [{min_coords_robot[2]:6.3f}, {max_coords_robot[2]:6.3f}] → height = {range_coords_robot[2]:.3f}")
            print(f"  📍 Center: [{center_robot[0]:.3f}, {center_robot[1]:.3f}, {center_robot[2]:.3f}]")

        # 如果有筛选，额外显示选中点的范围
        if has_selection and len(points_selected) > 0:
            min_coords_sel = np.min(points_selected, axis=0)
            max_coords_sel = np.max(points_selected, axis=0)
            range_coords_sel = max_coords_sel - min_coords_sel
            center_sel = (min_coords_sel + max_coords_sel) / 2

            print(f"  🎯 Selected points in robot frame:")
            print(f"     X: [{min_coords_sel[0]:6.3f}, {max_coords_sel[0]:6.3f}] → width  = {range_coords_sel[0]:.3f}")
            print(f"     Y: [{min_coords_sel[1]:6.3f}, {max_coords_sel[1]:6.3f}] → length = {range_coords_sel[1]:.3f}")
            print(f"     Z: [{min_coords_sel[2]:6.3f}, {max_coords_sel[2]:6.3f}] → height = {range_coords_sel[2]:.3f}")
            print(f"  📍 Center: [{center_sel[0]:.3f}, {center_sel[1]:.3f}, {center_sel[2]:.3f}]")

        # 物体局部坐标系下的点云（用于尺寸验证）
        points_object = point_cloud_3d_object_frame[env_id].cpu().numpy()
        if len(points_object) > 0:
            min_coords_obj = np.min(points_object, axis=0)
            max_coords_obj = np.max(points_object, axis=0)
            range_coords_obj = max_coords_obj - min_coords_obj
            center_obj = (min_coords_obj + max_coords_obj) / 2

            print(f"\n  📦 Point cloud in object local frame (for size verification):")
            print(f"     X: [{min_coords_obj[0]:6.3f}, {max_coords_obj[0]:6.3f}] → width  = {range_coords_obj[0]:.3f}")
            print(f"     Y: [{min_coords_obj[1]:6.3f}, {max_coords_obj[1]:6.3f}] → length = {range_coords_obj[1]:.3f}")
            print(f"     Z: [{min_coords_obj[2]:6.3f}, {max_coords_obj[2]:6.3f}] → height = {range_coords_obj[2]:.3f}")
            print(f"  📍 Center in object frame: [{center_obj[0]:.3f}, {center_obj[1]:.3f}, {center_obj[2]:.3f}]")

            # 如果有缩放信息，计算预期尺寸
            try:
                scale = object_scales[env_id].cpu().numpy()
                # 假设原始 cardbox 尺寸为 (0.7, 0.5, 0.5)
                original_size = np.array([0.7, 0.5, 0.5])
                expected_size = original_size * scale
                print(f"\n  ✅ Verification (in object local frame):")
                print(f"     Expected size: [{expected_size[0]:.3f}, {expected_size[1]:.3f}, {expected_size[2]:.3f}]")
                print(f"     Actual vs Expected:")
                print(f"       X: {range_coords_obj[0]:.3f} vs {expected_size[0]:.3f} (ratio: {range_coords_obj[0]/expected_size[0]:.2f})")
                print(f"       Y: {range_coords_obj[1]:.3f} vs {expected_size[1]:.3f} (ratio: {range_coords_obj[1]/expected_size[1]:.2f})")
                print(f"       Z: {range_coords_obj[2]:.3f} vs {expected_size[2]:.3f} (ratio: {range_coords_obj[2]/expected_size[2]:.2f})")

                # 推断缓存点云的基准尺寸
                if np.all(scale > 0):
                    inferred_base = range_coords_obj / scale
                    print(f"     🔍 Inferred base size: [{inferred_base[0]:.3f}, {inferred_base[1]:.3f}, {inferred_base[2]:.3f}]")
            except Exception as e:
                print(f"     ⚠️  Cannot verify: {e}")

            # 计算点云统计数据（使用所有点）
            mean = np.mean(points_all, axis=0)

            # 按高度分布统计
            height_distribution = points_all[:, 2]
            height_quartiles = np.percentile(height_distribution, [25, 50, 75])

            print(f"  📊 Statistical data (all points):")
            print(f"     Mean position: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
            print(f"     Height quartiles: 25%={height_quartiles[0]:.3f}, 50%={height_quartiles[1]:.3f}, 75%={height_quartiles[2]:.3f}]")

            # 如果有筛选，也显示选中点的统计数据
            if has_selection:
                mean_sel = np.mean(points_selected, axis=0)
                height_distribution_sel = points_selected[:, 2]
                height_quartiles_sel = np.percentile(height_distribution_sel, [25, 50, 75])

                print(f"  🎯 Statistical data (selected points):")
                print(f"     Mean position: [{mean_sel[0]:.3f}, {mean_sel[1]:.3f}, {mean_sel[2]:.3f}]")
                print(f"     Height quartiles: 25%={height_quartiles_sel[0]:.3f}, 50%={height_quartiles_sel[1]:.3f}, 75%={height_quartiles_sel[2]:.3f}]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPoint cloud image saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    return point_cloud_3d


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # ===== 添加点云测试代码 =====
    # print("Testing point cloud visualization...")
    # try:
    #     # 先初始化点云缓存（针对训练环境使用的参数）
    #     from omni.isaac.lab.managers import SceneEntityCfg
    #     import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations as obs_module

    #     print("\n初始化点云缓存...")

    #     # 重置初始化标志，以便可以为不同参数生成缓存
    #     obs_module._point_cloud_initialized = False

    #     # 直接调用生成函数为不同参数生成缓存
    #     # 为训练环境的参数生成缓存 (64点)
    #     obs_module.generate_all_point_clouds(
    #         env.unwrapped,
    #         object_cfg=SceneEntityCfg("interactive_object"),
    #         num_points=64,
    #         min_height=0.2,
    #         max_height=0.7,
    #         # surface_edge_margin = 0
    #     )
    #     print("✓ 64点点云缓存已初始化")

    #     # 重置标志以生成1024点的缓存
    #     obs_module._point_cloud_initialized = False

    #     # 为1024点生成缓存
    #     obs_module.generate_all_point_clouds(
    #         env.unwrapped,
    #         object_cfg=SceneEntityCfg("interactive_object"),
    #         num_points=1024,
    #         min_height=0.2,
    #         max_height=0.7,
    #         # surface_edge_margin = 0
    #     )
    #     print("✓ 1024点点云缓存已初始化\n")

    #     test_configs = [
    #         # {"min_height": 0.2, "max_height": 0.7, "num_points": 64, "name": "few_points", "use_optimized": True},
    #         {"min_height": 0.2, "max_height": 0.7, "num_points": 1024, "name": "many_points", "use_optimized": True},
    #     ]

    #     for i, config in enumerate(test_configs):
    #         print(f"\nTesting configuration {i+1}/{len(test_configs)}: {config['name']}")

    #         # 调用可视化函数
    #         point_cloud = visualize_point_cloud(
    #             env=env.unwrapped,
    #             object_name="interactive_object",
    #             num_points=config["num_points"],
    #             min_height=config["min_height"],
    #             max_height=config["max_height"],
    #             num_envs_to_show=3,  # 仅显示前6个环境
    #             save_path=f"/tmp/point_cloud_{config['name']}.png",
    #             show_plot=True,
    #             use_optimized=config.get("use_optimized", False)
    #         )

    #     # 等待用户确认继续
    #     input("\nPress Enter to continue the model playback process...")

    # except Exception as e:
    #     print(f"Point cloud visualization test error: {e}")
    #     import traceback
    #     traceback.print_exc()

    # ===== 点云测试代码结束 =====

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # door open
            if actions.shape[1] >= 18:
                actions[:, 16:18] = 0
            # print(actions)

            # ===== 冻结机械臂关节 =====
            # 获取机器人当前关节位置
            robot = env.unwrapped.scene["robot"]

            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            if args_cli.follow_camera and not args_cli.headless:
                robot_pos = robot.data.root_pos_w[0].tolist()
                env.unwrapped.sim.set_camera_view(
                    eye=(robot_pos[0] - 3.0, robot_pos[1] - 3.0, robot_pos[2] + 2.0),
                    target=(robot_pos[0], robot_pos[1], robot_pos[2] + 0.4),
                )

            # 每10步输出一次诊断信息
            # if timestep % 10 == 0:
            #     print(f"\n{'='*60}")
            #     print(f"Timestep: {timestep}")

            #     # 基座状态
            #     base_pos = robot.data.root_pos_w[0]
            #     base_vel = robot.data.root_lin_vel_w[0]
            #     print(f"基座位置: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f}")
            #     print(f"基座速度: x={base_vel[0]:.3f}, y={base_vel[1]:.3f}, z={base_vel[2]:.3f}")

            #     # 腿部状态
            #     leg_joint_vel = robot.data.joint_vel[0, :13]
            #     leg_joint_torques = robot.data.applied_torque[0, :13]
            #     print(f"\n腿部:")
            #     print(f"  速度: 平均={leg_joint_vel.abs().mean():.3f}, 最大={leg_joint_vel.abs().max():.3f} rad/s")
            #     print(f"  力矩: 平均={leg_joint_torques.abs().mean():.2f}, 最大={leg_joint_torques.abs().max():.2f} N·m")

            #     # 机械臂状态
            #     arm_joint_indices = [8, 13, 14, 15, 16, 17]
            #     arm_joint_vel = robot.data.joint_vel[0, arm_joint_indices]
            #     arm_joint_torques = robot.data.applied_torque[0, arm_joint_indices]
            #     arm_joint_pos = robot.data.joint_pos[0, arm_joint_indices]
            #     arm_joint_pos_target = robot.data.joint_pos_target[0, arm_joint_indices]

            #     print(f"\n机械臂:")
            #     print(f"  速度: 平均={arm_joint_vel.abs().mean():.3f}, 最大={arm_joint_vel.abs().max():.3f} rad/s")
            #     print(f"  力矩: 平均={arm_joint_torques.abs().mean():.2f}, 最大={arm_joint_torques.abs().max():.2f} N·m")
            #     print(f"  位置误差: 平均={((arm_joint_pos_target - arm_joint_pos).abs().mean()):.4f}, 最大={(arm_joint_pos_target - arm_joint_pos).abs().max():.4f} rad")

            #     # 详细的机械臂关节信息
            #     print(f"\n机械臂各关节详情 (索引: 速度 | 力矩 | 位置误差):")
            #     joint_names = ["zarx_j1(8)", "zarx_j2(13)", "zarx_j3(14)", "zarx_j4(15)", "zarx_j5(16)", "zarx_j6(17)"]
            #     for i, (idx, name) in enumerate(zip(arm_joint_indices, joint_names)):
            #         pos_err = abs(arm_joint_pos_target[i] - arm_joint_pos[i])
            #         print(f"  {name}: {arm_joint_vel[i]:6.3f} rad/s | {arm_joint_torques[i]:6.2f} N·m | {pos_err:.4f} rad")

            #     print(f"{'='*60}")
            # timestep += 1

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
