# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project ：organ_seg 
@File    ：multi_gen_group_data.py
@IDE     ：PyCharm 
@Author  ：杨鑫
@Date    ：2024/2/20 09:04 
"""

import os
import random
import uuid
import numpy as np
import torch
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import open3d as o3d


def compute_rotation_matrix(rv):
    angle_to_rad = np.pi / 180
    ax, ay, az = rv * angle_to_rad
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return np.dot(np.dot(rx, ry), rz)


def affine_optimized(points, tv, sv, rot_m):
    return np.dot(points, rot_m.T) * sv + tv


def random_str():
    return str(uuid.uuid4())[:8]


def seeding(row_space, row_num, plant_space, pnum_per_row):
    grid = np.zeros((row_num, pnum_per_row, 3))
    gx = np.linspace(0, (pnum_per_row - 1) * plant_space, pnum_per_row)
    gy = np.linspace(0, (row_num - 1) * row_space, row_num)

    for i in range(row_num):
        for j in range(pnum_per_row):
            r1, r2 = 2 * (np.random.rand() - 0.5), 2 * (np.random.rand() - 0.5)
            x, y = gx[j] + 0.05 * r1 * plant_space, gy[i] + 0.05 * r2 * row_space
            grid[i, j] = [x, y, 0]

    return grid


def farthest_point_sample_gpu(data, npoint=4096):
    """
    Optimized version of Farthest Point Sampling (FPS) algorithm for point cloud data.

    Input:
        data: pointcloud data, [N, 3]
        npoint: number of samples

    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    with torch.no_grad():
        xyz = torch.tensor(data, dtype=torch.float32).cuda()
        N, _ = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long).cuda()
        distance = torch.ones(N).cuda() * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda()[0]

        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest].unsqueeze(0)
            dist = torch.sum((xyz - centroid) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        centroids_numpy = centroids.cpu().numpy()

    return centroids_numpy


def voxel_downsample_with_indices(points, voxel_size):
    """
    手动实现体素下采样，并返回下采样后的点及其在原始点云中的索引。

    参数:
    - points: 原始点云数据，Nx3的numpy数组。
    - voxel_size: 体素的大小。

    返回:
    - downsampled_points: 下采样后的点云，Mx3的numpy数组。
    - indices: 下采样后的点在原始点云中的索引，长度为M的列表。
    """
    # 将点云坐标转换为体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 生成唯一的体素坐标，并找到每个体素内第一个点的索引
    _, indices = np.unique(voxel_indices, axis=0, return_index=True)

    # 按照原始点云的顺序排序索引，以保留点云的结构信息
    indices = np.sort(indices)

    # 使用索引提取下采样后的点
    downsampled_points = points[indices]

    return downsampled_points, indices


def calculate_normals(coords, k_neighbors=10, radius=None):
    """
    计算点云的法线。

    参数:
    - coords: Nx3的NumPy数组，表示点云的坐标。
    - k_neighbors: 用于估计每个点法线的最近邻的数量。
    - radius: 搜索半径，用于估计每个点的法线。如果指定了radius，则使用半径搜索而不是k最近邻搜索。

    返回:
    - normals: Nx3的NumPy数组，表示每个点的法线向量。
    """
    # 将coords数组转换为Open3D的点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # 计算法线
    if radius is not None:
        # 使用半径搜索来计算法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k_neighbors))
    else:
        # 使用k最近邻搜索来计算法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    # 可选：根据视点方向翻转法线
    # pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

    # 从点云对象中提取法线并转换为NumPy数组
    normals = np.asarray(pcd.normals)

    return normals


def process_point_cloud(idx, rot_m, tv, sv, data_set, organ_sem_label_trans=True):
    random_num = np.random.randint(0, len(data_set))
    plant_c = data_set[random_num]
    file_string = plant_c["file"]
    pts = plant_c["pts"]
    semantic_label = plant_c["semantic_labels"]

    # 找到茎的最低点
    stem_points = pts[semantic_label == 0]  # 筛选出茎部分的点云
    if len(stem_points) > 0:
        lowest_point = np.min(stem_points[:, 2])  # 假设Z轴为高度，找到最低点
    else:
        lowest_point = np.min(pts[:, 2])  # 如果没有茎部分，使用整个点云的最低点

    # 删除Z轴低于茎底以下0.05米的点
    threshold = lowest_point - 0.05
    pts_above_threshold = pts[pts[:, 2] >= threshold]
    semantic_label = semantic_label[pts[:, 2] >= threshold]  # 同步筛选语义标签
    rgb = plant_c["rgb"][pts[:, 2] >= threshold]  # 同步筛选RGB值
    instance_label = plant_c["instance_labels"][pts[:, 2] >= threshold]  # 同步筛选实例标签

    # 平移整个点云，使茎的最低点对齐到原点的Z轴
    pts_new = pts_above_threshold - [0, 0, lowest_point]

    # 应用仿射变换
    pts_new = affine_optimized(pts_new, tv, sv, rot_m)

    label = np.full(len(pts_new), idx)

    # 将雄穗变为叶子 1-> 1， 雌穗变为叶子 2-> 1, 叶子 3-> 1
    if organ_sem_label_trans:
        xs_mask = semantic_label == 1
        cs_mask = semantic_label == 2
        leaf_mask = semantic_label == 3
        semantic_label[xs_mask] = 1
        semantic_label[cs_mask] = 1
        semantic_label[leaf_mask] = 1

    # 给所有的实例标签重新编号
    instance_label = instance_label + idx * 1000
    return pts_new, label, rgb, semantic_label, instance_label, file_string


def process_iteration(args):
    data_set, target_dir, row_num, plant_num_row, row_space, plant_space, iter = args
    # terrain_points, terrain_rgb, field = generate_terrain_and_seeding(row_num, plant_num_row, row_space, plant_space, wave_height=0.05, noise_level=0.05)

    field = seeding(row_space, row_num, plant_space, plant_num_row)
    rotations = np.random.rand(row_num * plant_num_row, 3) * np.array([2, 2, 360]) - np.array([1, 1, 0])
    translations = field.reshape(-1, 3)
    scales = np.ones((row_num * plant_num_row, 3))
    for i_s in range(len(scales)):
        scales[i_s] *= random.uniform(0.9, 1.1)
    rotation_matrices = np.array([compute_rotation_matrix(rv) for rv in rotations])

    group_plant, group_labels, rgbs, semantic_labels, instance_labels = [], [], [], [], []
    file_strings = []


    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交任务到线程池
        futures = [executor.submit(process_point_cloud, idx, rot_m, tv, sv, data_set)
                   for idx, (rot_m, tv, sv) in enumerate(zip(rotation_matrices, translations, scales))]

        for future in as_completed(futures):
            pts_new, label, rgb, semantic_label, instance_label, file_string = future.result()
            group_plant.append(pts_new)
            group_labels.extend(label)
            rgbs.extend(rgb)
            semantic_labels.extend(semantic_label)
            instance_labels.extend(instance_label)
            file_strings.append(file_string)

    group_plant = np.vstack(group_plant)
    rgbs = np.array(rgbs)
    semantic_labels = np.array(semantic_labels)
    instance_labels = np.array(instance_labels)
    group_instance_labels = np.array(group_labels)
    group_semantic_labels = np.zeros(len(group_plant))

    _, instance_labels = np.unique(instance_labels, return_inverse=True)

    # 下采样
    _, v_indices = voxel_downsample_with_indices(group_plant, voxel_size=0.01)
    group_plant_land_xyz = group_plant[v_indices]
    group_plant_land_rgb = rgbs[v_indices]
    group_plant_land_organ_semantic_label = semantic_labels[v_indices]
    group_plant_land_organ_instance_label = instance_labels[v_indices]
    group_plant_land_semantic_label = group_semantic_labels[v_indices]
    group_plant_land_instance_label = group_instance_labels[v_indices]

    # 创建一个索引数组，用于打乱所有采样后的数据
    indices = np.arange(len(group_plant_land_xyz))
    np.random.shuffle(indices)

    # 使用打乱的索引重新组织数据
    group_plant_land_xyz_shuffled = group_plant_land_xyz[indices]
    group_plant_land_rgb_shuffled = group_plant_land_rgb[indices]
    group_plant_land_organ_semantic_label_shuffled = group_plant_land_organ_semantic_label[indices]
    group_plant_land_organ_instance_label_shuffled = group_plant_land_organ_instance_label[indices]
    group_plant_land_semantic_label_shuffled = group_plant_land_semantic_label[indices]
    group_plant_land_instance_label_shuffled = group_plant_land_instance_label[indices]

    random_name = f"group_r{row_num}_c{plant_num_row}_rs{row_space}_cs{plant_space}_{iter}_{random_str()}"

    normal = calculate_normals(group_plant_land_xyz_shuffled)
    torch.save(
        {
            "coord": group_plant_land_xyz_shuffled,
            "color": group_plant_land_rgb_shuffled,
            "normal": normal,
            "organ_semantic_gt": group_plant_land_organ_semantic_label_shuffled,
            "organ_instance_gt": group_plant_land_organ_instance_label_shuffled,
            "semantic_gt": group_plant_land_semantic_label_shuffled,
            "instance_gt": group_plant_land_instance_label_shuffled,
            "scene_id": random_name,
            # "superpoint": get_superpoint(group_plant_shuffled, normal)
        },
        os.path.join(target_dir, "data", f"{random_name}.pth")
    )


def load_dataset_from_npz(npz_file):
    with np.load(npz_file, allow_pickle=True) as data:
        data_set = data['data_set']
        # `data_set`是一个对象数组，每个对象是一个包含数据的字典
        return data_set.tolist()  # 将numpy数组转换为列表，以便后续处理


def save_dataset_as_npz(data_root, output_file, npoint=20000):
    data_set = []
    for name in tqdm(os.listdir(data_root)):
        if name.endswith('.txt'):
            try:
                data_path = os.path.join(data_root, name)
                data = np.loadtxt(data_path)
                pts = data[:, :3]
                pts_sample_idx = farthest_point_sample_gpu(data, npoint=npoint)
                pts = pts[pts_sample_idx]
                rgb = data[:, 3:6][pts_sample_idx]
                semantic_label = data[:, 6][pts_sample_idx]
                instance_label = data[:, 7][pts_sample_idx]
                data_set.append({
                    "file": name,
                    "pts": pts,
                    "rgb": rgb,
                    "semantic_labels": semantic_label,
                    "instance_labels": instance_label
                })
            except:
                print("error:", name)

    # 使用字典存储所有数据，便于后续保存为npz格式
    np.savez(output_file, data_set=data_set)
    print(f"Dataset saved as {output_file}")


def compute_edges(xyz, k=10):
    min_point = xyz.min(axis=0)
    # X = np.linalg.norm(xyz - min_point, axis=1)
    X = xyz - min_point
    kdt = KDTree(X, metric='euclidean')
    query_res = kdt.query(X, k=k)
    # knn_radius = query_res[0][:, k - 1]
    neighbors = query_res[1]

    return neighbors



def process_file(name, data_root, npoint=20000):
    sample = False
    try:
        data_path = os.path.join(data_root, name)
        data = np.loadtxt(data_path)
        pts = data[:, :3]
        if sample:
            pts_sample_idx = farthest_point_sample_gpu(pts, npoint=npoint)
            pts = pts[pts_sample_idx]
            rgb = data[:, 3:6][pts_sample_idx]
            semantic_label = data[:, 6][pts_sample_idx]
            instance_label = data[:, 7][pts_sample_idx]
        else:
            rgb = data[:, 3:6]
            semantic_label = data[:, 6]
            instance_label = data[:, 7]
        pts_min = pts[np.argmin(pts[:, 2])]
        pts -= pts_min
        return {
            "file": name,
            "pts": pts,
            "rgb": rgb,
            "semantic_labels": semantic_label,
            "instance_labels": instance_label
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


def save_dataset_as_npz_multithreaded(data_root, output_file, npoint=20000, max_workers=8):
    dataset = []
    names = [name for name in os.listdir(data_root) if name.endswith('.txt')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将所有任务提交到executor
        futures = [executor.submit(process_file, name, data_root, npoint) for name in names]

        # 初始化进度条
        with tqdm(total=len(futures), desc="Processing Files") as progress_bar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    dataset.append(result)
                # 对于每个完成的任务，更新进度条
                progress_bar.update(1)

    # 保存数据集为npz格式
    np.savez(output_file, data_set=dataset)
    print(f"Dataset saved as {output_file}")


def generate_datasets():
    data_root = "/home/yangxin/datasets/3d_qunti/txt/20240218_cs/data"  # 数据目录
    output_file = "/home/yangxin/datasets/3d_qunti/txt/20240218_cs/dataset_0326.npz"  # 输出文件名
    print(output_file)
    # save_dataset_as_npz(data_root, output_file, npoint=20000)
    save_dataset_as_npz_multithreaded(data_root, output_file, npoint=20000, max_workers=8)


def process_iteration_wrapper(args):
    while True:
        try:
            process_iteration(args)
            break
        except Exception as e:
            print(e, 'again')


def main():
    data_set_root = "/home/yangxin/datasets/3d_qunti/txt/20240218_cs/dataset_0326.npz"
    target_dir = "/data1/3d_qunti/exp_datasets/20240607_vega_800_cs_group"

    new_num = 200

    print(target_dir)
    os.makedirs(os.path.join(target_dir, "data"), exist_ok=True)
    # os.makedirs(os.path.join(target_dir, "items"), exist_ok=True)

    data_set = load_dataset_from_npz(data_set_root)
    print(len(data_set))

    for i in tqdm(range(new_num), desc='0.6, 0.6'):
        process_iteration_wrapper((data_set, target_dir, 4, 9, 0.6, 0.6, i + 1))
    for i in tqdm(range(new_num), desc='0.6, 0.45'):
        process_iteration_wrapper((data_set, target_dir, 4, 9, 0.6, 0.45, i + 1))
    for i in tqdm(range(new_num), desc='0.6, 0.2'):
        process_iteration_wrapper((data_set, target_dir, 4, 9, 0.6, 0.2, i + 1))
    for i in tqdm(range(new_num), desc='0.6, 0.1'):
        process_iteration_wrapper((data_set, target_dir, 4, 9, 0.6, 0.1, i + 1))


if __name__ == "__main__":
    # generate_datasets()
    main()