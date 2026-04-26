#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给定场景与目标物体，自动采样相机点位：
- 点位必须在“包含该物体”的 room polygon 内
- 不在任何物体内部
- 不离其他物体太近
- 与目标物体距离在合理范围
- 相机 forward 指向目标物体中心（物体在视野中央）
- 每个点位只生成 1 个视角
- 渲染所有视图并拼接

直接在 __main__ 里改 SCENE_ROOT 和 TARGET_ID 即可运行。
"""

import json
import math
import numpy as np
import imageio
from pathlib import Path

from ..bench_generation.batch_utils import (
    load_room_polys,
    load_scene_aabbs,
    point_in_poly,
    camtoworld_from_pos_target,
    occluded_area_on_image,
)


# ==========================================================
# 数据结构
# ==========================================================
class SceneObject:
    def __init__(self, obj_json):
        self.id = obj_json["ins_id"]
        self.label = obj_json["label"]

        xs = [p["x"] for p in obj_json["bounding_box"]]
        ys = [p["y"] for p in obj_json["bounding_box"]]
        zs = [p["z"] for p in obj_json["bounding_box"]]

        self.bbox_min = np.array([min(xs), min(ys), min(zs)], dtype=float)
        self.bbox_max = np.array([max(xs), max(ys), max(zs)], dtype=float)
        self.position = (self.bbox_min + self.bbox_max) / 2


# ==========================================================
# 工具函数
# ==========================================================
def is_inside_object(pt, obj: SceneObject) -> bool:
    """检查 pt 是否在物体 AABB 内"""
    return np.all(pt >= obj.bbox_min) and np.all(pt <= obj.bbox_max)


def is_too_close_to_any_object(pt, aabbs, min_dist=0.2) -> bool:
    """检查 pt 是否离任意物体中心太近"""
    for b in aabbs:
        center = 0.5 * (b.bmin + b.bmax)
        if np.linalg.norm(pt - center) < min_dist:
            return True
    return False


def _dist_point_to_segment_2d(p, a, b):
    """2D: point p to segment ab 最短欧氏距离"""
    # p, a, b: array-like (2,)
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ba = b - a
    pa = p - a
    denom = np.dot(ba, ba)
    if denom <= 1e-12:
        # a 和 b 非常接近，退化为点距
        return float(np.linalg.norm(p - a))
    t = np.dot(pa, ba) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ba
    return float(np.linalg.norm(p - proj))


def point_to_polygon_edge_distance(pt2d, polygon):
    """计算 2D 点到多边形各条边的最短欧氏距离。

    参数:
      pt2d: 可被 np.asarray 转为 (2,) 的点
      polygon: sequence 或 numpy array，形状 (N,2)

    返回: 最短距离（float）
    """
    p = np.asarray(pt2d, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be (N,2) array-like")
    n = poly.shape[0]
    if n == 0:
        return float('inf')
    min_d = float('inf')
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        d = _dist_point_to_segment_2d(p, a, b)
        if d < min_d:
            min_d = d
    return float(min_d)


def sample_points_in_room(poly, count=20, min_height: float = 0.6, max_height: float = 1.0):
    """在 room 多边形基底上生成接近均匀的 2D 网格采样并升成 3D 点 (x,y,z).

    策略：在包围盒上构造 nx * ny 的等间距格点（nx≈sqrt(count*aspect)），
    取落在多边形内部的点；如果过滤后点数多于 count，等距挑取 count 个；
    如果不足，则用随机采样补齐（最多尝试 count*60 次）。Z 使用中间高度
    (min_height+max_height)/2，保证高度落在指定范围内且对比随机采样更稳定。
    """
    poly = np.array(poly)
    xs, ys = poly[:, 0], poly[:, 1]

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    aspect = width / height

    # grid dims (try to make nx * ny >= count)
    nx = max(1, int(round(math.sqrt(count * aspect))))
    ny = max(1, int(math.ceil(count / float(nx))))

    xs_lin = np.linspace(xmin + 1e-6, xmax - 1e-6, nx)
    ys_lin = np.linspace(ymin + 1e-6, ymax - 1e-6, ny)
    gx, gy = np.meshgrid(xs_lin, ys_lin)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T

    inside_pts = []
    for (x, y) in grid_pts:
        try:
            if point_in_poly(float(x), float(y), poly):
                inside_pts.append((float(x), float(y)))
        except Exception:
            continue

    # If too many, pick evenly spaced indices
    if len(inside_pts) >= count:
        indices = np.linspace(0, len(inside_pts) - 1, num=count, dtype=int)
        selected = [inside_pts[i] for i in indices]
    else:
        # fallback: take all grid points inside, then random-fill until count
        selected = inside_pts.copy()
        tries = 0
        while len(selected) < count and tries < count * 60:
            tries += 1
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            try:
                if not point_in_poly(float(x), float(y), poly):
                    continue
            except Exception:
                continue
            selected.append((float(x), float(y)))

    # promote to 3D using mid-height for stability
    z_mid = float(0.5 * (min_height + max_height))
    pts3 = [np.array([x, y, z_mid], dtype=float) for (x, y) in selected]
    return pts3


# ==========================================================
# 相机点位生成（保证同一个 room）
# ==========================================================
def generate_camera_positions(scene_root: Path,
                              target_obj: SceneObject,
                              per_room_points: int = 20,
                              min_dist: float = 0.4,
                              max_dist: float = 3.5,
                              min_height: float = 0.6,
                              max_height: float = 1.2):
    """
    只在“包含该物体中心”的 room 中采样相机点位。
    """
    polys = load_room_polys(str(scene_root))
    aabbs = load_scene_aabbs(str(scene_root))

    obj_xy = target_obj.position[:2]

    # 1️⃣ 找出所有包含该物体的 room polygon
    rooms_for_object = []
    for poly in polys:
        if point_in_poly(float(obj_xy[0]), float(obj_xy[1]), np.array(poly)):
            rooms_for_object.append(poly)

    if not rooms_for_object:
        # print("[warn] 物体中心不在任何 room polygon 内，退化为使用所有 room 采样。")
        # rooms_for_object = polys
        return []
    # else:
    #     print(f"[info] 找到 {len(rooms_for_object)} 个包含该物体的 room，用于采样相机点位。")

    all_poses = []

    # 2️⃣ 只在这些 room 里采样
    for poly in rooms_for_object:
        pts = sample_points_in_room(poly, count=per_room_points, min_height=min_height, max_height=max_height)

        for pos in pts:
            # 不能落在任何一个物体里（包括目标物体与其他物体）
            inside_any = False
            eps = 0.2

            for b_any in aabbs:
                if (pos[0] >= b_any.bmin[0]-eps and pos[0] <= b_any.bmax[0]+eps and
                        pos[1] >= b_any.bmin[1]-eps and pos[1] <= b_any.bmax[1]+eps and
                        pos[2] >= b_any.bmin[2]-eps and pos[2] <= b_any.bmax[2]+eps):
                        inside_any = True
                        break
            if inside_any:
                continue
            # 距离房间边界也不能太近（水平距离 >= eps）
            try:
                pt2 = np.array([pos[0], pos[1]], dtype=float)
                # poly 是当前循环里的房间多边形（二维点序列）
                d_edge = point_to_polygon_edge_distance(pt2, poly)
                if d_edge < eps:
                    # 距离墙/房间边缘太近，跳过
                    continue
            except Exception:
                # 如果计算失败，保守地继续后面的判断（不直接接受）
                pass
            # 与其他物体太近就丢弃（避免撞上家具）
            if is_too_close_to_any_object(pos, aabbs, min_dist=min_dist):
                continue

            # 与目标物体距离范围限制（既不要太近，也不要太远）
            d = np.linalg.norm(pos - target_obj.position)
            if d < min_dist or d > max_dist:
                continue

            # forward 直接指向目标物体 —— 保证物体在视野中央附近
            forward = target_obj.position - pos
            forward = forward / (np.linalg.norm(forward) + 1e-6)

            target = pos + forward  # 用于你的 render_thumbnail_for_pose 的 "target"

            # 使用 occlusion helper 判断目标可见性（image-space occlusion <= 30% 视为可见）
            try:
                width = 400
                height = 400
                focal = float(width * 0.4)
                K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=float)
                camtoworld = camtoworld_from_pos_target(pos, target)
                res = occluded_area_on_image(pos, target_obj.bbox_min, target_obj.bbox_max, aabbs, K, camtoworld, width, height, target_id=target_obj.id, depth_mode='min', return_per_occluder=False)
                occ_ratio = float(res.get('occlusion_ratio_target', 0.0))
                print(occ_ratio)
            except Exception:
                occ_ratio = 1.0

            if occ_ratio > 0.3:
                # too occluded, skip
                continue

            all_poses.append({
                "position": pos.copy(),
                "target": target.copy(),
                "forward": forward.copy(),
                "object_id": target_obj.id,
                "object_label": target_obj.label,
                "occlusion_ratio": float(occ_ratio),
            })

    # print(f"[info] 共生成 {len(all_poses)} 个相机位姿（均在目标物体所在 room 内）")
    return all_poses


# ==========================================================
# 图像渲染 & 拼接
# ==========================================================
def render_and_mosaic(poses, scene_root, out_path="mosaic.png", thumb=256, per_row=8, gpu_id=0):
    if not poses:
        print("[warn] 没有可渲染的相机位姿。")
        return

    # import preview renderer lazily to avoid heavy imports at module import time
    try:
        from ..bench_generation.preview import render_thumbnail_for_pose
    except Exception:
        raise

    imgs = []
    for pose in poses:
        img = render_thumbnail_for_pose(
            scene_root,
            pose,
            thumb_size=thumb,
            gpu_id=gpu_id
        )
        imgs.append(img)

    H, W, C = imgs[0].shape
    rows = (len(imgs) + per_row - 1) // per_row
    canvas = np.zeros((rows * H, per_row * W, C), dtype=np.uint8)

    for i, img in enumerate(imgs):
        r = i // per_row
        c = i % per_row
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = img

    imageio.imwrite(out_path, canvas)
    print(f"[save] 拼接图已保存到: {out_path}")
    return canvas


# ==========================================================
# 主函数
# ==========================================================
def main(scene_root: Path, target_id: str, output_path="mosaic_views.png", gpu_id=0):
    # 这里假设 scene_root 直接是类似 .../data/0013_840910 这一层
    labels_path = scene_root / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json 不存在: {labels_path}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # 建立对象字典
    objects = {}
    for item in labels:
        if "bounding_box" in item and item.get("ins_id") and len(item["bounding_box"]) >= 4:
            obj = SceneObject(item)
            objects[obj.id] = obj

    if target_id not in objects:
        print(f"[error] 未找到目标物体 ID={target_id}（labels.json 中不存在该 ins_id）")
        return

    target_obj = objects[target_id]
    print(f"[info] 目标物体: id={target_obj.id}, label={target_obj.label}, center={target_obj.position}")

    poses = generate_camera_positions(scene_root, target_obj,
                                      per_room_points=20,
                                      min_dist=0.6,
                                      max_dist=3.5)

    # render mosaic and get image array
    canvas = None
    if poses:
        canvas = render_and_mosaic(poses, scene_root, out_path=output_path, gpu_id=gpu_id)
    else:
        print("[warn] 没有生成任何 pose，跳过渲染。")

    # assemble meta information
    meta = {
        'scene': str(scene_root.name),
        'target_id': str(target_obj.id),
        'label': str(target_obj.label),
        'bbox_min': [float(x) for x in target_obj.bbox_min.tolist()],
        'bbox_max': [float(x) for x in target_obj.bbox_max.tolist()],
        'poses': []
    }
    for p in poses:
        meta['poses'].append({
            'position': [float(x) for x in p['position'].tolist()],
            'target': [float(x) for x in p['target'].tolist()],
            'forward': [float(x) for x in p.get('forward', np.array([0.0,0.0,0.0])).tolist()],
            'occlusion_ratio': float(p.get('occlusion_ratio', 1.0)),
            'object_id': p.get('object_id'),
            'object_label': p.get('object_label'),
        })

    return canvas, meta


# ==========================================================
if __name__ == "__main__":
    # 这里的 SCENE_ROOT 要指向“包含 labels.json 的那一层目录”
    # 如果你的目录是 /data/.../ViewSuite/data/0013_840910/labels.json
    # 就写成 Path("/data/.../ViewSuite/data/0013_840910")
    SCENE_ROOT = Path("/data/liubinglin/jijiatong/ViewSuite/data/0013_840910")
    TARGET_ID = "89"
    OUTPUT_PATH = "views_89_same_room.png"
    GPU_ID = 0  # 指定使用的 GPU 编号

    main(SCENE_ROOT, TARGET_ID, OUTPUT_PATH, gpu_id=GPU_ID)
