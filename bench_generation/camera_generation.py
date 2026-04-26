#!/usr/bin/env python3
"""
语义相机系统 - 简化版
基于场景中的物体语义标签自动计算最佳相机位置

主要功能:
1. 从labels.json中提取场景物体
2. 根据用户选择的物体和视角自动计算相机位置
3. 生成可用于渲染的相机配置

使用方法:
    python semantic_camera.py [scene_path]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
from ..utils.occlusion import load_scene_aabbs, load_scene_wall_aabbs


@dataclass
class SceneObject:
    """场景物体"""
    id: str
    label: str
    position: np.ndarray  # 3D中心位置
    size: np.ndarray      # 3D尺寸
    bbox_min: np.ndarray  # 边界框最小点
    bbox_max: np.ndarray  # 边界框最大点


@dataclass
class CameraConfig:
    """相机配置"""
    object_id: str
    object_label: str
    preset: str
    camera_position: np.ndarray
    target_position: np.ndarray
    distance: float
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray


class SemanticCamera:
    """语义相机系统

    修复版：去除之前破损的逻辑，重建类结构，并实现用户要求的新的相机生成算法：
    从物体的 AABB 表面（沿预设方向）开始向外移动，直到：
      1. 物体全部八个顶点都在视野内（投影在图像范围且在前方）
      2. 移动路径上的任意位置不进入任何 AABB
    否则返回 None。
    """

    def __init__(self, scene_path: str):
        self.scene_path = Path(scene_path)
        self.objects: Dict[str, SceneObject] = {}
        try:
            self._aabbs_all = load_scene_aabbs(str(self.scene_path)) + load_scene_wall_aabbs(str(self.scene_path))
        except Exception:
            self._aabbs_all = []
        # 房间多边形（用于约束相机始终在某房间内部）
        self._room_polys: List[np.ndarray] = self._load_room_polys()

        # 预设方向（相对物体中心的位移向量，仅用于确定方向）
        # 方向说明 (右手系): X=右(+), Y=后(+), Z=上(+)
        # 新增水平与斜向预设：左前/右前/左后/右后，以及其上扬版本和低角度版本
        self.presets: Dict[str, np.ndarray] = {
            'front': np.array([0.0, -1.0, 0.3]),          # 前
            'back': np.array([0.0, 1.0, 0.3]),            # 后
            'left': np.array([-1.0, 0.0, 0.3]),           # 左
            'right': np.array([1.0, 0.0, 0.3]),           # 右
            'above': np.array([0.0, 0.0, 1.0]),           # 俯视
            'diagonal': np.array([1.0, -1.0, 0.6]),       # 原对角(右前上)
            'close': np.array([0.0, -0.5, 0.15]),         # 近距离前视
            'low': np.array([0.0, -1.0, -0.3]),           # 前低角度
            # --- 新增水平斜向 ---
            'front_left': np.array([-1.0, -1.0, 0.3]),    # 左前
            'front_right': np.array([1.0, -1.0, 0.3]),    # 右前
            'back_left': np.array([-1.0, 1.0, 0.3]),      # 左后
            'back_right': np.array([1.0, 1.0, 0.3]),      # 右后
            # --- 新增升高斜向 ---
            'front_left_up': np.array([-1.0, -1.0, 0.6]),
            'front_right_up': np.array([1.0, -1.0, 0.6]),
            'back_left_up': np.array([-1.0, 1.0, 0.6]),
            'back_right_up': np.array([1.0, 1.0, 0.6]),
            # --- 新增低角度斜向 ---
            'front_left_low': np.array([-1.0, -1.0, -0.2]),
            'front_right_low': np.array([1.0, -1.0, -0.2]),
            'back_left_low': np.array([-1.0, 1.0, -0.2]),
            'back_right_low': np.array([1.0, 1.0, -0.2])
        }
        self._load_objects()

    # ------------------------------- 数据加载 ---------------------------------
    def _load_objects(self):
        labels_file = self.scene_path / "labels.json"
        if not labels_file.exists():
            print(f"❌ 标签文件不存在: {labels_file}")
            return
        with open(labels_file, 'r') as f:
            data = json.load(f)
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            obj_id = item.get("ins_id")
            label = item.get("label")
            bbox = item.get("bounding_box", [])
            if not obj_id or not label or not bbox:
                continue
            if label.lower() in {"wall", "floor", "ceiling", "room"}:
                continue
            if len(bbox) < 8:
                continue
            xs = [p['x'] for p in bbox]; ys = [p['y'] for p in bbox]; zs = [p['z'] for p in bbox]
            bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
            bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
            pos = (bmin + bmax) / 2.0
            size = bmax - bmin
            self.objects[obj_id] = SceneObject(id=obj_id, label=label, position=pos, size=size, bbox_min=bmin, bbox_max=bmax)
            if count < 10:
                print(f"  📦 {label} (ID:{obj_id}) at {pos}")
            count += 1
        print(f"✅ 共加载 {count} 个物体")

    def _load_room_polys(self) -> List[np.ndarray]:
        """从 <scene>/structure.json 读取房间多边形，返回 [Nx2 numpy.array] 列表。
        若文件缺失或无有效房间，返回空列表（此时不强制房间约束）。"""
        p = self.scene_path / 'structure.json'
        if not p.exists():
            return []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rooms = data.get('rooms', [])
            polys: List[np.ndarray] = []
            for r in rooms:
                prof = r.get('profile')
                if not prof or len(prof) < 3:
                    continue
                arr = np.array(prof, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                polys.append(arr[:, :2])
            return polys
        except Exception:
            return []

    @staticmethod
    def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
        """射线法点在多边形内判定。poly: Nx2。"""
        if poly is None or len(poly) < 3:
            return False
        inside = False
        n = len(poly)
        px = poly[:, 0]
        py = poly[:, 1]
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def _inside_any_room(self, pos: np.ndarray) -> bool:
        """若存在房间多边形，要求 pos 的 XY 在任一房间内；若没有房间数据，则放宽为 True。"""
        if not self._room_polys:
            return True
        x, y = float(pos[0]), float(pos[1])
        for poly in self._room_polys:
            if self._point_in_poly(x, y, poly):
                return True
        return False

    # ------------------------------- 查询接口 ---------------------------------
    def list_objects(self) -> Dict[str, str]:
        return {oid: o.label for oid, o in self.objects.items()}

    def list_by_category(self) -> Dict[str, List[str]]:
        cats: Dict[str, List[str]] = {}
        for oid, o in self.objects.items():
            cats.setdefault(o.label, []).append(oid)
        return cats

    def get_object(self, obj_id: str) -> Optional[SceneObject]:
        return self.objects.get(obj_id)

    # ------------------------------- AABB辅助 ---------------------------------
    def _point_inside_any_aabb(self, p: np.ndarray) -> bool:
        for b in self._aabbs_all:
            try:
                if (p[0] >= b.bmin[0] and p[0] <= b.bmax[0] and
                    p[1] >= b.bmin[1] and p[1] <= b.bmax[1] and
                    p[2] >= b.bmin[2] and p[2] <= b.bmax[2]):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------- 相机逻辑 ---------------------------------
    def calculate_camera(self, obj_id: str, preset: str = 'front', distance_scale: float = 1.0,
                          height_offset: float = 0.0) -> Optional[CameraConfig]:
        """新的相机生成逻辑：从物体 AABB 表面沿预设方向向外移动寻找第一个完整可见视角。

        步骤:
          1. 选定方向 move_dir（来自 preset 向量归一化，若为零则使用 +Y）。
          2. 求从物体中心沿 move_dir 射线与 AABB 外表面的交点，作为起点 (稍微加 eps)。
          3. 迭代向外移动，期间任何一步进入任意 AABB -> 失败返回 None。
          4. 如果当前相机使物体全部角点投影在成像范围且 z>0 -> 返回配置。
        """
        if obj_id not in self.objects:
            print(f"❌ 物体 {obj_id} 不存在")
            return None
        if preset not in self.presets:
            print(f"❌ 预设 {preset} 不存在。可用: {list(self.presets.keys())}")
            return None

        obj = self.objects[obj_id]
        center = obj.position
        bmin = obj.bbox_min
        bmax = obj.bbox_max

        base_dir = self.presets[preset].astype(float)
        if np.linalg.norm(base_dir) < 1e-8:
            base_dir = np.array([0.0, 1.0, 0.0])  # 默认后视方向
        move_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)
        if preset == 'above':  # 俯视特殊处理：只向 Z+
            move_dir = np.array([0.0, 0.0, 1.0])

        # 射线与AABB表面交点（从中心出发沿 move_dir）
        # 对每个轴求 t，使得点离开盒子：
        ts = []
        for i in range(3):
            d = move_dir[i]
            if abs(d) < 1e-9:
                continue
            face = bmax[i] if d > 0 else bmin[i]
            t = (face - center[i]) / d
            if t > 0:
                ts.append(t)
        if not ts:
            # 没有轴能出盒，方向为零或异常
            return None
        t_exit = min(ts)  # 离开盒子的最早时间
        eps = max(1e-3, 0.01 * float(max(obj.size)))
        start_pos = center + move_dir * (t_exit + eps)  # 略微在外面
        start_pos[2] += height_offset

        if self._point_inside_any_aabb(start_pos):
            print("起始点仍在某个 AABB 内，放弃")
            return None
        # 起始点也必须在房间内部（若有房间数据）
        if not self._inside_any_room(start_pos):
            print("起始点不在任何房间内部，放弃")
            return None

        # 简单 pinhole 相机参数（与其它模块保持风格）
        width = 400
        height = 400
        focal = width * 0.4
        K = np.array([[focal, 0, width/2.0], [0, focal, height/2.0], [0, 0, 1.0]], dtype=float)

        def aabb_corners(bmin_: np.ndarray, bmax_: np.ndarray) -> np.ndarray:
            xs = [bmin_[0], bmax_[0]]; ys = [bmin_[1], bmax_[1]]; zs = [bmin_[2], bmax_[2]]
            return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

        def camtoworld(pos: np.ndarray, tgt: np.ndarray) -> np.ndarray:
            fwd = tgt - pos
            n = np.linalg.norm(fwd)
            fwd = np.array([0,0,1], dtype=float) if n < 1e-8 else fwd / n
            world_up = np.array([0,0,1], dtype=float)
            right = np.cross(fwd, world_up)
            rn = np.linalg.norm(right)
            right = np.array([1,0,0], dtype=float) if rn < 1e-6 else right / rn
            up = np.cross(right, fwd)
            M = np.eye(4, dtype=float)
            M[:3,0] = -right
            M[:3,1] = up
            M[:3,2] = fwd
            M[:3,3] = pos
            return M

        def world_to_camera(inv_view: np.ndarray, p: np.ndarray) -> np.ndarray:
            hp = np.array([p[0], p[1], p[2], 1.0], dtype=float)
            pc = inv_view @ hp
            return pc[:3]

        def project(pc: np.ndarray) -> Tuple[float, float, float]:
            x,y,z = pc
            if z <= 1e-8:
                return np.inf, np.inf, z
            u = K[0,0]*x/z + K[0,2]
            v = K[1,1]*y/z + K[1,2]
            return u,v,z

        corners = aabb_corners(bmin, bmax)
        step = max(0.2, 0.2 * float(max(obj.size))) * distance_scale
        max_dist = float(max(obj.size)) * 20.0 * distance_scale
        travelled = 0.0
        cam_pos = start_pos.copy()

        while travelled <= max_dist:
            # 可见性测试
            c2w = camtoworld(cam_pos, center)
            inv = np.linalg.inv(c2w)
            fully_visible = True
            for c in corners:
                pc = world_to_camera(inv, c)
                u,v,z = project(pc)
                if z <= 1e-6 or not (0 <= u < width and 0 <= v < height):
                    fully_visible = False
                    break
            if fully_visible:
                forward = center - cam_pos
                dist = np.linalg.norm(forward)
                if dist < 1e-8:
                    return None
                forward = forward / dist
                world_up = np.array([0,0,1], dtype=float)
                right = np.cross(forward, world_up)
                rn = np.linalg.norm(right)
                right = np.array([1,0,0], dtype=float) if rn < 1e-6 else right / rn
                up = np.cross(right, forward)
                return CameraConfig(
                    object_id=obj.id,
                    object_label=obj.label,
                    preset=preset,
                    camera_position=cam_pos.copy(),
                    target_position=center.copy(),
                    distance=dist,
                    forward=forward,
                    right=right,
                    up=up
                )

            # 前进一步，路径合法性检查
            next_pos = cam_pos + move_dir * step
            travelled += step
            if self._point_inside_any_aabb(next_pos):
                print("移动路径进入某个 AABB，失败返回 None")
                return None
            if not self._inside_any_room(next_pos):
                print("移动路径离开房间边界，失败返回 None")
                return None
            cam_pos = next_pos

        print("超过最大距离仍未获得完整视野，返回 None")
        return None

    # ------------------------------- 其它辅助 ---------------------------------
    def save_config(self, cfg: CameraConfig) -> str:
        out = {
            'object_id': cfg.object_id,
            'object_label': cfg.object_label,
            'preset': cfg.preset,
            'camera_position': cfg.camera_position.tolist(),
            'target_position': cfg.target_position.tolist(),
            'distance': cfg.distance,
            'forward': cfg.forward.tolist(),
            'right': cfg.right.tolist(),
            'up': cfg.up.tolist()
        }
        out_file = self.scene_path / f"camera_{cfg.object_id}_{cfg.preset}.json"
        with open(out_file, 'w') as f:
            json.dump(out, f, indent=2)
        return str(out_file)

    def visualize_camera(self, cfg: CameraConfig):
        # 简单 3D 可视化：物体 AABB + 相机位置 + 方向
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        obj = self.objects[cfg.object_id]
        bmin,bmax = obj.bbox_min, obj.bbox_max
        corners = np.array([[x,y,z] for x in [bmin[0], bmax[0]]
                                       for y in [bmin[1], bmax[1]]
                                       for z in [bmin[2], bmax[2]]])
        ax.scatter(corners[:,0], corners[:,1], corners[:,2], c='orange', s=20, label='AABB corners')
        ax.scatter([cfg.camera_position[0]],[cfg.camera_position[1]],[cfg.camera_position[2]], c='blue', s=40, label='Camera')
        ax.quiver(cfg.camera_position[0], cfg.camera_position[1], cfg.camera_position[2],
                  cfg.forward[0], cfg.forward[1], cfg.forward[2], length=0.5, color='red', label='Forward')
        ax.set_title(f"Camera view: {cfg.preset} -> {obj.label}")
        ax.legend()
        plt.show()

    def interactive_mode(self):
        print("进入交互模式 (简化版)。输入物体ID或部分标签，空行退出。")
        while True:
            q = input("object id / label > ").strip()
            if not q:
                break
            target = None
            for oid, o in self.objects.items():
                if q == oid or q.lower() in o.label.lower():
                    target = o
                    break
            if not target:
                print("未找到物体。")
                continue
            preset = input(f"preset ({','.join(self.presets.keys())}) > ").strip() or 'front'
            cfg = self.calculate_camera(target.id, preset=preset)
            if cfg:
                print("✅ 成功:", cfg.camera_position)
            else:
                print("❌ 失败: 无法生成符合要求的视角。")


# ------------------------------- CLI入口 ---------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="语义相机生成 (新的向外搜索逻辑)")
    p.add_argument('scene_path', type=str, help='场景路径')
    p.add_argument('--list-objects', action='store_true', help='列出所有物体')
    p.add_argument('--object', dest='obj_id', type=str, default=None, help='目标物体 (id 或标签片段)')
    p.add_argument('--preset', type=str, default='front', help='预设方向')
    p.add_argument('--distance', type=float, default=1.0, help='距离缩放')
    p.add_argument('--height-offset', type=float, default=0.0, help='高度偏移')
    p.add_argument('--all-presets', action='store_true', help='为目标物体生成所有预设')
    p.add_argument('--save-config', action='store_true', help='保存生成的相机配置')
    p.add_argument('-i', '--interactive', action='store_true', help='交互模式')
    p.add_argument('--visualize', action='store_true', help='可视化结果')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cam_sys = SemanticCamera(args.scene_path)

    if args.list_objects:
        print("📋 物体列表:")
        for oid, label in cam_sys.list_objects().items():
            print(f"  {oid}: {label}")
        return

    if args.interactive:
        cam_sys.interactive_mode()
        return

    if not args.obj_id:
        print("未指定 --object，使用 --list-objects 或 -i 进入交互模式。")
        return

    # 查找匹配物体
    target = None
    for oid, o in cam_sys.objects.items():
        if args.obj_id == oid or args.obj_id.lower() in o.label.lower():
            target = o
            break
    if not target:
        print(f"❌ 未找到物体: {args.obj_id}")
        return

    presets = list(cam_sys.presets.keys()) if args.all_presets else [args.preset]
    for pre in presets:
        cfg = cam_sys.calculate_camera(target.id, preset=pre, distance_scale=args.distance, height_offset=args.height_offset)
        if cfg:
            print(f"✅ {pre}: camera={cfg.camera_position}, dist={cfg.distance:.3f}")
            if args.visualize:
                cam_sys.visualize_camera(cfg)
            if args.save_config:
                fname = cam_sys.save_config(cfg)
                print(f"   已保存: {fname}")
        else:
            print(f"❌ {pre}: 失败，未找到符合要求的视角")


if __name__ == '__main__':
    main()