"""
choose_object.py

运行方式示例：
    python Data_generation/active_data/choose_object.py \
        --scenes_root /data/liubinglin/jijiatong/ViewSuite/data/InteriorGS --scene 0267_840790 \
        --num_objects 1 --max_results 50 --out /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/result/0267_1objects.json --debug_log /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/debug.log

参数说明：
    --scenes_root : 场景根目录，包含多个 scene 子文件夹（例如 `data`）
    --scene       : 场景文件夹名（例如 `0002_839955`），脚本将在该文件夹下读取 `labels.json` 和可选的 `structure.json`
    --num_objects : 每个 QA 需要的物体数量（1-5）
    --max_results : 输出的最大结果数
    --out         : 输出 JSON 文件路径（将写入 JSON 数组）
    --min_dist_to_wall : 物体边缘到房间墙体的最小距离阈值（米）

脚本说明：
    1) 从 `<scenes_root>/<scene>/labels.json` 读取物体列表（每个物体含 8 点 bbox）
    2) 对单个物体进行多项过滤（语义黑名单、尺寸阈值、体积、扁平度以及距墙距离）
    3) 对于需要多个物体的 QA，本脚本先要求组内所有物体属于同一房间（由 `structure.json` 中的房间多边形决定），
       然后基于物体的轴对齐包围盒（AABB）计算三维最小距离（非中心点距离），再用动态阈值判断是否可成对/成组。

距离计算细节：
    - 物体间距离采用 AABB 最小欧氏距离（3D）；若两个 AABB 在某一轴上重叠，则该轴的分离量为 0；
      最终距离 = sqrt(dx^2 + dy^2 + dz^2)。
    - 物体到墙的距离采用物体在地面投影（x,y）的 bbox 角点到房间多边形每条边的最短点到线段距离的最小值。

返回值：
    - 当 `--num_objects=1` 时，返回符合条件的单物体列表 [{'id': ..., 'label': ...}]
    - 当 `--num_objects>=2` 时，返回符合条件的物体组合（每组为 list of dict）

备注：此文件包含可复用的房间与几何辅助函数，可提取为公共模块以便其它脚本复用。
"""

import argparse
import json
import numpy as np
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Dict, Any

import math



# Objects to exclude from QA generation
BLACKLIST = {
    # ===== Structural elements =====
    "wall", "floor", "ceiling", "room",

    # ===== Carpet variations =====
    "carpet", "rug",

    # ===== Light fixtures =====
    "chandelier", "ceiling lamp", "spotlight", "lamp", "light",
    "downlights", "wall lamp", "table lamp", "strip light", "track light",
    "linear lamp", "decorative pendant",

    # ===== Generic / unclear categories =====
    "other", "curtain", "bread", "cigar", "wine", "fresh food", "pen",
    "medicine bottle", "toiletries", "chocolate", "paper",

    # ===== Small items that appear in large quantities =====
    "book", "boxed food", "bagged food", "medicine box",
    "vegetable", "fruit", "drinks", "canned food",

    # ===== Added categories =====
    "ice cubes", "cigarette", "straw", "candy", "chopsticks", "spoon", "fork", "knife",
    "lipstick", "nail polish", "hand cream", "cosmetic bottles", "makeup", "jewelry",
    "ring", "earrings", "bracelet", "necklace", "glasses", "watch", "key", "matches",
    "lighter", "candle", "soap", "toothpaste", "brush", "razor", "perfume", "medicines",
    "business card", "envelope", "cd", "dice", "rubiks cube", "toy blocks", "stapler",
    "paper clip", "pushpin", "eraser", "ruler", "pen holder", "tea scoop", "tea caddy",
    "tea clips", "tea needle", "chopstick holder", "cup lid", "bottle opener", "dropper",
    "test tube", "cruet", "spatula", "skimmer", "rolling pin", "peeler", "scissors",
    "clamp", "pliers", "hammer", "screwdriver", "knife sharpener", "track", "power strip",
    "mouse pad", "keyboard tray", "socket", "floor drain", "hanging hanger combination",
    "walnut", "hazelnut", "almond", "pistachio", "pinecone", "stone", "seal",
    "crucible", "stethoscope", "candle snuffer", "candle extinguisher",
    "aromatherapy", "sandalwood", "bow tie", "gloves", "laundry detergent",
    "canned beverage", "delicatessen", "meat product", "doughnut", "dessert",
    "biscuit", "flour", "paint", "washing and care combination",
    "toiletries combination", "cosmetics combination", "tool combination",
    "bottle combination", "cultural items", "couple", "pair", "set", "combination",
    "decorative painting", "wall design", "trappings", "tray", "storage rack",
    "wine glass", "apple", "pear", "peach", "cherry", "strawberry", "banana",
    "lemon", "lime", "orange", "kiwifruit", "mango", "pomegranate", "red jujube",
    "tomato", "cucumber", "potato", "onion", "garlic", "chili", "carrot",
    "chicken leg", "egg", "sushi", "broad bean", "coffee bean"
}

# -------------------- 过滤阈值（几何约束） --------------------
# 单个物体的几何约束（按每个维度/component 检查）
# MIN_DIM_COMPONENT: 任一边长小于此值则过滤（避免太小到无法在图像中辨认）
MIN_DIM_COMPONENT = 0.1   # 每个维度最小值（米），建议 >= 0.05~0.3 视数据集而定
# MAX_DIM_COMPONENT: 任一边长大于此值则过滤（避免占据整个房间或过大）
MAX_DIM_COMPONENT = 3   # 每个维度最大值（米）
# MIN_VOLUME: 体积下限（m^3），避免非常薄或小物体
MIN_VOLUME = 0.1     # 体积最小值（立方米）
# MIN_ASPECT_RATIO: 最短边 / 最长边 的最小比率，用于剔除极度扁平的物体（如海报、地垫）
MIN_ASPECT_RATIO = 0.05 # 最短/最长 边比，过小视为扁平
# MIN_DIST_TO_WALL: 物体最靠近房间边界（墙体）时允许的最小距离
# 计算方式：在地面投影 (x,y) 上，取 bbox 的角点到房间多边形每条边的点到线段距离，取最小值。
MIN_DIST_TO_WALL = 0.0  # 米

# 成对/成组物体的约束（主要用于 num_objects >= 2）
# 说明：脚本已使用 AABB（轴对齐包围盒）在三维空间中计算物体间最小距离，
# 因此下面的 MIN_PAIR_DIST/MAX_PAIR_DIST 为兼容/界限兜底，与动态阈值取更紧凑的。
MIN_PAIR_DIST = 0.1    # 参考最小中心/近邻距离（米）
MAX_PAIR_DIST = 3.0    # 参考最大允许距离（米）
# pair size-difference constraints: 控制尺寸差距，避免大物体与微小物体配对
MAX_PAIR_DIM_RATIO = 2.5  # 两物体最长边比率上限（较大表示允许尺寸差异更大）
MAX_PAIR_DIM_DIFF = 2.0   # 两物体最长边绝对差阈值（米），用于 num_objects==2 的严格控制
# 动态阈值：根据物体平均最大边长自动扩展/收缩配对允许的最小/最大距离
# 最终阈值计算方式：
#   avg_max_dim = mean(object.max_dim for object in group)
#   dyn_min = max(MIN_PAIR_DIST, DYN_MIN_MULT * avg_max_dim)
#   dyn_max = min(MAX_PAIR_DIST, DYN_MAX_MULT * avg_max_dim)
# 这样可以使得较大的物体允许更大的成对距离，较小物体要求更近的配对距离。
DYN_MIN_MULT = 0.2   # 最小乘子（乘以 avg_max_dim）
DYN_MAX_MULT = 1.5   # 最大乘子（乘以 avg_max_dim）


# -------------------- 房间多边形判定 -----------------
def point_in_poly(x: float, y: float, poly: List[List[float]]) -> bool:
    """判断点 (x,y) 是否位于二维多边形 `poly` 内。

    `poly` 为 [[x,y], ...] 的点序列（逆/顺时针均可）。使用经典射线法实现。
    """
    if poly is None or len(poly) < 3:
        return False
    px = [p[0] for p in poly]
    py = [p[1] for p in poly]
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def load_room_polys(scene_path: str) -> List[List[List[float]]]:
    """从场景文件夹中的 `structure.json` 加载房间多边形（floor-plan 轮廓）。

    返回值为多边形列表，每个多边形为 [[x,y], ...]。
    若没有 `structure.json` 或解析失败，则返回空列表（此时房间相关过滤会被跳过）。
    """
    p = Path(scene_path) / 'structure.json'
    if not p.exists():
        return []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rooms = data.get('rooms', [])
        polys: List[List[List[float]]] = []
        for r in rooms:
            profile = r.get('profile')
            if not profile or len(profile) < 3:
                continue
            # profile might be list of {'x':..,'y':..,'z':..} or [x,y,...]
            arr = []
            for pt in profile:
                if isinstance(pt, dict):
                    arr.append([float(pt.get('x', 0.0)), float(pt.get('y', 0.0))])
                else:
                    try:
                        arr.append([float(pt[0]), float(pt[1])])
                    except Exception:
                        pass
            if len(arr) >= 3:
                polys.append(arr)
        return polys
    except Exception:
        return []


# -------------------- 房间与几何辅助函数 --------------------
def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """点 P(px,py) 到线段 AB 的最短距离（二维）。

    计算步骤：将点 P 投影到 AB 对应的直线，计算投影参数 t 并截断到 [0,1]，
    得到线段上的投影点，再返回 P 与投影点之间的欧氏距离。
    """
    # project P onto AB, clamp to segment
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    if c == 0:
        # A==B
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)


def bbox_corners_2d(bbox_points: List[Dict[str, float]]) -> List[Tuple[float, float]]:
    """从 3D bbox 点列表提取地面投影的 (x,y) 角点集合并去重。

    输入通常为 8 个点（bounding box 的顶点），函数返回去重后的 (x,y) 列表。
    若输入格式异常（点数不足），返回空列表。
    """
    if not bbox_points or len(bbox_points) < 4:
        return []
    # Use unique (x,y) pairs
    pts = {(float(p['x']), float(p['y'])) for p in bbox_points}
    return list(pts)


def distance_bbox_to_polygon(bbox_points: List[Dict[str, float]], poly: List[List[float]]) -> float:
    """计算 bbox 在地面投影与房间多边形边界的最近距离（米）。

    方法：对每个 bbox 投影角点，计算到多边形每条边（线段）的点到线段距离，取全局最小值。
    若输入无效则返回大数（1e9）。
    """
    if not bbox_points or not poly:
        return 1e9
    corners = bbox_corners_2d(bbox_points)
    if not corners:
        return 1e9
    min_d = 1e9
    # iterate polygon edges
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        for (px, py) in corners:
            d = point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < min_d:
                min_d = d
    return float(min_d)


def get_room_index_for_point(x: float, y: float, room_polys: List[List[List[float]]]) -> int:
    """Return index of room polygon that contains (x,y) or None."""
    if not room_polys:
        return None
    for ri, poly in enumerate(room_polys):
        if point_in_poly(x, y, poly):
            return ri
    return None


def aabb_min_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    """计算两个轴对齐包围盒（AABB）在三维空间中的最小欧氏距离（米）。

    说明：若两个盒子在某一轴上有重叠，则该轴的分离量为 0；最终距离 = sqrt(dx^2+dy^2+dz^2)。
    返回值为浮点米数，如果盒子接触/重叠则返回 0.0。
    """
    # separation along each axis
    dx = 0.0
    if a_max[0] < b_min[0]:
        dx = b_min[0] - a_max[0]
    elif b_max[0] < a_min[0]:
        dx = a_min[0] - b_max[0]

    dy = 0.0
    if a_max[1] < b_min[1]:
        dy = b_min[1] - a_max[1]
    elif b_max[1] < a_min[1]:
        dy = a_min[1] - b_max[1]

    dz = 0.0
    if a_max[2] < b_min[2]:
        dz = b_min[2] - a_max[2]
    elif b_max[2] < a_min[2]:
        dz = a_min[2] - b_max[2]

    return float(math.sqrt(dx * dx + dy * dy + dz * dz))

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------------

class SceneObject:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("ins_id")
        self.label = data.get("label", "unknown")
        
        # Parse bounding box (list of 8 points)
        bbox_points = data.get("bounding_box", [])
        if not bbox_points:
            self.valid = False
            return

        xs = [p["x"] for p in bbox_points]
        ys = [p["y"] for p in bbox_points]
        zs = [p["z"] for p in bbox_points]

        self.bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
        self.bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
        self.center = (self.bmin + self.bmax) / 2.0
        self.dims = self.bmax - self.bmin
        self.valid = True

    @property
    def max_dim(self) -> float:
        return float(np.max(self.dims))

    @property
    def min_dim(self) -> float:
        return float(np.min(self.dims))

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

# -----------------------------------------------------------------------------
# Main Selection Function
# -----------------------------------------------------------------------------

def select_objects_for_qa(labels_json_path: str,
                         num_objects: int = 1,
                         min_dim_component: float = MIN_DIM_COMPONENT,
                         max_dim_component: float = MAX_DIM_COMPONENT,
                         max_pair_dim_ratio: float = MAX_PAIR_DIM_RATIO,
                         max_pair_dim_diff: float = MAX_PAIR_DIM_DIFF,
                         min_dist_to_wall: float = MIN_DIST_TO_WALL,
                         debug_log: str = None) -> List[Any]:
    """
    Selects suitable objects or pairs of objects for Spatial QA from a scene.

    Args:
        labels_json_path (str): Path to the labels.json file.
        num_objects (int): Number of objects required (1-5).

    Returns:
        List: 
            - If num_objects=1: List of dicts [{'id': '...', 'label': '...'}]
            - If num_objects=2: List of tuples [({'id': '...', 'label': '...'}, {'id': '...', 'label': '...'})]
    """
    path = Path(labels_json_path)
    if not path.exists():
        print(f"Error: File not found: {labels_json_path}")
        return []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return []

    # debug 收集结构
    from collections import Counter, defaultdict
    per_obj_reject = Counter()
    per_obj_reject_labels = defaultdict(Counter)  # reason -> Counter(label)
    group_reject = Counter()
    group_reject_samples = defaultdict(list)

    # 1. Parse and Filter Individual Objects
    candidates = []
    scene_dir = Path(labels_json_path).parent
    # load room polygons for this scene (may be empty)
    room_polys = load_room_polys(str(scene_dir))
    for item in data:
        # Skip non-object entries (some labels.json have metadata at the end)
        if not isinstance(item, dict) or "ins_id" not in item:
            continue

        obj = SceneObject(item)
        if not obj.valid:
            per_obj_reject['invalid_bbox'] += 1
            per_obj_reject_labels['invalid_bbox'][str(item.get('label', ''))] += 1
            continue

        # --- Filter 1: Blacklist ---
        if obj.label.lower() in BLACKLIST:
            per_obj_reject['blacklist'] += 1
            per_obj_reject_labels['blacklist'][obj.label] += 1
            continue

        # --- Filter 2: Size & Volume ---
        # reject if any component dimension too small or too large
        dims = obj.dims
        if float(np.min(dims)) < float(min_dim_component):
            per_obj_reject['too_small_component'] += 1
            per_obj_reject_labels['too_small_component'][obj.label] += 1
            continue
        if float(np.max(dims)) > float(max_dim_component):
            per_obj_reject['too_large_component'] += 1
            per_obj_reject_labels['too_large_component'][obj.label] += 1
            continue
        if obj.volume < MIN_VOLUME:
            per_obj_reject['too_small_volume'] += 1
            per_obj_reject_labels['too_small_volume'][obj.label] += 1
            continue

        # --- Filter 3: Shape (Flatness) ---
        # Avoid objects that are extremely flat relative to their size (e.g. posters, rugs)
        if obj.max_dim > 0:
            aspect_ratio = obj.min_dim / obj.max_dim
            if aspect_ratio < MIN_ASPECT_RATIO:
                per_obj_reject['too_flat'] += 1
                per_obj_reject_labels['too_flat'][obj.label] += 1
                continue

        # determine which room polygon contains the object center (if any)
        obj_room = None
        try:
            cx, cy = float(obj.center[0]), float(obj.center[1])
            obj_room = get_room_index_for_point(cx, cy, room_polys)
        except Exception:
            obj_room = None

        # 如果物体不在任何房间内，则直接剔除（根据需求）
        if obj_room is None:
            per_obj_reject['no_room'] += 1
            per_obj_reject_labels['no_room'][obj.label] += 1
            continue

        # store room index on object for later uniqueness checks
        obj.room_index = obj_room

        # --- Filter 4.5: distance to nearest wall ---
        # if room polygon exists, compute distance from bbox footprint to polygon edges
        if obj_room is not None and min_dist_to_wall is not None:
            try:
                poly = room_polys[obj_room]
                d_wall = distance_bbox_to_polygon(item.get('bounding_box', []), poly)
                # reject objects too close to wall
                if d_wall < float(min_dist_to_wall):
                    per_obj_reject['too_close_to_wall'] += 1
                    per_obj_reject_labels['too_close_to_wall'][obj.label] += 1
                    continue
            except Exception:
                pass

        candidates.append(obj)

    # 2. Return based on requested number of objects
    # --- Enforce label uniqueness rule:
    # keep object if (label count globally == 1) OR (label count within its room == 1)
    # compute label counts (case-insensitive)
    lbls = [getattr(c, 'label', '') or '' for c in candidates]
    lbls_norm = [l.strip().lower() for l in lbls]
    global_counts = Counter(lbls_norm)
    room_counts = defaultdict(Counter)  # room_index -> Counter
    for c, ln in zip(candidates, lbls_norm):
        room_counts[getattr(c, 'room_index', None)][ln] += 1

    filtered = []
    for c, ln in zip(candidates, lbls_norm):
        gcount = global_counts.get(ln, 0)
        rcount = room_counts[getattr(c, 'room_index', None)].get(ln, 0)
        if gcount == 1 or rcount == 1:
            filtered.append(c)

    # 哪些在 candidates 中但被唯一性规则剔除
    cand_ids = {c.id for c in candidates}
    filt_ids = {c.id for c in filtered}
    removed_by_uniqueness = cand_ids - filt_ids
    for c in candidates:
        if c.id in removed_by_uniqueness:
            per_obj_reject['not_unique_in_room'] += 1
            per_obj_reject_labels['not_unique_in_room'][c.label] += 1

    if num_objects == 1:
        results = [{"id": c.id, "label": c.label} for c in filtered]
        # write debug if requested
        if debug_log:
            try:
                dbg = {
                    'per_object_reject_counts': dict(per_obj_reject),
                    'per_object_reject_examples': {k: [x for x, _ in per_obj_reject_labels[k].most_common(10)] for k in per_obj_reject_labels},
                    'group_reject_counts': dict(group_reject),
                    'group_reject_examples': {k: group_reject_samples[k] for k in group_reject_samples},
                    'num_candidates': len(candidates),
                    'num_filtered': len(filtered),
                    'num_results': len(results)
                }
                outp = Path(debug_log)
                outp.parent.mkdir(parents=True, exist_ok=True)
                with open(outp, 'w', encoding='utf-8') as f:
                    json.dump(dbg, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return results

    # handle groups of size 2..5
    elif 2 <= num_objects <= 5:
        groups = []
        for group in combinations(filtered, num_objects):
            # per-group size checks: ratio between largest and smallest longest-dim
            longs = [float(o.max_dim) for o in group]
            max_long = max(longs)
            min_long = min(longs)
            ratio = max_long / (min_long + 1e-12)
            if ratio > float(max_pair_dim_ratio):
                group_reject['size_ratio'] += 1
                if len(group_reject_samples['size_ratio']) < 5:
                    group_reject_samples['size_ratio'].append([o.label for o in group])
                continue

            # for pair (2) we treat max_pair_dim_diff as max allowed absolute dim diff;
            # for groups >2 we treat max_pair_dim_diff as maximum allowed spatial spread
            if num_objects == 2:
                if abs(max_long - min_long) > float(max_pair_dim_diff):
                    group_reject['dim_diff'] += 1
                    if len(group_reject_samples['dim_diff']) < 5:
                        group_reject_samples['dim_diff'].append([o.label for o in group])
                    continue

            # require all group members to be in same room (if rooms available)
            room_idxs = [getattr(o, 'room_index', None) for o in group]
            if any(r is None for r in room_idxs):
                # if any object has no room assignment, skip group (should be rare since we removed such objects earlier)
                group_reject['no_room_in_group'] += 1
                if len(group_reject_samples['no_room_in_group']) < 5:
                    group_reject_samples['no_room_in_group'].append([o.label for o in group])
                continue
            if len(set(room_idxs)) != 1:
                group_reject['different_rooms'] += 1
                if len(group_reject_samples['different_rooms']) < 5:
                    group_reject_samples['different_rooms'].append([o.label for o in group])
                continue
            # (camera position unavailable — camera-room requirement removed)

            # compute pairwise minimum distances based on 3D bbox-to-bbox distances
            n = len(group)
            # build pairwise distance matrix using AABB min-distance
            dists = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    ai_min = group[i].bmin
                    ai_max = group[i].bmax
                    bj_min = group[j].bmin
                    bj_max = group[j].bmax
                    d = aabb_min_distance(ai_min, ai_max, bj_min, bj_max)
                    dists[i, j] = d
                    dists[j, i] = d
            # (ignore diagonal zeros by adding large to diag)
            if n > 1:
                diag_inf = np.eye(n, dtype=float) * 1e9
                min_pair = float(np.min(dists + diag_inf))
            else:
                min_pair = float(0.0)

            # dynamic thresholds based on average object size
            avg_max_dim = float(np.mean([o.max_dim for o in group]))
            dyn_min = max(float(MIN_PAIR_DIST), DYN_MIN_MULT * avg_max_dim)
            dyn_max = min(float(MAX_PAIR_DIST), DYN_MAX_MULT * avg_max_dim)

            if min_pair < dyn_min:
                group_reject['too_close'] += 1
                if len(group_reject_samples['too_close']) < 5:
                    group_reject_samples['too_close'].append([o.label for o in group])
                continue

            max_pair = float(np.max(dists))
            if num_objects == 2:
                if not (dyn_min <= max_pair <= dyn_max):
                    group_reject['out_of_dyn_range'] += 1
                    if len(group_reject_samples['out_of_dyn_range']) < 5:
                        group_reject_samples['out_of_dyn_range'].append([o.label for o in group])
                    continue
            else:
                # for groups >2, use provided max_pair_dim_diff as spread limit
                if max_pair > float(max_pair_dim_diff):
                    group_reject['spread_too_large'] += 1
                    if len(group_reject_samples['spread_too_large']) < 5:
                        group_reject_samples['spread_too_large'].append([o.label for o in group])
                    continue

            # group passed checks -> append as list of simple dicts
            groups.append([{"id": o.id, "label": o.label} for o in group])

        results = groups
        # write debug if requested
        if debug_log:
            try:
                dbg = {
                    'per_object_reject_counts': dict(per_obj_reject),
                    'per_object_reject_examples': {k: [x for x, _ in per_obj_reject_labels[k].most_common(10)] for k in per_obj_reject_labels},
                    'group_reject_counts': dict(group_reject),
                    'group_reject_examples': {k: group_reject_samples[k] for k in group_reject_samples},
                    'num_candidates': len(candidates),
                    'num_filtered': len(filtered),
                    'num_groups_considered': sum(1 for _ in combinations(filtered, num_objects)),
                    'num_groups_returned': len(results)
                }
                outp = Path(debug_log)
                outp.parent.mkdir(parents=True, exist_ok=True)
                with open(outp, 'w', encoding='utf-8') as f:
                    json.dump(dbg, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return results

    else:
        print(f"Warning: Logic for {num_objects} objects not implemented, returning empty list.")
        return []

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
def _write_json(out_path: Path, data: List[Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description='Choose objects for spatial QA from a scene')
    p.add_argument('--scenes_root', required=True, help='Root folder containing scene subfolders')
    p.add_argument('--scene', required=True, help='Scene folder name (e.g. 0002_839955)')
    p.add_argument('--num_objects', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Objects per QA (1-5)')
    p.add_argument('--max_results', type=int, default=100, help='Maximum results to return')
    p.add_argument('--out', required=True, help='Output JSON file path')
    # optional size params
    p.add_argument('--min_comp_dim', type=float, default=MIN_DIM_COMPONENT, help='Minimum per-dimension size (m)')
    p.add_argument('--max_comp_dim', type=float, default=MAX_DIM_COMPONENT, help='Maximum per-dimension size (m)')
    p.add_argument('--max_pair_ratio', type=float, default=MAX_PAIR_DIM_RATIO, help='Max allowed ratio between object longest dims')
    p.add_argument('--max_pair_diff', type=float, default=MAX_PAIR_DIM_DIFF, help='Max allowed absolute difference between longest dims (m)')
    p.add_argument('--min_dist_to_wall', type=float, default=MIN_DIST_TO_WALL, help='物体边缘到房间墙体的最小距离阈值（米）')
    p.add_argument('--debug_log', type=str, default=None, help='Debug JSON output path (default: <scene>/choose_object_debug.json)')
    args = p.parse_args()

    labels_path = Path(args.scenes_root) / args.scene / 'labels.json'
    if not labels_path.exists():
        print(f"Error: labels.json not found for scene: {labels_path}")
        return

    # determine debug log default path if not provided
    debug_log_path = args.debug_log
    if debug_log_path is None:
        debug_log_path = str(Path(args.scenes_root) / args.scene / 'choose_object_debug.json')

    results = select_objects_for_qa(
        str(labels_path),
        num_objects=args.num_objects,
        min_dim_component=args.min_comp_dim,
        max_dim_component=args.max_comp_dim,
        max_pair_dim_ratio=args.max_pair_ratio,
        max_pair_dim_diff=args.max_pair_diff,
        min_dist_to_wall=args.min_dist_to_wall,
        debug_log=debug_log_path,
    )
    # limit results
    limited = results[: args.max_results]

    _write_json(Path(args.out), limited)
    print(f"Wrote {len(limited)} results to {args.out}")


if __name__ == '__main__':
    main()