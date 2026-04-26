#!/usr/bin/env python3
"""
Batch QA generator: enumerate scene presets, collect visible objects per view,
and emit QA items for every meaningful combination (single objects, pairs, counts).

This module implements the "batch generation" pose-selection strategy you asked for:
- For each preset/view, find all objects that pass the 2D-corner visibility + occlusion checks.
- For each visible single object: emit 'nearest_object_mca' style items.
- For each visible pair: emit 'object_object_distance_mca'.
- For categories with multiple visible instances in the same view: emit 'object_count_mca'.
"""

from __future__ import annotations
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import math
import imageio
from .camera_generation import SemanticCamera
from .preview import compose_preview_for_item, render_thumbnail_for_pose
from ..utils.occlusion import (
    camtoworld_from_pos_target,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    is_occluded_by_any,
    is_box_occluded_by_any,
    ensure_visibility,
    is_target_in_fov,
    occluded_area_on_image,
    world_to_camera,
    aabb_corners,
    project_point,
    point_in_image,
)
# Support running as package (Data_generation.bench_generation) or as local module (-m bench_generation)
try:
    from ..motion.view_manipulator import ViewManipulator  # type: ignore
except Exception:  # pragma: no cover - fallback for local run
    from motion.view_manipulator import ViewManipulator  # type: ignore

WIDTH = 400
HEIGHT = 400


def create_intrinsics(width=WIDTH, height=HEIGHT):
    focal = float(width * 0.4)
    K = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    return {'width': int(width), 'height': int(height), 'focal': float(focal), 'K': K}


def load_structure_height_bounds(scene_path: str) -> Tuple[float | None, float | None]:
    """Read vertical bounds from scene occupancy file.

    This function now only reads `<scene>/occupancy.json` and returns
    the z components of the lower/upper (or min/max) bounds when present.
    Returns (min_h, max_h) or (None, None) if occupancy info not available.
    """
    try:
        p = Path(scene_path) / 'occupancy.json'
        if not p.exists():
            return None, None
        with open(p, 'r', encoding='utf-8') as f:
            occ = json.load(f)
        # prefer explicit lower/upper, fallback to min/max
        lower = occ.get('lower') or occ.get('min')
        upper = occ.get('upper') or occ.get('max')
        if isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple)) and len(lower) >= 3 and len(upper) >= 3:
            return float(lower[2]), float(upper[2])
    except Exception:
        pass
    return None, None


# When True, require sampled camera positions to lie inside one of the room
# polygons defined in <scene>/structure.json. This is set from the CLI
REQUIRE_IN_ROOM = True



def load_room_polys(scene_path: str) -> List[np.ndarray]:
    """Load room polygons from <scene>/structure.json. Returns list of Nx2 numpy arrays."""
    p = Path(scene_path) / 'structure.json'
    if not p.exists():
        return []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rooms = data.get('rooms', [])
        polys = []
        for r in rooms:
            profile = r.get('profile')
            if not profile or len(profile) < 3:
                continue
            arr = np.array(profile, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            polys.append(arr[:, :2])
        return polys
    except Exception:
        return []


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    """Point-in-polygon test (ray casting). poly: Nx2 array."""
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


def is_pos_inside_any_room(scene_path: str, pos: np.ndarray) -> bool:
    """Return True when XY of pos is strictly inside any room polygon."""
    polys = load_room_polys(scene_path)
    if not polys:
        return False
    x, y = float(pos[0]), float(pos[1])
    for poly in polys:
        if point_in_poly(x, y, poly):
            print('pos inside room poly')
            return True
        else :
            print('pos outside room poly')
    return False


def ensure_position_legal(scene_path: str, pos: np.ndarray, min_h: float | None = None, max_h: float | None = None, tol: float = 0.5) -> bool:
    """Return True when pos is legal: not inside any AABB (with tol) and within min/max heights if provided."""
    # inside-object check (reuse is_pos_inside_scene)
    if is_pos_inside_scene(scene_path, pos, tol=tol):
        print("Position inside object AABB")
        return False
    z = float(pos[2])
    if (min_h is not None) and (z < float(min_h) + 0.1):
        print("Position below min height")
        return False
    if (max_h is not None) and (z > float(max_h) - 0.1):
        print("Position above max height")
        return False
    # if the generator requested positions be inside a room, enforce that
    if not is_pos_inside_any_room(scene_path, pos):
        print("Position not inside any room polygon (rejected by --require-in-room)")
        return False
    print("Position legal")
    return True

# Blacklist of objects to exclude
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


def is_pos_inside_scene(scene_path: str, pos: np.ndarray, tol: float = 0.01) -> bool:
    """Return True if pos lies within any scene AABB expanded by tol."""
    try:
        aabbs = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    except Exception:
        return False
    q = np.array(pos, dtype=float)
    for b in aabbs:
        if (q[0] >= b.bmin[0] - tol and q[0] <= b.bmax[0] + tol and
            q[1] >= b.bmin[1] - tol and q[1] <= b.bmax[1] + tol and
            q[2] >= b.bmin[2] - tol and q[2] <= b.bmax[2] + tol):
            return True
    return False

def count_visible_corners_for_box(pos: np.ndarray, tgt: np.ndarray, K: np.ndarray, target_bmin: np.ndarray, target_bmax: np.ndarray, width: int, height: int, corner_threshold: int = 4, aabbs_all: List[Any] | None = None, target_id: str | None = None, require_unoccluded: bool = False) -> int:
    """Project the 8 corners of an AABB into the image and return how many corners
    are visible (in front of camera and inside image bounds).

    By default this counts geometric projections only (z>0 and inside image).
    If `require_unoccluded=True` and `aabbs_all` is provided, each corner is
    additionally tested with `is_occluded_by_any` and only counted when not occluded.

    Returns the integer count of visible corners (0..8).
    """
    c2w = camtoworld_from_pos_target(pos, tgt)
    view = np.linalg.inv(np.array(c2w, dtype=float))
    corners = aabb_corners(target_bmin, target_bmax)
    pcs = np.array([world_to_camera(view, c) for c in corners])
    uvz = [project_point(np.array(K, dtype=float), pc) for pc in pcs]
    cnt = 0
    for idx, (u, v, z) in enumerate(uvz):
        if z > 1e-6 and point_in_image(u, v, width, height, border=2):
            if require_unoccluded and aabbs_all is not None:
                # get the world-space corner coordinate
                corner_world = corners[idx]
                try:
                    if is_occluded_by_any(pos, corner_world, aabbs_all, target_id=target_id):
                        continue
                except Exception:
                    # If occlusion helper fails, fall back to counting projection only
                    pass
            cnt += 1
    return cnt

# Note: the previous helper `find_views_and_visible_objects` (which returned a
# list of sampled views) was intentionally removed because the main generation
# loop exhaustively iterates object × preset and performs the same visibility
# and legality checks inline. Keeping that helper here was redundant and it was
# not referenced by `main()`; removing it reduces dead code and confusion.

def simple_meters_to_choices(m: float, rng: random.Random | None = None) -> Tuple[List[str], str]:
    """Small utility: produce 3 choices and a correct letter for a meter value."""
    _rng = rng or random
    correct = round(float(m), 2)
    
    # Generate a set of delta values to create choices around the correct answer
    candidates = [round(correct + d, 2) for d in (0.0, 0.25, -0.25, 0.5, -0.5)]
    
    # Ensure that the candidates are unique
    seen = []
    for v in candidates:
        if v not in seen:
            seen.append(v)
        if len(seen) >= 3:
            break
    
    # Add the correct answer if not already in the candidates
    if correct not in seen:
        seen[-1] = correct
    
    # Shuffle the choices
    _rng.shuffle(seen)
    
    # Map the choices to labels A, B, C
    labels = ['A', 'B', 'C']
    formatted = [f"{v}m" for v in seen]
    correct_label = labels[formatted.index(f"{correct}m")]
    
    return formatted, correct_label


def meters_to_choices(m: float, rng: random.Random | None = None) -> Tuple[List[str], str]:
    """More featureful choice generator used by qa_generator: returns 3-4 choices and correct label."""
    _rng = rng or random
    def fmt_val(x: float) -> str:
        xv = round(float(x), 2)
        if abs(xv) < 1e-9:
            xv = 0.0
        s = ("%g" % xv)
        return f"{s}m"

    correct = round(float(m), 2)
    base_deltas = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0]
    numeric: List[float] = []
    for d in base_deltas:
        v = round(correct + d, 2)
        if m >= 0:
            v = max(0.0, v)
        numeric.append(v)
    numeric.append(correct)
    uniq_numeric: List[float] = []
    seen_nums = set()
    for v in numeric:
        if v not in seen_nums:
            uniq_numeric.append(v)
            seen_nums.add(v)
    _rng.shuffle(uniq_numeric)
    if correct not in uniq_numeric:
        uniq_numeric.append(correct)
    formatted: List[str] = []
    formatted_to_num: Dict[str, float] = {}
    for v in uniq_numeric:
        s = fmt_val(v)
        if s in formatted_to_num:
            continue
        formatted.append(s)
        formatted_to_num[s] = v
        if len(formatted) >= 4:
            break
    i = 0
    while len(formatted) < 3 and i < len(base_deltas):
        v = round(correct + base_deltas[i], 2)
        if m >= 0:
            v = max(0.0, v)
        s = fmt_val(v)
        if s not in formatted_to_num:
            formatted.append(s)
            formatted_to_num[s] = v
        i += 1
    correct_str = fmt_val(correct)
    if correct_str not in formatted_to_num:
        if len(formatted) >= 4:
            formatted[-1] = correct_str
        else:
            formatted.append(correct_str)
        formatted_to_num[correct_str] = correct
    _rng.shuffle(formatted)
    labels = ['A', 'B', 'C', 'D'][:len(formatted)]
    correct_label = labels[formatted.index(correct_str)]
    return formatted, correct_label


def setup_camtoworld(position: np.ndarray, target: np.ndarray, up_vec: np.ndarray = None):
    pos = np.array(position, dtype=float)
    tgt = np.array(target, dtype=float)
    forward = tgt - pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if up_vec is None:
        world_up = np.array([0.0, 0.0, 1.0])
    else:
        world_up = np.array(up_vec, dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) > 1e-6:
        right = right / np.linalg.norm(right)
    else:
        right = np.array([1.0, 0.0, 0.0])
    up_corrected = np.cross(right, forward)
    camtoworld = np.eye(4, dtype=float)
    camtoworld[:3, 0] = -right
    camtoworld[:3, 1] = up_corrected
    camtoworld[:3, 2] = forward
    camtoworld[:3, 3] = pos
    return camtoworld.tolist()


def _push_camera_from_point(pos: np.ndarray, tgt: np.ndarray, point: np.ndarray, min_dist: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array(pos, dtype=float)
    t = np.array(tgt, dtype=float)
    forward = t - p
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / n
    d = float(np.linalg.norm(p - point))
    if not np.isfinite(d):
        return p, t
    if d < min_dist:
        delta = (min_dist - d) + 1e-3
        p = p - forward * delta
        t = p + forward
    return p, t


def _ensure_camera_not_inside(scene_path: str, pos: np.ndarray, tgt: np.ndarray, max_iter: int = 8, step: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    aabbs = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    p = np.array(pos, dtype=float).copy()
    t = np.array(tgt, dtype=float).copy()

    # occupancy clamp
    occ_min, occ_max, _ = None, None, None
    try:
        occ_path = Path(scene_path) / 'occupancy.json'
        if occ_path.exists():
            with open(occ_path, 'r') as f:
                occ = json.load(f)
            occ_min = occ.get('min', occ.get('lower'))
            occ_max = occ.get('max', occ.get('upper'))
    except Exception:
        occ_min = occ_max = None
    if occ_min is not None and occ_max is not None:
        p = np.minimum(np.maximum(p, np.array(occ_min, dtype=float)), np.array(occ_max, dtype=float))
        t = np.minimum(np.maximum(t, np.array(occ_min, dtype=float)), np.array(occ_max, dtype=float))

    forward = t - p
    n = np.linalg.norm(forward)
    forward = np.array([0.0, 0.0, 1.0]) if n < 1e-8 else (forward / n)

    def inside_any(q: np.ndarray) -> bool:
        tol = float(0.5)
        for b in aabbs:
            if (q[0] >= b.bmin[0] - tol and q[0] <= b.bmax[0] + tol and
                q[1] >= b.bmin[1] - tol and q[1] <= b.bmax[1] + tol and
                q[2] >= b.bmin[2] - tol and q[2] <= b.bmax[2] + tol):
                return True
        return False

    def clamp_occ(q: np.ndarray) -> np.ndarray:
        if occ_min is None or occ_max is None:
            return q
        return np.minimum(np.maximum(q, np.array(occ_min, dtype=float)), np.array(occ_max, dtype=float))

    if inside_any(p):
        for d in [ -forward, +forward, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]) ]:
            q = p.copy()
            for _ in range(max_iter):
                if not inside_any(q):
                    p = clamp_occ(q)
                    t = p + forward
                    break
                q = q + step * d
            if not inside_any(p):
                break

    clear_dist = 0.2
    occluders_front = []
    for b in aabbs:
        c = 0.5 * (b.bmin + b.bmax)
        v = c - p
        dv = float(np.linalg.norm(v))
        if dv < 1e-8:
            continue
        if dv <= 1.2 and float(np.dot(v / dv, forward)) > 0.1:
            occluders_front.append(b)

    def front_blocked(q: np.ndarray) -> bool:
        return is_occluded_by_any(q, q + forward * clear_dist, occluders_front, target_id=None)

    if front_blocked(p):
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        offsets = [
            -forward,
            +right,
            -right,
            +up,
            -up,
            +2*right,
            -2*right,
            +2*up,
            -2*up,
        ]
        for k in range(min(len(offsets), max_iter)):
            q = p + offsets[k] * step
            q = clamp_occ(q)
            if inside_any(q):
                continue
            if not front_blocked(q):
                p = q
                t = p + forward
                break

    if np.linalg.norm(t - p) < 0.05:
        p = clamp_occ(p - forward * max(0.1, step))
        t = p + forward

    p = clamp_occ(p)
    t = p + forward
    return p, t


def is_aabb_occluded(
    pos: np.ndarray,
    aabb,
    aabbs_all: List[Any],
    K: np.ndarray | None = None,
    camtoworld: np.ndarray | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    area_threshold: float = 0.3,
) -> bool:
    """
    Determine if an AABB should be considered occluded from the current camera.

    Preferred method: use image-space occlusion area. If occluded area ratio on the
    target is >= area_threshold, we consider it occluded; otherwise it's allowed.

    Fallback: if K/camtoworld are not provided, fall back to ray/AABB test
    (is_box_occluded_by_any), preserving previous behavior.
    """
    # Prefer area-based occlusion if full camera params are available
    if (K is not None) and (camtoworld is not None):
        try:
            res = occluded_area_on_image(
                ray_o=np.asarray(pos, dtype=float),
                target_bmin=aabb.bmin,
                target_bmax=aabb.bmax,
                aabbs=aabbs_all,
                K=np.asarray(K, dtype=float),
                camtoworld=np.asarray(camtoworld, dtype=float),
                width=int(width),
                height=int(height),
                target_id=getattr(aabb, 'id', None),
            )
            ratio = float(res.get('occlusion_ratio_target', 0.0))
            return ratio >= float(area_threshold)
        except Exception:
            # On any failure, fall back to geometric check
            pass

    # Fallback: geometric occlusion using rays (previous behavior)
    return is_box_occluded_by_any(pos, aabb.bmin, aabb.bmax, aabbs_all, target_id=getattr(aabb, 'id', None))


def generate_distance_mca_item(scene_path: str, output_dir: str, seed: int | None = None, rng: random.Random | None = None) -> Dict[str, Any]:
    _rng = rng or random
    # Fallback-only implementation (no TVApproach used here to keep dependencies light)
    sc = SemanticCamera(scene_path)
    aabbs_objs = load_scene_aabbs(scene_path)
    if not aabbs_objs:
        raise RuntimeError('No objects for distance_mca')
    obj_ids = list(sc.list_objects().keys()) if hasattr(sc, 'list_objects') else [b.id for b in aabbs_objs]
    preset_names = list(sc.presets.keys())
    if not preset_names:
        raise RuntimeError('No presets for distance_mca')
    p1 = preset_names[_rng.getrandbits(32) % len(preset_names)]
    p2 = preset_names[(_rng.getrandbits(32) + 1) % len(preset_names)]
    if obj_ids:
        anchor_id = _rng.choice(obj_ids)
    else:
        anchor_id = aabbs_objs[0].id
    cfg1 = sc.calculate_camera(anchor_id, preset=p1) if hasattr(sc, 'calculate_camera') else None
    cfg2 = sc.calculate_camera(anchor_id, preset=p2) if hasattr(sc, 'calculate_camera') else None
    if cfg1 is None or cfg2 is None:
        raise RuntimeError('Failed to compute presets for distance_mca')
    s_pos = np.array(cfg1.camera_position, dtype=float)
    e_pos = np.array(cfg2.camera_position, dtype=float)
    if is_pos_inside_scene(scene_path, s_pos) or is_pos_inside_scene(scene_path, e_pos):
        raise RuntimeError('Failed to compute fallback camera configs (preset places camera inside)')
    N = 12
    path = []
    for i in range(N):
        t = i / (N - 1)
        pos = (1 - t) * s_pos + t * e_pos
        tgt = (1 - t) * np.array(cfg1.target_position) + t * np.array(cfg2.target_position)
        forward = tgt - pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1.0, 0.0, 0.0])
        up = np.cross(right, forward)
        path.append({'frame': i, 'position': pos.copy(), 'target': tgt.copy(), 'forward': forward.copy(), 'right': right.copy(), 'up': up.copy()})
    if len(path) < 4:
        idx = 0
    else:
        idx = _rng.randint(0, max(0, len(path)-3))
    current = path[idx]
    actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
    action = _rng.choice(actions)
    step_t = 0.3
    step_r_deg = 30.0
    curr_pos = np.array(current['position'], dtype=float)
    curr_tgt = np.array(current['target'], dtype=float)
    curr_pos, curr_tgt = _ensure_camera_not_inside(scene_path, curr_pos, curr_tgt)
    c2w_current = np.array(setup_camtoworld(curr_pos, curr_tgt), dtype=float)
    manip = ViewManipulator(step_translation=step_t, step_rotation_deg=step_r_deg, world_up_axis="Z", image_y_down=True)
    manip.reset(initial_extrinsic_c2w=c2w_current)
    if action == 'move_forward':
        manip.move_forward(+step_t)
    elif action == 'move_backward':
        manip.move_forward(-step_t)
    elif action == 'turn_left':
        manip.yaw_camera(-manip.step_r)
    elif action == 'turn_right':
        manip.yaw_camera(+manip.step_r)
    c2w_true = manip.get_pose(mode="c2w")
    pos_true = c2w_true[:3, 3]
    fwd_true = c2w_true[:3, :3] @ np.array([0.0, 0.0, 1.0])
    tgt_true = pos_true + fwd_true

    def make_pose_from_c2w(M: np.ndarray) -> Dict[str, Any]:
        p = M[:3, 3]
        f = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return {'frame': -1, 'position': p.copy(), 'target': (p + f).copy(), 'forward': f.copy(), 'right': (M[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy(), 'up': (M[:3, :3] @ np.array([0.0, 1.0, 0.0])).copy()}

    def apply_action_from_pose(c2w_pose: np.ndarray, act: str) -> np.ndarray:
        m = ViewManipulator(step_translation=step_t, step_rotation_deg=step_r_deg, world_up_axis="Z", image_y_down=True)
        m.reset(initial_extrinsic_c2w=c2w_pose)
        if act == 'move_forward':
            m.move_forward(+step_t)
        elif act == 'move_backward':
            m.move_forward(-step_t)
        elif act == 'turn_left':
            m.yaw_camera(-m.step_r)
        elif act == 'turn_right':
            m.yaw_camera(+m.step_r)
        return m.get_pose(mode='c2w')

    other_actions = [a for a in actions if a != action]
    _rng.shuffle(other_actions)
    dact1, dact2 = other_actions[:2]
    c2w_d1 = apply_action_from_pose(c2w_current, dact1)
    c2w_d2 = apply_action_from_pose(c2w_current, dact2)
    true_next = make_pose_from_c2w(c2w_true)
    d1 = make_pose_from_c2w(c2w_d1)
    d2 = make_pose_from_c2w(c2w_d2)
    candidates = [true_next, d1, d2]
    adjusted_candidates = []
    for c in candidates:
        pos = np.array(c['position'], dtype=float)
        tgt = np.array(c['target'], dtype=float)
        pos_adj, tgt_adj = _ensure_camera_not_inside(scene_path, pos, tgt)
        forward = tgt_adj - pos_adj
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1.0, 0.0, 0.0])
        up = np.cross(right, forward)
        cc = dict(c)
        cc['position'] = pos_adj
        cc['target'] = tgt_adj
        cc['forward'] = forward
        cc['right'] = right
        cc['up'] = up
        cc['visibility'] = {'checked': False}
        adjusted_candidates.append(cc)

    labels = ['A', 'B', 'C']
    order = list(range(3))
    _rng.shuffle(order)
    choices_frames = [adjusted_candidates[i] for i in order]
    correct_idx = order.index(0)
    correct_label = labels[correct_idx]

    def create_intrinsics_for_pose(cfg_pos, cfg_target, width, height):
        intr = create_intrinsics(width, height)
        intr['camtoworld'] = setup_camtoworld(np.array(cfg_pos), np.array(cfg_target))
        return intr

    augmented_choices = []
    for c in choices_frames:
        rc = dict(c)
        rc['render'] = create_intrinsics_for_pose(c['position'], c['target'])
        augmented_choices.append(rc)

    current_pos = curr_pos.tolist()
    current_tgt = curr_tgt.tolist()
    cf = np.array(current['target'], dtype=float) - np.array(current['position'], dtype=float)
    cf = cf / (np.linalg.norm(cf) + 1e-8)
    cr = np.cross(cf, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(cr) > 1e-6:
        cr = cr / np.linalg.norm(cr)
    else:
        cr = np.array([1.0, 0.0, 0.0])
    cu = np.cross(cr, cf)
    current_up = cu.tolist()

    item = {
        'qtype': 'action_next_frame_mca',
        'scene': str(scene_path),
        'seed': seed,
        'question': f"Given the current view and the action '{action}', which candidate view is the next most likely?",
        'choices': [
            {'frame': int(f['frame']), 'position': f['position'].tolist(), 'target': f['target'].tolist(), 'render': f['render']} for f in augmented_choices
        ],
        'answer': correct_label,
        'meta': {
            'current_frame': int(current['frame']),
            'action': action,
            'true_next_frame': -1,
            'current_pos': current_pos,
            'current_target': current_tgt,
            'current_up': current_up,
            'current_render': create_intrinsics_for_pose(current_pos, current_tgt)
        }
    }
    item['meta']['manipulator'] = {'applied': True, 'step_translation': step_t, 'step_rotation_deg': step_r_deg}
    return item


def generate_distance_mca_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None) -> Dict[str, Any] | None:
    """Generate a distance-style question based on the current view.
    This asks for the approximate distance (meters) between the camera and a visible object's center.
    Returns None if no suitable visible object is found.
    """
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        return None
    # Choose a visible object that is not blacklisted and not occluded
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    c2w = camtoworld_from_pos_target(pos, tgt)
    candidates = []
    for b in visible:
        if b.label in BLACKLIST:
            print(f"Skipping blacklisted object {b.label}")
            # continue
        if is_aabb_occluded(pos, b, aabbs_all, K=K, camtoworld=c2w, width=width, height=height):
            print(f"Skipping occluded object {b.label}")
            continue
        # ensure center is in fov
        if not is_target_in_fov(K, c2w, b.bmin, b.bmax, width, height, require_center=True):
            print(f"Skipping object {b.label} not in fov")
            # continue
        candidates.append(b)
    if not candidates:
        return None
    b = _rng.choice(candidates)
    center = 0.5 * (b.bmin + b.bmax)
    d = float(np.linalg.norm(pos - center))
    choices, correct = meters_to_choices(d, rng=_rng)
    item = {
        'qtype': 'distance_mca',
        'scene': str(scene_path),
        'seed': seed,
        'question': f'Approximately how far is the {b.label} from the camera?',
        'choices': choices,
        'answer': correct,
        'meta': {
            'object_id': b.id,
            'object_label': b.label,
            'distance_m': d,
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist()
        }
    }
    return item


def generate_action_next_frame_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None) -> Dict[str, Any] | None:
    """Generate action_next_frame_mca using the current view as the current pose.
    Returns None if generation fails (e.g., manipulator errors).
    """
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    # prepare current pose
    curr_pos = pos.copy()
    curr_tgt = tgt.copy()
    curr_pos, curr_tgt = _ensure_camera_not_inside(scene_path, curr_pos, curr_tgt)
    c2w_current = np.array(setup_camtoworld(curr_pos, curr_tgt), dtype=float)
    step_t = 0.3
    step_r_deg = 30.0
    manip = ViewManipulator(step_translation=step_t, step_rotation_deg=step_r_deg, world_up_axis="Z", image_y_down=True)
    manip.reset(initial_extrinsic_c2w=c2w_current)


    actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
    action = _rng.choice(actions)
    if action == 'move_forward':
        manip.move_forward(+step_t)
    elif action == 'move_backward':
        manip.move_forward(-step_t)
    elif action == 'turn_left':
        manip.yaw_camera(-manip.step_r)
    elif action == 'turn_right':
        manip.yaw_camera(+manip.step_r)
    c2w_true = manip.get_pose(mode="c2w")

    def make_pose_from_c2w(M: np.ndarray) -> Dict[str, Any]:
        p = M[:3, 3]
        f = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return {'frame': -1, 'position': p.copy(), 'target': (p + f).copy()}

    def apply_action_from_pose(c2w_pose: np.ndarray, act: str) -> np.ndarray:
        m = ViewManipulator(step_translation=step_t, step_rotation_deg=step_r_deg, world_up_axis="Z", image_y_down=True)
        m.reset(initial_extrinsic_c2w=c2w_pose)
        if act == 'move_forward':
            m.move_forward(+step_t)
        elif act == 'move_backward':
            m.move_forward(-step_t)
        elif act == 'turn_left':
            m.yaw_camera(-m.step_r)
        elif act == 'turn_right':
            m.yaw_camera(+m.step_r)
        return m.get_pose(mode='c2w')

    c2w_diffs = []
    other_actions = [a for a in actions if a != action]
    _rng.shuffle(other_actions)
    for a in other_actions[:2]:
        c2w_diffs.append(apply_action_from_pose(np.array(setup_camtoworld(curr_pos, curr_tgt), dtype=float), a))

    true_next = make_pose_from_c2w(c2w_true)
    d1 = make_pose_from_c2w(c2w_diffs[0])
    d2 = make_pose_from_c2w(c2w_diffs[1])
    candidates = [true_next, d1, d2]
    adjusted = []
    for c in candidates:
        pos_c = np.array(c['position'], dtype=float)
        tgt_c = np.array(c['target'], dtype=float)
        pos_adj, tgt_adj = _ensure_camera_not_inside(scene_path, pos_c, tgt_c)
        rc = dict(c)
        rc['position'] = pos_adj
        rc['target'] = tgt_adj
        adjusted.append(rc)

    labels = ['A', 'B', 'C']
    order = list(range(3))
    _rng.shuffle(order)
    choices_frames = [adjusted[i] for i in order]
    correct_idx = order.index(0)
    correct_label = labels[correct_idx]

    def create_intrinsics_for_pose(cfg_pos, cfg_target, width, height):
        intr = create_intrinsics(width, height)
        intr['camtoworld'] = setup_camtoworld(np.array(cfg_pos), np.array(cfg_target))
        return intr

    augmented_choices = []
    for c in choices_frames:
        rc = dict(c)
        rc['render'] = create_intrinsics_for_pose(c['position'], c['target'])
        augmented_choices.append(rc)

    item = {
        'qtype': 'action_next_frame_mca',
        'scene': str(scene_path),
        'seed': seed,
        'question': f"Given the current view and the action '{action}', which candidate view is the next most likely?",
        'choices': [
            {'position': f['position'].tolist(), 'target': f['target'].tolist(), 'render': f['render']} for f in augmented_choices
        ],
        'answer': correct_label,
        'meta': {
            'action': action,
            'current_pos': curr_pos.tolist(),
            'current_target': curr_tgt.tolist(),
            'current_render': create_intrinsics_for_pose(curr_pos, curr_tgt)
        }
    }
    item['meta']['manipulator'] = {'applied': True, 'step_translation': step_t, 'step_rotation_deg': step_r_deg}
    return item



def generate_object_size_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None) -> Dict[str, Any] | None:
    """Generate object_size_mca based on the current view's visible objects.
    Uses the current view camera (with small push if needed). Returns None if no suitable object.
    """
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        return None
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    # Prefer larger visible objects (so measurement is meaningful)
    c2w = camtoworld_from_pos_target(pos, tgt)
    candidates = [
        b for b in visible
        if b.label not in BLACKLIST and not is_aabb_occluded(pos, b, aabbs_all, K=K, camtoworld=c2w, width=width, height=height)
    ]
    if not candidates:
        return None
    # sort by longest dimension descending
    candidates.sort(key=lambda b: float(np.max(b.bmax - b.bmin)), reverse=True)
    # Try each candidate and ensure center in fov
    chosen = None
    cam_pos = None
    cam_tgt = None
    for b in candidates:
        pos_try, tgt_try = _push_camera_from_point(pos, tgt, 0.5*(b.bmin + b.bmax), min_dist=0.0)
        # ensure camera not inside after small push
        if is_pos_inside_scene(scene_path, pos_try):
            pos_try, tgt_try = _ensure_camera_not_inside(scene_path, pos_try, tgt_try)
        # require that the object's AABB projects fully into the image (all 8 corners)
            corners_visible = count_visible_corners_for_box(pos_try, tgt_try, K, b.bmin, b.bmax, width, height, corner_threshold=8, aabbs_all=aabbs_all)
        if corners_visible < 4:
            # skip objects that are not fully inside the view
            continue
        # also ensure the box is not occluded by other scene geometry (area-based)
        c2w_try = camtoworld_from_pos_target(pos_try, tgt_try)
        if is_aabb_occluded(pos_try, b, aabbs_all, K=K, camtoworld=c2w_try, width=width, height=height):
            continue

        chosen = b
        cam_pos, cam_tgt = pos_try, tgt_try
        break
    if chosen is None:
        return None
    b = chosen
    dims = (b.bmax - b.bmin)
    longest = float(np.max(dims)) * 100.0
    choices, correct = meters_to_choices(longest / 100.0, rng=_rng)
    choices_cm = [str(int(round(float(c[:-1]) * 100))) for c in choices]
    render = None
    if cam_pos is not None and cam_tgt is not None:
        render = create_intrinsics()
        render['camtoworld'] = setup_camtoworld(cam_pos, cam_tgt)
    item = {
        'qtype': 'object_size_mca',
        'scene': str(scene_path),
        'seed': seed,
        'question': f'What is the length of the longest dimension (cm) of the {b.label}?',
        'choices': choices_cm,
        'answer': correct,
        'meta': {
            'object_id': b.id,
            'label': b.label,
            'aabb_dims_m': dims.tolist(),
            'camera_pos': cam_pos.tolist() if cam_pos is not None else None,
            'camera_target': cam_tgt.tolist() if cam_tgt is not None else None,
            'render': render
        }
    }
    return item

def generate_items_from_view(view: Dict[str, Any], scene_path: str, rng: random.Random | None = None) -> List[Dict[str, Any]]:
    _rng = rng or random
    items: List[Dict[str, Any]] = []
    pos = view['pos']
    tgt = view['tgt']
    visible = view['visible']

    # For per-object questions (everything except object_count_mca) we must
    # remove categories that have multiple visible instances. This enforces the
    # constraint: if there are two cups visible, we do NOT ask cup-specific
    # questions such as size or cup-to-cabinet distance.
    label_counts: Dict[str, int] = {}
    for b in visible:
        label_counts[b.label] = label_counts.get(b.label, 0) + 1
    # unique_visible contains only instances whose label appears exactly once
    unique_visible = [b for b in visible if label_counts.get(b.label, 0) == 1]

    # Single-object items: nearest_object_mca-compatible
    if len(unique_visible) >= 1:
        print("unique visible objects counts for single-object items:", len(unique_visible))
        print("unique visible objects:", [b.id for b in unique_visible])
        labels = [b.label for b in unique_visible][:6]
        if len(labels) >= 3:
            choices = labels[:3]
            centers = [0.5 * (b.bmin + b.bmax) for b in unique_visible[:3]]
            dists = [float(np.linalg.norm(pos - c)) for c in centers]
            order = np.argsort(dists)
            correct_lab = choices[order[0]]
            shuffled = choices.copy()
            _rng.shuffle(shuffled)
            correct = ['A', 'B', 'C'][shuffled.index(correct_lab)]
            item = {
                'qtype': 'nearest_object_mca',
                'scene': str(scene_path),
                'seed': _rng.getrandbits(32),
                'question': 'Which of these objects is closest to the camera?',
                'choices': shuffled,
                'answer': correct,
                'meta': {
                    'camera_pos': pos.tolist(),
                    'camera_target': tgt.tolist(),
                    'visible_instances': [{'label': c, 'center': centers[i].tolist()} for i, c in enumerate(choices[:3])]
                }
            }
            items.append(item)

        # Note: relative direction questions were removed (we only produce
        # nearest_object_mca for single-object localization to avoid coarse
        # directional labels). If you want to re-enable a directional question
        # type later, add a new generator that produces more robust labels.

    # For pairwise distance questions only consider unique labels
    if len(unique_visible) >= 1:
        pairs_generated = 0
        max_pairs = 6
        for i in range(min(len(unique_visible), 8)):
            for j in range(i + 1, min(len(unique_visible), 8)):
                a = unique_visible[i]
                b = unique_visible[j]
                d = float(np.linalg.norm(0.5 * (a.bmin + a.bmax) - 0.5 * (b.bmin + b.bmax)))
                choices, correct_label = simple_meters_to_choices(d, rng=_rng)
                item = {
                    'qtype': 'object_object_distance_mca',
                    'scene': str(scene_path),
                    'seed': _rng.getrandbits(32),
                    'question': f'What is the distance between the {a.label} and the {b.label}?',
                    'choices': choices,
                    'answer': correct_label,
                    'meta': {
                        'object_a': a.id,
                        'object_b': b.id,
                        'label_a': a.label,
                        'label_b': b.label,
                        'camera_pos': pos.tolist(),
                        'camera_target': tgt.tolist()
                    }
                }
                items.append(item)
                pairs_generated += 1
                if pairs_generated >= max_pairs:
                    break
            if pairs_generated >= max_pairs:
                break

    if visible:
        cat_map: Dict[str, int] = {}
        for b in visible:
            cat_map[b.label] = cat_map.get(b.label, 0) + 1
        for lab, cnt in cat_map.items():
            if cnt >= 2:
                choices = [str(cnt), str(max(0, cnt - 1)), str(cnt + 1)]
                _rng.shuffle(choices)
                correct = ['A', 'B', 'C'][choices.index(str(cnt))]
                item = {
                    'qtype': 'object_count_mca',
                    'scene': str(scene_path),
                    'seed': _rng.getrandbits(32),
                    'question': f'How many {lab} are visible in this view?',
                    'choices': choices,
                    'answer': correct,
                    'meta': {
                        'preset': view.get('preset'),
                        'category': lab,
                        'camera_pos': pos.tolist(),
                        'camera_target': tgt.tolist()
                    }
                }
                items.append(item)

    # Also attempt richer QA types (size/distance/action-next-frame) using
    # view-based implementations. If a view cannot produce a given question
    # (no suitable visible object or other constraint), skip that question.
    try:
        it_size = generate_object_size_from_view(view, scene_path, seed=_rng.getrandbits(32), rng=_rng)
        if it_size is not None:
            items.append(it_size)
    except Exception:
        pass

    try:
        it_action = generate_action_next_frame_from_view(view, scene_path, seed=_rng.getrandbits(32), rng=_rng)
        if it_action is not None:
            items.append(it_action)
    except Exception:
        pass

    try:
        it_dist = generate_distance_mca_from_view(view, scene_path, seed=_rng.getrandbits(32), rng=_rng)
        if it_dist is not None:
            items.append(it_dist)
    except Exception:
        pass

    return items

def _item_sig(item: Dict[str, Any]) -> Tuple:
    q = item.get('qtype')
    m = item.get('meta', {})
    pos = m.get('camera_pos') or m.get('current_pos') or m.get('start_pos')
    pos_sig = None
    if pos:
        pos_sig = tuple(round(float(x), 2) for x in pos[:3])

    # --- 1. nearest_object_mca ---
    if q == 'nearest_object_mca':
        # sort object labels → ignore shuffle order
        choices = sorted(item.get('choices', []))
        return (q, tuple(choices), pos_sig)

    # --- 2. object_object_distance_mca ---
    if q == 'object_object_distance_mca':
        a = m.get('object_a')
        b = m.get('object_b')
        # order-agnostic pair
        pair = tuple(sorted([str(a), str(b)]))
        return (q, pair)

    # --- 3. object_count_mca ---
    if q == 'object_count_mca':
        cat = m.get('category')
        return (q, cat, pos_sig)

    # --- 4. object_size_mca ---
    if q == 'object_size_mca':
        oid = m.get('object_id')
        return (q, str(oid), pos_sig)

    # --- 5. distance_mca ---
    if q == 'distance_mca':
        oid = m.get('object_id')
        return (q, str(oid), pos_sig)

    # --- 6. action_next_frame_mca ---
    if q == 'action_next_frame_mca':
        act = m.get('action')
        return (q, act, pos_sig)

    return (q,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out-dir', required=False, help='When provided, save each item as a folder under this directory with a json and rendered images')
    parser.add_argument('--render', action='store_true', help='Attempt strict rendering for thumbnails (uses camera_test / PLYGaussianLoader if available)')
    parser.add_argument('--corner-threshold', type=int, default=4)
    parser.add_argument('--max_items_per_view', type=int, default=10, help='Maximum number of items to emit from a single view')
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200, help='Maximum number of items to emit per scene')
    parser.add_argument('--question_type', type=str, default='all', help="Comma-separated qtypes to generate (e.g. 'distance_mca,object_count_mca') or 'all'")
    args = parser.parse_args()

    # apply CLI option to module-level flag
    global REQUIRE_IN_ROOM
    REQUIRE_IN_ROOM = bool(getattr(args, 'require_in_room', False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Instead of a single scene, accept either a single scene or a directory
    # containing many scene subfolders. Iterate each scene and emit up to
    # --max_items_per_scene items per scene, respecting the global --max_items cap.
    rng = random.Random(42)
    items: List[Dict[str, Any]] = []
    seen = set()

    root_scene = Path(args.scene)
    if root_scene.is_dir():
        scene_paths = [p for p in sorted(root_scene.iterdir()) if p.is_dir()]
    else:
        scene_paths = [root_scene]

    corner_threshold = int(getattr(args, 'corner_threshold', 4))

    for scene_path in scene_paths:
        if len(items) >= args.max_items:
            break
        print(f"===========Processing scene: {scene_path}================")
        sc = SemanticCamera(str(scene_path))
        aabbs_objs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs_objs + load_scene_wall_aabbs(str(scene_path))
        presets = list(sc.presets.keys())

        # optional height bounds for this scene
        min_h, max_h = load_structure_height_bounds(str(scene_path))
        if min_h is not None or max_h is not None:
            print(f'Using height bounds from occupancy for {scene_path.name}: {min_h} - {max_h}')

        per_scene_limit = int(getattr(args, 'max_items_per_scene', 200))
        per_scene_count = 0
        cnt = 0
        # iterate over every object, each preset and multiple distance scales for this scene
        for obj in aabbs_objs:
            if per_scene_count >= per_scene_limit or len(items) >= args.max_items:
                break
            # try a small set of distance scales for each preset to produce near/medium/far views
            distance_scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.5]

            for preset in presets:
                for scale in distance_scales:
                    if per_scene_count >= per_scene_limit or len(items) >= args.max_items:
                        break
                    try:
                        cfg = sc.calculate_camera(obj.id, preset=preset, distance_scale=scale) if hasattr(sc, 'calculate_camera') else None
                        cnt += 1
                    except Exception as e:
                        print(f"calculate_camera raised exception for obj={obj.id} preset={preset} scale={scale}: {e}")
                        cfg = None
                    if cfg is None:
                        print(f"[skip] camera config None for obj={obj.id} preset={preset} scale={scale}")
                        continue
                    # Found a usable cfg for this (preset, scale); proceed with view construction
                    pos = np.array(cfg.camera_position, dtype=float)
                    tgt = np.array(cfg.target_position, dtype=float)

                    # check position legality (inside AABBs and height bounds)
                    if not ensure_position_legal(str(scene_path), pos, min_h=min_h, max_h=max_h):
                        print(f"[illegal] camera position rejected for obj={obj.id} preset={preset} scale={scale}")
                        continue

                    # build visible list for this view
                    visible = []
                    K = np.array(create_intrinsics()['K'], dtype=float)
                    width = WIDTH; height = WIDTH
                    for b in aabbs_objs:
                        corners = count_visible_corners_for_box(pos, tgt, K, b.bmin, b.bmax, width, height, corner_threshold=corner_threshold)
                        if corners < corner_threshold:
                            continue
                        c2w = camtoworld_from_pos_target(pos, tgt)
                        K = np.array(create_intrinsics()['K'], dtype=float)
                        res = occluded_area_on_image(
                            ray_o=pos,
                            target_bmin=b.bmin,
                            target_bmax=b.bmax,
                            aabbs=aabbs_all,
                            K=K,
                            camtoworld=c2w,
                            width=width,
                            height=height,
                            target_id=b.id,
                            depth_mode="mean",
                        )
                        if res['occlusion_ratio_target'] > 0.4:
                            continue
                        if b.label in BLACKLIST:
                            continue
                        visible.append(b)
                    print(f"View preset={preset} scale={scale} pos={pos} tgt={tgt} visible_count={len(visible)}")

                    if not visible:
                        print(f"[skip] no visible objects for obj={obj.id} preset={preset} scale={scale}")
                        continue

                    view = {'preset': preset, 'pos': pos, 'tgt': tgt, 'visible': visible}
                    gen = generate_items_from_view(view, str(scene_path), rng=rng)
                    if isinstance(gen, list) and len(gen) == 0:
                        print(f"[info] generate_items_from_view produced 0 items for obj={obj.id} preset={preset} scale={scale}")
                    # filter by requested question types
                    if args.question_type and args.question_type.lower() != 'all':
                        allowed_qtypes = set([s.strip() for s in args.question_type.split(',') if s.strip()])
                        gen = (it for it in gen if it.get('qtype') in allowed_qtypes)
                    per_view_limit = int(getattr(args, 'max_items_per_view', args.max_items_per_view if hasattr(args, 'max_items_per_view') else 10))
                    per_view_count = 0
                    for it in gen:
                        if per_view_count >= per_view_limit:
                            break
                        sig = (scene_path.name,) + _item_sig(it)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        it['scene'] = str(scene_path)
                        items.append(it)
                        per_view_count += 1
                        per_scene_count += 1
                        if len(items) >= args.max_items:
                            break
                # end for scale loop
                # continue to next preset
                continue

        print("tried {} times".format(cnt))


    # write JSONL
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    print(f"Wrote {len(items)} QA items to {out_path}")

    # If requested, create per-item folders with reduced JSON and rendered images
    if getattr(args, 'out_dir', None):
        out_base = Path(args.out_dir)
        out_base.mkdir(parents=True, exist_ok=True)

        # Prepare strict renderer assets if requested (--render). Mirrors qa_preview.main behavior.
        means = quats = scales = opacities = colors = sh_degree = None
        device = None
        assets_cache: Dict[str, dict] = {}

        if getattr(args, 'render', False):
            from utils.camera_test import prepare_gaussian_data as ct_prepare_gaussian_data
            from ply_gaussian_loader import PLYGaussianLoader
            import torch
            loader = PLYGaussianLoader()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ... 下面在按 item 渲染时：
        for idx, it in enumerate(items):
            scene_p = Path(it.get('scene'))          # 这里 it['scene'] 在前面你已经设成 str(scene_path)
            scene_key = scene_p.as_posix()           # 或者用 scene_p.name 也行，看你想怎么区分

            if getattr(args, 'render', False):
                if scene_key not in assets_cache:
                    ply_path = scene_p / '3dgs_compressed.ply'
                    gs_data = loader.load_ply(str(ply_path))
                    means_, quats_, scales_, opacities_, colors_, sh_degree_ = ct_prepare_gaussian_data(
                        gs_data, device, use_sh=True
                    )
                    assets_cache[scene_key] = {
                        "means": means_, "quats": quats_, "scales": scales_,
                        "opacities": opacities_, "colors": colors_, "sh_degree": sh_degree_
                    }
                    print(f"[cache] Prepared gaussian data for scene {scene_key}")

                cached = assets_cache[scene_key]
                means = cached["means"]
                quats = cached["quats"]
                scales = cached["scales"]
                opacities = cached["opacities"]
                colors = cached["colors"]
                sh_degree = cached["sh_degree"]

            # 接下来照原来逻辑调用 compose_preview_for_item / render_thumbnail_for_pose(...)


        
        # helpers: poses_for_item (small, qtypes produced by this generator)
        def poses_for_item(it: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            poses = {}
            qtype = it.get('qtype')
            m = it.get('meta', {})
            p = m.get('camera_pos') or m.get('current_pos') or m.get('start_pos')
            t = m.get('camera_target') or m.get('current_target') or m.get('start_target')
            if p is not None:
                poses['view'] = {'position': np.array(p).tolist(), 'target': np.array(t if t is not None else (np.array(p)+np.array([0,1,0]))).tolist()}
                choices = it.get('choices')
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    if 'position' in c and 'target' in c:
                        key = chr(ord('A') + idx)   # A, B, C
                        poses[key] = {'position': c['position'], 'target': c['target']}
            return poses

        def extract_positions_for_item(it: Dict[str, Any]) -> Dict[str, Any]:
            meta = it.get('meta', {})
            # For meta.json we prefer to store just the scene folder name (not full path)
            scene_name = Path(it.get('scene') or '').name
            out = {'qtype': it.get('qtype'), 'scene': scene_name, 'seed': it.get('seed'), 'question': it.get('question')}
            # positional fields
            pos_fields = {}
            for k, v in meta.items():
                if k.endswith('_pos') or k.endswith('_position') or k in ('camera_pos', 'start_pos', 'end_pos', 'current_pos'):
                    try:
                        pos_fields[k] = (np.array(v).tolist() if v is not None else None)
                    except Exception:
                        pos_fields[k] = v

            # collect per-choice position/target when present
            choices_meta = []
            choices = it.get('choices')
            if isinstance(choices, list):
                for c in choices:
                    if isinstance(c, dict):
                        cd = {}
                        if 'position' in c:
                            cd['position'] = (np.array(c['position']).tolist())
                        if 'target' in c:
                            cd['target'] = (np.array(c['target']).tolist())
                        if cd:
                            choices_meta.append(cd)
                        else:
                            choices_meta.append(None)
                    else:
                        choices_meta.append(None)
            # only include choices_pos if at least one entry has actual data
            if choices_meta and any(cm is not None for cm in choices_meta):
                out['choices_pos'] = choices_meta
            if pos_fields:
                out['meta_positions'] = pos_fields

            # correct answer label (may be letter A/B/C or semantic key)
            out['answer'] = it.get('answer')

            # Build pose-based image mapping using semantic pose keys from poses_for_item
            poses = poses_for_item(it)
            pose_images = {}
            if poses:
                # Strict schema: use pose keys as the image keys (e.g. 'view','start','end')
                for key in poses.keys():
                    pose_images[key] = f"{key}.png"
                out['choice_images'] = pose_images

            # Keep human-readable choice text keyed by letters when choices are present
            # (We intentionally do NOT remap the original 'answer' here. Consumers
            # should interpret 'answer' according to the question type. This keeps
            # the schema strict: image keys use semantic names while choices remain
            # lettered when applicable.)
            choice_text = {}
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    lab = chr(ord('A') + idx)
                    # text label
                    if isinstance(c, dict):
                        # try common keys for human-readable text
                        text = c.get('label') or c.get('frame') or c.get('choice') or None
                    else:
                        text = str(c)
                    choice_text[lab] = text
                if choice_text:
                    out['choice_text'] = choice_text

            # include target info when available
            targ = {}
            if 'object_id' in meta:
                targ['id'] = meta.get('object_id')
                targ['label'] = meta.get('object_label') or meta.get('label')
            elif 'target_id' in meta:
                targ['id'] = meta.get('target_id')
                targ['label'] = meta.get('target_label') or meta.get('label')
            # visible_instances (e.g., nearest_object) -- include as-is if present
            if 'visible_instances' in meta:
                targ['visible_instances'] = meta.get('visible_instances')
            if targ:
                out['target'] = targ

            if poses:
                out['poses'] = poses

            return out

        # iterate and write
        for idx, it in enumerate(items):
            # prefix folder names with the scene name so that items from multiple
            # scenes are grouped and easily identifiable
            scene_name = Path(it.get('scene') or '').name
            folder = out_base / f'{scene_name}_item_{idx:04d}_{it.get("qtype","unknown")}'
            folder.mkdir(parents=True, exist_ok=True)
            # write reduced JSON
            meta_json = extract_positions_for_item(it)
            with open(folder / 'meta.json', 'w', encoding='utf-8') as jf:
                json.dump(meta_json, jf, indent=2, ensure_ascii=False)

            # If strict preview composite desired, call compose_preview_for_item to create a full PNG
            scene_p = Path(it.get('scene'))
            base_scene_path = Path(args.scene)
            if getattr(args, 'render', False) and compose_preview_for_item is not None:
                try:
                    compose_preview_for_item(it, base_scene_path /scene_p, folder / 'preview.png', tvgen=None, means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, sh_degree=sh_degree, device=device, thumb_size=256)
                except Exception as e:
                    print(f'Warning: compose_preview_for_item failed for item {idx}: {e}')

            # render images for poses (A.png, B.png...); try strict renderer first
            poses = poses_for_item(it)
            for label, pose in poses.items():
                img = None
                img = render_thumbnail_for_pose(base_scene_path /scene_p, pose, thumb_size=400)
                imageio.imwrite(folder / f'{label}.png', img)
                print(f'Wrote {folder / f"{label}.png"}')

                


if __name__ == '__main__':
    main()
