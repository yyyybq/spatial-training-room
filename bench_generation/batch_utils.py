#!/usr/bin/env python3
"""Shared helpers for batch generation tasks.

This module pulls together commonly-used constants and helper functions
extracted from `qa_batch_generator.py` so other generators can import them
without depending on the large monolithic file.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Any, Tuple, Dict
import json
import numpy as np
import random

from ..utils.occlusion import (
    camtoworld_from_pos_target,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    is_occluded_by_any,
    is_box_occluded_by_any,
    is_target_in_fov,
    occluded_area_on_image,
    world_to_camera,
    aabb_corners,
    project_point,
    point_in_image,
)

# Image/Intrinsics defaults
WIDTH = 400
HEIGHT = 400


def create_intrinsics(width=WIDTH, height=HEIGHT):
    focal = float(width * 0.4)
    K = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    return {'width': int(width), 'height': int(height), 'focal': float(focal), 'K': K}


def load_structure_height_bounds(scene_path: str) -> Tuple[float | None, float | None]:
    try:
        p = Path(scene_path) / 'occupancy.json'
        if not p.exists():
            return None, None
        with open(p, 'r', encoding='utf-8') as f:
            occ = json.load(f)
        lower = occ.get('lower') or occ.get('min')
        upper = occ.get('upper') or occ.get('max')
        if isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple)) and len(lower) >= 3 and len(upper) >= 3:
            return float(lower[2]), float(upper[2])
    except Exception:
        pass
    return None, None


def load_room_polys(scene_path: str) -> List[np.ndarray]:
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
    polys = load_room_polys(scene_path)
    if not polys:
        return False
    x, y = float(pos[0]), float(pos[1])
    for poly in polys:
        if point_in_poly(x, y, poly):
            return True
    return False


def is_pos_inside_scene(scene_path: str, pos: np.ndarray, tol: float = 0.01) -> bool:
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


def ensure_position_legal(scene_path: str, pos: np.ndarray, min_h: float | None = None, max_h: float | None = None, tol: float = 0.5) -> bool:
    if is_pos_inside_scene(scene_path, pos, tol=tol):
        return False
    z = float(pos[2])
    if (min_h is not None) and (z < float(min_h) + 0.1):
        return False
    if (max_h is not None) and (z > float(max_h) - 0.1):
        return False
    if not is_pos_inside_any_room(scene_path, pos):
        return False
    return True


def count_visible_corners_for_box(pos: np.ndarray, tgt: np.ndarray, K: np.ndarray, target_bmin: np.ndarray, target_bmax: np.ndarray, width: int, height: int, corner_threshold: int = 4, aabbs_all: List[Any] | None = None, target_id: str | None = None, require_unoccluded: bool = False) -> int:
    c2w = camtoworld_from_pos_target(pos, tgt)
    view = np.linalg.inv(np.array(c2w, dtype=float))
    corners = aabb_corners(target_bmin, target_bmax)
    pcs = np.array([world_to_camera(view, c) for c in corners])
    uvz = [project_point(np.array(K, dtype=float), pc) for pc in pcs]
    cnt = 0
    for idx, (u, v, z) in enumerate(uvz):
        if z > 1e-6 and point_in_image(u, v, width, height, border=2):
            if require_unoccluded and aabbs_all is not None:
                corner_world = corners[idx]
                try:
                    if is_occluded_by_any(pos, corner_world, aabbs_all, target_id=target_id):
                        continue
                except Exception:
                    pass
            cnt += 1
    return cnt


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
            pass
    return is_box_occluded_by_any(pos, aabb.bmin, aabb.bmax, aabbs_all, target_id=getattr(aabb, 'id', None))


# BLACKLIST copied from original generator
BLACKLIST = {
    "wall", "floor", "ceiling", "room",
    "carpet", "rug",
    "chandelier", "ceiling lamp", "spotlight", "lamp", "light",
    "downlights", "wall lamp", "table lamp", "strip light", "track light",
    "linear lamp", "decorative pendant",
    "other", "curtain", "bread", "cigar", "wine", "fresh food", "pen",
    "medicine bottle", "toiletries", "chocolate", "paper",
    "book", "boxed food", "bagged food", "medicine box",
    "vegetable", "fruit", "drinks", "canned food",
}


def simple_meters_to_choices(m: float, rng: random.Random | None = None):
    _rng = rng or random
    correct = round(float(m), 2)
    candidates = [round(correct + d, 2) for d in (0.0, 0.25, -0.25, 0.5, -0.5)]
    seen_vals = []
    for v in candidates:
        if v not in seen_vals:
            seen_vals.append(v)
        if len(seen_vals) >= 3:
            break
    if correct not in seen_vals:
        seen_vals[-1] = correct
    _rng.shuffle(seen_vals)
    labels = ['A', 'B', 'C']
    formatted = [f"{v}m" for v in seen_vals]
    correct_label = labels[formatted.index(f"{correct}m")]
    return formatted, correct_label


def meters_to_choices(m: float, rng: random.Random | None = None):
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


def generate_distance_mca_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None):
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        return None
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    c2w = camtoworld_from_pos_target(pos, tgt)
    candidates = []
    for b in visible:
        if b.label in BLACKLIST:
            # keep same behavior as original generator
            pass
        if is_aabb_occluded(pos, b, aabbs_all, K=K, camtoworld=c2w, width=width, height=height):
            continue
        if not is_target_in_fov(K, c2w, b.bmin, b.bmax, width, height, require_center=True):
            continue
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
