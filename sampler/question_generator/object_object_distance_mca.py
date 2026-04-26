#!/usr/bin/env python3
"""
Generate `object_object_distance_mca` items using sampler-style view sampling.

This module samples camera poses per object using the same helper
(`generate_camera_positions`) as `sampler.batch_generate_views`, then
for each sampled view selects a pair of visible objects and emits an
`object_object_distance_mca` question asking for the distance between them.

Rendering is mandatory (no lazy imports). Per-item folders contain
`meta.json`, `preview.png` and `view.png`.

Usage example:
  python -m Data_generation.sampler.question_generator.object_object_distance_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/object_object_distance.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/ood_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer (no lazy import)
from ...bench_generation.preview import compose_preview_for_item, compose_view_map

# shared helpers/constants
from ...bench_generation.batch_utils import (
    create_intrinsics,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    camtoworld_from_pos_target,
    occluded_area_on_image,
    is_aabb_occluded,
    is_occluded_by_any,
    count_visible_corners_for_box,
    BLACKLIST,
    WIDTH,
    HEIGHT,
)
from types import SimpleNamespace


# -------------------- 房间多边形判定（从 choose_object.py 复制） -----------------
def point_in_poly(x: float, y: float, poly: List[List[float]]) -> bool:
    """判断点 (x,y) 是否位于二维多边形 `poly` 内（射线法）。"""
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
    """从 `structure.json` 加载房间多边形，返回多边形列表，每个为 [[x,y], ...]。

    若文件缺失或解析失败返回空列表。
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


def get_room_index_for_point(x: float, y: float, room_polys: List[List[List[float]]]) -> int:
    """Return index of room polygon that contains (x,y) or None."""
    if not room_polys:
        return None
    for ri, poly in enumerate(room_polys):
        if point_in_poly(x, y, poly):
            return ri
    return None


def meters_to_choices(m: float, rng: random.Random | None = None) -> Tuple[List[str], str]:
    """Return 3-4 human-readable choices and the correct label for a meter value."""
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


def generate_object_object_distance_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None, verbose: bool = False) -> Dict[str, Any] | None:
    """Given a sampled view dict (pos,tgt,visible list), pick a pair and build an item.
    Returns None when no suitable pair found.
    """
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible or len(visible) < 2:
        if verbose:
            print(f"[skip] view has fewer than 2 visible objects (visible={len(visible)}) for scene={scene_path}")
        return None

    # Filter visible objects by blacklist and occlusion
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    c2w = camtoworld_from_pos_target(pos, tgt)

    candidates = []
    for b in visible:
        if b.label in BLACKLIST:
            continue
        # create a thin wrapper with expected attributes (bmin/bmax/id/label)
        bmin_val = getattr(b, 'bbox_min', None)
        if bmin_val is None:
            bmin_val = getattr(b, 'bmin', None)
        bmax_val = getattr(b, 'bbox_max', None)
        if bmax_val is None:
            bmax_val = getattr(b, 'bmax', None)
        wrapped = SimpleNamespace(bmin=bmin_val,
                                  bmax=bmax_val,
                                  id=getattr(b, 'id', None),
                                  label=getattr(b, 'label', None))
        if wrapped.bmin is None or wrapped.bmax is None:
            # cannot evaluate occlusion without bbox info
            if verbose:
                print(f"[skip] object missing bbox for occlusion check: {b}")
            continue
        if is_aabb_occluded(pos, wrapped, aabbs_all, K=K, camtoworld=c2w, width=width, height=height):
            continue
        candidates.append(wrapped)
    if len(candidates) < 2:
        if verbose:
            print(f"[skip] after blacklist/occlusion filtering fewer than 2 candidates (candidates={len(candidates)}) for scene={scene_path}")
        return None

    # If any object label appears more than twice among candidates, skip this view
    label_counts_view = {}
    for c in candidates:
        label_counts_view[c.label] = label_counts_view.get(c.label, 0) + 1
    for lab, cnt in label_counts_view.items():
        if cnt > 2:
            if verbose:
                print(f"[skip] more than 2 visible instances of label='{lab}' ({cnt}) for scene={scene_path}")
            return None

    # Assign unique labels to candidates based on their order in the visible list
    # Assign unique labels to candidates based on left-to-right ordering in the image
    # Compute camera right vector and sort candidate centers by projected x (left->right)
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v * 0.0
        return v / n

    forward = tgt - pos
    forward = _unit(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = _unit(right)

    # compute center x projection for each candidate
    cand_with_x = []
    for c in candidates:
        center = 0.5 * (np.array(c.bmin, dtype=float) + np.array(c.bmax, dtype=float))
        xproj = float(np.dot(center - pos, right))
        cand_with_x.append((c, xproj))

    # sort left->right (ascending xproj)
    cand_with_x.sort(key=lambda t: t[1])

    # count occurrences per label among candidates
    label_counts = {}
    for c, _ in cand_with_x:
        label_counts[c.label] = label_counts.get(c.label, 0) + 1

    # assign ordinal among same-label candidates in left->right order
    seen_idx: Dict[str, int] = {}
    for c, _ in cand_with_x:
        lbl = c.label
        seen_idx[lbl] = seen_idx.get(lbl, 0) + 1
        cnt = label_counts.get(lbl, 0)
        if cnt > 1:
            idxn = seen_idx[lbl]
            if 10 <= (idxn % 100) <= 20:
                suf = 'th'
            else:
                suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(idxn % 10, 'th')
            c.unique_label = f"the {idxn}{suf} {lbl} from left to right"
        else:
            c.unique_label = lbl

    # ----- Same-room filtering: keep only candidates that are in the same room
    # as the anchor object (anchor defined as candidate nearest to the view target)
    try:
        room_polys = load_room_polys(scene_path)
    except Exception:
        room_polys = []

    if room_polys:
        # find anchor = candidate whose center is nearest to the view target
        tgt_xy = np.array([tgt[0], tgt[1]], dtype=float)
        best_c = None
        best_d = float('inf')
        for c in candidates:
            center = 0.5 * (np.array(c.bmin, dtype=float) + np.array(c.bmax, dtype=float))
            d_xy = float(np.linalg.norm(center[:2] - tgt_xy))
            if d_xy < best_d:
                best_d = d_xy
                best_c = c

        if best_c is not None:
            ar = get_room_index_for_point(float(0.5 * (best_c.bmin[0] + best_c.bmax[0])), float(0.5 * (best_c.bmin[1] + best_c.bmax[1])), room_polys)
            if ar is not None:
                kept = []
                for c in candidates:
                    ri = get_room_index_for_point(float(0.5 * (c.bmin[0] + c.bmax[0])), float(0.5 * (c.bmin[1] + c.bmax[1])), room_polys)
                    if ri == ar:
                        kept.append(c)
                if len(kept) < 2:
                    if verbose:
                        print(f"[skip] after same-room filtering fewer than 2 candidates (kept={len(kept)}) for scene={scene_path}")
                    return None
                # replace candidates with kept list and rebuild cand_with_x ordering
                candidates = kept
                cand_with_x = []
                for c in candidates:
                    center = 0.5 * (np.array(c.bmin, dtype=float) + np.array(c.bmax, dtype=float))
                    xproj = float(np.dot(center - pos, right))
                    cand_with_x.append((c, xproj))
                cand_with_x.sort(key=lambda t: t[1])
                # reassign unique labels among remaining candidates
                label_counts = {}
                for c, _ in cand_with_x:
                    label_counts[c.label] = label_counts.get(c.label, 0) + 1
                seen_idx = {}
                for c, _ in cand_with_x:
                    lbl = c.label
                    seen_idx[lbl] = seen_idx.get(lbl, 0) + 1
                    cnt = label_counts.get(lbl, 0)
                    if cnt > 1:
                        idxn = seen_idx[lbl]
                        if 10 <= (idxn % 100) <= 20:
                            suf = 'th'
                        else:
                            suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(idxn % 10, 'th')
                        c.unique_label = f"the {idxn}{suf} {lbl} from left to right"
                    else:
                        c.unique_label = lbl

    # Build all unordered pairs and select the pair with largest center-to-center distance
    pairs: List[Tuple[Any, Any, float]] = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            ca = 0.5 * (np.array(a.bmin, dtype=float) + np.array(a.bmax, dtype=float))
            cb = 0.5 * (np.array(b.bmin, dtype=float) + np.array(b.bmax, dtype=float))
            d = float(np.linalg.norm(ca - cb))
            pairs.append((a, b, d))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[2], reverse=True)
    a, b, dist = pairs[0]

    # Use meters_to_choices to produce choices in meters
    choices, correct_label = meters_to_choices(dist, rng=_rng)

    render = None
    # prepare render intrinsics if camera pose available
    render = create_intrinsics()
    render['camtoworld'] = setup_camtoworld(pos, tgt)

    # Check for duplicate labels and assign unique identifiers
    # Assign readable unique labels only when there are duplicates of the same label
    label_counts = {}
    for obj in visible:
        label_counts[obj.label] = label_counts.get(obj.label, 0) + 1
    seen_idx: Dict[str, int] = {}
    for obj in visible:
        lbl = obj.label
        seen_idx[lbl] = seen_idx.get(lbl, 0) + 1
        count = label_counts.get(lbl, 0)
        if count > 1:
            idxn = seen_idx[lbl]
            # simple English ordinal suffix
            if 10 <= (idxn % 100) <= 20:
                suf = 'th'
            else:
                suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(idxn % 10, 'th')
            obj.unique_label = f"the {idxn}{suf} {lbl} from left to right"
        else:
            obj.unique_label = lbl

    # Modify the question to include unique labels
    item = {
        'qtype': 'object_object_distance',
        'scene': str(scene_path),
        'seed': seed,
        'question': f"What is the distance (m) between the {a.unique_label} and the {b.unique_label} in the image?",
        'choices': choices,
        'answer': correct_label,
        'meta': {
            'objectA_id': a.id,
            'objectB_id': b.id,
            'labelA': a.unique_label,
            'labelB': b.unique_label,
            'centerA_m': (0.5 * (a.bmin + a.bmax)).tolist(),
            'centerB_m': (0.5 * (b.bmin + b.bmax)).tolist(),
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist(),
            'distance_m': float(dist),
            'render': render,
        }
    }

    return item


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(42)
    items: List[Dict[str, Any]] = []

    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")

        labels_file = scene_path / 'labels.json'
        if not labels_file.exists():
            print(f"[warn] no labels.json in {scene_path}, skip")
            continue
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        objects: Dict[str, SceneObject] = {}
        for it in labels:
            if not isinstance(it, dict):
                continue
            if not it.get('ins_id') or 'bounding_box' not in it:
                continue
            try:
                obj = SceneObject(it)
                objects[obj.id] = obj
            except Exception:
                continue

        if not objects:
            print(f"[info] no valid objects in {scene_path}")
            continue

        aabbs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

        per_scene_count = 0

        for obj_id, obj in objects.items():
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break

            poses = generate_camera_positions(scene_path, obj, per_room_points=per_room_points, min_dist=min_dist, max_dist=max_dist)
            if not poses:
                if verbose:
                    print(f"[info] no poses for scene={scene_path.name} obj={obj_id} ({obj.label})")
                continue

            for pi, p in enumerate(poses):
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

                pos = np.array(p['position'], dtype=float)
                tgt = np.array(p['target'], dtype=float)

                # build visible list using occlusion + corner visibility checks
                visible: List[SceneObject] = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                for oid, so in objects.items():
                    if so.id == obj_id:
                        # the target object is always included if visible under occlusion checks
                        pass 
                    # corners visibility and occlusion
                    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, WIDTH, HEIGHT, corner_threshold=4, aabbs_all=aabbs_all)
                    if corners_visible < 2:
                        if verbose:
                            print(f"[skip-corner] scene={scene_path.name} obj={so.id} corners_visible={corners_visible} pose_index={pi}")
                        continue
                    # area occlusion

                    camtoworld = camtoworld_from_pos_target(pos, tgt)
                    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    target_px = float(occ.get('target_area_px'))
                    target_ratio = float(target_px) / float(WIDTH * HEIGHT)
                    # print(target_px, target_ratio, occluded_area_on_image)
                    if target_ratio <= 0.1:
                        # if verbose:
                            # print(target_px, target_ratio ,f"[skip-area] scene={scene_path.name} obj={so.id} target_px=0 pose_index={pi}")
                        continue
                    visible.append(so)

                if not visible:
                    # if verbose:
                    #     print(f"[skip] no visible objects after filtering for scene={scene_path.name} pose_index={pi}")
                    continue

                view = {'pos': pos.tolist(), 'tgt': tgt.tolist(), 'visible': visible, 'pose_meta': p}

                item = generate_object_object_distance_from_view(view, str(scene_path), seed=None, rng=rng, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_object_object_distance_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    # deterministic file name using counters
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_object_object_distance_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)
                    # render preview and view (thumb size driven by WIDTH constant)
                    try:
                        # compose_preview_for_item writes directly to out_path
                        compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    except Exception as e:
                        # print reason then re-raise (render is mandatory)
                        print(f"[error] preview render failed for {item_dir}: {e}")
                        raise
                    try:
                        compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)
                    except Exception as e:
                        print(f"[error] view render failed for {item_dir}: {e}")
                        raise

                # Annotate the preview image with object labels
                try:
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH, annotations=[
                        {'label': item['meta']['labelA'], 'position': item['meta']['centerA_m']},
                        {'label': item['meta']['labelB'], 'position': item['meta']['centerB_m']}
                    ])
                except Exception as e:
                    print(f"[error] preview render failed for {item_dir}: {e}")
                    raise

                items.append(item)
                per_scene_count += 1

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    # final summary
    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate object_object_distance_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=7)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    parser.add_argument('--verbose', action='store_true', help='Print debug/skip reasons during generation')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render, verbose=args.verbose)
    print(f"Wrote {len(items)} object_object_distance_mca items to {out_path}")


if __name__ == '__main__':
    main()
