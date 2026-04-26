#!/usr/bin/env python3
"""
Generate `object_size_mca` items using sampler-style view sampling.

This module samples camera poses per object using the same helper
(`generate_camera_positions`) as `sampler.batch_generate_views`, then
applies the `generate_object_size_from_view` logic from the main
`qa_batch_generator` to decide whether a pose yields a valid
`object_size_mca` item. When rendering is enabled (default), a preview
image is produced for each generated item using the project's preview
composer.

Usage example:
  python -m Data_generation.sampler.question_generator.object_size_mca    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data --out /tmp/object_size.jsonl --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/obj_size_items --per_room_points 12 --max_items 200
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
        'qtype': 'object_size',
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

def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None, per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5, max_items: int = 200, max_items_per_scene: int = 200, render: bool = True) -> List[Dict[str, Any]]:
    rng = random.Random(42)
    items: List[Dict[str, Any]] = []

    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    # preview composer is imported at module top; render failures will propagate

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")
        # load labels.json and build SceneObject list
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

        # scene geometry for occlusion checks
        aabbs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

        per_scene_count = 0

        for obj_id, obj in objects.items():
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break

            poses = generate_camera_positions(scene_path, obj, per_room_points=per_room_points, min_dist=min_dist, max_dist=max_dist)
            if not poses:
                continue

            for p in poses:
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

                pos = np.array(p['position'], dtype=float)
                tgt = np.array(p['target'], dtype=float)

                # basic legality: ensure camera not inside objects (reuse existing helper in batch_utils via is_aabb_occluded/ensure_position_legal not exposed here)
                # build visible list using same logic as qa_batch_generator
                visible = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                width = WIDTH; height = HEIGHT
                for b in aabbs:
                    corners = count_visible_corners_for_box(pos, tgt, K, b.bmin, b.bmax, width, height)
                    if corners < 4:
                        continue
                    c2w = camtoworld_from_pos_target(pos, tgt)
                    res = occluded_area_on_image(
                        ray_o=pos,
                        target_bmin=b.bmin,
                        target_bmax=b.bmax,
                        aabbs=aabbs_all,
                        K=K,
                        camtoworld=c2w,
                        width=width,
                        height=height,
                        target_id=getattr(b, 'id', None),
                        depth_mode='mean',
                    )
                    if res.get('occlusion_ratio_target', 1.0) > 0.4:
                        continue
                    if b.label in BLACKLIST:
                        continue
                    visible.append(b)

                if not visible:
                    continue

                view = {'pos': pos.tolist(), 'tgt': tgt.tolist(), 'visible': visible}

                it = generate_object_size_from_view(view, str(scene_path), seed=rng.getrandbits(32), rng=rng)

                if it is None:
                    continue

                items.append(it)
                per_scene_count += 1

                # write per-item folder and preview if requested
                if out_dir is not None:
                    base = Path(out_dir)
                    scene_name = scene_path.name
                    folder = base / f"{scene_name}_item_{len(items)-1:04d}_{it.get('qtype','unknown')}"
                    folder.mkdir(parents=True, exist_ok=True)
                    with open(folder / 'meta.json', 'w', encoding='utf-8') as jf:
                        json.dump(it, jf, indent=2, ensure_ascii=False)
                    if render:
                        out_png = folder / 'preview.png'
                        compose_preview_for_item(it, scene_path, out_png)
                        # also write a view-only map image
                        view_png = folder / 'view.png'
                        compose_view_map(it, scene_path, view_png)

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    return items


def main():
    parser = argparse.ArgumentParser(description='Generate object_size_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    # rendering default on; allow --no-render to disable
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render)
    print(f"Wrote {len(items)} object_size_mca items to {out_path}")


if __name__ == '__main__':
    main()
