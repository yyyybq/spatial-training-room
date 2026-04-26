#!/usr/bin/env python3
"""
Generate `relative_position_mca` items using sampler-style view sampling.

This generator samples camera poses per object (using `generate_camera_positions`)
and for each sampled view selects a pair of visible objects, then asks the
relative position of one object with respect to the other in the image (left,
right, front (closer to camera), back).

Per-item outputs (when --out-dir provided):
  - meta.json (contains camera pose, object ids, and answer)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)

Run example:
  python -m Data_generation.sampler.question_generator.relative_position_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/relative_position.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/relative_pos_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# ------------------------- small copied helpers -------------------------
from types import SimpleNamespace


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
    return camtoworld


def is_pos_inside_scene(scene_path: str, pos: np.ndarray, tol: float = 0.01) -> bool:
    aabbs = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    q = np.array(pos, dtype=float)
    for b in aabbs:
        if (q[0] >= b.bmin[0] - tol and q[0] <= b.bmax[0] + tol and
            q[1] >= b.bmin[1] - tol and q[1] <= b.bmax[1] + tol and
            q[2] >= b.bmin[2] - tol and q[2] <= b.bmax[2] + tol):
            return True
    return False


# ------------------------- core logic -------------------------

def determine_relative(a_center: np.ndarray, b_center: np.ndarray, cam_pos: np.ndarray, camtoworld: np.ndarray,
                       left_right_thresh: float = 0.05, depth_thresh: float = 0.05, vertical_thresh: float = 0.05) -> str | None:
    """Determine relative direction of B w.r.t A from the camera viewpoint.

    This returns one of a set including single-axis directions (left,right,front,back,above,below)
    or two-axis combined directions such as 'left-front', 'right-above', etc. The function
    projects object centers into the camera-aligned basis (right, forward, up) and compares
    the relative offsets between A and B along each axis using separate thresholds.

    Returns None if the relative position is ambiguous (no axis exceeds thresholds).
    """
    cam_pos = np.array(cam_pos, dtype=float)
    # camera basis: right, up, forward (note setup_camera uses -right in column 0)
    right = -np.array(camtoworld[:3, 0], dtype=float)
    up = np.array(camtoworld[:3, 1], dtype=float)
    forward = np.array(camtoworld[:3, 2], dtype=float)

    vA = np.array(a_center, dtype=float) - cam_pos
    vB = np.array(b_center, dtype=float) - cam_pos

    xA = float(np.dot(vA, right)); xB = float(np.dot(vB, right))
    yA = float(np.dot(vA, up));    yB = float(np.dot(vB, up))
    zA = float(np.dot(vA, forward)); zB = float(np.dot(vB, forward))

    delta_x = xB - xA
    delta_y = yB - yA
    delta_z = zB - zA

    horiz = None
    depth = None
    vert = None

    # left/right
    if abs(delta_x) > left_right_thresh:
        # flipped: positive delta_x -> object B appears to the LEFT in image
        horiz = 'left' if delta_x > 0 else 'right'

    # front/back: smaller projected z means closer to camera -> 'front'
    if abs(delta_z) > depth_thresh:
        depth = 'front' if zB < zA else 'back'

    # above/below based on camera up vector
    if abs(delta_y) > vertical_thresh:
        vert = 'above' if delta_y > 0 else 'below'

    # prefer two-axis combined labels if available (choose meaningful combos)
    # ordering: horiz-depth, horiz-vert, depth-vert
    if horiz and depth:
        return f"{horiz}-{depth}"
    if horiz and vert:
        return f"{horiz}-{vert}"
    if depth and vert:
        return f"{depth}-{vert}"

    # otherwise return single-axis if any
    if horiz:
        return horiz
    if depth:
        return depth
    if vert:
        return vert

    return None


def generate_relative_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None, verbose: bool = False) -> Dict[str, Any] | None:
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible or len(visible) < 2:
        if verbose:
            print(f"[skip] fewer than 2 visible objects for scene={scene_path}")
        return None

    # build candidate list similar to object_object_distance_mca
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    c2w = camtoworld_from_pos_target(pos, tgt)

    candidates = []
    for b in visible:
        if b.label in BLACKLIST:
            continue
        bmin_val = getattr(b, 'bbox_min', None)
        if bmin_val is None:
            bmin_val = getattr(b, 'bmin', None)
        bmax_val = getattr(b, 'bbox_max', None)
        if bmax_val is None:
            bmax_val = getattr(b, 'bmax', None)
        if bmin_val is None or bmax_val is None:
            continue
        wrapped = SimpleNamespace(bmin=bmin_val, bmax=bmax_val, id=getattr(b, 'id', None), label=getattr(b, 'label', None))
        if is_aabb_occluded(pos, wrapped, aabbs_all, K=K, camtoworld=c2w, width=width, height=height):
            continue
        candidates.append(wrapped)

    if len(candidates) < 2:
        if verbose:
            print(f"[skip] fewer than 2 candidates after occlusion for scene={scene_path}")
        return None

    # assign unique labels for display if duplicates
    label_counts = {}
    for obj in candidates:
        label = obj.label
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            obj.unique_label = f"{label} #{label_counts[label]}"
        else:
            obj.unique_label = label

    # pick pair with largest center-to-center distance (deterministic)
    pairs: List[Tuple[Any, Any, float]] = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            ca = 0.5 * (np.array(a.bmin, dtype=float) + np.array(a.bmax, dtype=float))
            cb = 0.5 * (np.array(b.bmin, dtype=float) + np.array(b.bmax, dtype=float))
            d = float(np.linalg.norm(ca - cb))
            pairs.append((a, b, d))
    pairs.sort(key=lambda x: x[2], reverse=True)
    a, b, _ = pairs[0]
    ca = 0.5 * (np.array(a.bmin, dtype=float) + np.array(a.bmax, dtype=float))
    cb = 0.5 * (np.array(b.bmin, dtype=float) + np.array(b.bmax, dtype=float))

    # compute cam pose and determine relative
    camtoworld = camtoworld_from_pos_target(pos, tgt)
    rel = determine_relative(ca, cb, camtoworld[:3, 3], camtoworld)
    if rel is None:
        if verbose:
            print(f"[skip] ambiguous relative position for pair {a.id},{b.id} in scene={scene_path}")
        return None

    # Build a pool of possible directional labels (including diagonals and verticals)
    possible_labels = [
        'left','right','front','back','above','below',
        'left-front','left-back','right-front','right-back',
        'left-above','left-below','right-above','right-below',
        'front-above','front-below'
    ]

    # Ensure the correct label is present in the options; pick 3 distinct distractors
    other_choices = [x for x in possible_labels if x != rel]
    # If rng has insufficient pool (unlikely), fallback to sampling with replacement
    try:
        distractors = _rng.sample(other_choices, k=3)
    except ValueError:
        # fallback: fill remaining with random choices allowing duplicates
        distractors = []
        while len(distractors) < 3 and other_choices:
            distractors.append(_rng.choice(other_choices))

    options_shuffled = [rel] + distractors
    _rng.shuffle(options_shuffled)
    correct_label = options_shuffled.index(rel)
    labels = ['A', 'B', 'C', 'D']
    answer = labels[correct_label]

    render = create_intrinsics()
    render['camtoworld'] = camtoworld.tolist()

    item = {
                    'qtype': 'relative_position',
        'scene': str(scene_path),
        'question': f"In the image, which is the relative position of the {a.unique_label} to  the {b.unique_label}?",
        'choices': options_shuffled,
        'answer': answer,
        'meta': {
            'objectA_id': a.id,
            'objectB_id': b.id,
            'labelA': a.unique_label,
            'labelB': b.unique_label,
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist(),
            'render': render,
            'choices_map': options_shuffled,
        }
    }
    return item


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(202312)
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
            obj = SceneObject(it)
            objects[obj.id] = obj

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
                    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, WIDTH, HEIGHT, corner_threshold=4, aabbs_all=aabbs_all)
                    camtoworld = camtoworld_from_pos_target(pos, tgt)
                    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    target_px = float(occ.get('target_area_px', 0.0))
                    target_ratio = float(target_px) / float(WIDTH * HEIGHT)
                    if target_ratio <= 0.01:
                        continue
                    visible.append(so)

                if not visible:
                    if verbose:
                        print(f"[skip] no visible objects after filtering for scene={scene_path.name} pose_index={pi}")
                    continue

                view = {'pos': pos.tolist(), 'tgt': tgt.tolist(), 'visible': visible, 'pose_meta': p}
                view['pose_meta']['target_obj_id'] = obj_id

                item = generate_relative_from_view(view, str(scene_path), seed=None, rng=rng, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_relative_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_relative_position_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # compose preview and view map using preview utilities (thumb_size uses WIDTH)
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                items.append(item)
                per_scene_count += 1

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate relative_position_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    parser.add_argument('--verbose', action='store_true', help='Print debug/skip reasons during generation')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render, verbose=args.verbose)
    print(f"Wrote {len(items)} relative_position_mca items to {out_path}")


if __name__ == '__main__':
    main()
