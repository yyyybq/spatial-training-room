#!/usr/bin/env python3
"""
Generate `object_after_rotation_mca` items using sampler-style view sampling.

Description:
  - Sample legal start views using `generate_camera_positions` from the sampler.
  - Pick a start view where a target object is visible.
  - Rotate the camera in-place by a small angle (abs < 100 deg). Select a rotation
    where AFTER rotation the target object becomes not visible.
  - Ask: "After the rotation the target object is on which side of the camera?"
    (choices: Left / Right / Front / Back)

Per-item outputs (when --out-dir provided):
  - meta.json
  - preview.png (composed preview using project's preview composer)
  - view.png (map / rendered view)

Run example:
  python -m Data_generation.sampler.question_generator.object_after_rotation_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/object_after_rot.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/object_after_rot_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import imageio

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer (no lazy import)
from ...bench_generation.preview import compose_preview_for_item, compose_view_map, render_thumbnail_for_pose

# shared helpers/constants (use WIDTH for thumbnail size as requested)
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


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def rotate_around_z(vec: np.ndarray, deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c = math.cos(rad); s = math.sin(rad)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    xr = c * x - s * y
    yr = s * x + c * y
    return np.array([xr, yr, z], dtype=float)


def signed_yaw_deg_between_forward(start_pos: np.ndarray, start_tgt: np.ndarray, end_pos: np.ndarray, end_tgt: np.ndarray) -> float:
    """Compute signed yaw (degrees) of camera rotation from start->end.

    Projects forward vectors to XY plane and computes signed angle via atan2(det, dot).
    Positive means rotated to the left (counter-clockwise looking from above).
    """
    f0 = unit(np.array(start_tgt) - np.array(start_pos))
    f1 = unit(np.array(end_tgt) - np.array(end_pos))
    v0 = np.array([f0[0], f0[1], 0.0], dtype=float)
    v1 = np.array([f1[0], f1[1], 0.0], dtype=float)
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    v0 = v0 / n0
    v1 = v1 / n1
    dot = float(np.dot(v0, v1))
    det = float(v0[0]*v1[1] - v0[1]*v1[0])
    ang = math.degrees(math.atan2(det, dot))
    # clamp
    if ang > 100.0:
        ang = 100.0
    if ang < -100.0:
        ang = -100.0
    return ang


def is_instance_visible(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    """Return True if ALL four bbox corners are visible and target occupies >1% of image."""
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 4:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(width * height)
    return target_ratio > 0.01


def classify_relative_side(end_pos: np.ndarray, end_tgt: np.ndarray, target_pos: np.ndarray) -> str:
    """Classify target side into four buckets suitable for this question:

    Returns one of: 'Left', 'Right', 'LeftBack', 'RightBack'.
    Logic:
      - Compute signed angle (degrees) from camera forward to vector->target (positive = left).
      - If abs(angle) > 90 -> behind hemisphere: choose LeftBack/RightBack based on sign.
      - Else -> front hemisphere: choose Left/Right based on sign (ang==0 -> Right).
    """
    fwd = unit(np.array(end_tgt) - np.array(end_pos))
    vec = unit(np.array(target_pos) - np.array(end_pos))
    # project to XY
    v0 = np.array([fwd[0], fwd[1], 0.0], dtype=float)
    v1 = np.array([vec[0], vec[1], 0.0], dtype=float)
    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
        return 'Right'
    v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    det = float(v0[0]*v1[1] - v0[1]*v1[0])
    dot = float(np.dot(v0, v1))
    ang = math.degrees(math.atan2(det, dot))
    # behind hemisphere if abs(angle) > 90
    if abs(ang) > 90.0:
        return 'LeftBack' if ang > 0 else 'RightBack'
    # front hemisphere
    return 'Left' if ang > 0 else 'Right'


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(202502)
    items: List[Dict[str, Any]] = []

    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")

        labels_file = scene_path / 'labels.json'
        if not labels_file.exists():
            if verbose:
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
            if verbose:
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

                start_pos = np.array(p['position'], dtype=float)
                start_tgt = np.array(p['target'], dtype=float)

                # build visible list (all objects visible from this start view)
                visible: List[SceneObject] = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                for oid, so in objects.items():
                    if so.label in BLACKLIST:
                        continue
                    if is_instance_visible(start_pos, start_tgt, so, aabbs_all, K, WIDTH, HEIGHT):
                        visible.append(so)

                # target must be visible in the start view
                if obj not in visible:
                    if verbose:
                        print(f"[skip] target not visible at start for scene={scene_path.name} pose_index={pi} obj={obj.label}")
                    continue

                # assign unique labels for repeated categories (left-to-right order of `visible` list)
                label_counts: Dict[str, int] = {}
                for so in visible:
                    lab = getattr(so, 'label', '')
                    if lab not in label_counts:
                        label_counts[lab] = 0
                    label_counts[lab] += 1
                    if label_counts[lab] > 1:
                        so.unique_label = f"The {label_counts[lab]}th {lab} from left to right"
                    else:
                        so.unique_label = lab

                # candidate rotation angles (deg). Try small to moderate angles (<100).
                candidate_angles = [10, -10, 15, -15, 20, -20, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
                rng.shuffle(candidate_angles)
                chosen_ang = None
                chosen_end = None

                d0 = np.array(start_tgt - start_pos, dtype=float)

                for ang in candidate_angles:
                    if abs(ang) >= 100.0:
                        continue
                    end_pos = start_pos.copy()
                    d1 = rotate_around_z(d0, ang)
                    end_tgt = start_pos + d1
                    # We need the object to become NOT visible after rotation
                    visible_after = is_instance_visible(end_pos, end_tgt, obj, aabbs_all, K, WIDTH, HEIGHT)
                    if visible_after:
                        # still visible -> not acceptable
                        continue
                    # accept this rotation
                    chosen_ang = float(ang)
                    chosen_end = {'position': end_pos.tolist(), 'target': end_tgt.tolist()}
                    break

                if chosen_end is None:
                    if verbose:
                        print(f"[skip] no rotation that hides object for scene={scene_path.name} pose_index={pi} obj={obj.label}")
                    continue

                end_pos = np.array(chosen_end['position'], dtype=float)
                end_tgt = np.array(chosen_end['target'], dtype=float)

                # classify side relative to end camera forward
                side = classify_relative_side(end_pos, end_tgt, np.array(obj.position, dtype=float))

                # build choices and answer (English labels only)
                choices = ['Left', 'Right', 'LeftBack', 'RightBack']
                rng.shuffle(choices)
                labels_ABC = ['A', 'B', 'C', 'D'][:len(choices)]
                # side is one of the internal keys 'Left','Right','LeftBack','RightBack'
                if side in choices:
                    correct_idx = choices.index(side)
                else:
                    correct_idx = 0
                answer = labels_ABC[correct_idx]

                # prepare render metadata
                camtworld_start = camtoworld_from_pos_target(start_pos, start_tgt)
                camtworld_end = camtoworld_from_pos_target(end_pos, end_tgt)
                render_start = create_intrinsics()
                render_start['camtoworld'] = camtworld_start.tolist()
                render_end = create_intrinsics()
                render_end['camtoworld'] = camtworld_end.tolist()

                # Use the disambiguated (unique) label in the question so humans see which
                # instance we refer to when multiple objects share the same category.
                target_label_text = getattr(obj, 'unique_label', getattr(obj, 'label', 'the object'))
                question_text = f"After the rotation the {target_label_text} is on which side of the camera?"
                item = {
                    'qtype': 'object_after_rotation',
                    'scene': str(scene_path),
                    'question': question_text,
                    'choices': choices,
                    'answer': answer,
                    'meta': {
                        'begin_pos': start_pos.tolist(),
                        'begin_target': start_tgt.tolist(),
                        'begin_render': render_start,
                        'end_pos': end_pos.tolist(),
                        'end_target': end_tgt.tolist(),
                        'end_render': render_end,
                            'target_object_id': getattr(obj, 'id', None),
                            'target_label': getattr(obj, 'unique_label', getattr(obj, 'label', None)),
                        'rotation_deg': float(chosen_ang),
                        'in_place_rotation': True,
                        'end_sees_target': False,
                        'choices_map': choices,
                    }
                }

                items.append(item)
                per_scene_count += 1

                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    fname_base = f"{scene_path.name}_item_{idx:04d}_object_after_rotation_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta.json
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render begin/end thumbnails and preview/view
                    img_begin = render_thumbnail_for_pose(scene_path, {'position': start_pos, 'target': start_tgt}, thumb_size=WIDTH)
                    img_end = render_thumbnail_for_pose(scene_path, {'position': end_pos, 'target': end_tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'begin.png'), img_begin)
                    imageio.imwrite(str(item_dir / 'end.png'), img_end)

                    # preview (composed) and view map
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                if len(items) >= max_items:
                    break

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate object_after_rotation_mca QA items using sampler views.')
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
    print(f"Wrote {len(items)} object_after_rotation_mca items to {out_path}")


if __name__ == '__main__':
    main()
