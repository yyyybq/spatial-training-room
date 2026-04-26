#!/usr/bin/env python3
"""
Generate `view_rotation_mca` items using sampler-style view sampling.

Description:
  - Sample legal start views using `generate_camera_positions`.
  - From a start view pick a visible object (prefer large projected area).
  - Sample end views targeting the same object and pick an end view that also
    sees the object and yields a meaningful yaw rotation relative to the start.
  - Compute the signed yaw rotation (degrees) of the camera from begin->end
    in the horizontal plane (positive = rotate to the right). The reported
    angle is clamped to [-100, +100].
  - Produce a 4-way MCQ asking: "By how many degrees has the camera rotated from left to right from begin to end?"

Per-item outputs (when --out-dir provided):
  - meta.json
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)

Run example:
  python -m Data_generation.sampler.question_generator.view_rotation_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/view_rot.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/view_rot_items \
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


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def signed_yaw_deg(start_pos: np.ndarray, start_tgt: np.ndarray, end_pos: np.ndarray, end_tgt: np.ndarray) -> float:
    """Compute signed yaw (degrees) of camera rotation from start->end.

    Projects forward vectors to XY plane and computes signed angle via atan2(det, dot).
    Positive means rotated to the left (counter-clockwise looking from above) relative to start.
    """
    f0 = unit(np.array(start_tgt) - np.array(start_pos))
    f1 = unit(np.array(end_tgt) - np.array(end_pos))
    # project to XY plane
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
    # convention: positive = rotate left (CCW). atan2(det,dot) yields a positive angle when v1 is
    # counter-clockwise from v0, which matches "positive = left" semantics, so return ang.
    angle_left = ang
    # clamp
    if angle_left > 100.0:
        angle_left = 100.0
    if angle_left < -100.0:
        angle_left = -100.0
    return angle_left


def is_instance_visible(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 4:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(width * height)
    return target_ratio > 0.01


def make_numeric_distractors(center: int) -> List[int]:
    """Create 3 integer distractors around center (in degrees), within [-100,100]."""
    deltas = [10, -10, 25, -25, 40, -40]
    out = []
    for d in deltas:
        v = center + d
        if v < -100 or v > 100:
            continue
        if v == center:
            continue
        out.append(int(round(v)))
        if len(out) >= 3:
            break
    # if not enough, fill with evenly spaced values
    cand = list(range(-90, 101, 30))
    for c in cand:
        if c == center or c in out:
            continue
        out.append(c)
        if len(out) >= 3:
            break
    return out[:3]


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

                # build visible list
                visible: List[SceneObject] = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                for oid, so in objects.items():
                    if so.label in BLACKLIST:
                        continue
                    if is_instance_visible(start_pos, start_tgt, so, aabbs_all, K, WIDTH, HEIGHT):
                        visible.append(so)

                if not visible:
                    if verbose:
                        print(f"[skip] no visible objects after filtering for scene={scene_path.name} pose_index={pi}")
                    continue

                # pick anchor object among visible (prefer largest projected area)
                area_scores = []
                for so in visible:
                    camtworld = camtoworld_from_pos_target(start_pos, start_tgt)
                    occ = occluded_area_on_image(start_pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    area_scores.append((so, float(occ.get('target_area_px', 0.0))))
                area_scores.sort(key=lambda x: -x[1])
                target_obj = area_scores[0][0]

                # Instead of moving the camera position, rotate in-place around Z (turn left/right).
                # Build a set of candidate yaw angles (degrees) to try and pick one that still
                # sees the target object. Positive = rotate to the right.
                candidate_angles = [15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 10, -10, 20, -20]
                rng.shuffle(candidate_angles)
                chosen_end = None
                chosen_angle = None

                # vector from camera to target
                d0 = np.array(start_tgt - start_pos, dtype=float)
                def rotate_around_z(vec: np.ndarray, deg: float) -> np.ndarray:
                    rad = math.radians(deg)
                    c = math.cos(rad); s = math.sin(rad)
                    x, y, z = vec[0], vec[1], vec[2]
                    xr = c * x - s * y
                    yr = s * x + c * y
                    return np.array([xr, yr, z], dtype=float)

                for ang_deg in candidate_angles:
                    if abs(ang_deg) < 4.0:
                        continue
                    end_pos = start_pos.copy()
                    d1 = rotate_around_z(d0, ang_deg)
                    end_tgt = start_pos + d1
                    # accept this rotation even if the object becomes partially occluded;
                    # user requested to drop the end-visibility requirement
                    chosen_end = {'position': end_pos.tolist(), 'target': end_tgt.tolist()}
                    # invert left/right: store negated angle so that positive values now
                    # indicate rotation to the left (user-requested flip of left/right).
                    chosen_angle = -float(ang_deg)
                    break

                if chosen_end is None:
                    if verbose:
                        print(f"[skip] no end pose with meaningful rotation for scene={scene_path.name} start_pose_index={pi}")
                    continue

                end_pos = np.array(chosen_end['position'], dtype=float)
                end_tgt = np.array(chosen_end['target'], dtype=float)

                # prepare choices: correct angle (rounded int) and 3 distractors
                true_angle = int(round(chosen_angle))
                distractors = make_numeric_distractors(true_angle)
                opts = [true_angle] + distractors[:3]
                rng.shuffle(opts)
                choices = [f"{int(v)}°" for v in opts]
                labels_ABC = ['A', 'B', 'C', 'D']
                answer = labels_ABC[choices.index(f"{true_angle}°")]

                # prepare render metadata
                camtworld_start = camtoworld_from_pos_target(start_pos, start_tgt)
                camtworld_end = camtoworld_from_pos_target(end_pos, end_tgt)
                render_start = create_intrinsics()
                render_start['camtoworld'] = camtworld_start.tolist()
                render_end = create_intrinsics()
                render_end['camtoworld'] = camtworld_end.tolist()

                question_text = "By how many degrees has the camera rotated from left to right from begin to end?"
                item = {
                    'qtype': 'view_rotation',
                    'scene': str(scene_path),
                    'question': question_text,
                    'choices': [],
                    'answer': answer,
                    'meta': {
                        'begin_pos': start_pos.tolist(),
                        'begin_target': start_tgt.tolist(),
                        'begin_render': render_start,
                        'current_pos': start_pos.tolist(),
                        'current_target': start_tgt.tolist(),
                        'current_render': render_start,
                        'camera_pos': start_pos.tolist(),
                        'camera_target': start_tgt.tolist(),
                        'end_pos': end_pos.tolist(),
                        'end_target': end_tgt.tolist(),
                        'end_render': render_end,
                        'target_object_id': getattr(target_obj, 'id', None),
                        'target_label': getattr(target_obj, 'label', None),
                        'rotation_deg': float(chosen_angle),
                        'choices_map': choices,
                    }
                }

                items.append(item)
                per_scene_count += 1

                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    fname_base = f"{scene_path.name}_item_{idx:04d}_view_rotation_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    img_begin = render_thumbnail_for_pose(scene_path, {'position': start_pos, 'target': start_tgt}, thumb_size=WIDTH)
                    img_end = render_thumbnail_for_pose(scene_path, {'position': end_pos, 'target': end_tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'begin.png'), img_begin)
                    imageio.imwrite(str(item_dir / 'end.png'), img_end)

                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)

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
    parser = argparse.ArgumentParser(description='Generate view_rotation_mca QA items using sampler views.')
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
    print(f"Wrote {len(items)} view_rotation_mca items to {out_path}")


if __name__ == '__main__':
    main()
