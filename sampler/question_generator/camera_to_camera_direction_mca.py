#!/usr/bin/env python3
"""
Generate `camera_to_camera_direction_mca` items using sampler-style view sampling.

Workflow:
 1. Sample legal start views using the sampler (`generate_camera_positions`).
 2. For a sampled start view, choose one object that is visible in that view.
 3. Sample end views that target the same object (using `generate_camera_positions` on that object)
    and pick an end view that also has the object visible. This guarantees both views
    contain the same object.
 4. Compute the relative camera-to-camera direction of the end view w.r.t. the start view
    (front/back/left/right/above/below and combinations like "front-left", "above-front-right", ...)
 5. Produce a 4-way multiple-choice question asking the relative direction.

Per-item outputs (when --out-dir provided):
    - meta.json
    - preview.png (composed preview using project's preview composer)

Run example:
  python -m Data_generation.sampler.question_generator.camera_to_camera_direction_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/c2c_dir.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/c2c_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import imageio

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer (no lazy import)
from ...bench_generation.preview import compose_preview_for_item, render_thumbnail_for_pose

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


# ----------------- small helpers -----------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def camera_local_axes(pos: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return forward, right, up axes for a camera given pos and target.

    Uses world-up = [0,0,1] as reference to build a stable right/up basis.
    forward points from pos -> tgt.
    """
    forward = tgt - pos
    forward = unit(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    # right = cross(forward, world_up)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        # forward nearly parallel to world_up; choose an alternative up
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, world_up)
        right = unit(right)
    else:
        right = right / rn
    up = np.cross(right, forward)
    up = unit(up)
    return forward, right, up


def compute_camera_to_camera_direction(start_pos: np.ndarray, start_tgt: np.ndarray, end_pos: np.ndarray) -> str:
    """Compute a human-readable direction of end_pos relative to start camera frame.

    Returns labels like 'front', 'left', 'front-right', 'above-front-left', etc.
    The rule: project vector d = end_pos - start_pos onto start camera axes
    (forward, right, up). Include an axis in the label if its normalized absolute
    component >= 0.30 (empirical). Order of components: vertical (above/below),
    forward/back, left/right. If none pass threshold, use the axis with largest magnitude.
    """
    d = end_pos - start_pos
    L = np.linalg.norm(d)
    if L < 1e-6:
        return 'same'
    f, r, u = camera_local_axes(start_pos, start_tgt)
    # components along axes (signed)
    cf = float(np.dot(d, f) / (L + 1e-12))
    cr = float(np.dot(d, r) / (L + 1e-12))
    cu = float(np.dot(d, u) / (L + 1e-12))

    abs_cf, abs_cr, abs_cu = abs(cf), abs(cr), abs(cu)
    threshold = 0.30
    parts: List[str] = []
    # vertical first
    if abs_cu >= threshold:
        parts.append('above' if cu > 0 else 'below')
    # forward/back
    if abs_cf >= threshold:
        parts.append('front' if cf > 0 else 'back')
    # left/right
    if abs_cr >= threshold:
        parts.append('right' if cr > 0 else 'left')

    if not parts:
        # pick the largest component
        comps = [('front' if cf > 0 else 'back', abs_cf), ('right' if cr > 0 else 'left', abs_cr), ('above' if cu > 0 else 'below', abs_cu)]
        comps_sorted = sorted(comps, key=lambda x: -x[1])
        parts = [comps_sorted[0][0]]

    # normalize order: vertical, forward/back, left/right
    order = ['above', 'below', 'front', 'back', 'left', 'right']
    parts_sorted = sorted(parts, key=lambda x: order.index(x) if x in order else 999)
    return '-'.join(parts_sorted)


def make_distractors_for_direction(correct: str) -> List[str]:
    """Given a correct direction label, return up to 3 plausible distractors.

    Heuristics: flip one component at a time, prefer single-axis variants, then other
    orthogonal axes.
    """
    axes = {
        'front': 'back', 'back': 'front',
        'left': 'right', 'right': 'left',
        'above': 'below', 'below': 'above'
    }
    all_base = ['front','back','left','right','above','below']
    parts = correct.split('-') if correct else []
    candidates = []
    # flip each component individually
    for i, p in enumerate(parts):
        if p in axes:
            flipped = parts.copy()
            flipped[i] = axes[p]
            candidates.append('-'.join(sorted(flipped, key=lambda x: all_base.index(x) if x in all_base else 999)))
    # add single-axis neighbors (components of the correct label)
    for p in parts:
        if p not in candidates:
            candidates.append(p)
    # add orthogonal single axes
    for b in all_base:
        if b not in parts and b not in candidates:
            candidates.append(b)
        if len(candidates) >= 10:
            break

    # remove correct itself
    candidates = [c for c in candidates if c != correct]
    # return first 3 unique
    seen = set()
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= 3:
            break
    # fallback: pick arbitrary other labels
    if len(out) < 3:
        for b in all_base:
            if b == correct or b in out:
                continue
            out.append(b)
            if len(out) >= 3:
                break
    return out


# ----------------- generation logic -----------------
def is_instance_visible(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 4:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(width * height)
    return target_ratio > 0.01


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(202401)
    items: List[Dict[str, Any]] = []
    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]
    total_scenes = len(scenes)

    def _print_progress(idx: int, total: int, scene_name: str, prefix: str = 'Processing') -> None:
        """Print a short inline progress bar for scene processing.

        Uses carriage return to update the same terminal line.
        """
        try:
            width = 30
            frac = (idx + 1) / total if total > 0 else 1.0
            filled = int(frac * width)
            bar = '█' * filled + '-' * (width - filled)
            print(f"{prefix} {scene_name} [{idx+1}/{total}] [{bar}] {int(frac*100):3d}%", end='\r', flush=True)
        except Exception:
            # fallback to simple progress if any terminal encoding issues occur
            print(f"{prefix} {scene_name} [{idx+1}/{total}] {int(frac*100):3d}%", end='\r', flush=True)

    for idx, scene_path in enumerate(scenes):
        if len(items) >= max_items:
            break
        _print_progress(idx, total_scenes, scene_path.name)

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

        # Precompute label -> list of object ids for sampling labels
        label_to_objs: Dict[str, List[SceneObject]] = {}
        for so in objects.values():
            label = str(so.label)
            label_to_objs.setdefault(label, []).append(so)

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

                # build visible list using occlusion + corner visibility checks
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

                # choose a visible object to anchor (prefer one with larger projected area)
                # compute projected area for ordering
                area_scores = []
                for so in visible:
                    camtworld = camtoworld_from_pos_target(start_pos, start_tgt)
                    occ = occluded_area_on_image(start_pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    area_scores.append((so, float(occ.get('target_area_px', 0.0))))
                area_scores.sort(key=lambda x: -x[1])
                target_obj = area_scores[0][0]

                # Now sample end poses that target the same object
                end_poses = generate_camera_positions(scene_path, target_obj, per_room_points=per_room_points, min_dist=min_dist, max_dist=max_dist)
                if not end_poses:
                    if verbose:
                        print(f"[skip] no end poses for target {target_obj.id} in scene {scene_path.name}")
                    continue

                chosen_end = None
                for q in end_poses:
                    end_pos = np.array(q['position'], dtype=float)
                    end_tgt = np.array(q['target'], dtype=float)
                    # require the target_obj to also be visible from end view
                    if is_instance_visible(end_pos, end_tgt, target_obj, aabbs_all, K, WIDTH, HEIGHT):
                        # ensure the two camera poses are not nearly identical
                        if np.linalg.norm(end_pos - start_pos) < 0.05:
                            continue
                        chosen_end = q
                        break

                if chosen_end is None:
                    if verbose:
                        print(f"[skip] no end pose that also sees the target for scene={scene_path.name} start_pose_index={pi}")
                    continue

                end_pos = np.array(chosen_end['position'], dtype=float)
                end_tgt = np.array(chosen_end['target'], dtype=float)

                # compute direction label: end relative to start
                dir_label = compute_camera_to_camera_direction(start_pos, start_tgt, end_pos)

                # prepare plausible distractors
                distractors = make_distractors_for_direction(dir_label)
                opts = [dir_label] + distractors[:3]
                rng.shuffle(opts)
                choices = [str(x) for x in opts]
                labels_ABC = ['A', 'B', 'C', 'D']
                answer = labels_ABC[choices.index(dir_label)]

                # prepare render metadata
                camtworld_start = camtoworld_from_pos_target(start_pos, start_tgt)
                camtworld_end = camtoworld_from_pos_target(end_pos, end_tgt)
                render_start = create_intrinsics()
                render_start['camtoworld'] = camtworld_start.tolist()
                render_end = create_intrinsics()
                render_end['camtoworld'] = camtworld_end.tolist()

                # Build item using frame_frame_action_mca layout so preview shows two thumbs
                question_text = "Which direction is the end view relative to the begin view?"
                item = {
                    'qtype': 'camera_to_camera_direction',
                    'scene': str(scene_path),
                    'question': question_text,
                    'choices': [],  # we display textual choices via choices_map in meta
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
                        'choices_map': choices,
                    }
                }

                # accept and record
                items.append(item)
                per_scene_count += 1

                # save per-item outputs if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    fname_base = f"{scene_path.name}_item_{idx:04d}_camera_to_camera_direction_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render begin and end thumbnails and save using WIDTH as thumb size
                    img_begin = render_thumbnail_for_pose(scene_path, {'position': start_pos, 'target': start_tgt}, thumb_size=WIDTH)
                    img_end = render_thumbnail_for_pose(scene_path, {'position': end_pos, 'target': end_tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'begin.png'), img_begin)
                    imageio.imwrite(str(item_dir / 'end.png'), img_end)

                    # compose preview using preview utilities (thumb_size uses WIDTH)
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)

                if len(items) >= max_items:
                    break

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
    parser = argparse.ArgumentParser(description='Generate camera_to_camera_direction_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=False, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    parser.add_argument('--min_cam_dist_to_wall', type=float, default=0.0, help='Minimum camera distance to room walls (meters); requires structure.json')
    parser.add_argument('--verbose', action='store_true', help='Print debug/skip reasons during generation')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render, verbose=args.verbose)
    print(f"Wrote {len(items)} camera_to_camera_direction_mca items to {out_path}")


if __name__ == '__main__':
    main()
