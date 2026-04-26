#!/usr/bin/env python3
"""
Generate `middle_frame_mca` items using sampler-style view sampling.

Question: Which frame is the correct middle frame between Image 1 and Image 3?
 - Image1 -> Image2 is one action, Image2 -> Image3 is another action.
 - The two actions must be in the same direction family (translation forward-family or rotation same-sign)
 - Actions must be continuous and not cancelling each other.

Per-item outputs (when --out-dir provided):
  - meta.json
  - image1.png (start)
  - image2.png (ground-truth middle)
  - image3.png (end)
  - A.png, B.png, C.png, D.png (candidate middle frames)
  - preview.png (composed preview)
  - view.png (top-down view)

Run example:
  python -m Data_generation.sampler.question_generator.middle_frame_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/middle_frame.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/middle_frame_items \
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

# sampler helpers
from ..generate_view import SceneObject, generate_camera_positions

# preview composer
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


def is_inside_any_aabb(pos: np.ndarray, aabbs) -> bool:
    q = np.array(pos, dtype=float)
    for b in aabbs:
        if (q[0] >= b.bmin[0] and q[0] <= b.bmax[0] and
            q[1] >= b.bmin[1] and q[1] <= b.bmax[1] and
            q[2] >= b.bmin[2] and q[2] <= b.bmax[2]):
            return True
    return False


def scene_bounds(aabbs_all) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array([1e9, 1e9, 1e9], dtype=float)
    maxs = np.array([-1e9, -1e9, -1e9], dtype=float)
    for b in aabbs_all:
        mins = np.minimum(mins, np.array(b.bmin, dtype=float))
        maxs = np.maximum(maxs, np.array(b.bmax, dtype=float))
    return mins, maxs


def clamp_within_bounds(pos: np.ndarray, mins: np.ndarray, maxs: np.ndarray, pad: float = 0.5) -> bool:
    # Return True if pos within expanded bounds
    return (pos[0] >= mins[0] - pad and pos[0] <= maxs[0] + pad and
            pos[1] >= mins[1] - pad and pos[1] <= maxs[1] + pad and
            pos[2] >= mins[2] - pad and pos[2] <= maxs[2] + pad)


def apply_action_to_pose(pos: np.ndarray, tgt: np.ndarray, action: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an action to camera pose and return new (pos,tgt).
    Actions are either {'type':'move','forward':d,'side':s} in meters (side positive = right)
    or {'type':'turn','yaw_deg':y} rotate target around camera position by yaw_deg.
    """
    pos = np.array(pos, dtype=float)
    tgt = np.array(tgt, dtype=float)
    forward = unit(tgt - pos)
    # world up z positive
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right = right / (np.linalg.norm(right) + 1e-9)

    if action['type'] == 'move':
        f = float(action.get('forward', 0.0))
        s = float(action.get('side', 0.0))
        dp = forward * f + right * s
        new_pos = pos + dp
        new_tgt = tgt + dp
        return new_pos, new_tgt
    elif action['type'] == 'turn':
        yaw = float(action.get('yaw_deg', 0.0))
        # rotate target around pos by yaw about Z
        v = tgt - pos
        rad = math.radians(yaw)
        c = math.cos(rad); s = math.sin(rad)
        x, y, z = v[0], v[1], v[2]
        xr = c * x - s * y
        yr = s * x + c * y
        new_tgt = pos + np.array([xr, yr, z], dtype=float)
        return pos.copy(), new_tgt
    else:
        return pos.copy(), tgt.copy()


def is_pose_valid(pos: np.ndarray, tgt: np.ndarray, aabbs_all, mins, maxs, K) -> bool:
    # 1) camera not inside any object
    if is_inside_any_aabb(pos, aabbs_all):
        return False
    # 2) inside scene bounds
    if not clamp_within_bounds(pos, mins, maxs, pad=0.5):
        return False
    # 3) target coordinates finite
    if not np.isfinite(pos).all() or not np.isfinite(tgt).all():
        return False
    return True


def is_instance_visible_local(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    # same logic as other generators
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 2:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(width * height)
    return target_ratio > 0.01


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 8,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(1234)
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
        mins, maxs = scene_bounds(aabbs_all)

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

                p1_pos = np.array(p['position'], dtype=float)
                p1_tgt = np.array(p['target'], dtype=float)

                # require target visible at p1
                K = np.array(create_intrinsics()['K'], dtype=float)
                if not is_instance_visible_local(p1_pos, p1_tgt, obj, aabbs_all, K, WIDTH, HEIGHT):
                    if verbose:
                        print(f"[skip] target not fully visible at start for scene={scene_path.name} pose_index={pi}")
                    continue

                # New logic per requirement:
                # - From Image1, move to four directions (forward/backward/left/right) to get 4 middle frames.
                # - Pick one direction as the gold path; from that middle, continue the same-direction move to get Image3.
                # - The four middle frames are A/B/C/D options; only one is correct (the gold direction's middle).

                success = False
                # Try a few step distances to robustly get 4 valid/visible middles
                step_candidates = [0.25, 0.3, 0.35, 0.4]
                for attempt in range(12):
                    if success:
                        break
                    step = step_candidates[min(attempt, len(step_candidates)-1)]

                    # define four move actions from start
                    actions_4 = [
                        {'type': 'move', 'forward': step, 'side': 0.0, 'name': 'forward'},
                        {'type': 'move', 'forward': -step, 'side': 0.0, 'name': 'backward'},
                        {'type': 'move', 'forward': 0.0, 'side': step, 'name': 'right'},
                        {'type': 'move', 'forward': 0.0, 'side': -step, 'name': 'left'},
                    ]

                    middles: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
                    for a in actions_4:
                        m_pos, m_tgt = apply_action_to_pose(p1_pos, p1_tgt, a)
                        if not is_pose_valid(m_pos, m_tgt, aabbs_all, mins, maxs, K):
                            continue
                        if not is_instance_visible_local(m_pos, m_tgt, obj, aabbs_all, K, WIDTH, HEIGHT):
                            continue
                        middles.append((m_pos, m_tgt, a))

                    if len(middles) < 4:
                        # not all directions produce valid/visible middles; try another step
                        continue

                    # choose one direction as gold
                    gold_idx = rng.randrange(4)
                    p2_pos, p2_tgt, a1 = middles[gold_idx]
                    # continue same-direction move to get Image3
                    a2 = {'type': 'move', 'forward': a1.get('forward', 0.0), 'side': a1.get('side', 0.0), 'name': a1.get('name', 'same')}
                    p3_pos, p3_tgt = apply_action_to_pose(p2_pos, p2_tgt, a2)

                    if not is_pose_valid(p3_pos, p3_tgt, aabbs_all, mins, maxs, K):
                        # try another gold choice or step
                        continue

                    # Build choices from the four middles
                    choices_frames = [(m[0], m[1]) for m in middles]
                    dir_names = [m[2].get('name','') for m in middles]
                    # shuffle to remove positional bias
                    indices = list(range(4))
                    rng.shuffle(indices)
                    labels_ABC = ['A', 'B', 'C', 'D']
                    choices_render_cfgs: List[Dict[str, Any]] = []
                    choices_texts: List[str] = []
                    answer_letter = 'A'
                    for ci, idx_choice in enumerate(indices):
                        cf_pos, cf_tgt = choices_frames[idx_choice]
                        choices_render_cfgs.append({'position': cf_pos.tolist(), 'target': cf_tgt.tolist()})
                        choices_texts.append(dir_names[idx_choice])
                        if idx_choice == gold_idx:
                            answer_letter = labels_ABC[ci]

                    # Build item
                    render_start = create_intrinsics()
                    render_start['camtoworld'] = camtoworld_from_pos_target(p1_pos, p1_tgt).tolist()
                    render_mid = create_intrinsics()
                    render_mid['camtoworld'] = camtoworld_from_pos_target(p2_pos, p2_tgt).tolist()
                    render_end = create_intrinsics()
                    render_end['camtoworld'] = camtoworld_from_pos_target(p3_pos, p3_tgt).tolist()

                    question_text = "Which frame is the correct middle frame between Image 1 and Image 3?"
                    item = {
                        'qtype': 'middle_frame',
                        'scene': str(scene_path),
                        'question': question_text,
                        'choices': labels_ABC,
                        'answer': answer_letter,
                        'meta': {
                            'image1': {'position': p1_pos.tolist(), 'target': p1_tgt.tolist(), 'render': render_start},
                            'image2': {'position': p2_pos.tolist(), 'target': p2_tgt.tolist(), 'render': render_mid},
                            'image3': {'position': p3_pos.tolist(), 'target': p3_tgt.tolist(), 'render': render_end},
                            'choices_map': choices_render_cfgs,
                            'choices_texts': choices_texts,
                            'target_object_id': getattr(obj, 'id', None),
                            'target_label': getattr(obj, 'label', None),
                            'action1': a1,
                            'action2': a2,
                            'step_meters': float(step),
                        }
                    }

                    # write files
                    items.append(item)
                    per_scene_count += 1

                    if out_dir is not None:
                        scene_out = out_dir
                        scene_out.mkdir(parents=True, exist_ok=True)
                        idx = len(items) - 1
                        fname_base = f"{scene_path.name}_item_{idx:04d}_middle_frame_mca"
                        item_dir = scene_out / fname_base
                        item_dir.mkdir(parents=True, exist_ok=True)
                        with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                            json.dump(item, mf, indent=2, ensure_ascii=False)

                        # render image1, image2 (ground truth), image3
                        img1 = render_thumbnail_for_pose(scene_path, {'position': p1_pos, 'target': p1_tgt}, thumb_size=WIDTH)
                        img2 = render_thumbnail_for_pose(scene_path, {'position': p2_pos, 'target': p2_tgt}, thumb_size=WIDTH)
                        img3 = render_thumbnail_for_pose(scene_path, {'position': p3_pos, 'target': p3_tgt}, thumb_size=WIDTH)
                        imageio.imwrite(str(item_dir / 'image1.png'), img1)
                        imageio.imwrite(str(item_dir / 'image2.png'), img2)
                        imageio.imwrite(str(item_dir / 'image3.png'), img3)

                        # render candidate choice images A/B/C/D
                        for ci, rc in enumerate(choices_render_cfgs):
                            imgc = render_thumbnail_for_pose(scene_path, rc, thumb_size=WIDTH)
                            imageio.imwrite(str(item_dir / f"{labels_ABC[ci]}.png"), imgc)

                        # preview and view
                        compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                        # compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                    success = True
                    break

                if success:
                    # one item produced for this pose
                    pass

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate middle_frame_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + images)')
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
    print(f"Wrote {len(items)} middle_frame_mca items to {out_path}")


if __name__ == '__main__':
    main()
