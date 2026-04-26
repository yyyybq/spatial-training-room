#!/usr/bin/env python3
"""
Generate `frame_frame_action_mca` items using sampler-style view sampling.

This generator samples legal begin-views (using the same sampler helper as
`sampler.batch_generate_views` / `generate_camera_positions`). For each begin
view it simulates four discrete actions using `ViewManipulator`:
    - move_forward
    - move_backward
    - turn_left
    - turn_right

We pick one action as the ground-truth, compute the resulting end view, and
produce a multiple-choice question: given the begin and end views, which action
most likely produced the transition?

Per-item outputs (when --out-dir provided):
  - meta.json (contains begin/end poses and choice metadata)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)


Run example:
  python -m Data_generation.sampler.question_generator.frame_frame_action_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/frame_frame_action.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/frame_frame_items \
    --per_room_points 6 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np
import math

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

# motion/controller for discrete actions
from ...motion.view_manipulator import ViewManipulator


# ------------------------- small copied helpers -------------------------
def load_room_polys(scene_path: str) -> List[np.ndarray]:
    p = Path(scene_path) / 'structure.json'
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rooms = data.get('rooms', [])
    polys: List[np.ndarray] = []
    for r in rooms:
        profile = r.get('profile')
        if not profile or len(profile) < 3:
            continue
        arr = np.array(profile, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        polys.append(arr[:, :2])
    return polys


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


def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """2D point to segment distance."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    if c == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)


def distance_point_to_polygon(px: float, py: float, poly: np.ndarray) -> float:
    """Minimum distance from point to polygon edges (2D)."""
    if poly is None or len(poly) < 2:
        return float('inf')
    min_d = float('inf')
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i, 0]), float(poly[i, 1])
        bx, by = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        d = point_to_segment_dist(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
    return float(min_d)


def is_pos_inside_scene(scene_path: str, pos: np.ndarray, tol: float = 0.01) -> bool:
    aabbs = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    q = np.array(pos, dtype=float)
    for b in aabbs:
        if (q[0] >= b.bmin[0] - tol and q[0] <= b.bmax[0] + tol and
            q[1] >= b.bmin[1] - tol and q[1] <= b.bmax[1] + tol and
            q[2] >= b.bmin[2] - tol and q[2] <= b.bmax[2] + tol):
            return True
    return False


# ------------------------- generation logic -------------------------
def simulate_action(scene_path: str, c2w: np.ndarray, action: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a discrete action via ViewManipulator and return (new_c2w, cam_pos, cam_tgt)."""
    vm = ViewManipulator()
    vm.reset(c2w)
    amap = {
        'move_forward': 'w',
        'move_backward': 's',
        'turn_left': 'e',
        'turn_right': 'q',
    }
    if action not in amap:
        raise ValueError(f"Unsupported action: {action}")
    new_c2w = vm.step(amap[action])
    cam_pos = new_c2w[:3, 3].astype(float)
    forward = np.array(new_c2w[:3, 2], dtype=float)
    cam_tgt = cam_pos + forward
    return new_c2w, cam_pos, cam_tgt


def generate_frame_frame_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None, verbose: bool = False) -> Dict[str, Any] | None:
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        if verbose:
            print(f"[skip] no visible objects for scene={scene_path}")
        return None

    # prefer the originally sampled object when present
    target = None
    for v in visible:
        if getattr(v, 'id', None) == view.get('pose_meta', {}).get('target_obj_id'):
            target = v
            break
    if target is None:
        target = _rng.choice(visible)

    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)

    # starting camera-to-world
    start_c2w = camtoworld_from_pos_target(pos, tgt)

    actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right']

    # choose a ground-truth action and compute end view
    gt_action = _rng.choice(actions)
    gt_c2w, gt_cam_pos, gt_cam_tgt = simulate_action(scene_path, np.array(start_c2w, dtype=float), gt_action)

    # Additional checks: camera must be inside a room polygon and be away from walls
    room_polys = load_room_polys(scene_path)
    min_cam_dist_to_wall = 0.0
    # try to read threshold from outer scope args if present (will be passed in via outer parser)
    try:
        # when called from iterate_and_generate we can read a global var; fallback to 0.0
        min_cam_dist_to_wall = float(view.get('pose_meta', {}).get('min_cam_dist_to_wall', 0.0))
    except Exception:
        min_cam_dist_to_wall = 0.0

    illegal_gt = False
    if room_polys:
        # find start room and end room
        start_room = None
        end_room = None
        for ri, poly in enumerate(room_polys):
            if point_in_poly(pos[0], pos[1], poly):
                start_room = ri
            if point_in_poly(float(gt_cam_pos[0]), float(gt_cam_pos[1]), poly):
                end_room = ri
        if start_room is None or end_room is None or start_room != end_room:
            illegal_gt = True
            if verbose:
                print(f"[skip] camera not inside same room for scene={scene_path} start_room={start_room} end_room={end_room}")
        else:
            # check distance to wall for end_cam_pos
            d_wall = distance_point_to_polygon(float(gt_cam_pos[0]), float(gt_cam_pos[1]), room_polys[end_room])
            if d_wall < min_cam_dist_to_wall:
                illegal_gt = True
                if verbose:
                    print(f"[skip] end camera too close to wall: {d_wall:.3f} < {min_cam_dist_to_wall} for scene={scene_path}")
    else:
        # no room polys available; fallback to only geometry-inside check
        illegal_gt = is_pos_inside_scene(scene_path, gt_cam_pos)

    if illegal_gt:
        return None

    # build choices by simulating all actions and measuring target visibility
    choices = []
    choice_labels = []
    K = np.array(create_intrinsics()['K'], dtype=float)

    for a in actions:
        new_c2w, cam_pos_c, cam_tgt_c = simulate_action(scene_path, np.array(start_c2w, dtype=float), a)
        camtoworld = new_c2w
        # compute occlusion/visible area for target
        occ = occluded_area_on_image(cam_pos_c, np.array(target.bbox_min), np.array(target.bbox_max), aabbs_all, K, camtoworld, WIDTH, HEIGHT, target_id=target.id, depth_mode='min', return_per_occluder=False)
        target_px = float(occ.get('target_area_px', 0.0))
        # mark illegal if camera inside scene or target not visible
        illegal = is_pos_inside_scene(scene_path, cam_pos_c)
        if illegal or target_px <= 0.0:
            # still include but mark score low by attaching score
            score = -1.0
        else:
            score = float(target_px)
        render_cfg = create_intrinsics()
        render_cfg['camtoworld'] = camtoworld.tolist()
        choices.append({'position': cam_pos_c.tolist(), 'target': cam_tgt_c.tolist(), 'render': render_cfg, 'score': score})
        choice_labels.append(a)

    # we map actions to A..D in the order defined in `actions`
    correct_idx = actions.index(gt_action)
    labels = ['A', 'B', 'C', 'D']
    correct_label = labels[correct_idx]

    # prepare item
    render = create_intrinsics()
    render['camtoworld'] = np.array(start_c2w).tolist()

    question_text = "Given the begin view (left) and the end view (right), which action most likely produced the transition?"

    item = {
        'qtype': 'frame_frame_action',
        'scene': str(scene_path),
        'seed': seed,
        'question': question_text,
        'choices': [{'position': c['position'], 'target': c['target'], 'render': c['render']} for c in choices],
        'answer': correct_label,
        'meta': {
            'begin_pos': pos.tolist(),
            'begin_target': tgt.tolist(),
            'begin_render': render,
            'current_pos': pos.tolist(),
            'current_target': tgt.tolist(),
            'current_render': render,
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist(),
            'end_pos': gt_cam_pos.tolist(),
            'end_target': gt_cam_tgt.tolist(),
            'end_render': {'camtoworld': gt_c2w.tolist()},
            'target_object_id': getattr(target, 'id', None),
            'target_label': getattr(target, 'label', None),
            'choices_map': choice_labels,
        }
    }
    return item


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False, min_cam_dist_to_wall: float = 0.0) -> List[Dict[str, Any]]:
    rng = random.Random(12345)
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
                # propagate min_cam_dist_to_wall into pose_meta for downstream checks
                view['pose_meta']['target_obj_id'] = obj_id
                view['pose_meta']['min_cam_dist_to_wall'] = float(min_cam_dist_to_wall)

                item = generate_frame_frame_from_view(view, str(scene_path), seed=None, rng=rng, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_frame_frame_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_frame_frame_action_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render begin/current and end thumbnails and save begin.png / end.png using WIDTH as thumb size
                    begin_pos = np.array(item['meta'].get('current_pos', item['meta'].get('begin_pos')), dtype=float)
                    begin_tgt = np.array(item['meta'].get('current_target', item['meta'].get('begin_target', begin_pos + np.array([0.0, 1.0, 0.0]))), dtype=float)
                    end_pos = np.array(item['meta'].get('end_pos'), dtype=float)
                    end_tgt = np.array(item['meta'].get('end_target', end_pos + np.array([0.0, 1.0, 0.0])), dtype=float)
                    img_begin = render_thumbnail_for_pose(scene_path, {'position': begin_pos, 'target': begin_tgt}, thumb_size=WIDTH)
                    img_end = render_thumbnail_for_pose(scene_path, {'position': end_pos, 'target': end_tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'begin.png'), img_begin)
                    imageio.imwrite(str(item_dir / 'end.png'), img_end)

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
    parser = argparse.ArgumentParser(description='Generate frame_frame_action_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png + A.png..C.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--min_cam_dist_to_wall', type=float, default=0.0, help='Minimum camera distance to room walls (meters); requires structure.json')
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    parser.add_argument('--verbose', action='store_true', help='Print debug/skip reasons during generation')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render, verbose=args.verbose, min_cam_dist_to_wall=args.min_cam_dist_to_wall)
    print(f"Wrote {len(items)} frame_frame_action_mca items to {out_path}")


if __name__ == '__main__':
    main()
