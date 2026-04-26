#!/usr/bin/env python3
"""
Generate `next_frame_mca` items using sampler-style view sampling.

This generator samples camera poses (using the same helper as
`sampler.batch_generate_views`) and for each legal pose simulates four
discrete actions using `ViewManipulator`: move_forward, move_backward,
turn_left, turn_right. The action that results in the largest visible
area of a chosen target object is considered the correct answer.

Per-item outputs (when --out-dir provided):
  - meta.json (contains camera poses for current view and choices A-D)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)
  - A.png, B.png, C.png, D.png (thumbnails for each action)

Usage example:
  python -m Data_generation.sampler.question_generator.next_frame_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/next_frame.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/next_frame_items \
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
from ...bench_generation.preview import compose_preview_for_item, compose_view_map, render_thumbnail_for_pose

# shared helpers/constants (use WIDTH as thumb size)
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


# ------------------------- copied helpers -------------------------
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
            return True
    return False


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


# ------------------------- generation logic -------------------------
def simulate_action_and_measure(scene_path: str, c2w: np.ndarray, action: str, target_obj, aabbs_all, K, width, height) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Simulate action using ViewManipulator starting from c2w (camera-to-world).
    Returns (new_c2w, target_area_px, cam_pos, cam_target).
    """
    vm = ViewManipulator()
    vm.reset(c2w)
    # map action names to discrete keys used by ViewManipulator.step
    amap = {
        'move_forward': 'w',
        'move_backward': 's',
        # map turn_left -> 'q', turn_right -> 'e' so action names match visual direction
        # 好像和视觉是反的
        'turn_left': 'e',
        'turn_right': 'q',
    }
    if action not in amap:
        raise ValueError(f"Unsupported action: {action}")
    new_c2w = vm.step(amap[action])
    cam_pos = new_c2w[:3, 3].astype(float)
    # forward vector in camera-to-world is column 2
    forward = np.array(new_c2w[:3, 2], dtype=float)
    cam_tgt = cam_pos + forward

    # measure target visible area using occlusion helper
    occ = occluded_area_on_image(cam_pos, np.array(target_obj.bbox_min), np.array(target_obj.bbox_max), aabbs_all, np.array(K, dtype=float), new_c2w, width, height, target_id=target_obj.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    return new_c2w, target_px, cam_pos, cam_tgt


def generate_next_frame_from_view(view: Dict[str, Any], scene_path: str, seed: int | None = None, rng: random.Random | None = None, verbose: bool = False) -> Dict[str, Any] | None:
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        if verbose:
            print(f"[skip] no visible objects for scene={scene_path}")
        return None

    # choose a target object from visible set (prefer the originally sampled object if present)
    target = None
    for v in visible:
        if getattr(v, 'id', None) == view.get('pose_meta', {}).get('target_obj_id'):
            target = v
            break
    if target is None:
        target = _rng.choice(visible)

    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT

    # starting camera-to-world
    start_c2w = camtoworld_from_pos_target(pos, tgt)

    actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
    results = []  # list of tuples (action, new_c2w, target_px, cam_pos, cam_tgt)
    for a in actions:
        new_c2w, target_px, cam_pos, cam_tgt = simulate_action_and_measure(scene_path, np.array(start_c2w, dtype=float), a, target, aabbs_all, K, width, height)
        # check legality: camera not inside scene geometry
        illegal = is_pos_inside_scene(scene_path, cam_pos)
        if illegal:
            # mark as very low score so it won't be chosen as correct
            target_px = -1.0
        results.append((a, new_c2w, float(target_px), cam_pos, cam_tgt))

    # choose the action with maximum target_px
    results.sort(key=lambda x: x[2], reverse=True)
    if results[0][2] <= 0:
        if verbose:
            print(f"[skip] no action yields positive visibility for target in scene={scene_path}")
        return None

    correct_action = results[0][0]

    # prepare choices format expected by preview composer: list of dicts with position/target/render
    choices = []
    choice_labels = []
    render_cfgs = []
    for (a_name, c2w_mat, score, cam_pos, cam_tgt) in results:
        render_cfg = create_intrinsics()
        render_cfg['camtoworld'] = c2w_mat.tolist()
        choices.append({'position': cam_pos.tolist(), 'target': cam_tgt.tolist(), 'render': render_cfg})
        choice_labels.append(a_name)

    # choices are sorted by target_px descending; map to letters A..D
    correct_idx = 0
    labels = ['A', 'B', 'C', 'D']
    correct_label = labels[correct_idx]

    # prepare item
    render = create_intrinsics()
    # ensure camtoworld is JSON-serializable (convert ndarray -> list)
    render['camtoworld'] = np.array(start_c2w).tolist()

    # Phrase the question to include the chosen/correct action
    question_text = f"Given the current view and the action '{correct_action}', which candidate view is the next most likely?"

    item = {
        'qtype': 'next_frame',
        'scene': str(scene_path),
        'seed': seed,
        'question': question_text,
        'choices': choices,
        'answer': correct_label,
        'meta': {
            # include both 'current_pos'/'current_target' and keys expected by compose_view_map
            'current_pos': pos.tolist(),
            'current_target': tgt.tolist(),
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist(),
            'current_render': render,
            'target_object_id': getattr(target, 'id', None),
            'target_label': getattr(target, 'label', None),
            'choices_map': choice_labels,  # action names in order of A..D
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
                # attach target_obj_id for generate_next_frame_from_view to prefer
                view['pose_meta']['target_obj_id'] = obj_id

                item = generate_next_frame_from_view(view, str(scene_path), seed=None, rng=rng, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_next_frame_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_next_frame_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render thumbnails for each choice and save A/B/C/D.png
                    for ci, c in enumerate(item['choices']):
                        pos_c = np.array(c['position'], dtype=float)
                        tgt_c = np.array(c['target'], dtype=float)
                        # reuse render_thumbnail_for_pose
                        img = render_thumbnail_for_pose(scene_path, {'position': pos_c, 'target': tgt_c}, thumb_size=WIDTH)
                        img_path = item_dir / (chr(ord('A') + ci) + '.png')
                        imageio.imwrite(str(img_path), img)

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
    parser = argparse.ArgumentParser(description='Generate next_frame_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png + A.png..D.png)')
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
    print(f"Wrote {len(items)} next_frame_mca items to {out_path}")


if __name__ == '__main__':
    main()
