#!/usr/bin/env python3
"""
Generate `nearest_object_mca` items using sampler-style view sampling.

This generator samples camera poses per object using `generate_camera_positions` (same
helper used by batch_generate_views), then for each sampled pose identifies visible
objects and asks: "Which object is closest to the camera?" Distance is measured to
the object surface (nearest point on the object's AABB), not its center.

Per-item outputs (when --out-dir provided):
  - meta.json (contains camera pose, visible instances, and answer)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)

Run example:
  python -m Data_generation.sampler.question_generator.nearest_object_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/nearest_object.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/nearest_items \
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
from types import SimpleNamespace


def point_to_aabb_distance(p: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Return Euclidean distance from point p to axis-aligned box [bmin,bmax].
    If p is inside the box, distance is 0.0."""
    p = np.array(p, dtype=float)
    bmin = np.array(bmin, dtype=float)
    bmax = np.array(bmax, dtype=float)
    # closest point on box to p
    closest = np.minimum(np.maximum(p, bmin), bmax)
    return float(np.linalg.norm(p - closest))


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


def generate_nearest_from_view(view: Dict[str, Any], scene_path: str, rng: random.Random | None = None, verbose: bool = False) -> Dict[str, Any] | None:
    _rng = rng or random
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        if verbose:
            print(f"[skip] no visible objects for scene={scene_path}")
        return None

    # Gather aabbs and intrinsics used by occlusion tests
    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = WIDTH; height = HEIGHT
    c2w = camtoworld_from_pos_target(pos, tgt)

    candidates = []
    for so in visible:
        if so.label in BLACKLIST:
            continue
        bmin_val = getattr(so, 'bbox_min', None)
        if bmin_val is None:
            bmin_val = getattr(so, 'bmin', None)
        bmax_val = getattr(so, 'bbox_max', None)
        if bmax_val is None:
            bmax_val = getattr(so, 'bmax', None)
        if bmin_val is None or bmax_val is None:
            continue

        # create a lightweight wrapper with bmin/bmax attributes for occlusion checks
        wrapped = SimpleNamespace(bmin=np.array(bmin_val, dtype=float), bmax=np.array(bmax_val, dtype=float), id=getattr(so, 'id', None), label=getattr(so, 'label', ''))
        # occlusion check: skip if AABB fully occluded
        if is_aabb_occluded(pos, wrapped, aabbs_all, K=K, camtoworld=c2w, width=width, height=height):
            continue
        candidates.append((so, bmin_val, bmax_val))

    if len(candidates) < 1:
        if verbose:
            print(f"[skip] no candidates after occlusion for scene={scene_path}")
        return None

    # compute distance-to-surface for each candidate and pick the nearest
    dists = []
    for so, bmin_val, bmax_val in candidates:
        d = point_to_aabb_distance(pos, np.array(bmin_val, dtype=float), np.array(bmax_val, dtype=float))
        dists.append((so, float(d), bmin_val, bmax_val))
    dists.sort(key=lambda x: x[1])
    nearest = dists[0]
    nearest_obj = nearest[0]

    # Prepare visible_instances for preview (centers and labels).
    # To ensure preview's A/B/C/D markers correspond to the displayed choices,
    # we build `visible_instances` in the same order as `options` (choices).
    vis_by_label = {}
    for so, _, bmin_val, bmax_val in dists:
        center = 0.5 * (np.array(bmin_val, dtype=float) + np.array(bmax_val, dtype=float))
        lab = str(getattr(so, 'label', ''))
        if lab not in vis_by_label:
            vis_by_label[lab] = {'id': getattr(so, 'id', None), 'label': lab, 'center': center.tolist()}

    # Build visible_instances aligned with options order below (after options computed)
    # (placeholder list for now; will be filled after options determined)
    vis_instances = []

    # Build choices: nearest object's label + three other visible object labels (if available)
    unique_labels = []
    for v in vis_instances:
        lab = str(v.get('label',''))
        if lab not in unique_labels:
            unique_labels.append(lab)

    # ensure at least 4 choices by sampling from other objects in scene if needed
    choices_pool = unique_labels.copy()
    if str(nearest_obj.label) not in choices_pool:
        choices_pool.insert(0, str(nearest_obj.label))
    # remove duplicates, keep order
    seen = set(); choices_pool = [x for x in choices_pool if not (x in seen or seen.add(x))]

    # fill distractors
    distractors = [x for x in choices_pool if x != str(nearest_obj.label)]
    if len(distractors) < 3:
        # try to add labels from visible set again (allow duplicates if necessary)
        extra = [str(getattr(so, 'label','')) for so in visible if str(getattr(so, 'label','')) != str(nearest_obj.label)]
        for e in extra:
            if e not in distractors and e != str(nearest_obj.label):
                distractors.append(e)
            if len(distractors) >= 3:
                break

    distractors = distractors[:3]
    options = [str(nearest_obj.label)] + distractors
    # ensure 4 options
    while len(options) < 4:
        options.append(f"other_{len(options)}")

    _rng.shuffle(options)
    labels = ['A','B','C','D']
    answer = labels[options.index(str(nearest_obj.label))]

    camtworld = setup_camtoworld(pos, tgt)
    render = create_intrinsics()
    render['camtoworld'] = camtworld.tolist()

    # Now align vis_instances to the shuffled options so preview markers match choice labels
    choices_vis = []
    for opt in options:
        if opt in vis_by_label:
            choices_vis.append(vis_by_label[opt])
        else:
            # placeholder (no exact instance with that label found)
            choices_vis.append({'id': None, 'label': opt, 'center': [float('nan'), float('nan'), float('nan')]})

    item = {
        'qtype': 'nearest_object',
        'scene': str(scene_path),
        'question': 'Which object is closest to the camera (nearest surface)?',
        'choices': options,
        'answer': answer,
        'meta': {
            'camera_pos': pos.tolist(),
            'camera_target': tgt.tolist(),
            'render': render,
            'visible_instances': choices_vis,
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
                    # require at least 2 visible corners for nearest-object candidate
                    if corners_visible < 2:
                        continue
                    camtworld = camtoworld_from_pos_target(pos, tgt)
                    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
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

                item = generate_nearest_from_view(view, str(scene_path), rng=rng, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_nearest_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_nearest_object_mca"
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
    parser = argparse.ArgumentParser(description='Generate nearest_object_mca QA items using sampler views.')
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
    print(f"Wrote {len(items)} nearest_object_mca items to {out_path}")


if __name__ == '__main__':
    main()
